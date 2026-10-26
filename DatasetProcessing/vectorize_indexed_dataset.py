"""
Vectorize an indexed dataset: states.npy (float16, one row of FEATURE_DIM per index entry), win_rates.npy and
groups.npy in the output directory. State vectors come from the collector's vector chunks where they exist, the rest
is vectorized from the YAML samples. Every label is checked before the long part starts.

Usage: python vectorize_indexed_dataset.py <index.csv> [--base-path .] [--workers 8] [--output-dir .]
"""
import argparse
import os
import sys
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from rich.progress import (
    track,
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from state_to_vector import yaml_to_numpy, FEATURE_DIM
from vector_chunks import load_vector_chunks
    
def load_index_file(index_path: str):
    print(f"Loading index from \"{os.path.realpath(index_path)}\".")

    index = []

    # Index aus CSV auslesen
    with open(index_path, "r") as index_file:
        # Header skippen
        index_file.readline()

        while line := index_file.readline():
            entry_idx, group_idx, subdir_name, entry_name = line.rstrip().split(",")
            entry_idx = int(entry_idx)
            group_idx = int(group_idx)

            index.append([entry_idx, group_idx, subdir_name, entry_name])

    return index


def _process_entry(args):
    # Worker Process: YAML Datei laden und vektorisieren
    entry_idx, base_path, subdir_name, entry_name = args
    yaml_file_path = base_path / subdir_name / "input" / f"{entry_name}.yaml"
    vec = yaml_to_numpy(str(yaml_file_path))  # float16
    return entry_idx, vec


def check_labels(base_path: Path, index):
    """Raises if a sample of the index has no TXT label, before anything is vectorized"""
    missing = [f"{subdir_name}/{entry_name}" for _, _, subdir_name, entry_name in index
               if not (base_path / subdir_name / "output" / (entry_name + ".txt")).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} sample(s) of the index have no TXT label, e.g. {missing[:5]}")


def vectorize_states(base_path: Path, index, max_workers=None, chunksize=64, output_dir: Path = Path(".")):
    # Die Matrix wird direkt in die Datei geschrieben (f16 <Index Length>,<Feature vector length>), damit große Datensätze nicht doppelt in den RAM müssen
    output_path = output_dir / "states.npy"
    state_matrix = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float16, shape=(len(index), FEATURE_DIM))

    # Vektoren, die der Collector schon als Chunks geschrieben hat, direkt übernehmen, die Chunks eines Datensatzes nach dem anderen
    entries_by_subdir = {}
    for entry in index:
        entries_by_subdir.setdefault(entry[2], []).append(entry)

    # Nur die übrigen State YAMLs (z.B. ältere Datensätze ohne Chunks oder mit Chunks eines älteren Layouts) multi-processed vektorisieren
    tasks = []
    outdated = 0
    for subdir_name, entries in entries_by_subdir.items():
        chunk_vectors = load_vector_chunks(str(base_path / subdir_name))
        for entry_idx, group_idx, _, entry_name in entries:
            vec = chunk_vectors.get(entry_name)
            if vec is not None and len(vec) == FEATURE_DIM:
                state_matrix[entry_idx, :] = vec
            else:
                outdated += vec is not None
                tasks.append((entry_idx, base_path, subdir_name, entry_name))
        del chunk_vectors

    print(f"{len(index) - len(tasks)} state vectors taken from vector chunks, {len(tasks)} to vectorize from YAML"
          f"{f' ({outdated} with chunks of an older layout)' if outdated else ''}.")
 
    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ]
 
    # Einmal parallel durch die übrigen Einträge gehen
    if tasks:
        with Progress(*progress_columns) as progress:
            task_id = progress.add_task("Processing Game States", total=len(tasks))

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for entry_idx, vec in executor.map(_process_entry, tasks, chunksize=chunksize):
                    # Vektor in Matrix schreiben
                    state_matrix[entry_idx, :] = vec
                    progress.advance(task_id)
 
    state_matrix.flush()
    del state_matrix
    print(f"Saved state matrix as \"{output_path}\".")
    return


def vectorize_win_rates(base_path: Path, index, output_dir: Path = Path(".")):
    # Win Percentages vektorisieren
    # NP Array erstellen: f16 <Index Length>,4
    win_rate_matrix = np.zeros((len(index), 4), dtype=np.float16)

    # Einmal durch den ganzen Index gehen
    for entry in track(index, description="Processing Win Probabilities"):
        entry_idx, group_idx, subdir_name, entry_name = entry
        txt_file_path = base_path / subdir_name / "output" / (entry_name + ".txt")

        # TXT Datei laden und vektorisieren
        win_rates = None
        with txt_file_path.open("r") as txt_file:
            win_rates = list(map(float, txt_file.read().rstrip().split("\n")))
            win_rates = np.array(win_rates, dtype=np.float16)
        # Vektor in Matrix schreiben
        win_rate_matrix[entry_idx, :] = win_rates

    # Array der Win-Percentage-Matrix als Datei speichern
    output_path = output_dir / "win_rates.npy"
    np.save(output_path, win_rate_matrix)
    # Speicherbestätigung ausgeben
    print(f"Saved win rate matrix as \"{output_path}\".")
    return


def vectorize_groups(base_path: Path, index, output_dir: Path = Path(".")):
    # Groups vektorisieren
    # NP Array erstellen: uint32 <Index Length>,
    group_vector = np.zeros((len(index),), dtype=np.uint32)

    # Einmal durch den ganzen Index gehen
    for entry in track(index, description="Processing Group Labels"):
        entry_idx, group_idx, subdir_name, entry_name = entry
        # Group value direkt in den Vektor schreiben
        group_vector[entry_idx] = group_idx
    
    # Array des Group-Vektors als Datei speichern
    output_path = output_dir / "groups.npy"
    np.save(output_path, group_vector)
    # Speicherbestätigung ausgeben
    print(f"Saved group labels as \"{output_path}\".")
    return


def vectorize_dataset(base_path: Path, index, max_workers=None, chunksize=64, output_dir: Path = Path(".")):
    output_dir.mkdir(parents=True, exist_ok=True)
    check_labels(base_path, index)
    vectorize_states(base_path, index, max_workers, chunksize, output_dir)
    vectorize_win_rates(base_path, index, output_dir)
    vectorize_groups(base_path, index, output_dir)

    # Finale Bestätigung ausgeben
    print("Dataset vectorization completed!")
    return


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Vectorize an indexed dataset into states.npy, win_rates.npy and groups.npy.")
    parser.add_argument("index", help="index CSV from build_dataset_index.py")
    parser.add_argument("--base-path", default=".", help="base path the index's dataset paths are relative to")
    parser.add_argument("--workers", type=int, default=8, help="processes for samples without vector chunks")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args(argv)

    index = load_index_file(args.index)
    num_groups = max(entry[1] for entry in index) + 1 if index else 0
    print(f"Loaded index file containing {len(index)} entries across {num_groups} groups and {len({e[2] for e in index})} datasets.")

    try:
        vectorize_dataset(Path(args.base_path), index, args.workers, output_dir=Path(args.output_dir))
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1

    for name in ("states", "win_rates", "groups"):
        array = np.load(Path(args.output_dir) / f"{name}.npy", mmap_mode="r")
        print(f"{name}.npy: shape {array.shape}, dtype {array.dtype}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
