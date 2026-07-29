import os
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


def vectorize_states(base_path: Path, index, max_workers=None, chunksize=64):
    # State YAMLs multi-processed vektorisieren
    tasks = [
        (entry_idx, base_path, subdir_name, entry_name)
        for entry_idx, group_idx, subdir_name, entry_name in index
    ]
 
    # NP Array erstellen: f16 <Index Length>,<Feature vector length>
    state_matrix = np.zeros((len(index), FEATURE_DIM), dtype=np.float16)
 
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
 
    # Einmal parallel durch den ganzen Index gehen
    with Progress(*progress_columns) as progress:
        task_id = progress.add_task("Processing Game States", total=len(tasks))
 
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for entry_idx, vec in executor.map(_process_entry, tasks, chunksize=chunksize):
                # Vektor in Matrix schreiben
                state_matrix[entry_idx, :] = vec
                progress.advance(task_id)
 
    # Array der State-Matrix als Datei speichern
    output_path = Path(".") / "states.npy"
    np.save(output_path, state_matrix)
    # Speicherbestätigung ausgeben
    print(f"Saved state matrix as \"{output_path}\".")
    return


def vectorize_win_rates(base_path: Path, index):
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
    output_path = Path(".") / "win_rates.npy"
    np.save(output_path, win_rate_matrix)
    # Speicherbestätigung ausgeben
    print(f"Saved win rate matrix as \"{output_path}\".")
    return


def vectorize_groups(base_path: Path, index):
    # Groups vektorisieren
    # NP Array erstellen: uint32 <Index Length>,
    group_vector = np.zeros((len(index),), dtype=np.uint32)

    # Einmal durch den ganzen Index gehen
    for entry in track(index, description="Processing Group Labels"):
        entry_idx, group_idx, subdir_name, entry_name = entry
        # Group value direkt in den Vektor schreiben
        group_vector[entry_idx] = group_idx
    
    # Array des Group-Vektors als Datei speichern
    output_path = Path(".") / "groups.npy"
    np.save(output_path, group_vector)
    # Speicherbestätigung ausgeben
    print(f"Saved group labels as \"{output_path}\".")
    return


def vectorize_dataset(base_path: Path, index, max_workers=None, chunksize=64):
    vectorize_states(base_path, index, max_workers, chunksize)
    vectorize_win_rates(base_path, index)
    vectorize_groups(base_path, index)

    # Finale Bestätigung ausgeben
    print("Dataset vectorization completed!")
    return


if __name__ == "__main__":
    index_path = input("Enter path of the index CSV file for the dataset.\n")
    index_path = index_path.strip()
    print()

    index = load_index_file(index_path)
    num_groups = max(index,key=lambda entry:entry[1])[1]+1
    subdirectories = []
    for entry in index:
        subdir_name = entry[2]
        if subdir_name not in subdirectories:
            subdirectories.append(subdir_name)

    print(f"Loaded index file containing {len(index)} entries across {num_groups} groups and {len(subdirectories)} subdirectories.\n")

    # Show which subdirectories are expected
    print("Dataset subdirectories expected by the index:")
    for subdirectory in subdirectories:
        print(f"\t- \"{subdirectory}\"")
    print()

    base_path = input("Enter base path of the dataset containing the subdirectories listed above. The immediate subdirectories should EACH contain an 'input' directory for YAML game states and an 'output' directory for TXT win rates). Default: Current\n")
    if base_path.strip() == "":
        base_path = "."
    base_path = Path(base_path)

    # Determine worker count
    worker_count = input("Worker count? Default: 8\n").strip()
    if worker_count == "":
        worker_count = 8
    else:
        worker_count = int(worker_count)

    input(f"Processing dataset at \"{os.path.abspath(base_path)}\" for vectorization. Press [Enter] to continue.\n")
    vectorize_dataset(base_path, index, worker_count)

    input("Output file verification:  Press [Enter] to continue.\n")
    state_matrix = np.load(Path(".") / "states.npy")
    print(f"State matrix: shape {state_matrix.shape}, dtype {state_matrix.dtype}")
    print(state_matrix)
    print()

    win_rate_matrix = np.load(Path(".") / "win_rates.npy")
    print(f"Win rate matrix: shape {win_rate_matrix.shape}, dtype {win_rate_matrix.dtype}")
    print(win_rate_matrix)
    print()

    groups_vector = np.load(Path(".") / "groups.npy")
    print(f"Group label vector: shape {groups_vector.shape}, dtype {groups_vector.dtype}")
    print(groups_vector)
    print()

    print("\nVerification done!")
