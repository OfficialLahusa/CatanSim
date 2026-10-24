"""
Read the state vectors the dataset collector writes next to the YAML samples.

Each collector worker appends to chunk files in the dataset's `vectors` directory:
`w<worker>_c<chunk>.f16` holds one row of little-endian float16 values per sample,
`w<worker>_c<chunk>.txt` lists the sample name (e.g. `r12_s3`) of each row, one per line.
The C# StateVectorizer and state_to_vector.py produce the same vectors (checked by the parity fixtures),
so these rows replace vectorizing the YAML files again.

The row width is the collector's feature_dim from metadata.json (FEATURE_DIM without it). Chunks of the base
encoding (BASE_FEATURE_DIM, written before the derived features existed) are extended with the derived features;
chunks of any other width are ignored, so their samples get vectorized from the YAML files.
"""

import glob
import json
import os
import warnings
from typing import Dict

import numpy as np

from state_to_vector import BASE_FEATURE_DIM, FEATURE_DIM, derived_features


def chunk_feature_dim(dataset_dir: str) -> int:
    """Width of the dataset's vector rows, from the collector's metadata.json (FEATURE_DIM for older datasets without it)"""
    path = os.path.join(dataset_dir, "metadata.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as file:
            return int(json.load(file).get("feature_dim", FEATURE_DIM))
    return FEATURE_DIM


def load_vector_chunks(dataset_dir: str) -> Dict[str, np.ndarray]:
    """Returns the vector of every sample in the dataset's chunk files by sample name (empty without usable chunks)."""
    vectors: Dict[str, np.ndarray] = {}
    data_paths = sorted(glob.glob(os.path.join(dataset_dir, "vectors", "*.f16")))
    if not data_paths:
        return vectors

    width = chunk_feature_dim(dataset_dir)
    if width not in (FEATURE_DIM, BASE_FEATURE_DIM):
        warnings.warn(f"The vector chunks of \"{dataset_dir}\" have {width} features, the current encoding {FEATURE_DIM}: "
                      "ignoring them, the samples are vectorized from their YAML files.")
        return vectors

    for data_path in data_paths:
        index_path = data_path[: -len(".f16")] + ".txt"

        values = np.fromfile(data_path, dtype="<f2")
        if values.size % width != 0:
            raise ValueError(f"\"{data_path}\" holds {values.size} values, not a whole number of rows of {width}.")
        rows = values.reshape(-1, width)

        with open(index_path, "r") as index_file:
            names = [line.strip() for line in index_file if line.strip()]

        # The collector writes a row before its name, so an interrupted run can leave rows without a name, never the other way round
        if len(names) > len(rows):
            raise ValueError(f"\"{index_path}\" lists {len(names)} samples, but \"{data_path}\" only holds {len(rows)} rows.")
        if len(rows) > len(names):
            warnings.warn(f"\"{data_path}\" holds {len(rows) - len(names)} row(s) without a sample name (interrupted run?), ignoring them.")
        rows = rows[: len(names)]

        # Chunks of the base encoding get the derived features, computed from the base features like the vectorizers do
        if width == BASE_FEATURE_DIM:
            rows = np.concatenate([rows, derived_features(rows).astype(np.float16)], axis=1)

        for name, row in zip(names, rows):
            if name in vectors:
                raise ValueError(f"Sample \"{name}\" appears in more than one vector chunk of \"{dataset_dir}\".")
            vectors[name] = row

    return vectors
