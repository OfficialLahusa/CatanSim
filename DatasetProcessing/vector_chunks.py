"""
Read the state vectors the dataset collector writes next to the YAML samples.

Each collector worker appends to chunk files in the dataset's `vectors` directory:
`w<worker>_c<chunk>.f16` holds one row of FEATURE_DIM little-endian float16 values per sample,
`w<worker>_c<chunk>.txt` lists the sample name (e.g. `r12_s3`) of each row, one per line.
The C# StateVectorizer and state_to_vector.py produce the same vectors (checked by the parity fixtures),
so these rows replace vectorizing the YAML files again.
"""

import glob
import os
import warnings
from typing import Dict

import numpy as np

from state_to_vector import FEATURE_DIM


def load_vector_chunks(dataset_dir: str) -> Dict[str, np.ndarray]:
    """Returns the vector of every sample in the dataset's chunk files by sample name (empty without chunks)."""
    vectors: Dict[str, np.ndarray] = {}

    for data_path in sorted(glob.glob(os.path.join(dataset_dir, "vectors", "*.f16"))):
        index_path = data_path[: -len(".f16")] + ".txt"

        values = np.fromfile(data_path, dtype="<f2")
        if values.size % FEATURE_DIM != 0:
            raise ValueError(f"\"{data_path}\" holds {values.size} values, not a whole number of rows of {FEATURE_DIM}.")
        rows = values.reshape(-1, FEATURE_DIM)

        with open(index_path, "r") as index_file:
            names = [line.strip() for line in index_file if line.strip()]

        # The collector writes a row before its name, so an interrupted run can leave rows without a name, never the other way round
        if len(names) > len(rows):
            raise ValueError(f"\"{index_path}\" lists {len(names)} samples, but \"{data_path}\" only holds {len(rows)} rows.")
        if len(rows) > len(names):
            warnings.warn(f"\"{data_path}\" holds {len(rows) - len(names)} row(s) without a sample name (interrupted run?), ignoring them.")

        for name, row in zip(names, rows):
            if name in vectors:
                raise ValueError(f"Sample \"{name}\" appears in more than one vector chunk of \"{dataset_dir}\".")
            vectors[name] = row

    return vectors
