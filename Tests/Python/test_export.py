"""ONNX export of the state value net: shapes, names and parity with PyTorch.

Skipped when torch, lightning or onnxruntime are not installed.
Run from the repository root:  python -m unittest discover -s Tests/Python -v
"""

import os
import sys
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "DatasetProcessing"))
sys.path.insert(0, os.path.join(REPO, "Agents", "Training"))

# conda's MKL and torch both ship an OpenMP runtime in Agents/env
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from state_to_vector import FEATURE_DIM  # noqa: E402

try:
    import numpy as np
    import onnxruntime as ort
    import torch
    from export_onnx import export_model
    from state_value_net import StateValueNet
except ImportError as error:  # pragma: no cover - depends on the environment
    IMPORT_ERROR = error
else:
    IMPORT_ERROR = None


@unittest.skipIf(IMPORT_ERROR is not None, f"training dependencies missing: {IMPORT_ERROR}")
class ExportTest(unittest.TestCase):
    def test_exported_model_matches_pytorch_and_the_csharp_contract(self):
        torch.manual_seed(0)
        model = StateValueNet(FEATURE_DIM).eval()

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state_value_net.onnx")
            max_diff = export_model(model, path, FEATURE_DIM)
            self.assertLess(max_diff, 1e-4)

            session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            (state_input,) = session.get_inputs()
            (logits_output,) = session.get_outputs()

            # The C# side feeds "state" [batch, FeatureDim] and reads 4 logits per row
            self.assertEqual("state", state_input.name)
            self.assertEqual(FEATURE_DIM, state_input.shape[1])
            self.assertEqual("logits", logits_output.name)
            self.assertEqual(4, logits_output.shape[1])

            batch = np.random.default_rng(0).standard_normal((7, FEATURE_DIM)).astype(np.float32)
            self.assertEqual((7, 4), session.run(["logits"], {"state": batch})[0].shape)


if __name__ == "__main__":
    unittest.main()
