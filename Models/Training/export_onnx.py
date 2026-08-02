import argparse
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import lightning as L
from state_value_net import StateValueModule


def export(ckpt_path: str, out_path: str, in_dim: int):
    # Lightning-Module aus Checkpoint laden
    module = StateValueModule.load_from_checkpoint(ckpt_path, in_dim=in_dim, map_location="cpu")

    # Dropout deaktivieren und freezen
    module.eval()

    # Kern-Modell statt Lightning-Module für ONNX-Export nötig
    model = module.model

    dummy = torch.randn(1, in_dim)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["state"], output_names=["logits"],
        dynamic_axes={"state": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    print(f"Exported to {out_path}")

    # Paritätscheck zwischen Torch und ONNX auf Random Inputs
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(out_path, sess_options=so, providers=["CPUExecutionProvider"])

    x_np = np.random.randn(16, in_dim).astype(np.float32)
    with torch.no_grad():
        torch_out = model(torch.from_numpy(x_np)).numpy()
    onnx_out = sess.run(["logits"], {"state": x_np})[0]

    max_diff = np.abs(torch_out - onnx_out).max()
    print(f"Max abs diff, PyTorch vs ONNX (batch=16): {max_diff:.2e}")
    assert max_diff < 1e-4, "ONNX export does not match PyTorch output -- do not ship this .onnx file"
    print("Parity check passed.")

    # Und nochmal mit batch=1 wegen Relevanz für MCTS
    x1 = np.random.randn(1, in_dim).astype(np.float32)
    with torch.no_grad():
        torch_out1 = model(torch.from_numpy(x1)).numpy()
    onnx_out1 = sess.run(["logits"], {"state": x1})[0]
    print(f"Max abs diff, batch=1: {np.abs(torch_out1 - onnx_out1).max():.2e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="state_value_net.onnx")
    parser.add_argument("--in-dim", type=int, default=1611)
    args = parser.parse_args()
    export(args.ckpt, args.out, args.in_dim)