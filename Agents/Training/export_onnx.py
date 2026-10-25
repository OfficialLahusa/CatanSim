import argparse
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import lightning as L
from state_value_net import StateValueModule


class ExportedValueNet(nn.Module):
    """Outputs für den Export: die Value Logits und, bei Modellen mit Aux Heads, die Platzierungs-Logits [batch, 4 Spieler, 4 Plätze]"""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model.forward_all(x)
        return outputs["logits"], outputs["placements"]


def export(ckpt_path: str, out_path: str, in_dim: int):
    # Lightning-Module aus Checkpoint laden
    module = StateValueModule.load_from_checkpoint(ckpt_path, in_dim=in_dim, map_location="cpu")

    # Dropout deaktivieren und freezen
    module.eval()

    # Kern-Modell statt Lightning-Module für ONNX-Export nötig
    export_model(module.model, out_path, in_dim)


def export_model(model: nn.Module, out_path: str, in_dim: int) -> float:
    """Exportiert ein StateValueNet (im eval-Modus) nach ONNX und prüft die Parität.
    Modelle mit trainiertem Platzierungs-Head (--placement-weight > 0) bekommen einen zweiten Output "placements"
    (der C#-Code liest "logits" über den Namen); Aux Heads nur für andere Losses werden nicht exportiert.
    Gibt die maximale Abweichung zwischen PyTorch und ONNX zurück."""
    with_placements = getattr(model, "aux", None) is not None and getattr(model, "placements_trained", False)
    exported = ExportedValueNet(model).eval() if with_placements else model
    output_names = ["logits", "placements"] if with_placements else ["logits"]

    dummy = torch.randn(1, in_dim)
    torch.onnx.export(
        exported, dummy, out_path,
        input_names=["state"], output_names=output_names,
        dynamic_axes={"state": {0: "batch"}, **{name: {0: "batch"} for name in output_names}},
        opset_version=17, dynamo=False,
    )
    print(f"Exported to {out_path} (outputs: {', '.join(output_names)})")

    # Paritätscheck zwischen Torch und ONNX auf Random Inputs, für jeden Output
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(out_path, sess_options=so, providers=["CPUExecutionProvider"])

    def max_difference(batch: int) -> float:
        x_np = np.random.randn(batch, in_dim).astype(np.float32)
        with torch.no_grad():
            torch_out = exported(torch.from_numpy(x_np))
        torch_outputs = [t.numpy() for t in torch_out] if with_placements else [torch_out.numpy()]
        onnx_outputs = sess.run(output_names, {"state": x_np})
        return max(float(np.abs(t - o).max()) for t, o in zip(torch_outputs, onnx_outputs))

    max_diff = max_difference(16)
    print(f"Max abs diff, PyTorch vs ONNX (batch=16): {max_diff:.2e}")
    assert max_diff < 1e-4, "ONNX export does not match PyTorch output -- do not ship this .onnx file"
    print("Parity check passed.")

    # Und nochmal mit batch=1 wegen Relevanz für MCTS
    max_diff1 = max_difference(1)
    print(f"Max abs diff, batch=1: {max_diff1:.2e}")

    return float(max(max_diff, max_diff1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="state_value_net.onnx")
    parser.add_argument("--in-dim", type=int, default=1764)
    args = parser.parse_args()
    export(args.ckpt, args.out, args.in_dim)
