import argparse
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import lightning as L
from state_value_net import StateValueModule


class ExportedValueNet(nn.Module):
    """Outputs für den Export: die Value Logits, bei Modellen mit trainiertem Platzierungs-Head die Platzierungs-Logits
    [batch, 4 Spieler, 4 Plätze], bei Modellen mit Policy Head die Queries [batch, Typen * dim] und die Embedding-Tabelle
    der Zug-Tokens [Vokabular, dim] (konstant, für alle Batches gleich). Die Queries kommen als float16: sie sind 300-mal so
    groß wie die Value Logits, und die Kopie von der GPU bremst sonst die Suche"""

    def __init__(self, model: nn.Module, with_placements: bool, with_policy: bool):
        super().__init__()
        self.model = model
        self.with_placements, self.with_policy = with_placements, with_policy

    def forward(self, x):
        outputs = self.model.forward_all(x)
        result = [outputs["logits"]]
        if self.with_placements:
            result.append(outputs["placements"])
        if self.with_policy:
            result.append(outputs["policy_query"].flatten(1).half())
            result.append(self.model.policy_embeddings.weight * 1.0)
        return tuple(result)


def export(ckpt_path: str, out_path: str, in_dim: int = None):
    # Lightning-Module aus Checkpoint laden, die Inputgröße steht in seinen Hyperparametern
    overrides = {"in_dim": in_dim} if in_dim else {}
    module = StateValueModule.load_from_checkpoint(ckpt_path, map_location="cpu", weights_only=True, **overrides)
    in_dim = module.hparams.in_dim

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
    with_policy = getattr(model, "policy_query", None) is not None
    exported = ExportedValueNet(model, with_placements, with_policy).eval()
    output_names = ["logits"] + (["placements"] if with_placements else []) + (["policy_query", "policy_embeddings"] if with_policy else [])
    batch_outputs = [name for name in output_names if name != "policy_embeddings"]

    dummy = torch.randn(1, in_dim)
    torch.onnx.export(
        exported, dummy, out_path,
        input_names=["state"], output_names=output_names,
        dynamic_axes={"state": {0: "batch"}, **{name: {0: "batch"} for name in batch_outputs}},
        opset_version=17, dynamo=False,
    )
    if with_policy:
        _name_constant_outputs(out_path, output_names)
    print(f"Exported to {out_path} (outputs: {', '.join(output_names)})")

    # Paritätscheck zwischen Torch und ONNX auf Random Inputs, für jeden Output
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(out_path, sess_options=so, providers=["CPUExecutionProvider"])

    def max_difference(batch: int) -> float:
        x_np = np.random.randn(batch, in_dim).astype(np.float32)
        with torch.no_grad():
            torch_out = exported(torch.from_numpy(x_np))
        torch_outputs = [t.numpy() for t in torch_out]
        onnx_outputs = sess.run(output_names, {"state": x_np})

        def difference(t, o):
            # float16-Outputs relativ zu ihrer Größe (ein paar Rundungsschritte von float16 sind erlaubt)
            if t.dtype == np.float16:
                t, o = t.astype(np.float32), o.astype(np.float32)
                return float((np.abs(t - o) / np.maximum(np.abs(t), 1.0)).max()) * 1e-2
            return float(np.abs(t - o).max())
        return max(difference(t, o) for t, o in zip(torch_outputs, onnx_outputs))

    max_diff = max_difference(16)
    print(f"Max abs diff, PyTorch vs ONNX (batch=16): {max_diff:.2e}")
    assert max_diff < 1e-4, "ONNX export does not match PyTorch output -- do not ship this .onnx file"
    print("Parity check passed.")

    # Und nochmal mit batch=1 wegen Relevanz für MCTS
    max_diff1 = max_difference(1)
    print(f"Max abs diff, batch=1: {max_diff1:.2e}")

    return float(max(max_diff, max_diff1))


def _name_constant_outputs(path: str, output_names):
    """Der Export faltet die konstante Embedding-Tabelle zu einem Initializer und verliert dabei den Output-Namen: die
    Outputs der Reihe nach über Identity-Knoten wieder benennen"""
    import onnx
    model = onnx.load(path)
    for output, name in zip(model.graph.output, output_names):
        if output.name != name:
            model.graph.node.append(onnx.helper.make_node("Identity", [output.name], [name]))
            output.name = name
    onnx.save(model, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="state_value_net.onnx")
    # Nur für Checkpoints ohne gespeicherte Inputgröße nötig
    parser.add_argument("--in-dim", type=int, default=None)
    args = parser.parse_args()
    export(args.ckpt, args.out, args.in_dim)
