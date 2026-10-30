"""Training additions: batches that keep sibling groups together, the auxiliary and ranking losses, and that a model with
auxiliary heads still exports only the value logits.

Skipped when torch, lightning or onnxruntime are not installed.
Run from the repository root:  python -m unittest discover -s Tests/Python -v
"""

import json
import os
import sys
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "DatasetProcessing"))
sys.path.insert(0, os.path.join(REPO, "Agents", "Training"))

# conda's MKL and torch both ship an OpenMP runtime in Agents/env
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import numpy as np
    import onnxruntime as ort
    import torch
    from export_onnx import export_model
    from state_value_net import StateValueModule, masked_kl, policy_cross_entropy, sibling_ranking
    from train import GroupBatchSampler, MixedDataset, MixSampler, StateDataset, widen_state_dict
except ImportError as error:  # pragma: no cover - depends on the environment
    IMPORT_ERROR = error
else:
    IMPORT_ERROR = None


def write_source(directory, rows, features, label_dtype, self_play):
    """A prepared data directory with random states and labels, plus root_q.npy and winners.npy for self-play data."""
    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(rows)
    np.save(os.path.join(directory, "states.npy"), rng.random((rows, features)).astype(np.float16))
    np.save(os.path.join(directory, "win_rates.npy"), rng.dirichlet(np.ones(4), rows).astype(label_dtype))
    if self_play:
        np.save(os.path.join(directory, "root_q.npy"), rng.dirichlet(np.ones(4), rows).astype(np.float32))
        np.save(os.path.join(directory, "winners.npy"), np.array([3, 2, 1] * (rows // 3 + 1), dtype=np.int8)[:rows])


@unittest.skipIf(IMPORT_ERROR is not None, f"training dependencies missing: {IMPORT_ERROR}")
class TrainingTest(unittest.TestCase):
    def test_a_widened_checkpoint_computes_the_same_on_the_old_features(self):
        # The view layout appends features: the old net's weights go on, the new columns start at 0
        old = StateValueModule(in_dim=8, hidden=16, normalize_inputs=True, placement_weight=0.1)
        old.model.input_mean.copy_(torch.arange(8, dtype=torch.float32))
        old.model.input_scale.copy_(torch.full((8,), 0.5))
        wide = StateValueModule(in_dim=11, hidden=16, normalize_inputs=True, placement_weight=0.1)
        widened = widen_state_dict(old.state_dict(), 11, new_mean=np.array([1.0, 2.0, 3.0]), new_scale=np.array([0.1, 0.2, 0.3]))
        wide.load_state_dict(widened, strict=True)
        np.testing.assert_allclose(wide.model.input_mean[8:].numpy(), [1, 2, 3])
        np.testing.assert_allclose(wide.model.input_scale[8:].numpy(), [0.1, 0.2, 0.3], rtol=1e-6)

        old.eval()
        wide.eval()
        x = torch.rand(5, 11) * 4
        with torch.no_grad():
            torch.testing.assert_close(wide.model(x), old.model(x[:, :8]))

        # A checkpoint of the same width stays as it is, a wider one can't be narrowed
        self.assertIs(old.state_dict()["model.net.0.weight"].shape[1], 8)
        same = widen_state_dict(old.state_dict(), 8)
        torch.testing.assert_close(same["model.net.0.weight"], old.state_dict()["model.net.0.weight"])
        with self.assertRaises(ValueError):
            widen_state_dict(wide.state_dict(), 8)

    def test_a_source_can_give_another_feature_file(self):
        with tempfile.TemporaryDirectory() as directory:
            write_source(directory, 6, 5, np.float32, self_play=False)
            np.save(os.path.join(directory, "views.npy"), np.ones((6, 7), dtype=np.float16))
            dataset = StateDataset(os.path.join(directory, "views.npy"), os.path.join(directory, "win_rates.npy"), directory)
            self.assertEqual((6, 7), dataset.states.shape)
            # Windows keeps memory-mapped files open until the arrays are gone
            del dataset
            import gc
            gc.collect()

    def test_batches_keep_sibling_groups_together(self):
        groups = np.array([-1, 0, 0, 0, -1, 1, 1, -1, 2, 2, 2, 2, -1, -1, 3, 3])
        indices = np.arange(1, 16)  # sample 0 is not in this split
        np.random.seed(0)

        for shuffle in (True, False):
            batches = list(GroupBatchSampler(indices, groups, batch_size=4, shuffle=shuffle))
            self.assertEqual(indices.tolist(), sorted(i for batch in batches for i in batch))
            for group in range(4):
                self.assertEqual(1, sum(any(groups[i] == group for i in batch) for batch in batches), f"group {group} split")

    def test_losses_skip_missing_labels(self):
        uniform = torch.full((2, 4), 0.25)
        target = torch.tensor([[0.25] * 4, [float("nan")] * 4])
        self.assertAlmostEqual(0.0, masked_kl(torch.zeros(2, 4), target).item(), places=6)
        self.assertEqual(0.0, masked_kl(torch.zeros(1, 4), torch.full((1, 4), float("nan"))).item())
        self.assertGreater(masked_kl(torch.tensor([[5.0, 0, 0, 0]]), uniform[:1]).item(), 0)

    def test_ranking_compares_the_actors_value_within_groups(self):
        labels = torch.tensor([[0.1, 0.6, 0.2, 0.1], [0.1, 0.4, 0.4, 0.1], [0.7, 0.1, 0.1, 0.1], [0.25] * 4])
        groups, actors = torch.tensor([5, 5, -1, -1]), torch.tensor([1, 1, 0, 0])

        # Logits that rank the actor's (player 1's) value the same way as the labels: right order, smaller loss
        right = torch.tensor([[0, 2.0, 0, 0], [0, 1.0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        loss_right, accuracy_right, pairs = sibling_ranking(right, labels, groups, actors)
        loss_wrong, accuracy_wrong, _ = sibling_ranking(right[[1, 0, 2, 3]], labels, groups, actors)

        self.assertEqual(1, pairs)
        self.assertEqual(1.0, accuracy_right.item())
        self.assertEqual(0.0, accuracy_wrong.item())
        self.assertLess(loss_right.item(), loss_wrong.item())

        # Without groups in the batch there is nothing to compare
        self.assertEqual(0, sibling_ranking(right, labels, torch.full((4,), -1), actors)[2])

    def test_the_placement_score_lets_the_ranking_learn_far_behind(self):
        # The actor (player 1) never wins in either sibling, but finishes second more often in the first
        win_rates = torch.tensor([[0.5, 0.0, 0.3, 0.2], [0.5, 0.0, 0.3, 0.2]])
        placement_labels = torch.tensor([[[0.5, 0.2, 0.2, 0.1], [0.0, 0.6, 0.3, 0.1], [0.3, 0.1, 0.4, 0.2], [0.2, 0.1, 0.1, 0.6]],
                                         [[0.5, 0.2, 0.2, 0.1], [0.0, 0.2, 0.3, 0.5], [0.3, 0.4, 0.2, 0.1], [0.2, 0.2, 0.3, 0.3]]])
        groups, actors = torch.tensor([3, 3]), torch.tensor([1, 1])
        logits = torch.zeros(2, 4)

        # On the win rates alone there is nothing to rank, with the placement score the first sibling has to come out ahead
        self.assertEqual(0.0, sibling_ranking(logits, win_rates, groups, actors)[0].item())
        right = torch.zeros(2, 4, 4)
        right[0, 1, 1] = 3.0  # the net puts player 1 in second place in the first sibling
        loss_right = sibling_ranking(logits, win_rates, groups, actors, right, placement_labels, 0.5)[0]
        loss_flat = sibling_ranking(logits, win_rates, groups, actors, torch.zeros(2, 4, 4), placement_labels, 0.5)[0]
        self.assertGreater(loss_flat.item(), 0.0)
        self.assertLess(loss_right.item(), loss_flat.item())

    def test_a_model_with_auxiliary_heads_exports_the_value_and_the_placements(self):
        torch.manual_seed(0)
        module = StateValueModule(in_dim=32, placement_weight=1.0, vp_share_weight=1.0, rounds_weight=1.0, ranking_weight=1.0)
        model = module.model.eval()
        outputs = model.forward_all(torch.randn(3, 32))
        self.assertEqual((3, 4), outputs["logits"].shape)
        self.assertEqual((3, 4, 4), outputs["placements"].shape)
        self.assertEqual((3,), outputs["rounds_left"].shape)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state_value_net.onnx")
            self.assertLess(export_model(model, path, 32), 1e-4)
            # The C# side reads "logits" by name and the placement distribution logits from "placements"
            session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.assertEqual(["logits", "placements"], [output.name for output in session.get_outputs()])
            logits, placements = session.run(None, {"state": np.zeros((2, 32), dtype=np.float32)})
            self.assertEqual((2, 4), logits.shape)
            self.assertEqual((2, 4, 4), placements.shape)

        # Input normalization is part of the model, so the export applies it too
        normalized = StateValueModule(in_dim=32, normalize_inputs=True)
        states = np.random.default_rng(1).integers(0, 2, (500, 32)).astype(np.float32)
        states[:, :4] = np.random.default_rng(2).normal(5, 3, (500, 4))
        self.assertEqual(4, normalized.set_input_normalization(states))
        standardized = normalized.model._normalize(torch.from_numpy(states)).numpy()
        np.testing.assert_allclose(standardized[:, :4].mean(axis=0), 0, atol=1e-4)
        np.testing.assert_allclose(standardized[:, :4].std(axis=0), 1, atol=1e-3)
        np.testing.assert_array_equal(standardized[:, 4:], states[:, 4:])
        with tempfile.TemporaryDirectory() as directory:
            self.assertLess(export_model(normalized.model.eval(), os.path.join(directory, "normalized.onnx"), 32), 1e-4)

        # Aux heads for other losses only: the placements head is untrained, so the export leaves it out
        rounds_only = StateValueModule(in_dim=32, rounds_weight=1.0)
        self.assertIsNotNone(rounds_only.model.aux)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "rounds_only.onnx")
            self.assertLess(export_model(rounds_only.model.eval(), path, 32), 1e-4)
            session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.assertEqual(["logits"], [output.name for output in session.get_outputs()])

        # Without auxiliary losses the model has exactly the parameters of earlier checkpoints
        plain = StateValueModule(in_dim=32)
        self.assertIsNone(plain.model.aux)
        self.assertTrue(all(name.startswith("model.net.") for name in plain.state_dict()))

    def test_self_play_targets_mix_the_result_and_the_root_values(self):
        # The datasets keep their arrays memory-mapped, which Windows doesn't let the cleanup delete
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            write_source(directory, rows=3, features=8, label_dtype=np.float32, self_play=True)
            dataset = StateDataset(os.path.join(directory, "states.npy"), os.path.join(directory, "win_rates.npy"), directory, target_lambda=0.25)
            batch = dataset.__getitems__([2, 0])
            root_q = np.load(os.path.join(directory, "root_q.npy"))
            expected = 0.75 * root_q[[2, 0]] + 0.25 * np.eye(4)[[1, 3]]
            np.testing.assert_allclose(batch["y"].numpy(), expected, rtol=1e-6)

            # Without λ, or without the self-play arrays, the stored labels stay
            stored = StateDataset(os.path.join(directory, "states.npy"), os.path.join(directory, "win_rates.npy"), directory)
            np.testing.assert_array_equal(stored.__getitems__([2])["y"].numpy(), np.load(os.path.join(directory, "win_rates.npy"))[[2]])

    def test_mixed_sources_are_drawn_by_their_shares(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            first, second = os.path.join(directory, "a"), os.path.join(directory, "b")
            write_source(first, rows=10, features=8, label_dtype=np.float32, self_play=True)
            write_source(second, rows=50, features=8, label_dtype=np.float16, self_play=False)
            datasets = [StateDataset(os.path.join(d, "states.npy"), os.path.join(d, "win_rates.npy"), d, target_lambda=0.5) for d in (first, second)]
            mixed = MixedDataset(datasets)
            self.assertEqual(60, len(mixed))

            # The first source's 10 rows at share 1, the second's 50 at share 3: 40 per epoch, 10 and 30
            sampler = MixSampler([np.arange(10), np.arange(50) + 10], [1, 3], epoch_samples=40)
            drawn = np.array(list(sampler))
            self.assertEqual(40, len(drawn))
            self.assertEqual(10, int((drawn < 10).sum()))
            self.assertEqual(10, len(set(drawn[drawn < 10].tolist())), "a source with enough rows is drawn without replacement")

            batch = mixed.__getitems__([0, 15, 59, 3])
            self.assertEqual(torch.float32, batch["y"].dtype)
            self.assertEqual((4, 8), tuple(batch["x"].shape))
            # Rows come back per source; the second source's rows are its stored states
            np.testing.assert_array_equal(batch["x"][2:].numpy(), np.load(os.path.join(second, "states.npy"))[[5, 49]])


    def test_board_symmetries_move_the_move_tokens_like_the_features(self):
        from augmentation import BOARD_INDEX, BOARD_SYMMETRY_COUNT, _EDGE_SIZE, _EDGE_START, move_token_symmetries
        from state_to_vector import _INTERSECTION_SIZE, _INTERSECTION_START, _TILE_SIZE, _TILE_START
        table = move_token_symmetries(POLICY_LAYOUT)
        for t in range(BOARD_SYMMETRY_COUNT):
            for name, start, size in (("intersections", _INTERSECTION_START, _INTERSECTION_SIZE), ("edges", _EDGE_START, _EDGE_SIZE),
                                      ("tiles", _TILE_START, _TILE_SIZE)):
                first, count = POLICY_LAYOUT[name]
                for old in range(count):
                    new = table[t, first + old] - first
                    # The feature block of the element moves to the same place as its token
                    self.assertEqual(start + old * size, BOARD_INDEX[t][start + new * size], f"{name} {old} under symmetry {t}")
            # Move types, resources and victims stay
            np.testing.assert_array_equal(table[t, :POLICY_LAYOUT["intersections"][0]], np.arange(POLICY_LAYOUT["intersections"][0]))
            victims = POLICY_LAYOUT["victims"][0]
            np.testing.assert_array_equal(table[t, victims:], np.arange(victims, POLICY_LAYOUT["vocabulary"][1]))

    def test_the_policy_loss_covers_only_legal_moves_of_rows_with_moves(self):
        torch.manual_seed(1)
        module = StateValueModule(in_dim=8, hidden=16, policy_weight=1.0, policy_types=19, policy_vocabulary=213, policy_dim=4)
        query = module.model.forward_all(torch.randn(2, 8))["policy_query"]
        tokens = torch.tensor([[[0, -1], [5, 30], [-1, -1]], [[-1, -1], [-1, -1], [-1, -1]]])
        batch = {"policy_tokens": tokens, "policy_weights": (tokens >= 0).float(),
                 "policy_targets": torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
                 "policy_mask": torch.tensor([[True, True, False], [False, False, False]])}
        loss, _ = policy_cross_entropy(module.model, query, batch)
        logits = module.model.policy_logits(query, tokens, batch["policy_weights"])[0, :2]
        self.assertAlmostEqual(-torch.log_softmax(logits, dim=0)[1].item(), loss.item(), places=5)
        loss.backward()
        self.assertTrue(torch.isfinite(module.model.policy_embeddings.weight.grad).all())

    def test_mixed_sources_pad_the_moves_of_sources_without_policy(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            first, second = os.path.join(directory, "a"), os.path.join(directory, "b")
            write_source(first, rows=3, features=8, label_dtype=np.float32, self_play=True)
            write_source(second, rows=4, features=8, label_dtype=np.float16, self_play=False)
            np.save(os.path.join(first, "policy_offsets.npy"), np.array([0, 2, 5, 6]))
            np.save(os.path.join(first, "policy_tokens.npy"), np.array([[0, -1], [5, 20], [1, -1], [2, -1], [3, 21], [4, -1]], dtype=np.int16))
            np.save(os.path.join(first, "policy_weights.npy"), np.ones((6, 2), dtype=np.uint8))
            np.save(os.path.join(first, "policy_targets.npy"), np.array([0.5, 0.5, 1, 0, 0, 1], dtype=np.float32))
            with open(os.path.join(first, "policy_layout.json"), "w") as file:
                json.dump(POLICY_LAYOUT | {"max_tokens": [2]}, file)
            mixed = MixedDataset([StateDataset(os.path.join(d, "states.npy"), os.path.join(d, "win_rates.npy"), d) for d in (first, second)])
            batch = mixed.__getitems__([1, 4, 0])
            self.assertEqual((3, 3, 2), tuple(batch["policy_tokens"].shape))
            np.testing.assert_array_equal(batch["policy_mask"].sum(dim=1).numpy(), [3, 2, 0])
            np.testing.assert_array_equal(batch["policy_tokens"][2].numpy(), -1)
            np.testing.assert_allclose(batch["policy_targets"][0].numpy(), [1, 0, 0])

    def test_a_model_with_a_policy_head_exports_queries_and_embeddings(self):
        module = StateValueModule(in_dim=32, hidden=16, policy_weight=1.0, policy_types=19, policy_vocabulary=213, policy_dim=4)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "policy.onnx")
            self.assertLess(export_model(module.model.eval(), path, 32), 1e-4)
            session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.assertEqual(["logits", "policy_query", "policy_embeddings"], [output.name for output in session.get_outputs()])
            _, queries, embeddings = session.run(None, {"state": np.zeros((3, 32), dtype=np.float32)})
            self.assertEqual((3, 19 * 4), queries.shape)
            np.testing.assert_allclose(embeddings, module.model.policy_embeddings.weight.detach().numpy(), rtol=1e-6)


# The token layout of Agents/Inference/ActionEncoding.cs (as SelfPlayCollector writes it into policy_layout.json)
POLICY_LAYOUT = {"move_types": [0, 19], "intersections": [19, 54], "edges": [73, 72], "tiles": [145, 49], "victims": [194, 4],
                 "given_resources": [198, 5], "received_resources": [203, 5], "discarded_resources": [208, 5], "vocabulary": [0, 213],
                 "max_tokens": [6]}


if __name__ == "__main__":
    unittest.main()
