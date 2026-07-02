from __future__ import annotations

import json
import math
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import torch

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from lightgcn_cf.artifacts import ArtifactPaths, load_tensor
from lightgcn_cf.attribute_perturbation import (
    run_candidate_attribute_drops,
    run_support_dict_attribute_drops,
)
from lightgcn_cf.data import load_dataset
from lightgcn_cf.graph import (
    build_normalized_adjacency,
    propagate,
    remove_user_item_edges,
)
from lightgcn_cf.metrics import evaluate
from lightgcn_cf.perturbation import run_edge_drop
from train import run_training


class FixedModel:
    def __init__(self, users: torch.Tensor, items: torch.Tensor) -> None:
        self.users = users
        self.items = items
        self.training = True

    def eval(self) -> None:
        self.training = False

    def train(self) -> None:
        self.training = True

    def __call__(self, _adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.users, self.items


class GraphTests(unittest.TestCase):
    def test_adjacency_is_symmetric_deduplicated_and_normalized(self) -> None:
        pairs = torch.tensor([[0, 0], [0, 0], [0, 1], [1, 1]])
        adjacency = build_normalized_adjacency(2, 2, pairs).to_dense()
        expected = torch.tensor(
            [
                [0.0, 0.0, 1.0 / math.sqrt(2.0), 0.5],
                [0.0, 0.0, 0.0, 1.0 / math.sqrt(2.0)],
                [1.0 / math.sqrt(2.0), 0.0, 0.0, 0.0],
                [0.5, 1.0 / math.sqrt(2.0), 0.0, 0.0],
            ]
        )
        self.assertTrue(torch.allclose(adjacency, expected))
        self.assertTrue(torch.allclose(adjacency, adjacency.T))

    def test_edge_drop_is_exact_and_isolated_nodes_remain_finite(self) -> None:
        pairs = torch.tensor([[0, 0], [1, 1]])
        perturbed_pairs = remove_user_item_edges(pairs, 0, [0])
        adjacency = build_normalized_adjacency(2, 2, perturbed_pairs)
        embeddings = propagate(torch.ones((4, 3)), adjacency, num_layers=2)
        self.assertTrue(torch.isfinite(embeddings).all())
        self.assertEqual(perturbed_pairs.tolist(), [[1, 1]])
        with self.assertRaisesRegex(ValueError, "not history edges"):
            remove_user_item_edges(pairs, 0, [1])


class DataAndMetricTests(unittest.TestCase):
    def test_csv_loader_deduplicates_train_pairs_and_skips_seen_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            train_csv = root / "train.csv"
            validation_csv = root / "val.csv"
            train_csv.write_text(
                "user_id,item_id\nu1,i1\nu1,i1\nu1,i2\n", encoding="utf-8"
            )
            validation_csv.write_text(
                "user_id,item_id\nu1,i1\nu1,i3\nu2,i1\n", encoding="utf-8"
            )
            dataset = load_dataset(train_csv, validation_csv)
            self.assertEqual(dataset.train_pairs.shape, (2, 2))
            self.assertEqual(dataset.summary["validation_rows_skipped_seen_in_train"], 1)
            self.assertEqual(dataset.summary["validation_rows_skipped_unknown_item"], 1)
            self.assertEqual(dataset.summary["validation_rows_skipped_unknown_user"], 1)

    def test_full_ranking_metrics_mask_training_history(self) -> None:
        model = FixedModel(
            users=torch.tensor([[1.0]]),
            items=torch.tensor([[100.0], [3.0], [2.0]]),
        )
        metrics = evaluate(
            model,
            torch.empty(0),
            validation_pairs={0: {2}},
            train_history={0: {0}},
            recall_k=(1, 2),
            ndcg_k=2,
        )
        self.assertEqual(metrics["recall@1"], 0.0)
        self.assertEqual(metrics["recall@2"], 1.0)
        self.assertAlmostEqual(metrics["ndcg@2"], 1.0 / math.log2(3.0))
        self.assertTrue(model.training)


class AttributePerturbationTests(unittest.TestCase):
    def test_support_dict_attribute_drops_preserve_pickle_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            paths = ArtifactPaths(artifact_dir)
            mappings = {
                "user_to_index": {"u1": 0},
                "item_to_index": {"i0": 0, "i1": 1, "target": 2, "other": 3},
                "index_to_user": ["u1"],
                "index_to_item": ["i0", "i1", "target", "other"],
            }
            paths.mappings.write_text(json.dumps(mappings), encoding="utf-8")
            paths.user_history.write_text(json.dumps({"0": [0, 1]}), encoding="utf-8")
            train_pairs = torch.tensor([[0, 0], [0, 1]], dtype=torch.long)
            torch.save(train_pairs, paths.train_pairs)
            torch.save(torch.tensor([[0.0]]), paths.user_ego_embeddings)
            torch.save(torch.tensor([[7.0], [-5.0], [2.0], [1.0]]), paths.item_ego_embeddings)

            support_path = root / "support.pkl"
            with support_path.open("wb") as output_file:
                pickle.dump(
                    {
                        (0, 2): {
                            10: {
                                "attr_name": "supported",
                                "candidate_score": 1.5,
                                "support_items": [0],
                                "support_scores": [0.9],
                                "item_tau": [0.1],
                            },
                            11: {
                                "attr_name": "empty",
                                "candidate_score": 0.2,
                                "support_items": [],
                                "support_scores": [],
                                "item_tau": [],
                            },
                            12: {
                                "attr_name": "supported-2",
                                "candidate_score": 0.8,
                                "support_items": [1],
                                "support_scores": [0.7],
                                "item_tau": [0.2],
                            },
                        }
                    },
                    output_file,
                )

            result = run_support_dict_attribute_drops(
                artifact_dir,
                support_path,
                num_layers=1,
                device="cpu",
                save_path=root / "drop_effects.pkl",
                save_every_pairs=1,
                perturbation_batch_size=2,
            )

            with (root / "drop_effects.pkl").open("rb") as input_file:
                saved_result = pickle.load(input_file)

        self.assertEqual(result, saved_result)
        self.assertEqual(set(result), {(0, 2)})
        self.assertEqual(set(result[(0, 2)]), {10, 11, 12})
        expected_fields = {
            "attr_name",
            "candidate_score",
            "score_drop",
            "baseline_rank",
            "perturbed_rank",
            "rank_drop",
        }
        supported = result[(0, 2)][10]
        empty = result[(0, 2)][11]
        supported_2 = result[(0, 2)][12]
        self.assertEqual(set(supported), expected_fields)
        self.assertEqual(set(empty), expected_fields)
        self.assertEqual(set(supported_2), expected_fields)
        self.assertEqual(supported["attr_name"], "supported")
        self.assertEqual(supported["candidate_score"], 1.5)
        self.assertEqual(supported["baseline_rank"], 1)
        self.assertEqual(supported["perturbed_rank"], 2)
        self.assertEqual(supported["rank_drop"], 1)
        self.assertGreater(supported["score_drop"], 0.0)
        self.assertEqual(empty["attr_name"], "empty")
        self.assertEqual(empty["candidate_score"], 0.2)
        self.assertIsNone(empty["score_drop"])
        self.assertIsNone(empty["baseline_rank"])
        self.assertIsNone(empty["perturbed_rank"])
        self.assertIsNone(empty["rank_drop"])
        self.assertEqual(supported_2["attr_name"], "supported-2")
        self.assertEqual(supported_2["candidate_score"], 0.8)
        self.assertIsNotNone(supported_2["score_drop"])
        self.assertIsNotNone(supported_2["baseline_rank"])
        self.assertIsNotNone(supported_2["perturbed_rank"])
        self.assertIsNotNone(supported_2["rank_drop"])

    def test_candidate_attribute_drops_emit_minimal_schema_and_nulls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            paths = ArtifactPaths(artifact_dir)
            mappings = {
                "user_to_index": {"u1": 0},
                "item_to_index": {"i0": 0, "i1": 1, "target": 2, "other": 3},
                "index_to_user": ["u1"],
                "index_to_item": ["i0", "i1", "target", "other"],
            }
            paths.mappings.write_text(json.dumps(mappings), encoding="utf-8")
            paths.user_history.write_text(json.dumps({"0": [0, 1]}), encoding="utf-8")
            train_pairs = torch.tensor([[0, 0], [0, 1]], dtype=torch.long)
            torch.save(train_pairs, paths.train_pairs)

            user_ego = torch.tensor([[0.0]])
            item_ego = torch.tensor([[7.0], [-5.0], [2.0], [1.0]])
            adjacency = build_normalized_adjacency(1, 4, train_pairs)
            baseline = propagate(torch.cat((user_ego, item_ego), dim=0), adjacency, num_layers=1)
            baseline_users, baseline_items = torch.split(baseline, (1, 4), dim=0)
            torch.save(user_ego, paths.user_ego_embeddings)
            torch.save(item_ego, paths.item_ego_embeddings)
            torch.save(baseline_users, paths.user_baseline_embeddings)
            torch.save(baseline_items, paths.item_baseline_embeddings)

            candidates_path = root / "candidates.json"
            candidates_path.write_text(
                json.dumps(
                    {
                        "u1,target": [
                            {"attr_id": 1, "attr_name": "supported", "score": 99.0},
                            {"attr_id": 2, "attr_name": "empty"},
                            {"attr_id": 3, "attr_name": "missing"},
                        ],
                        "missing-user,target": [
                            {"attr_id": 4, "attr_name": "supported"},
                        ],
                        "u1,missing-target": [
                            {"attr_id": 5, "attr_name": "supported"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            support_path = root / "support.jsonl"
            support_path.write_text(
                json.dumps(
                    {
                        "user_id": "u1",
                        "target_item_id": "target",
                        "supported_items_by_attribute": {
                            "supported": [
                                {
                                    "item_id": "i0",
                                    "item_index": 0,
                                    "score": 1.0,
                                    "matched_attribute": "supported",
                                }
                            ],
                            "empty": [],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = run_candidate_attribute_drops(
                artifact_dir,
                candidates_path,
                support_path,
                num_layers=1,
                device="cpu",
            )

        self.assertEqual(set(result), {"u1,target", "missing-user,target", "u1,missing-target"})
        expected_fields = {
            "attr_id",
            "attr_name",
            "score_drop",
            "baseline_rank",
            "perturbed_rank",
            "rank_drop",
        }
        for entry in result["u1,target"]:
            self.assertEqual(set(entry), expected_fields)

        supported, empty, missing = result["u1,target"]
        self.assertEqual(supported["attr_id"], 1)
        self.assertEqual(supported["attr_name"], "supported")
        self.assertEqual(supported["baseline_rank"], 1)
        self.assertEqual(supported["perturbed_rank"], 2)
        self.assertEqual(supported["rank_drop"], 1)
        self.assertGreater(supported["score_drop"], 0.0)
        for entry in (empty, missing):
            self.assertIsNone(entry["score_drop"])
            self.assertIsNone(entry["baseline_rank"])
            self.assertIsNone(entry["perturbed_rank"])
            self.assertIsNone(entry["rank_drop"])
        for pair_key in ("missing-user,target", "u1,missing-target"):
            self.assertEqual(len(result[pair_key]), 1)
            entry = result[pair_key][0]
            self.assertEqual(set(entry), expected_fields)
            self.assertIsNone(entry["score_drop"])
            self.assertIsNone(entry["baseline_rank"])
            self.assertIsNone(entry["perturbed_rank"])
            self.assertIsNone(entry["rank_drop"])


class EndToEndTests(unittest.TestCase):
    def test_smoke_train_exports_reproducible_baseline_and_supports_edge_drop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            train_csv = root / "train.csv"
            validation_csv = root / "val.csv"
            artifact_dir = root / "artifacts"
            config_path = root / "config.yaml"
            train_csv.write_text(
                "user_id,item_id\n"
                "u1,i1\n"
                "u1,i2\n"
                "u2,i2\n"
                "u2,i3\n"
                "u3,i1\n"
                "u3,i3\n",
                encoding="utf-8",
            )
            validation_csv.write_text(
                "user_id,item_id\nu1,i3\nu2,i1\nu3,i2\n", encoding="utf-8"
            )
            config = {
                "_config_path": str(config_path),
                "train_csv": str(train_csv),
                "validation_csv": str(validation_csv),
                "artifact_dir": str(artifact_dir),
                "embedding_dim": 4,
                "num_layers": 1,
                "batch_size": 2,
                "learning_rate": 0.01,
                "l2_regularization": 0.0001,
                "epochs": 2,
                "early_stopping_patience": 2,
                "evaluation_every": 1,
                "evaluation_user_batch_size": 2,
                "seed": 7,
                "device": "cpu",
                "recall_k": [10, 20],
                "ndcg_k": 20,
            }
            result = run_training(config)
            self.assertEqual(result["users_evaluated"], 3)
            paths = ArtifactPaths(artifact_dir)
            for path in (
                paths.checkpoint,
                paths.mappings,
                paths.train_pairs,
                paths.user_history,
                paths.user_ego_embeddings,
                paths.item_ego_embeddings,
                paths.user_baseline_embeddings,
                paths.item_baseline_embeddings,
                paths.validation_metrics,
            ):
                self.assertTrue(path.exists(), path)

            checkpoint = torch.load(paths.checkpoint, map_location="cpu")
            user_ego = load_tensor(paths.user_ego_embeddings)
            item_ego = load_tensor(paths.item_ego_embeddings)
            self.assertTrue(
                torch.allclose(
                    checkpoint["model_state_dict"]["user_embedding.weight"], user_ego
                )
            )
            pairs = load_tensor(paths.train_pairs)
            adjacency = build_normalized_adjacency(3, 3, pairs)
            reproduced = propagate(torch.cat((user_ego, item_ego)), adjacency, num_layers=1)
            baseline = torch.cat(
                (
                    load_tensor(paths.user_baseline_embeddings),
                    load_tensor(paths.item_baseline_embeddings),
                )
            )
            self.assertTrue(torch.allclose(reproduced, baseline))

            perturbation = run_edge_drop(
                artifact_dir,
                user_id="u1",
                drop_item_ids=["i1"],
                num_layers=1,
                top_k=2,
                device="cpu",
            )
            self.assertEqual(perturbation["user_id"], "u1")
            self.assertEqual(perturbation["dropped_item_ids"], ["i1"])
            self.assertEqual(len(perturbation["perturbed_user_embedding"]), 4)
            with self.assertRaisesRegex(ValueError, "not history edges"):
                run_edge_drop(
                    artifact_dir,
                    user_id="u1",
                    drop_item_ids=["i3"],
                    num_layers=1,
                    device="cpu",
                )


if __name__ == "__main__":
    unittest.main()
