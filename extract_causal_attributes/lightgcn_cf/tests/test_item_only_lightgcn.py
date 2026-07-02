from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from lightgcn_cf.artifacts import ArtifactPaths, load_tensor
from lightgcn_cf.graph import build_normalized_adjacency
from lightgcn_cf.item_only_model import ItemOnlyLightGCN
from lightgcn_cf.perturbation import run_edge_drop
from train import set_seed
from train_item_only import MODEL_VARIANT, run_item_only_training


class ItemOnlyLightGCNTests(unittest.TestCase):
    def test_optimizer_step_keeps_user_embeddings_fixed_and_updates_items(self) -> None:
        torch.manual_seed(11)
        model = ItemOnlyLightGCN(num_users=2, num_items=3, embedding_dim=4, num_layers=1)
        adjacency = build_normalized_adjacency(
            2,
            3,
            torch.tensor([[0, 0], [0, 1], [1, 1], [1, 2]], dtype=torch.long),
        )
        initial_users = model.user_embedding.weight.detach().clone()
        initial_items = model.item_embedding.weight.detach().clone()
        optimizer = torch.optim.Adam(model.item_embedding.parameters(), lr=0.05)

        loss, _ranking_loss, _regularization_loss = model.bpr_loss(
            adjacency,
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([0, 2], dtype=torch.long),
            torch.tensor([2, 0], dtype=torch.long),
            l2_regularization=0.0001,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.assertFalse(model.user_embedding.weight.requires_grad)
        self.assertTrue(torch.allclose(model.user_embedding.weight, initial_users))
        self.assertFalse(torch.allclose(model.item_embedding.weight, initial_items))


class ItemOnlyTrainingTests(unittest.TestCase):
    def _write_fixture(self, root: Path) -> dict:
        train_csv = root / "train.csv"
        validation_csv = root / "val.csv"
        artifact_dir = root / "artifacts_item_only"
        config_path = root / "config_item_only.yaml"
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
        return {
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

    def test_smoke_train_exports_reproducible_frozen_users_and_supports_edge_drop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = self._write_fixture(root)
            result = run_item_only_training(config)
            paths = ArtifactPaths(Path(config["artifact_dir"]))

            self.assertEqual(result["model_variant"], MODEL_VARIANT)
            self.assertEqual(result["users_evaluated"], 3)
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

            saved_metrics = json.loads(paths.validation_metrics.read_text(encoding="utf-8"))
            self.assertEqual(saved_metrics["model_variant"], MODEL_VARIANT)
            self.assertIn(
                f"model_variant: {MODEL_VARIANT}",
                paths.run_config.read_text(encoding="utf-8"),
            )

            checkpoint = torch.load(paths.checkpoint, map_location="cpu")
            user_ego = load_tensor(paths.user_ego_embeddings)
            item_ego = load_tensor(paths.item_ego_embeddings)
            self.assertEqual(checkpoint["model_variant"], MODEL_VARIANT)
            self.assertTrue(
                torch.allclose(
                    checkpoint["model_state_dict"]["user_embedding.weight"], user_ego
                )
            )

            set_seed(int(config["seed"]))
            initialized_model = ItemOnlyLightGCN(
                num_users=3,
                num_items=3,
                embedding_dim=int(config["embedding_dim"]),
                num_layers=int(config["num_layers"]),
            )
            self.assertTrue(torch.allclose(initialized_model.user_embedding.weight, user_ego))
            self.assertFalse(
                torch.allclose(initialized_model.item_embedding.weight, item_ego)
            )

            perturbation = run_edge_drop(
                config["artifact_dir"],
                user_id="u1",
                drop_item_ids=["i1"],
                num_layers=1,
                top_k=2,
                device="cpu",
            )
            self.assertEqual(perturbation["user_id"], "u1")
            self.assertEqual(perturbation["dropped_item_ids"], ["i1"])
            self.assertEqual(len(perturbation["perturbed_user_embedding"]), 4)


if __name__ == "__main__":
    unittest.main()
