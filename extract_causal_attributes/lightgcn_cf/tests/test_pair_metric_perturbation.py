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

from lightgcn_cf.artifacts import ArtifactPaths
from lightgcn_cf.graph import build_normalized_adjacency
from lightgcn_cf.pair_metric_perturbation import (
    FixedDegreeAdjacencyBuilder,
    run_top_m_attribute_perturbation_evaluation,
)


class PairMetricPerturbationTests(unittest.TestCase):
    def _write_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        artifact_dir = root / "artifacts_item_only"
        artifact_dir.mkdir()
        paths = ArtifactPaths(artifact_dir)
        paths.mappings.write_text(
            json.dumps(
                {
                    "user_to_index": {"100": 0},
                    "item_to_index": {
                        "10": 0,
                        "20": 1,
                        "30": 2,
                        "40": 3,
                        "50": 4,
                    },
                    "index_to_user": ["100"],
                    "index_to_item": ["10", "20", "30", "40", "50"],
                }
            ),
            encoding="utf-8",
        )
        paths.user_history.write_text(json.dumps({"0": [0, 1, 3]}), encoding="utf-8")
        torch.save(torch.tensor([[0, 0], [0, 1], [0, 3]], dtype=torch.long), paths.train_pairs)

        # These ego embeddings determine perturbed propagation after edge drops.
        torch.save(torch.tensor([[1.0, 0.0]]), paths.user_ego_embeddings)
        torch.save(
            torch.tensor(
                [
                    [10.0, 0.0],
                    [0.0, 10.0],
                    [1.0, 0.0],
                    [-10.0, 0.0],
                    [0.0, 1.0],
                ]
            ),
            paths.item_ego_embeddings,
        )

        # These baseline embeddings are intentionally not the propagated ego result.
        torch.save(torch.tensor([[1.0, 0.0]]), paths.user_baseline_embeddings)
        torch.save(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [2.0, 0.0],
                    [0.0, 0.0],
                    [1.0, 0.0],
                ]
            ),
            paths.item_baseline_embeddings,
        )
        paths.validation_metrics.write_text(
            json.dumps({"recall@10": 0.5, "recall@20": 0.75, "ndcg@20": 0.6}),
            encoding="utf-8",
        )

        chosen_path = root / "new_chosen_sorted_attributes.pkl"
        with chosen_path.open("wb") as output_file:
            pickle.dump(
                {
                    (100, 30): [
                        ("a", 0.9),
                        ("b", 0.8),
                        ("c", 0.7),
                    ]
                },
                output_file,
            )

        support_path = root / "attribute_items_mapping.pkl"
        with support_path.open("wb") as output_file:
            pickle.dump(
                {
                    (100, 30): {
                        "a": [10],
                        "b": [20],
                        "c": [40],
                    }
                },
                output_file,
            )

        return artifact_dir, chosen_path, support_path

    def test_cumulative_top_m_pair_metrics_use_raw_id_mapping_and_ego_propagation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            artifact_dir, chosen_path, support_path = self._write_fixture(root)
            result = run_top_m_attribute_perturbation_evaluation(
                artifact_dir,
                chosen_path,
                support_path,
                num_layers=1,
                recall_k=(1, 2),
                ndcg_k=2,
                max_m=3,
                device="cpu",
                perturbation_batch_size=2,
            )

        self.assertEqual(result["coverage"]["pairs_seen"], 1)
        self.assertEqual(result["coverage"]["pairs_with_attributes"], 1)
        self.assertEqual(result["coverage"]["valid_pairs_by_m"], {"1": 1, "2": 1, "3": 1})
        self.assertEqual(result["coverage"]["unknown_support_items"], 0)
        self.assertEqual(result["coverage"]["support_items_not_in_history"], 0)

        for m in ("1", "2", "3"):
            self.assertEqual(result["origin_by_m"][m]["pairs_evaluated"], 1)
            self.assertEqual(result["origin_by_m"][m]["recall@1"], 1.0)
            self.assertEqual(result["origin_by_m"][m]["recall@2"], 1.0)
            self.assertEqual(result["origin_by_m"][m]["ndcg@2"], 1.0)

        expected_rank_2_ndcg = 1.0 / math.log2(3.0)
        self.assertEqual(result["perturbed_by_m"]["1"]["recall@1"], 0.0)
        self.assertEqual(result["perturbed_by_m"]["1"]["recall@2"], 1.0)
        self.assertAlmostEqual(result["perturbed_by_m"]["1"]["ndcg@2"], expected_rank_2_ndcg)
        self.assertEqual(result["perturbed_by_m"]["2"]["recall@1"], 0.0)
        self.assertEqual(result["perturbed_by_m"]["2"]["recall@2"], 1.0)
        self.assertAlmostEqual(result["perturbed_by_m"]["2"]["ndcg@2"], expected_rank_2_ndcg)
        self.assertEqual(result["perturbed_by_m"]["3"]["recall@1"], 1.0)
        self.assertEqual(result["perturbed_by_m"]["3"]["recall@2"], 1.0)
        self.assertEqual(result["perturbed_by_m"]["3"]["ndcg@2"], 1.0)

        self.assertEqual(result["delta_by_m"]["1"]["recall@1"], -1.0)
        self.assertAlmostEqual(
            result["delta_by_m"]["1"]["ndcg@2"],
            expected_rank_2_ndcg - 1.0,
        )
        self.assertEqual(result["reference_validation_metrics"]["recall@10"], 0.5)
        self.assertEqual(result["metadata"]["perturbation_mode"], "global-batched")
        self.assertEqual(result["metadata"]["normalization"], "fixed-original-degree")
        self.assertEqual(result["metadata"]["perturbation_batch_size"], 2)
        self.assertIn("user_ego_embeddings.pt", result["metadata"]["embedding_source_for_perturbation"][0])
        self.assertIn("user_baseline_embeddings.pt", result["metadata"]["embedding_source_for_origin"][0])

    def test_fixed_degree_adjacency_masks_a_without_recomputing_d(self) -> None:
        train_pairs = torch.tensor([[0, 0], [0, 1]], dtype=torch.long)
        builder = FixedDegreeAdjacencyBuilder.from_train_pairs(
            num_users=1,
            num_items=2,
            train_pairs=train_pairs,
            device=torch.device("cpu"),
        )
        indices, values = builder.perturbed_components(
            user_index=0,
            drop_item_indices=(0,),
            offset=0,
        )
        fixed_degree_adjacency = torch.sparse_coo_tensor(
            indices,
            values,
            (3, 3),
        ).to_dense()
        recomputed_adjacency = build_normalized_adjacency(
            1,
            2,
            torch.tensor([[0, 1]], dtype=torch.long),
        ).to_dense()

        self.assertAlmostEqual(float(fixed_degree_adjacency[0, 2]), 1.0 / math.sqrt(2.0))
        self.assertAlmostEqual(float(fixed_degree_adjacency[2, 0]), 1.0 / math.sqrt(2.0))
        self.assertAlmostEqual(float(recomputed_adjacency[0, 2]), 1.0)
        self.assertFalse(torch.allclose(fixed_degree_adjacency, recomputed_adjacency))

    def test_save_path_writes_json_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            artifact_dir, chosen_path, support_path = self._write_fixture(root)
            output_path = root / "metrics.json"
            result = run_top_m_attribute_perturbation_evaluation(
                artifact_dir,
                chosen_path,
                support_path,
                num_layers=1,
                recall_k=(1,),
                ndcg_k=1,
                max_m=1,
                device="cpu",
                save_path=output_path,
            )
            saved = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(saved, result)


if __name__ == "__main__":
    unittest.main()
