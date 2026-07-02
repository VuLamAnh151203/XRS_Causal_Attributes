from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from lightgcn_cf.artifacts import ArtifactPaths
from lightgcn_cf.local_perturbation import LocalPerturbationScorer
from score_item_only_attribute_support_jsonl import (
    derive_candidate_attributes_from_support_jsonl,
    run_item_only_attribute_support_jsonl,
)


class ItemOnlyAttributeSupportJsonlTests(unittest.TestCase):
    def _write_artifacts_and_support(self, root: Path) -> tuple[dict, Path]:
        artifact_dir = root / "artifacts_item_only"
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
        torch.save(torch.tensor([[0, 0], [0, 1]], dtype=torch.long), paths.train_pairs)
        torch.save(torch.tensor([[0.0]]), paths.user_ego_embeddings)
        torch.save(torch.tensor([[7.0], [-5.0], [2.0], [1.0]]), paths.item_ego_embeddings)

        support_jsonl = root / "tst_attribute_support.jsonl"
        support_jsonl.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "user_id": "u1",
                    "user_index": 0,
                    "target_item_id": "target",
                    "target_item_index": 2,
                    "target_attributes": ["supported", "empty", "missing"],
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
        config = {
            "_config_path": str(root / "config_item_only.yaml"),
            "artifact_dir": str(artifact_dir),
            "num_layers": 1,
            "device": "cpu",
            "model_variant": "item_only_fixed_random_users",
        }
        return config, support_jsonl

    def _write_two_pair_artifacts_and_support(self, root: Path) -> tuple[dict, Path]:
        artifact_dir = root / "artifacts_item_only"
        artifact_dir.mkdir()
        paths = ArtifactPaths(artifact_dir)
        mappings = {
            "user_to_index": {"u1": 0, "u2": 1},
            "item_to_index": {
                "i0": 0,
                "i1": 1,
                "target": 2,
                "other": 3,
                "target2": 4,
            },
            "index_to_user": ["u1", "u2"],
            "index_to_item": ["i0", "i1", "target", "other", "target2"],
        }
        paths.mappings.write_text(json.dumps(mappings), encoding="utf-8")
        paths.user_history.write_text(json.dumps({"0": [0, 1], "1": [1, 3]}), encoding="utf-8")
        torch.save(
            torch.tensor([[0, 0], [0, 1], [1, 1], [1, 3]], dtype=torch.long),
            paths.train_pairs,
        )
        torch.save(torch.tensor([[0.0], [0.0]]), paths.user_ego_embeddings)
        torch.save(
            torch.tensor([[7.0], [-5.0], [2.0], [6.0], [1.0]]),
            paths.item_ego_embeddings,
        )

        support_jsonl = root / "tst_attribute_support.jsonl"
        records = [
            {
                "schema_version": 2,
                "user_id": "u1",
                "user_index": 0,
                "target_item_id": "target",
                "target_item_index": 2,
                "target_attributes": ["supported"],
                "supported_items_by_attribute": {
                    "supported": [
                        {
                            "item_id": "i0",
                            "item_index": 0,
                            "score": 1.0,
                            "matched_attribute": "supported",
                        }
                    ]
                },
            },
            {
                "schema_version": 2,
                "user_id": "u2",
                "user_index": 1,
                "target_item_id": "target2",
                "target_item_index": 4,
                "target_attributes": ["supported2"],
                "supported_items_by_attribute": {
                    "supported2": [
                        {
                            "item_id": "other",
                            "item_index": 3,
                            "score": 1.0,
                            "matched_attribute": "supported2",
                        }
                    ]
                },
            },
        ]
        support_jsonl.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        config = {
            "_config_path": str(root / "config_item_only.yaml"),
            "artifact_dir": str(artifact_dir),
            "num_layers": 1,
            "device": "cpu",
            "model_variant": "item_only_fixed_random_users",
        }
        return config, support_jsonl

    def test_derives_candidates_from_target_attributes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            _config, support_jsonl = self._write_artifacts_and_support(root)
            candidates = derive_candidate_attributes_from_support_jsonl(support_jsonl)

        self.assertEqual(set(candidates), {"u1,target"})
        self.assertEqual(
            candidates["u1,target"],
            [
                {"attr_name": "supported"},
                {"attr_name": "empty"},
                {"attr_name": "missing"},
            ],
        )

    def test_jsonl_only_mode_scores_attribute_drops(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config, support_jsonl = self._write_artifacts_and_support(root)
            result = run_item_only_attribute_support_jsonl(
                config,
                support_jsonl,
                num_layers=1,
                device="cpu",
            )

        self.assertEqual(set(result), {"u1,target"})
        supported, empty, missing = result["u1,target"]
        expected_fields = {
            "attr_id",
            "attr_name",
            "baseline_score",
            "ratios",
            "score_drop",
            "baseline_rank",
            "perturbed_rank",
            "rank_drop",
        }
        for entry in (supported, empty, missing):
            self.assertEqual(set(entry), expected_fields)

        self.assertIsNone(supported["attr_id"])
        self.assertEqual(supported["attr_name"], "supported")
        self.assertIsNone(supported["baseline_rank"])
        self.assertIsNone(supported["perturbed_rank"])
        self.assertIsNone(supported["rank_drop"])
        self.assertIsNotNone(supported["baseline_score"])
        self.assertIsNotNone(supported["ratios"])
        self.assertGreater(supported["score_drop"], 0.0)

        for entry in (empty, missing):
            self.assertIsNone(entry["baseline_score"])
            self.assertIsNone(entry["ratios"])
            self.assertIsNone(entry["score_drop"])
            self.assertIsNone(entry["baseline_rank"])
            self.assertIsNone(entry["perturbed_rank"])
            self.assertIsNone(entry["rank_drop"])

    def test_local_lhop_matches_full_graph_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config, support_jsonl = self._write_artifacts_and_support(root)
            local_result = run_item_only_attribute_support_jsonl(
                config,
                support_jsonl,
                num_layers=1,
                device="cpu",
                propagation_mode="local-lhop",
            )
            full_result = run_item_only_attribute_support_jsonl(
                config,
                support_jsonl,
                num_layers=1,
                device="cpu",
                propagation_mode="full",
            )

        local_supported = local_result["u1,target"][0]
        full_supported = full_result["u1,target"][0]
        self.assertAlmostEqual(
            local_supported["score_drop"],
            full_supported["score_drop"],
            places=6,
        )
        self.assertEqual(local_supported["baseline_rank"], full_supported["baseline_rank"])
        self.assertEqual(local_supported["perturbed_rank"], full_supported["perturbed_rank"])
        self.assertEqual(local_supported["rank_drop"], full_supported["rank_drop"])
        self.assertAlmostEqual(
            local_supported["baseline_score"],
            full_supported["baseline_score"],
            places=6,
        )
        self.assertAlmostEqual(local_supported["ratios"], full_supported["ratios"], places=6)

    def test_local_score_matches_full_score_and_leaves_ranks_null(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config, support_jsonl = self._write_artifacts_and_support(root)
            local_score_result = run_item_only_attribute_support_jsonl(
                config,
                support_jsonl,
                num_layers=1,
                device="cpu",
                propagation_mode="local-score",
            )
            full_result = run_item_only_attribute_support_jsonl(
                config,
                support_jsonl,
                num_layers=1,
                device="cpu",
                propagation_mode="full",
            )

        local_supported = local_score_result["u1,target"][0]
        full_supported = full_result["u1,target"][0]
        self.assertAlmostEqual(
            local_supported["score_drop"],
            full_supported["score_drop"],
            places=6,
        )
        self.assertIsNone(local_supported["baseline_rank"])
        self.assertIsNone(local_supported["perturbed_rank"])
        self.assertIsNone(local_supported["rank_drop"])
        self.assertAlmostEqual(
            local_supported["baseline_score"],
            full_supported["baseline_score"],
            places=6,
        )
        self.assertAlmostEqual(local_supported["ratios"], full_supported["ratios"], places=6)

    def test_local_score_reuses_repeated_drop_set_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config, _support_jsonl = self._write_artifacts_and_support(root)
            scorer = LocalPerturbationScorer.from_artifacts(
                config["artifact_dir"],
                num_layers=1,
                device="cpu",
            )
            scores = scorer.score_target_many(0, 2, ([0], [0]))

        self.assertEqual(scores[0], scores[1])
        self.assertEqual(scorer.score_cache_miss_count, 1)
        self.assertEqual(scorer.score_cache_hit_count, 1)

    def test_save_steps_writes_final_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config, support_jsonl = self._write_two_pair_artifacts_and_support(root)
            output_path = root / "drop_effects.json"
            result = run_item_only_attribute_support_jsonl(
                config,
                support_jsonl,
                num_layers=1,
                device="cpu",
                save_path=output_path,
                save_steps=2,
                propagation_mode="local-lhop",
            )
            saved = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(saved, result)
        self.assertEqual(set(result), {"u1,target", "u2,target2"})

    def test_resume_skips_completed_pairs(self) -> None:
        sentinel = [
            {
                "attr_id": None,
                "attr_name": "already_done",
                "baseline_score": None,
                "ratios": None,
                "score_drop": None,
                "baseline_rank": None,
                "perturbed_rank": None,
                "rank_drop": None,
            }
        ]
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config, support_jsonl = self._write_two_pair_artifacts_and_support(root)
            output_path = root / "drop_effects.json"
            output_path.write_text(json.dumps({"u1,target": sentinel}), encoding="utf-8")
            result = run_item_only_attribute_support_jsonl(
                config,
                support_jsonl,
                num_layers=1,
                device="cpu",
                save_path=output_path,
                save_steps=1,
                resume=True,
            )

        self.assertEqual(result["u1,target"], sentinel)
        self.assertEqual(result["u2,target2"][0]["attr_name"], "supported2")

    def test_no_resume_overwrites_existing_output(self) -> None:
        sentinel = [
            {
                "attr_id": None,
                "attr_name": "already_done",
                "baseline_score": None,
                "ratios": None,
                "score_drop": None,
                "baseline_rank": None,
                "perturbed_rank": None,
                "rank_drop": None,
            }
        ]
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config, support_jsonl = self._write_two_pair_artifacts_and_support(root)
            output_path = root / "drop_effects.json"
            output_path.write_text(json.dumps({"u1,target": sentinel}), encoding="utf-8")
            result = run_item_only_attribute_support_jsonl(
                config,
                support_jsonl,
                num_layers=1,
                device="cpu",
                save_path=output_path,
                save_steps=1,
                resume=False,
            )

        self.assertEqual(result["u1,target"][0]["attr_name"], "supported")
        self.assertNotEqual(result["u1,target"], sentinel)

    def test_invalid_save_steps_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config, support_jsonl = self._write_artifacts_and_support(root)
            with self.assertRaisesRegex(ValueError, "save_steps"):
                run_item_only_attribute_support_jsonl(
                    config,
                    support_jsonl,
                    num_layers=1,
                    device="cpu",
                    save_steps=0,
                )

    def test_candidate_json_mode_preserves_attribute_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config, support_jsonl = self._write_artifacts_and_support(root)
            candidate_json = root / "candidates.json"
            candidate_json.write_text(
                json.dumps(
                    {
                        "u1,target": [
                            {"attr_id": 10, "attr_name": "supported"},
                            {"attr_id": 11, "attr_name": "supported"},
                            {"attr_id": 12, "attr_name": "empty"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            result = run_item_only_attribute_support_jsonl(
                config,
                support_jsonl,
                candidate_json,
                num_layers=1,
                device="cpu",
            )

        self.assertEqual([entry["attr_id"] for entry in result["u1,target"]], [10, 11, 12])
        self.assertEqual(result["u1,target"][0]["attr_name"], "supported")
        self.assertGreater(result["u1,target"][0]["score_drop"], 0.0)
        self.assertEqual(
            result["u1,target"][0]["score_drop"],
            result["u1,target"][1]["score_drop"],
        )
        self.assertIsNone(result["u1,target"][2]["score_drop"])


if __name__ == "__main__":
    unittest.main()
