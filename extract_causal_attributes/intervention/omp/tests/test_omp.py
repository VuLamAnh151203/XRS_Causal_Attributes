import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from extract_causal_attributes.intervention.omp.core import (
    InterventionShardCache,
    OmpConfig,
    OmpError,
    RecoveryResult,
    _manifest_record,
    _resume_configs_match,
    _stats_from_manifests,
    generate_omp_artifacts,
    load_intervention_slice,
    recover_pair,
    signed_omp,
    write_coefficient_shard,
)


def make_config(root: Path, sparsity_level: int = 2) -> OmpConfig:
    return OmpConfig(
        intervention_dir=root,
        output_dir=root / "omp",
        sparsity_level=sparsity_level,
        minimum_intervention_multiplier=2,
        require_strictly_more_than_minimum=True,
        correlation_tolerance=1.0e-12,
        residual_tolerance=1.0e-10,
        pairs_per_shard=100,
        source_shard_cache_max_entries=2,
    )


def write_intervention_shard(path: Path, rows, y_delta, removed_items, width=6):
    indices = []
    indptr = [0]
    removed_ids = []
    removed_indptr = [0]
    for row, removed in zip(rows, removed_items):
        indices.extend(row)
        indptr.append(len(indices))
        removed_ids.extend(removed)
        removed_indptr.append(len(removed_ids))
    np.savez_compressed(
        path,
        A_data=np.ones(len(indices), dtype=np.int8),
        A_indices=np.asarray(indices, dtype=np.int64),
        A_indptr=np.asarray(indptr, dtype=np.int64),
        A_shape=np.asarray([len(rows), width], dtype=np.int64),
        y_delta=np.asarray(y_delta, dtype=np.float32),
        removed_item_ids=np.asarray(removed_ids, dtype=np.int64),
        removed_item_indptr=np.asarray(removed_indptr, dtype=np.int64),
    )


class OmpAlgorithmTests(unittest.TestCase):
    def test_recovers_signed_sparse_coefficients(self):
        matrix = np.asarray(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
            ],
            dtype=np.float64,
        )
        outcomes = matrix @ np.asarray([0.8, -0.3, 0.0])
        fit = signed_omp(matrix, outcomes, [4, 2, 5], 2, 1.0e-12, 1.0e-12)
        self.assertIsNotNone(fit)
        self.assertEqual(fit.global_indices, (2, 4))
        self.assertAlmostEqual(fit.coefficients[0], -0.3)
        self.assertAlmostEqual(fit.coefficients[1], 0.8)

    def test_preserves_small_selected_coefficient(self):
        matrix = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.float64)
        outcomes = matrix @ np.asarray([1.0, 1.0e-8])
        fit = signed_omp(matrix, outcomes, [0, 1], 2, 1.0e-12, 1.0e-12)
        self.assertIsNotNone(fit)
        self.assertEqual(fit.global_indices, (0, 1))
        self.assertAlmostEqual(fit.coefficients[1], 1.0e-8)

    def test_zero_signal_is_valid_empty_vector(self):
        fit = signed_omp(np.eye(2), np.zeros(2), [0, 1], 2, 1.0e-12, 1.0e-10)
        self.assertTrue(fit.zero_signal)
        self.assertEqual(fit.global_indices, ())
        self.assertEqual(fit.coefficients, ())

    def test_ties_are_broken_by_global_vocabulary_index(self):
        matrix = np.asarray([[1, 1], [1, 1]], dtype=np.float64)
        fit = signed_omp(matrix, np.asarray([1.0, 1.0]), [7, 3], 1, 1.0e-12, 1.0e-12)
        self.assertEqual(fit.global_indices, (3,))


class SliceAndRecoveryTests(unittest.TestCase):
    def test_rejects_stale_schema_version_1_intervention_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(OmpError, "Regenerate schema-version-2 intervention"):
                recover_pair(
                    {
                        "schema_version": 1,
                        "pair_index": 1,
                        "user_id": 2,
                        "target_item_id": 3,
                        "skip_reason": "upstream",
                    },
                    InterventionShardCache(root, 2),
                    vocabulary_size=6,
                    config=make_config(root),
                )

    def test_reads_pair_slice_from_multi_pair_shard(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            shards = root / "shards"
            shards.mkdir()
            write_intervention_shard(
                shards / "source.npz",
                rows=[(0,), (1,), (2,), (0, 1)],
                y_delta=[0.1, 0.2, 0.3, 0.4],
                removed_items=[(5,), (6,), (7,), (5, 6)],
            )
            cache = InterventionShardCache(root, 2)
            sliced = load_intervention_slice(
                {"shard": "shards/source.npz", "row_start": 1, "row_end": 4},
                cache,
                expected_vocabulary_size=6,
            )
            self.assertEqual(sliced.matrix.shape, (3, 3))
            self.assertEqual(sliced.active_columns, (0, 1, 2))
            self.assertEqual(sliced.unique_removed_item_signature_count, 3)
            self.assertEqual(sliced.y_delta.tolist(), np.asarray([0.2, 0.3, 0.4], dtype=np.float32).tolist())

    def test_skips_upstream_failure_without_loading_shard(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result = recover_pair(
                {
                    "schema_version": 2,
                    "pair_index": 1,
                    "user_id": 2,
                    "user_index": 12,
                    "target_item_id": 3,
                    "target_item_index": 13,
                    "skip_reason": "Pair target item ID is outside item embedding rows.",
                },
                InterventionShardCache(root, 2),
                vocabulary_size=6,
                config=make_config(root),
            )
            self.assertEqual(result.status, "skipped")
            self.assertEqual(result.skip_reason, "upstream_skip")

    def test_skips_when_interventions_are_not_strictly_more_than_2s(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            shards = root / "shards"
            shards.mkdir()
            write_intervention_shard(
                shards / "source.npz",
                rows=[(0,), (1,), (0, 1), (0,)],
                y_delta=[0.1, 0.2, 0.3, 0.1],
                removed_items=[(1,), (2,), (1, 2), (1,)],
            )
            result = recover_pair(
                {
                    "schema_version": 2,
                    "pair_index": 0,
                    "user_id": 1,
                    "user_index": 11,
                    "target_item_id": 2,
                    "target_item_index": 12,
                    "shard": "shards/source.npz",
                    "row_start": 0,
                    "row_end": 4,
                },
                InterventionShardCache(root, 2),
                vocabulary_size=6,
                config=make_config(root),
            )
            self.assertEqual(result.skip_reason, "insufficient_interventions")

    def test_recovers_pair_and_maps_coefficients_to_global_columns(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            shards = root / "shards"
            shards.mkdir()
            rows = [(1,), (4,), (1, 4), (1,), (4,), (1, 4)]
            write_intervention_shard(
                shards / "source.npz",
                rows=rows,
                y_delta=[0.8, -0.3, 0.5, 0.8, -0.3, 0.5],
                removed_items=[(1,), (2,), (1, 2), (3,), (4,), (3, 4)],
            )
            config = make_config(root)
            config = OmpConfig(
                **{
                    **config.__dict__,
                    "minimum_intervention_multiplier": 1,
                }
            )
            result = recover_pair(
                {
                    "schema_version": 2,
                    "pair_index": 0,
                    "user_id": 1,
                    "user_index": 11,
                    "target_item_id": 2,
                    "target_item_index": 12,
                    "shard": "shards/source.npz",
                    "row_start": 0,
                    "row_end": 6,
                    "generated_intervention_count": 6,
                },
                InterventionShardCache(root, 2),
                vocabulary_size=6,
                config=config,
            )
            self.assertEqual(result.status, "recovered")
            self.assertEqual(result.global_indices, (1, 4))
            self.assertAlmostEqual(result.coefficients[0], 0.8)
            self.assertAlmostEqual(result.coefficients[1], -0.3)

    def test_skips_when_unique_rows_are_not_strictly_more_than_2s(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            shards = root / "shards"
            shards.mkdir()
            write_intervention_shard(
                shards / "source.npz",
                rows=[(0,), (1,), (0, 1), (0,), (1,), (0, 1)],
                y_delta=[0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
                removed_items=[(1,), (2,), (1, 2), (3,), (4,), (3, 4)],
            )
            result = recover_pair(
                {
                    "schema_version": 2,
                    "pair_index": 0,
                    "user_id": 1,
                    "user_index": 11,
                    "target_item_id": 2,
                    "target_item_index": 12,
                    "shard": "shards/source.npz",
                    "row_start": 0,
                    "row_end": 6,
                },
                InterventionShardCache(root, 2),
                vocabulary_size=6,
                config=make_config(root),
            )
            self.assertEqual(result.skip_reason, "insufficient_unique_rows")


class ArtifactTests(unittest.TestCase):
    def test_coefficient_shard_round_trip_preserves_small_values(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "coefficients.npz"
            write_coefficient_shard(
                path,
                [
                    RecoveryResult(
                        pair_index=8,
                        user_id=1,
                        user_index=11,
                        target_item_id=2,
                        target_item_index=12,
                        status="recovered",
                        global_indices=(1, 4),
                        coefficients=(0.8, -1.0e-9),
                    )
                ],
                vocabulary_size=6,
            )
            with np.load(path) as shard:
                self.assertEqual(shard["coef_indices"].tolist(), [1, 4])
                self.assertEqual(shard["coef_shape"].tolist(), [1, 6])
                self.assertAlmostEqual(float(shard["coef_data"][1]), -1.0e-9)
                self.assertEqual(shard["user_index"].tolist(), [11])
                self.assertEqual(shard["target_item_index"].tolist(), [12])

    def test_skipped_manifest_has_no_vector_pointer(self):
        record = _manifest_record(
            RecoveryResult(
                pair_index=1,
                user_id=2,
                user_index=12,
                target_item_id=3,
                target_item_index=13,
                status="skipped",
                skip_reason="insufficient_interventions",
                skip_details={"observed": 0, "required": "> 10"},
            ),
            None,
            None,
        )
        self.assertIsNone(record["vector_shard"])
        self.assertIsNone(record["vector_row"])

    def test_resume_allows_checkpoint_granularity_change(self):
        previous = {
            "config": {"pairs_per_shard": 100, "sparsity_level": 5},
            "source_checksums": {"manifest": "abc"},
        }
        expected = {
            "config": {"pairs_per_shard": 1, "sparsity_level": 5},
            "source_checksums": {"manifest": "abc"},
        }
        self.assertTrue(_resume_configs_match(previous, expected))

    def test_resume_stats_are_rebuilt_from_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            summary_path = Path(temporary_directory) / "summary.json"
            summary_path.write_text(json.dumps({"full_time_seconds": 12.5}), encoding="utf-8")
            stats = _stats_from_manifests(
                [
                    {
                        "status": "recovered",
                        "vector_shard": "shards/a.npz",
                        "selected_attribute_count": 2,
                        "zero_signal": False,
                        "diagnostics": {
                            "relative_residual": 0.1,
                            "reconstruction_r_squared": 0.9,
                        },
                    },
                    {
                        "status": "skipped",
                        "skip_reason": "insufficient_interventions",
                    },
                ],
                summary_path,
            )
            self.assertEqual(stats.processed_pair_count, 2)
            self.assertEqual(stats.recovered_pair_count, 1)
            self.assertEqual(stats.skipped_pair_count, 1)
            self.assertEqual(stats.written_shard_count, 1)
            self.assertEqual(stats.previous_full_time_seconds, 12.5)

    def test_generation_writes_aligned_manifest_and_resumes_without_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            shards = root / "shards"
            shards.mkdir()
            (root / "vocabulary.json").write_text(
                json.dumps({"attributes": ["a", "b"], "attribute_to_index": {"a": 0, "b": 1}}),
                encoding="utf-8",
            )
            (root / "run_config.json").write_text(json.dumps({"source": "test"}), encoding="utf-8")
            write_intervention_shard(
                shards / "source.npz",
                rows=[(0,), (1,), (0, 1)],
                y_delta=[0.8, -0.3, 0.5],
                removed_items=[(1,), (2,), (1, 2)],
                width=2,
            )
            source_records = [
                {
                    "schema_version": 2,
                    "pair_index": 0,
                    "user_id": 1,
                    "user_index": 11,
                    "target_item_id": 2,
                    "target_item_index": 12,
                    "shard": "shards/source.npz",
                    "row_start": 0,
                    "row_end": 3,
                    "generated_intervention_count": 3,
                    "skip_reason": None,
                },
                {
                    "schema_version": 2,
                    "pair_index": 1,
                    "user_id": 2,
                    "user_index": 12,
                    "target_item_id": 99,
                    "target_item_index": None,
                    "skip_reason": "Pair target item ID 99 is outside item embedding rows.",
                },
            ]
            (root / "manifest.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in source_records),
                encoding="utf-8",
            )
            config = OmpConfig(
                intervention_dir=root,
                output_dir=root / "omp",
                sparsity_level=2,
                minimum_intervention_multiplier=1,
                require_strictly_more_than_minimum=True,
                correlation_tolerance=1.0e-12,
                residual_tolerance=1.0e-10,
                pairs_per_shard=100,
                source_shard_cache_max_entries=2,
            )

            first_summary = generate_omp_artifacts(config)
            resumed_summary = generate_omp_artifacts(config, resume=True)
            manifests = [
                json.loads(line)
                for line in (config.output_dir / "manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

            self.assertEqual(first_summary["processed_pair_count"], 2)
            self.assertEqual(first_summary["recovered_pair_count"], 1)
            self.assertEqual(first_summary["skipped_pair_count"], 1)
            self.assertEqual(resumed_summary["processed_pair_count"], 2)
            self.assertEqual(len(manifests), 2)
            self.assertEqual(manifests[0]["status"], "recovered")
            self.assertEqual(manifests[1]["skip_reason"], "upstream_skip")


if __name__ == "__main__":
    unittest.main()
