import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from extract_causal_attributes.intervention.core import (
    HybridLocalLightGCNScorer,
    INTERVENTION_GENERATION_VERSION,
    InterventionConfig,
    InterventionError,
    InterventionRow,
    PairResult,
    RunStats,
    ScoreCache,
    Vocabulary,
    _resume_configs_match,
    build_graph,
    build_pair_result,
    generate_intervention_artifacts,
    load_vocabulary,
    parse_support_record,
    resolve_device,
    sample_interventions,
    write_shard,
)
from extract_causal_attributes.id_mappings import IdMappings

try:
    import torch
except ImportError:
    torch = None


def make_mappings(users=(), items=()):
    index_to_user = tuple(str(value) for value in users)
    index_to_item = tuple(str(value) for value in items)
    return IdMappings(
        user_to_index={value: index for index, value in enumerate(index_to_user)},
        item_to_index={value: index for index, value in enumerate(index_to_item)},
        index_to_user=index_to_user,
        index_to_item=index_to_item,
    )


def write_mappings(path, users=(), items=()):
    mappings = make_mappings(users, items)
    path.write_text(
        json.dumps(
            {
                "user_to_index": mappings.user_to_index,
                "item_to_index": mappings.item_to_index,
                "index_to_user": list(mappings.index_to_user),
                "index_to_item": list(mappings.index_to_item),
            }
        ),
        encoding="utf-8",
    )


class ResolveDeviceTests(unittest.TestCase):
    @patch("extract_causal_attributes.intervention.core._load_torch")
    def test_indexed_cuda_device_is_accepted_when_available(self, load_torch):
        load_torch.return_value.cuda.is_available.return_value = True
        load_torch.return_value.cuda.device_count.return_value = 2

        self.assertEqual(resolve_device("cuda:1"), "cuda:1")

    @patch("extract_causal_attributes.intervention.core._load_torch")
    def test_indexed_cuda_device_fails_when_index_is_out_of_range(self, load_torch):
        load_torch.return_value.cuda.is_available.return_value = True
        load_torch.return_value.cuda.device_count.return_value = 1

        with self.assertRaisesRegex(InterventionError, "only 1 CUDA device"):
            resolve_device("cuda:1")

    @patch("extract_causal_attributes.intervention.core._load_torch")
    def test_indexed_cuda_device_fails_when_cuda_is_unavailable(self, load_torch):
        load_torch.return_value.cuda.is_available.return_value = False

        with self.assertRaisesRegex(InterventionError, "CUDA is unavailable"):
            resolve_device("cuda:0")

    @patch("extract_causal_attributes.intervention.core._load_torch")
    def test_invalid_cuda_device_syntax_fails_clearly(self, load_torch):
        with self.assertRaisesRegex(InterventionError, "must use"):
            resolve_device("cuda:one")

    @patch("extract_causal_attributes.intervention.core._load_torch")
    def test_auto_and_cpu_behavior_is_preserved(self, load_torch):
        load_torch.return_value.cuda.is_available.return_value = False
        self.assertEqual(resolve_device("auto"), "cpu")
        self.assertEqual(resolve_device("cpu"), "cpu")

        load_torch.return_value.cuda.is_available.return_value = True
        self.assertEqual(resolve_device("auto"), "cuda")


def brute_force_score(graph, user_embeddings, item_embeddings, layer_count, user_id, item_id, removed):
    removed_edges = set(removed)
    user_count = user_embeddings.shape[0]
    item_count = item_embeddings.shape[0]
    user_degrees = [
        sum((candidate_user_id, candidate_item_id) not in removed_edges for candidate_item_id in graph.user_items.get(candidate_user_id, ()))
        for candidate_user_id in range(user_count)
    ]
    item_degrees = [
        sum((candidate_user_id, candidate_item_id) not in removed_edges for candidate_user_id in graph.item_users.get(candidate_item_id, ()))
        for candidate_item_id in range(item_count)
    ]
    user_layers = [user_embeddings.float()]
    item_layers = [item_embeddings.float()]
    for _ in range(layer_count):
        next_users = torch.zeros_like(user_embeddings, dtype=torch.float32)
        next_items = torch.zeros_like(item_embeddings, dtype=torch.float32)
        for candidate_user_id, items in graph.user_items.items():
            for candidate_item_id in items:
                if (candidate_user_id, candidate_item_id) in removed_edges:
                    continue
                normalization = math.sqrt(
                    user_degrees[candidate_user_id] * item_degrees[candidate_item_id]
                )
                next_users[candidate_user_id] += item_layers[-1][candidate_item_id] / normalization
                next_items[candidate_item_id] += user_layers[-1][candidate_user_id] / normalization
        user_layers.append(next_users)
        item_layers.append(next_items)
    user_embedding = torch.stack(user_layers).mean(dim=0)[user_id]
    item_embedding = torch.stack(item_layers).mean(dim=0)[item_id]
    return float(torch.dot(user_embedding, item_embedding).item())


class VocabularyTests(unittest.TestCase):
    def test_load_vocabulary_preserves_list_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "vocabulary.json"
            path.write_text(json.dumps(["beta", "alpha"]), encoding="utf-8")
            vocabulary = load_vocabulary(path)
        self.assertEqual(vocabulary.attributes, ("beta", "alpha"))
        self.assertEqual(vocabulary.attribute_to_index, {"beta": 0, "alpha": 1})

    def test_load_vocabulary_supports_index_mapping_and_rejects_gaps(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "vocabulary.json"
            path.write_text(json.dumps({"beta": 1, "alpha": 0}), encoding="utf-8")
            self.assertEqual(load_vocabulary(path).attributes, ("alpha", "beta"))
            path.write_text(json.dumps({"alpha": 0, "beta": 2}), encoding="utf-8")
            with self.assertRaises(InterventionError):
                load_vocabulary(path)

    def test_load_vocabulary_supports_metadata_wrapper(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "vocabulary.json"
            path.write_text(
                json.dumps({"attribute_to_index": {"beta": 1, "alpha": 0}, "size": 2}),
                encoding="utf-8",
            )
            vocabulary = load_vocabulary(path)
        self.assertEqual(vocabulary.attributes, ("alpha", "beta"))

    def test_load_vocabulary_supports_index_to_attribute_mapping(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "vocabulary.json"
            path.write_text(json.dumps({"0": "alpha", "1": "beta"}), encoding="utf-8")
            vocabulary = load_vocabulary(path)
        self.assertEqual(vocabulary.attributes, ("alpha", "beta"))


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.vocabulary = Vocabulary(
            attributes=("a", "b", "c"),
            attribute_to_index={"a": 0, "b": 1, "c": 2},
        )

    def sample(self):
        return sample_interventions(
            supported_items_by_attribute={"a": [1], "b": [2], "c": [3, 4, 5]},
            eligible_history_item_ids=[1, 2, 3, 4, 5, 6],
            vocabulary=self.vocabulary,
            requested_count=15,
            max_history_drop_fraction=0.50,
            subset_probability=0.50,
            random_seed=42,
            pair_index=7,
            user_id=9,
            target_item_id=10,
            exhaustive_attribute_limit=20,
            max_sampling_attempts=100,
        )

    def test_sampling_is_deterministic_distinct_and_enforces_history_cap(self):
        first = self.sample()
        second = self.sample()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(
            tuple(sample.attributes for sample in first),
            (("a",), ("b",), ("c",), ("a", "b")),
        )
        self.assertEqual(len({sample.attributes for sample in first}), len(first))
        self.assertTrue(all(len(sample.removed_item_ids) <= 3 for sample in first))

    def test_sampling_keeps_all_singletons_when_they_exceed_requested_count(self):
        samples = sample_interventions(
            supported_items_by_attribute={"a": [1], "b": [2], "c": [3]},
            eligible_history_item_ids=[1, 2, 3, 4],
            vocabulary=self.vocabulary,
            requested_count=2,
            max_history_drop_fraction=1.0,
            subset_probability=0.50,
            random_seed=42,
            pair_index=0,
            user_id=0,
            target_item_id=0,
            exhaustive_attribute_limit=20,
            max_sampling_attempts=100,
        )

        self.assertEqual(
            tuple(sample.attributes for sample in samples),
            (("a",), ("b",), ("c",)),
        )

    def test_random_sampling_adds_multi_attribute_subsets_after_singletons(self):
        samples = sample_interventions(
            supported_items_by_attribute={"a": [1], "b": [2], "c": [3]},
            eligible_history_item_ids=[1, 2, 3, 4],
            vocabulary=self.vocabulary,
            requested_count=4,
            max_history_drop_fraction=1.0,
            subset_probability=0.50,
            random_seed=42,
            pair_index=0,
            user_id=0,
            target_item_id=0,
            exhaustive_attribute_limit=2,
            max_sampling_attempts=100,
        )

        self.assertEqual(
            tuple(sample.attributes for sample in samples[:3]),
            (("a",), ("b",), ("c",)),
        )
        self.assertEqual(len(samples), 4)
        self.assertGreaterEqual(len(samples[3].attributes), 2)

    def test_sampling_rejects_support_attribute_missing_from_vocabulary(self):
        with self.assertRaises(InterventionError):
            sample_interventions(
                supported_items_by_attribute={"missing": [1]},
                eligible_history_item_ids=[1, 2],
                vocabulary=self.vocabulary,
                requested_count=1,
                max_history_drop_fraction=0.50,
                subset_probability=0.50,
                random_seed=42,
                pair_index=0,
                user_id=0,
                target_item_id=0,
                exhaustive_attribute_limit=20,
                max_sampling_attempts=10,
            )


class IdSpaceTests(unittest.TestCase):
    def test_stale_schema_version_1_support_record_is_rejected(self):
        with self.assertRaisesRegex(InterventionError, "Regenerate schema-version-2 support"):
            parse_support_record(
                {
                    "schema_version": 1,
                    "pair_index": 0,
                    "user_id": 241,
                    "target_item_id": 5371,
                    "supported_items_by_attribute": {},
                },
                1,
                make_mappings(users=[241], items=[5371]),
            )

    def test_scoring_uses_internal_indices_while_result_preserves_raw_ids(self):
        class RecordingScorer:
            def __init__(self):
                self.calls = []

            def validate_pair_ids(self, user_id, target_item_id):
                self.calls.append(("validate", user_id, target_item_id))

            def score(self, user_id, target_item_id, removed_item_ids):
                self.calls.append(("score", user_id, target_item_id, tuple(removed_item_ids)))
                return 1.0 - 0.1 * len(tuple(removed_item_ids))

            def score_many(self, user_id, target_item_id, removed_item_id_groups):
                groups = tuple(tuple(item_ids) for item_ids in removed_item_id_groups)
                self.calls.append(("score_many", user_id, target_item_id, groups))
                return tuple(1.0 - 0.1 * len(item_ids) for item_ids in groups)

        scorer = RecordingScorer()
        config = InterventionConfig(
            attribute_support_path=Path("support.jsonl"),
            vocabulary_path=Path("vocabulary.json"),
            user_history_path=Path("history.json"),
            id_mappings_path=Path("id_mappings.json"),
            user_ego_embeddings_path=Path("users.pt"),
            item_ego_embeddings_path=Path("items.pt"),
            lightgcn_config_path=Path("lightgcn.yaml"),
            output_dir=Path("output"),
            sparsity_level=1,
            sample_multiplier=1,
            max_history_drop_fraction=0.50,
            subset_probability=0.50,
            random_seed=42,
            exhaustive_attribute_limit=20,
            max_sampling_attempts=100,
            device="cpu",
            pairs_per_shard=1,
            score_cache_max_entries=10,
        )
        result = build_pair_result(
            pair_index=0,
            user_id=241,
            user_index=673,
            target_item_id=5371,
            target_item_index=5361,
            supports={"a": [400]},
            graph=build_graph({673: [400, 401, 5361]}),
            vocabulary=Vocabulary(("a",), {"a": 0}),
            score_cache=ScoreCache(scorer, 10),
            config=config,
            stats=RunStats(),
        )

        self.assertEqual((result.user_id, result.user_index), (241, 673))
        self.assertEqual((result.target_item_id, result.target_item_index), (5371, 5361))
        self.assertTrue(all(call[1:3] == (673, 5361) for call in scorer.calls))
        self.assertEqual(
            scorer.calls,
            [
                ("validate", 673, 5361),
                ("score_many", 673, 5361, ((),)),
                ("score_many", 673, 5361, ((400,),)),
            ],
        )
        self.assertEqual(result.rows[0].removed_item_ids, (400,))


class ScoreCacheTests(unittest.TestCase):
    def test_score_many_batches_unique_misses_and_preserves_order(self):
        class RecordingScorer:
            def __init__(self):
                self.calls = []

            def score_many(self, user_id, target_item_id, removed_item_id_groups):
                groups = tuple(tuple(item_ids) for item_ids in removed_item_id_groups)
                self.calls.append((user_id, target_item_id, groups))
                return tuple(float(sum(item_ids)) for item_ids in groups)

            def validate_pair_ids(self, user_id, target_item_id):
                pass

        scorer = RecordingScorer()
        cache = ScoreCache(scorer, 10)
        self.assertEqual(cache.score(1, 2, [9]), 9.0)

        scores = cache.score_many(1, 2, ([9], [2, 1], [1, 2], [3]))

        self.assertEqual(scores, (9.0, 3.0, 3.0, 3.0))
        self.assertEqual(
            scorer.calls,
            [
                (1, 2, ((9,),)),
                (1, 2, ((1, 2), (3,))),
            ],
        )
        self.assertEqual(cache.hit_count, 2)
        self.assertEqual(cache.miss_count, 3)


@unittest.skipIf(torch is None, "PyTorch is not installed.")
class HybridScoringTests(unittest.TestCase):
    def setUp(self):
        # Collaborative paths exist through user 1 and prove this is more than an isolated user star.
        self.graph = build_graph({0: [0, 1, 2], 1: [1, 2, 3], 2: [0, 3]})
        self.user_embeddings = torch.tensor(
            [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=torch.float32
        )
        self.item_embeddings = torch.tensor(
            [[0.2, 0.8], [1.0, 0.1], [0.3, 0.7], [0.9, 0.4]], dtype=torch.float32
        )

    def test_hybrid_local_score_matches_brute_force_propagation(self):
        for layer_count in (0, 1, 2, 3):
            scorer = HybridLocalLightGCNScorer(
                self.graph, self.user_embeddings, self.item_embeddings, layer_count
            )
            removed_edges = frozenset({(0, 1), (0, 2)})
            actual = scorer.score_with_removed_edges(0, 0, removed_edges)
            expected = brute_force_score(
                self.graph,
                self.user_embeddings,
                self.item_embeddings,
                layer_count,
                0,
                0,
                removed_edges,
            )
            self.assertAlmostEqual(actual, expected, places=6)

    def test_batched_hybrid_local_scores_match_brute_force_propagation(self):
        removed_edge_groups = (
            frozenset(),
            frozenset({(0, 1)}),
            frozenset({(0, 1), (0, 2)}),
        )
        for layer_count in (0, 1, 2, 3):
            scorer = HybridLocalLightGCNScorer(
                self.graph, self.user_embeddings, self.item_embeddings, layer_count
            )
            actual = scorer.score_many_with_removed_edges(0, 0, removed_edge_groups)
            expected = tuple(
                brute_force_score(
                    self.graph,
                    self.user_embeddings,
                    self.item_embeddings,
                    layer_count,
                    0,
                    0,
                    removed_edges,
                )
                for removed_edges in removed_edge_groups
            )
            for actual_score, expected_score in zip(actual, expected, strict=True):
                self.assertAlmostEqual(actual_score, expected_score, places=6)

    def test_score_removes_target_edge_for_leave_one_out_baseline(self):
        scorer = HybridLocalLightGCNScorer(
            self.graph, self.user_embeddings, self.item_embeddings, layer_count=2
        )
        self.assertAlmostEqual(
            scorer.score(0, 0, ()),
            scorer.score_with_removed_edges(0, 0, frozenset({(0, 0)})),
            places=6,
        )

    def test_zero_degree_node_after_edge_removal_is_supported(self):
        graph = build_graph({0: [0], 1: [0]})
        scorer = HybridLocalLightGCNScorer(
            graph,
            torch.tensor([[1.0], [2.0]]),
            torch.tensor([[3.0]]),
            layer_count=2,
        )
        score = scorer.score_with_removed_edges(0, 0, frozenset({(0, 0)}))
        self.assertTrue(math.isfinite(score))

    def test_batched_score_supports_mixed_zero_and_non_zero_degrees(self):
        graph = build_graph({0: [0], 1: [0]})
        scorer = HybridLocalLightGCNScorer(
            graph,
            torch.tensor([[1.0], [2.0]]),
            torch.tensor([[3.0]]),
            layer_count=2,
        )
        scores = scorer.score_many_with_removed_edges(
            0,
            0,
            (frozenset({(0, 0)}), frozenset()),
        )
        expected = tuple(
            brute_force_score(
                graph,
                torch.tensor([[1.0], [2.0]]),
                torch.tensor([[3.0]]),
                2,
                0,
                0,
                removed_edges,
            )
            for removed_edges in (frozenset({(0, 0)}), frozenset())
        )
        for actual_score, expected_score in zip(scores, expected, strict=True):
            self.assertTrue(math.isfinite(actual_score))
            self.assertAlmostEqual(actual_score, expected_score, places=6)

    def test_generation_can_resume_from_completed_shard(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            support_path = root / "support.jsonl"
            vocabulary_path = root / "vocabulary.json"
            history_path = root / "history.json"
            id_mappings_path = root / "id_mappings.json"
            user_embeddings_path = root / "users.pt"
            item_embeddings_path = root / "items.pt"
            lightgcn_config_path = root / "lightgcn.yaml"
            output_dir = root / "output"

            support_records = [
                {
                    "schema_version": 2,
                    "pair_index": 0,
                    "user_id": 0,
                    "user_index": 0,
                    "target_item_id": 0,
                    "target_item_index": 0,
                    "supported_items_by_attribute": {"a": [{"item_id": 1, "item_index": 1}]},
                },
                {
                    "schema_version": 2,
                    "pair_index": 1,
                    "user_id": 0,
                    "user_index": 0,
                    "target_item_id": 2,
                    "target_item_index": 2,
                    "supported_items_by_attribute": {"a": [{"item_id": 1, "item_index": 1}]},
                },
                {
                    "schema_version": 2,
                    "pair_index": 2,
                    "user_id": 0,
                    "user_index": 0,
                    "target_item_id": 99,
                    "target_item_index": None,
                    "supported_items_by_attribute": {"a": [{"item_id": 1, "item_index": 1}]},
                },
            ]
            support_path.write_text(
                "".join(json.dumps(record) + "\n" for record in support_records),
                encoding="utf-8",
            )
            vocabulary_path.write_text(json.dumps(["a"]), encoding="utf-8")
            history_path.write_text(json.dumps({"0": [0, 1, 2, 3], "1": [1]}), encoding="utf-8")
            write_mappings(id_mappings_path, users=[0, 1], items=[0, 1, 2, 3])
            torch.save(torch.tensor([[1.0], [2.0]]), user_embeddings_path)
            torch.save(torch.tensor([[1.0], [2.0], [3.0], [4.0]]), item_embeddings_path)
            lightgcn_config_path.write_text("num_layers: 1\n", encoding="utf-8")

            config = InterventionConfig(
                attribute_support_path=support_path,
                vocabulary_path=vocabulary_path,
                user_history_path=history_path,
                id_mappings_path=id_mappings_path,
                user_ego_embeddings_path=user_embeddings_path,
                item_ego_embeddings_path=item_embeddings_path,
                lightgcn_config_path=lightgcn_config_path,
                output_dir=output_dir,
                sparsity_level=1,
                sample_multiplier=1,
                max_history_drop_fraction=0.50,
                subset_probability=0.50,
                random_seed=42,
                exhaustive_attribute_limit=20,
                max_sampling_attempts=100,
                device="cpu",
                pairs_per_shard=1,
                score_cache_max_entries=10,
            )

            first_summary = generate_intervention_artifacts(config, limit=1)
            resumed_summary = generate_intervention_artifacts(config, resume=True)
            manifests = [
                json.loads(line)
                for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            run_config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))

            self.assertEqual(first_summary["processed_pair_count"], 1)
            self.assertEqual(resumed_summary["processed_pair_count"], 3)
            self.assertEqual(
                run_config["intervention_generation_version"],
                INTERVENTION_GENERATION_VERSION,
            )
            self.assertEqual(resumed_summary["written_shard_count"], 3)
            self.assertEqual(resumed_summary["skipped_pair_count"], 1)
            self.assertEqual(resumed_summary["invalid_pair_id_skip_count"], 1)
            self.assertEqual(len(manifests), 3)
            self.assertEqual(manifests[0]["generated_intervention_count"], 1)
            self.assertEqual(manifests[1]["generated_intervention_count"], 1)
            self.assertEqual(manifests[2]["generated_intervention_count"], 0)
            self.assertIn("absent from LightGCN mappings", manifests[2]["skip_reason"])
            self.assertGreater(resumed_summary["full_time_seconds"], 0.0)
            self.assertAlmostEqual(
                resumed_summary["average_time_per_pair_seconds"],
                resumed_summary["full_time_seconds"] / 3,
            )
            self.assertNotIn("previous_full_time_seconds", resumed_summary)


class ShardTests(unittest.TestCase):
    def test_resume_allows_checkpoint_granularity_change(self):
        previous = {
            "intervention_generation_version": INTERVENTION_GENERATION_VERSION,
            "config": {"pairs_per_shard": 100, "random_seed": 42},
            "source_checksums": {"support": "abc"},
        }
        expected = {
            "intervention_generation_version": INTERVENTION_GENERATION_VERSION,
            "config": {"pairs_per_shard": 1, "random_seed": 42},
            "source_checksums": {"support": "abc"},
        }
        self.assertTrue(_resume_configs_match(previous, expected))

    def test_resume_rejects_missing_intervention_generation_version(self):
        previous = {
            "config": {"pairs_per_shard": 100, "random_seed": 42},
            "source_checksums": {"support": "abc"},
        }
        expected = {
            "intervention_generation_version": INTERVENTION_GENERATION_VERSION,
            "config": {"pairs_per_shard": 100, "random_seed": 42},
            "source_checksums": {"support": "abc"},
        }
        self.assertFalse(_resume_configs_match(previous, expected))

    def test_write_shard_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            shard_path = Path(temporary_directory) / "interventions_000000.npz"
            manifests = write_shard(
                shard_path=shard_path,
                shard_relative_path="shards/interventions_000000.npz",
                pair_results=[
                    PairResult(
                        pair_index=5,
                        user_id=1,
                        user_index=11,
                        target_item_id=2,
                        target_item_index=12,
                        baseline_score=0.9,
                        eligible_history_count=4,
                        supported_target_attribute_count=2,
                        requested_intervention_count=15,
                        rows=(
                            InterventionRow(
                                attribute_indices=(1, 3),
                                removed_item_ids=(7, 8),
                                y_h=0.4,
                                y_delta=0.5,
                            ),
                        ),
                    )
                ],
                vocabulary_size=6,
            )
            with np.load(shard_path) as shard:
                self.assertEqual(shard["A_shape"].tolist(), [1, 6])
                self.assertEqual(shard["A_indices"].tolist(), [1, 3])
                self.assertEqual(shard["A_indptr"].tolist(), [0, 2])
                self.assertEqual(shard["removed_item_ids"].tolist(), [7, 8])
                self.assertEqual(shard["removed_item_indptr"].tolist(), [0, 2])
                self.assertAlmostEqual(float(shard["y_delta"][0]), 0.5)
            self.assertEqual(manifests[0]["row_start"], 0)
            self.assertEqual(manifests[0]["row_end"], 1)
            self.assertEqual(manifests[0]["user_index"], 11)
            self.assertEqual(manifests[0]["target_item_index"], 12)


if __name__ == "__main__":
    unittest.main()
