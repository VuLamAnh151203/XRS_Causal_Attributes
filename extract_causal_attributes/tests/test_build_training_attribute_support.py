import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from extract_causal_attributes.build_training_attribute_support import (
    AttributeMatcher,
    EmbeddingIndex,
    GenerationStats,
    GeneratorConfig,
    SchemaError,
    TrainingPair,
    build_support_record,
    convert_internal_user_histories_to_raw,
    generate_artifact,
    load_item_attributes,
    parse_training_pairs,
)
from extract_causal_attributes.id_mappings import IdMappingError, IdMappings, load_id_mappings


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


class StubMatcher:
    def __init__(self, matches):
        self.matches = matches

    def best_match(self, target_attribute, history_item_id):
        return self.matches.get((target_attribute, history_item_id))


class FakeEmbedder:
    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts, batch_size):
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


class FakeDataFrame:
    columns = ("user_id", "item_id")

    def to_dict(self, orient):
        if orient != "records":
            raise AssertionError("Expected record conversion.")
        return [{"user_id": "7", "item_id": "8"}]


class PositionalFakeDataFrame:
    columns = (0, 1)

    def to_dict(self, orient):
        raise AssertionError("Positional DataFrame fallback should use itertuples.")

    def itertuples(self, index, name):
        if index or name is not None:
            raise AssertionError("Expected positional tuple conversion.")
        return [(11, "12")]


class AttributeSupportTests(unittest.TestCase):
    def test_mapping_loader_resolves_raw_user_241_to_internal_row_673(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "id_mappings.json"
            users = [f"user-{index}" for index in range(674)]
            users[673] = "241"
            write_mappings(path, users=users, items=[400])

            mappings = load_id_mappings(path)

        self.assertEqual(mappings.user_index(241), 673)
        self.assertEqual(mappings.raw_user_id(673), "241")

    def test_mapping_loader_rejects_inconsistent_inverse(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "id_mappings.json"
            path.write_text(
                json.dumps(
                    {
                        "user_to_index": {"241": 0},
                        "item_to_index": {"400": 0},
                        "index_to_user": ["999"],
                        "index_to_item": ["400"],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(IdMappingError):
                load_id_mappings(path)

    def test_internal_history_indices_convert_to_raw_item_ids_before_matching(self):
        mappings = make_mappings(users=[349, 241], items=[5371, 400])
        raw_histories = convert_internal_user_histories_to_raw({"1": [1]}, mappings)
        record = build_support_record(
            pair=TrainingPair(0, 241, 5371),
            user_histories=raw_histories,
            item_attributes={400: ["related"], 5371: ["target"]},
            matcher=StubMatcher({("target", 400): (0.9, "related")}),
            threshold=0.70,
            stats=GenerationStats(),
            id_mappings=mappings,
        )

        self.assertEqual(raw_histories, {"241": [400]})
        self.assertEqual(record["user_index"], 1)
        self.assertEqual(
            record["supported_items_by_attribute"]["target"],
            [{"item_id": 400, "item_index": 1, "score": 0.9, "matched_attribute": "related"}],
        )

    def test_item_attribute_loader_concatenates_normalizes_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "attributes.json"
            path.write_text(
                json.dumps(
                    {
                        "10": {
                            "explicit_attributes": [" lightweight ", "", "durable"],
                            "implicit_attributes": ["durable", "easy to clean"],
                        }
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                load_item_attributes(path),
                {10: ["lightweight", "durable", "easy to clean"]},
            )

    def test_training_pair_parser_supports_records_columns_tuples_and_dataframes(self):
        self.assertEqual(
            parse_training_pairs([{"user_id": "1", "target_item_id": "2"}, (3, "4")]),
            [TrainingPair(0, "1", 2), TrainingPair(1, 3, 4)],
        )
        self.assertEqual(
            parse_training_pairs({"user_id": ["5"], "item_id": ["6"]}),
            [TrainingPair(0, "5", 6)],
        )
        self.assertEqual(
            parse_training_pairs({"user_id": "9", "item_id": "10"}),
            [TrainingPair(0, "9", 10)],
        )
        self.assertEqual(parse_training_pairs(FakeDataFrame()), [TrainingPair(0, "7", 8)])
        self.assertEqual(parse_training_pairs(PositionalFakeDataFrame()), [TrainingPair(0, 11, 12)])
        self.assertEqual(parse_training_pairs(np.asarray([[13, 14]])), [TrainingPair(0, 13, 14)])

    def test_training_pair_parser_rejects_unsupported_schema(self):
        with self.assertRaises(SchemaError):
            parse_training_pairs([{"unknown": 1}])

    def test_embedding_matcher_selects_best_history_attribute(self):
        attributes = ["lightweight", "portable", "low weight", "heavy"]
        vectors = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.6],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        )
        matcher = AttributeMatcher(
            EmbeddingIndex(attributes, vectors),
            {2: ["heavy", "low weight"]},
            comparison_chunk_size=1,
            cache_max_entries=10,
        )

        score, matched_attribute = matcher.best_match("lightweight", 2)
        self.assertAlmostEqual(score, 0.8)
        self.assertEqual(matched_attribute, "low weight")

    def test_support_record_uses_strict_threshold_and_excludes_target_and_duplicates(self):
        matcher = StubMatcher(
            {
                ("lightweight", 2): (0.70, "portable"),
                ("lightweight", 3): (0.9, "low weight"),
                ("lightweight", 4): (0.8, "small"),
            }
        )
        stats = GenerationStats()
        record = build_support_record(
            pair=TrainingPair(0, 1, 9),
            user_histories={"1": [9, 2, 3, 3, 4]},
            item_attributes={
                2: ["portable"],
                3: ["low weight"],
                4: ["small"],
                9: ["lightweight"],
            },
            matcher=matcher,
            threshold=0.70,
            stats=stats,
            id_mappings=make_mappings(users=[1], items=[9, 2, 3, 4]),
        )

        self.assertEqual(
            record["supported_items_by_attribute"]["lightweight"],
            [
                {"item_id": 3, "item_index": 2, "score": 0.9, "matched_attribute": "low weight"},
                {"item_id": 4, "item_index": 3, "score": 0.8, "matched_attribute": "small"},
            ],
        )
        self.assertEqual(stats.semantic_match_count, 2)

    def test_support_record_emits_empty_values_and_counts_missing_data(self):
        stats = GenerationStats()
        record = build_support_record(
            pair=TrainingPair(0, "missing-user", 99),
            user_histories={},
            item_attributes={},
            matcher=StubMatcher({}),
            threshold=0.70,
            stats=stats,
            id_mappings=make_mappings(),
        )

        self.assertEqual(record["target_attributes"], [])
        self.assertEqual(record["supported_items_by_attribute"], {})
        self.assertEqual(stats.missing_history_count, 1)
        self.assertEqual(stats.missing_user_mapping_count, 1)
        self.assertEqual(stats.missing_target_item_mapping_count, 1)
        self.assertEqual(stats.missing_target_attribute_count, 1)

    def test_support_record_uses_history_position_to_break_score_ties(self):
        matcher = StubMatcher(
            {
                ("target", 2): (0.8, "related"),
                ("target", 3): (0.8, "related"),
            }
        )
        record = build_support_record(
            pair=TrainingPair(0, 1, 9),
            user_histories={"1": [3, 2]},
            item_attributes={2: ["related"], 3: ["related"], 9: ["target"]},
            matcher=matcher,
            threshold=0.70,
            stats=GenerationStats(),
            id_mappings=make_mappings(users=[1], items=[9, 3, 2]),
        )

        self.assertEqual(
            [item["item_id"] for item in record["supported_items_by_attribute"]["target"]],
            [3, 2],
        )

    def test_generate_artifact_writes_aligned_jsonl_and_summary(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            training_pairs_path = root / "trn.pkl"
            item_attributes_path = root / "attributes.json"
            user_history_path = root / "history.json"
            id_mappings_path = root / "id_mappings.json"
            output_path = root / "output.jsonl"
            summary_path = root / "summary.json"

            with training_pairs_path.open("wb") as output_file:
                pickle.dump([(1, 9), (2, 10)], output_file)
            item_attributes_path.write_text(
                json.dumps(
                    {
                        "9": {"explicit": ["lightweight"], "implicit": []},
                        "10": {"explicit": ["missing history target"], "implicit": []},
                        "3": {"explicit": ["low weight"], "implicit": []},
                    }
                ),
                encoding="utf-8",
            )
            user_history_path.write_text(json.dumps({"0": [2]}), encoding="utf-8")
            write_mappings(id_mappings_path, users=[1], items=[9, 10, 3])

            config = GeneratorConfig(
                training_pairs_path=training_pairs_path,
                item_attributes_path=item_attributes_path,
                user_history_path=user_history_path,
                id_mappings_path=id_mappings_path,
                output_path=output_path,
                summary_output_path=summary_path,
                model_name="fake",
                threshold=0.70,
                batch_size=2,
                device="cpu",
                comparison_chunk_size=2,
                comparison_cache_max_entries=10,
            )
            summary = generate_artifact(
                config,
                embedder=FakeEmbedder(
                    {
                        "lightweight": [1.0, 0.0],
                        "missing history target": [0.0, 1.0],
                        "low weight": [0.8, 0.6],
                    }
                ),
            )

            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            persisted_summary = json.loads(summary_path.read_text(encoding="utf-8"))

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["supported_items_by_attribute"]["lightweight"][0]["item_id"], 3)
            self.assertEqual(rows[0]["user_index"], 0)
            self.assertEqual(rows[0]["target_item_index"], 0)
            self.assertEqual(
                rows[0]["supported_items_by_attribute"]["lightweight"][0]["item_index"], 2
            )
            self.assertEqual(rows[1]["supported_items_by_attribute"], {"missing history target": []})
            self.assertEqual(summary["processed_pair_count"], 2)
            self.assertEqual(persisted_summary["emitted_row_count"], 2)
            self.assertEqual(persisted_summary["missing_history_count"], 1)
            self.assertEqual(persisted_summary["missing_user_mapping_count"], 1)


if __name__ == "__main__":
    unittest.main()
