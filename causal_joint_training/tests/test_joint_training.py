import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from causal_joint_training.config import (
    GenerationConfig,
    JointTrainingConfig,
    ModelConfig,
    NegativeSamplingConfig,
    PathsConfig,
    TrainingConfig,
)
from causal_joint_training.data import (
    ArtifactError,
    CausalLabel,
    ExplanationExample,
    IdMappings,
    load_omp_labels,
)
from causal_joint_training.model import (
    CausalJointModel,
    FrozenSoftPromptLM,
    hard_prompt,
    signed_target_distribution,
)
from causal_joint_training.negative_sampling import (
    causal_attribute_similarity,
    precompute_top_cf_unseen,
    select_semihard_negative,
)
from causal_joint_training.training import (
    ExplanationDataset,
    RecommendationDataset,
    _identity_collate,
    build_labeled_pair_split,
    checkpoint_payload,
    examples_to_pair_tensor,
    examples_to_evaluation_pairs,
    merge_history_with_examples,
    select_omp_labeled_explanations,
    train_alternating_epoch,
    train_causal_warmup,
)


def mappings():
    return IdMappings(
        user_to_index={"241": 0, "7": 1},
        item_to_index={"5371": 0, "400": 1, "401": 2},
        index_to_user=("241", "7"),
        index_to_item=("5371", "400", "401"),
    )


def label(pair_index=0, user_index=0, item_index=0):
    return CausalLabel(
        pair_index=pair_index,
        user_id="241",
        user_index=user_index,
        item_id="5371",
        item_index=item_index,
        attribute_indices=(0, 1),
        coefficients=(0.8, -0.2),
        relative_residual=0.1,
    )


def model_config():
    return ModelConfig(
        embedding_dim=4,
        extractor_hidden_dim=8,
        preference_hidden_dim=6,
        preference_dim=4,
    )


def full_config(root: Path, warmup_epochs=1):
    paths = PathsConfig(**{key: root / key for key in PathsConfig.__annotations__})
    return JointTrainingConfig(
        paths=paths,
        model=model_config(),
        negative_sampling=NegativeSamplingConfig(
            candidate_pool_size=2,
            predicted_attribute_count=2,
            similarity_threshold=0.5,
            user_batch_size=2,
        ),
        generation=GenerationConfig(
            model_name="fake",
            load_in_4bit=False,
            causal_prompt_tokens=4,
            user_prompt_tokens=2,
            item_prompt_tokens=2,
            prompt_hidden_dim=8,
            max_text_tokens=32,
            max_new_tokens=2,
        ),
        training=TrainingConfig(
            causal_warmup_epochs=warmup_epochs,
            joint_epochs=1,
            recommendation_batch_size=1,
            causal_batch_size=2,
            generation_batch_size=1,
            gradient_accumulation_steps=1,
            alternating_update_order=("recommendation", "generation"),
        ),
    )


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, add_special_tokens=True):
        values = [3 + (ord(character) % 10) for character in text]
        return ([1] if add_special_tokens else []) + values

    def batch_decode(self, values, skip_special_tokens=True):
        return ["generated" for _ in values]


class FakeLM(nn.Module):
    def __init__(self, hidden_size=6):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(32, hidden_size)
        self.last_labels = None

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask, labels):
        self.last_labels = labels.detach().clone()
        return SimpleNamespace(loss=inputs_embeds.square().mean())

    def generate(self, inputs_embeds, attention_mask, max_new_tokens):
        return torch.tensor([[5, 6]] * inputs_embeds.shape[0], device=inputs_embeds.device)


class ArtifactTests(unittest.TestCase):
    def _write_omp(self, root: Path, schema_version=2, residual=0.2):
        (root / "shards").mkdir()
        (root / "vocabulary.json").write_text(json.dumps({"attributes": ["a", "b"]}), encoding="utf-8")
        np.savez_compressed(
            root / "shards" / "vectors.npz",
            coef_data=np.asarray([0.8, -0.2], dtype=np.float32),
            coef_indices=np.asarray([0, 1], dtype=np.int64),
            coef_indptr=np.asarray([0, 2], dtype=np.int64),
            coef_shape=np.asarray([1, 2], dtype=np.int64),
            pair_index=np.asarray([0], dtype=np.int64),
            user_index=np.asarray([0], dtype=np.int64),
            target_item_index=np.asarray([0], dtype=np.int64),
        )
        record = {
            "schema_version": schema_version,
            "pair_index": 0,
            "user_id": 241,
            "user_index": 0,
            "target_item_id": 5371,
            "target_item_index": 0,
            "status": "recovered",
            "vector_shard": "shards/vectors.npz",
            "vector_row": 0,
            "diagnostics": {"relative_residual": residual},
        }
        (root / "manifest.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    def test_loads_schema_v2_omp_and_preserves_signed_coefficients(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_omp(root)
            labels, stats = load_omp_labels(root, mappings(), ("a", "b"), 0.5)
        self.assertEqual(labels[0].coefficients, (0.800000011920929, -0.20000000298023224))
        self.assertEqual((labels[0].user_index, labels[0].item_index), (0, 0))
        self.assertEqual(stats["accepted_rows"], 1)

    def test_rejects_stale_omp_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_omp(root, schema_version=1)
            with self.assertRaisesRegex(ArtifactError, "Regenerate schema-version-2"):
                load_omp_labels(root, mappings(), ("a", "b"), 0.5)

    def test_filters_noisy_omp_label(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_omp(root, residual=0.6)
            labels, stats = load_omp_labels(root, mappings(), ("a", "b"), 0.5)
        self.assertEqual(labels, {})
        self.assertEqual(stats["skipped_quality_rows"], 1)

    def test_rejects_inconsistent_raw_and_internal_ids(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_omp(root)
            manifest_path = root / "manifest.jsonl"
            record = json.loads(manifest_path.read_text(encoding="utf-8"))
            record["user_index"] = 1
            manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactError, "does not match raw ID"):
                load_omp_labels(root, mappings(), ("a", "b"), 0.5)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.users = torch.randn(2, 4)
        self.items = torch.randn(3, 4)
        self.model = CausalJointModel(self.users, self.items, vocabulary_size=3, config=model_config())

    def test_signed_target_distribution_uses_two_channels(self):
        target = signed_target_distribution([label()], vocabulary_size=3)
        self.assertAlmostEqual(float(target[0, 0]), 0.8)
        self.assertAlmostEqual(float(target[0, 4]), 0.2)
        self.assertAlmostEqual(float(target.sum()), 1.0)

    def test_model_shapes_and_frozen_embeddings(self):
        users = torch.tensor([0, 1])
        items = torch.tensor([0, 1])
        logits, probabilities, dense = self.model.causal_outputs(users, items)
        preference, _, _, _ = self.model.preference(users, items)
        self.assertEqual(tuple(logits.shape), (2, 6))
        self.assertEqual(tuple(probabilities.shape), (2, 6))
        self.assertEqual(tuple(dense.shape), (2, 4))
        self.assertEqual(tuple(preference.shape), (2, 4))
        self.assertFalse(self.model.user_embeddings.weight.requires_grad)
        self.assertFalse(self.model.item_embeddings.weight.requires_grad)

    def test_combined_score_reduces_to_cf_when_causal_projection_is_zero(self):
        nn.init.zeros_(self.model.causal_signal.weight)
        nn.init.zeros_(self.model.causal_signal.bias)
        users = torch.tensor([0, 1])
        items = torch.tensor([0, 1])
        expected = (self.users[users] * self.items[items]).sum(dim=-1)
        self.assertTrue(torch.allclose(self.model.score_pairs(users, items), expected))

    def test_checkpoint_excludes_frozen_embeddings_and_llm(self):
        generator = FrozenSoftPromptLM(FakeLM(), FakeTokenizer(), 4, 4, full_config(Path(".")).generation)
        payload = checkpoint_payload(self.model, generator, {"epoch": 1})
        self.assertNotIn("user_embeddings", payload["model"])
        self.assertNotIn("item_embeddings", payload["model"])
        self.assertNotIn("llm", payload["prompt_generators"])


class NegativeSamplingTests(unittest.TestCase):
    def test_similarity_matches_average_maximum_cosine_formula(self):
        semantic = torch.eye(3)
        similarity = causal_attribute_similarity([0, 1], [0, 2], semantic)
        self.assertAlmostEqual(similarity, 0.5)

    def test_selects_first_high_cf_candidate_below_threshold(self):
        semantic = torch.eye(3)
        selection = select_semihard_negative(
            [1, 2], [0], {1: (0,), 2: (2,)}, semantic, similarity_threshold=0.5
        )
        self.assertEqual(selection.item_index, 2)
        self.assertEqual(selection.strategy, "threshold_match")

    def test_uses_least_similar_and_missing_attribute_fallbacks(self):
        semantic = torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.6, 0.8]])
        least = select_semihard_negative([1, 2], [0], {1: (1,), 2: (2,)}, semantic, 0.5)
        missing = select_semihard_negative([1, 2], [0], {}, semantic, 0.5)
        self.assertEqual((least.item_index, least.strategy), (2, "least_similar_fallback"))
        self.assertEqual((missing.item_index, missing.strategy), (1, "missing_attribute_fallback"))

    def test_precomputes_only_unseen_high_cf_items(self):
        pools = precompute_top_cf_unseen(
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[1.0, 0.0], [0.8, 0.0], [0.5, 0.0]]),
            {0: {0}},
            pool_size=2,
            user_batch_size=1,
        )
        self.assertEqual(pools[0], (1, 2))


class GenerationAndTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = CausalJointModel(torch.randn(2, 4), torch.randn(3, 4), 3, model_config())
        self.config = full_config(Path("."))
        self.generator = FrozenSoftPromptLM(FakeLM(), FakeTokenizer(), 4, 4, self.config.generation)
        self.example = ExplanationExample(
            pair_index=0,
            user_id="241",
            user_index=0,
            item_id="5371",
            item_index=0,
            title="Book",
            user_profile="User summary",
            item_profile="Item summary",
            explanation="Target text",
        )

    def _optimizer(self):
        return AdamW(
            [
                {"params": [p for p in self.model.parameters() if p.requires_grad]},
                {
                    "params": [
                        p
                        for module in (
                            self.generator.causal_prompt,
                            self.generator.user_prompt,
                            self.generator.item_prompt,
                        )
                        for p in module.parameters()
                    ]
                },
            ],
            lr=1.0e-3,
        )

    def _prompt_vector(self):
        return torch.cat(
            [
                parameter.detach().cpu().flatten()
                for module in (
                    self.generator.causal_prompt,
                    self.generator.user_prompt,
                    self.generator.item_prompt,
                )
                for parameter in module.parameters()
            ]
        )

    def test_soft_prefix_dimensions_and_explanation_only_masking(self):
        users, items = torch.tensor([0]), torch.tensor([0])
        preference, _, user_embeddings, item_embeddings = self.model.preference(users, items)
        prompt = hard_prompt("Explain.", "Book", "User", "Item")
        output = self.generator(preference, user_embeddings, item_embeddings, [prompt], ["Target"])
        prefix = self.generator.prefix_embeddings(preference, user_embeddings, item_embeddings)
        labels = self.generator.llm.last_labels[0]
        self.assertEqual(tuple(prefix.shape), (1, 8, 6))
        self.assertTrue(torch.all(labels[:8] == -100))
        self.assertTrue(torch.any(labels[8:] != -100))
        self.assertTrue(output.loss.requires_grad)
        self.assertTrue(all(not parameter.requires_grad for parameter in self.generator.llm.parameters()))

    def test_language_batch_preserves_target_labels_when_prompt_is_truncated(self):
        batch = self.generator.language_batch(["P" * 100], ["Target"], torch.device("cpu"))
        self.assertEqual(batch.input_ids.shape[1], self.config.generation.max_text_tokens)
        self.assertTrue(torch.any(batch.labels != -100))

    def test_selects_omp_labeled_subset_for_option_b_training(self):
        unlabeled = ExplanationExample(
            pair_index=9,
            user_id="7",
            user_index=1,
            item_id="400",
            item_index=1,
            title="Other",
            user_profile="User summary",
            item_profile="Item summary",
            explanation="Other target",
        )
        selected, stats = select_omp_labeled_explanations([self.example, unlabeled], {0: label(0, 0, 0)})
        pairs = examples_to_pair_tensor(selected)
        history = merge_history_with_examples({0: {2}, 1: set()}, selected)
        self.assertEqual([example.pair_index for example in selected], [0])
        self.assertEqual(stats["skipped_without_omp_label"], 1)
        self.assertEqual(pairs.tolist(), [[0, 0]])
        self.assertEqual(history[0], {0, 2})

    def test_rejects_omp_label_example_mismatch(self):
        mismatched = label(0, user_index=1, item_index=0)
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            select_omp_labeled_explanations([self.example], {0: mismatched})

    def test_labeled_pair_split_is_deterministic_and_disjoint(self):
        examples = [
            replace(self.example, pair_index=index, user_index=index % 2, item_index=index % 3)
            for index in range(10)
        ]
        labels = {
            example.pair_index: label(example.pair_index, example.user_index, example.item_index)
            for example in examples
        }
        first = build_labeled_pair_split(examples, labels, validation_ratio=0.2, seed=7)
        second = build_labeled_pair_split(examples, labels, validation_ratio=0.2, seed=7)
        train_ids = {example.pair_index for example in first.train_examples}
        validation_ids = {example.pair_index for example in first.validation_examples}
        self.assertEqual(
            [example.pair_index for example in first.validation_examples],
            [example.pair_index for example in second.validation_examples],
        )
        self.assertEqual(len(first.validation_examples), 2)
        self.assertFalse(train_ids & validation_ids)
        self.assertEqual(first.stats["labeled_train_count"], 8)

    def test_labeled_pair_split_keeps_validation_when_possible(self):
        examples = [self.example, replace(self.example, pair_index=1, user_index=1, item_index=1)]
        labels = {0: label(0, 0, 0), 1: label(1, 1, 1)}
        split = build_labeled_pair_split(examples, labels, validation_ratio=0.1, seed=0)
        self.assertEqual(len(split.train_examples), 1)
        self.assertEqual(len(split.validation_examples), 1)

    def test_omp_validation_pairs_skip_seen_history(self):
        validation_examples = [
            self.example,
            replace(self.example, pair_index=1, item_index=1),
        ]
        pairs, stats = examples_to_evaluation_pairs(validation_examples, {0: {0}})
        self.assertEqual(pairs, {0: {1}})
        self.assertEqual(stats["validation_skipped_seen"], 1)
        self.assertEqual(stats["validation_evaluable_pairs"], 1)

    def test_warmup_and_alternating_epoch_smoke(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = full_config(Path(temporary_directory))
            labels = [label(0, 0, 0), label(1, 1, 1)]
            history = train_causal_warmup(self.model, labels, labels[:1], config, torch.device("cpu"))
            recommendation_loader = DataLoader(
                RecommendationDataset(torch.tensor([[0, 0], [1, 1]], dtype=torch.long)),
                batch_size=1,
                shuffle=False,
            )
            explanation_loader = DataLoader(
                ExplanationDataset([self.example]),
                batch_size=1,
                shuffle=False,
                collate_fn=_identity_collate,
            )
            metrics = train_alternating_epoch(
                self.model,
                self.generator,
                recommendation_loader,
                explanation_loader,
                torch.tensor([1, 2]),
                labels,
                {0: labels[0]},
                self._optimizer(),
                config,
                torch.device("cpu"),
            )
        self.assertEqual(len(history), 1)
        self.assertGreater(metrics["loss"], 0.0)
        self.assertGreater(metrics["recommendation_kl_loss"], 0.0)
        self.assertGreater(metrics["generation_kl_loss"], 0.0)
        self.assertEqual(metrics["recommendation_updates"], 2.0)
        self.assertEqual(metrics["generation_updates"], 1.0)

    def test_prompt_generators_update_only_in_generation_stage(self):
        recommendation_loader = DataLoader(
            RecommendationDataset(torch.tensor([[0, 0]], dtype=torch.long)),
            batch_size=1,
            shuffle=False,
        )
        explanation_loader = DataLoader(
            ExplanationDataset([self.example]),
            batch_size=1,
            shuffle=False,
            collate_fn=_identity_collate,
        )
        rec_only = replace(
            self.config,
            training=replace(self.config.training, alternating_update_order=("recommendation",)),
        )
        before = self._prompt_vector()
        train_alternating_epoch(
            self.model,
            self.generator,
            recommendation_loader,
            explanation_loader,
            torch.tensor([1]),
            [label(0, 0, 0)],
            {0: label(0, 0, 0)},
            self._optimizer(),
            rec_only,
            torch.device("cpu"),
        )
        after_recommendation = self._prompt_vector()
        self.assertTrue(torch.allclose(before, after_recommendation))

        gen_only = replace(
            self.config,
            training=replace(self.config.training, alternating_update_order=("generation",)),
        )
        train_alternating_epoch(
            self.model,
            self.generator,
            recommendation_loader,
            explanation_loader,
            torch.tensor([1]),
            [label(0, 0, 0)],
            {0: label(0, 0, 0)},
            self._optimizer(),
            gen_only,
            torch.device("cpu"),
        )
        after_generation = self._prompt_vector()
        self.assertFalse(torch.allclose(after_recommendation, after_generation))


if __name__ == "__main__":
    unittest.main()
