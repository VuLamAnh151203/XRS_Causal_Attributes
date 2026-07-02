"""Warm-up, joint optimization, checkpointing, and top-level training orchestration."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from .config import JointTrainingConfig
from .data import CausalLabel, ExplanationExample, LoadedArtifacts, load_artifacts
from .evaluation import evaluate_causal, evaluate_generation_loss, evaluate_recommendation
from .model import CausalJointModel, FrozenSoftPromptLM, hard_prompt, signed_target_distribution
from .negative_sampling import precompute_top_cf_unseen, refresh_semihard_negatives


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RecommendationDataset(Dataset):
    def __init__(self, pairs: torch.Tensor) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return self.pairs.shape[0]

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor, torch.Tensor]:
        return index, self.pairs[index, 0], self.pairs[index, 1]


class CausalDataset(Dataset):
    def __init__(self, labels: Sequence[CausalLabel]) -> None:
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> CausalLabel:
        return self.labels[index]


class ExplanationDataset(Dataset):
    def __init__(self, examples: Sequence[ExplanationExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> ExplanationExample:
        return self.examples[index]


def _identity_collate(values: list[Any]) -> list[Any]:
    return values


@dataclass(frozen=True)
class LabeledPairSplit:
    train_examples: list[ExplanationExample]
    validation_examples: list[ExplanationExample]
    train_labels: list[CausalLabel]
    validation_labels: list[CausalLabel]
    stats: dict[str, Any]


def select_omp_labeled_explanations(
    examples: Sequence[ExplanationExample],
    causal_labels: Mapping[int, CausalLabel],
) -> tuple[list[ExplanationExample], dict[str, int]]:
    """Keep only explanation rows whose trn.pkl pair_index has an accepted OMP label."""

    selected: list[ExplanationExample] = []
    stats = {
        "input_explanation_rows": len(examples),
        "omp_labeled_rows": 0,
        "skipped_without_omp_label": 0,
    }
    for example in examples:
        label = causal_labels.get(example.pair_index)
        if label is None:
            stats["skipped_without_omp_label"] += 1
            continue
        if label.user_index != example.user_index or label.item_index != example.item_index:
            raise RuntimeError(
                "OMP label and explanation row disagree for "
                f"pair_index {example.pair_index}: "
                f"label=({label.user_index}, {label.item_index}), "
                f"example=({example.user_index}, {example.item_index})."
            )
        selected.append(example)
    stats["omp_labeled_rows"] = len(selected)
    return selected, stats


def build_labeled_pair_split(
    examples: Sequence[ExplanationExample],
    causal_labels: Mapping[int, CausalLabel],
    *,
    validation_ratio: float,
    seed: int,
) -> LabeledPairSplit:
    if validation_ratio <= 0.0 or validation_ratio >= 1.0:
        raise RuntimeError("labeled_validation_ratio must be greater than 0 and less than 1.")
    labeled_examples, selection_stats = select_omp_labeled_explanations(examples, causal_labels)
    if not labeled_examples:
        raise RuntimeError("No trn.pkl explanation rows have accepted OMP causal labels.")

    ordered = list(labeled_examples)
    random.Random(seed).shuffle(ordered)
    validation_count = max(1, round(validation_ratio * len(ordered))) if len(ordered) > 1 else 0
    validation_count = min(validation_count, len(ordered) - 1) if len(ordered) > 1 else 0
    validation_examples = ordered[:validation_count]
    train_examples = ordered[validation_count:]
    if not train_examples:
        raise RuntimeError("No OMP-labeled training rows remain after validation split.")

    train_labels = [causal_labels[example.pair_index] for example in train_examples]
    validation_labels = [causal_labels[example.pair_index] for example in validation_examples]
    stats = {
        **selection_stats,
        "labeled_pair_count": len(labeled_examples),
        "labeled_train_count": len(train_examples),
        "labeled_validation_count": len(validation_examples),
        "labeled_validation_ratio": validation_ratio,
        "split_seed": seed,
    }
    return LabeledPairSplit(train_examples, validation_examples, train_labels, validation_labels, stats)


def examples_to_pair_tensor(examples: Sequence[ExplanationExample]) -> torch.Tensor:
    if not examples:
        return torch.empty((0, 2), dtype=torch.long)
    return torch.tensor(
        [[example.user_index, example.item_index] for example in examples],
        dtype=torch.long,
    )


def merge_history_with_examples(
    history: Mapping[int, set[int]],
    examples: Sequence[ExplanationExample],
) -> dict[int, set[int]]:
    merged = {int(user): set(items) for user, items in history.items()}
    for example in examples:
        merged.setdefault(example.user_index, set()).add(example.item_index)
    return merged


def examples_to_evaluation_pairs(
    examples: Sequence[ExplanationExample],
    train_history: Mapping[int, set[int]],
) -> tuple[dict[int, set[int]], dict[str, int]]:
    pairs: dict[int, set[int]] = {}
    stats = {
        "validation_candidate_pairs": len(examples),
        "validation_skipped_seen": 0,
        "validation_evaluable_pairs": 0,
        "validation_evaluable_users": 0,
    }
    for example in examples:
        if example.item_index in train_history.get(example.user_index, set()):
            stats["validation_skipped_seen"] += 1
            continue
        pairs.setdefault(example.user_index, set()).add(example.item_index)
        stats["validation_evaluable_pairs"] += 1
    stats["validation_evaluable_users"] = len(pairs)
    return pairs, stats


def train_causal_warmup(
    model: CausalJointModel,
    train_labels: Sequence[CausalLabel],
    validation_labels: Sequence[CausalLabel],
    config: JointTrainingConfig,
    device: torch.device,
) -> list[dict[str, Any]]:
    loader = DataLoader(
        CausalDataset(train_labels),
        batch_size=config.training.causal_batch_size,
        shuffle=True,
        collate_fn=_identity_collate,
    )
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.training.causal_learning_rate,
        weight_decay=config.training.weight_decay,
    )
    history: list[dict[str, Any]] = []
    model.train()
    for epoch in range(config.training.causal_warmup_epochs):
        total_loss = 0.0
        rows = 0
        for labels in loader:
            users = torch.tensor([label.user_index for label in labels], dtype=torch.long, device=device)
            items = torch.tensor([label.item_index for label in labels], dtype=torch.long, device=device)
            target = signed_target_distribution(labels, model.vocabulary_size, device)
            optimizer.zero_grad()
            loss = model.kl_loss(users, items, target)
            loss.backward()
            clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            rows += len(labels)
        history.append(
            {
                "phase": "causal_warmup",
                "epoch": epoch + 1,
                "train_kl_loss": total_loss / rows if rows else 0.0,
                "validation": evaluate_causal(
                    model, validation_labels, config.training.causal_batch_size, device
                ),
            }
        )
    return history


def _prompt_parameters(generator: FrozenSoftPromptLM) -> Iterable[torch.nn.Parameter]:
    return (
        parameter
        for module in (generator.causal_prompt, generator.user_prompt, generator.item_prompt)
        for parameter in module.parameters()
    )


def train_alternating_epoch(
    model: CausalJointModel,
    generator: FrozenSoftPromptLM,
    recommendation_loader: DataLoader,
    explanation_loader: DataLoader,
    negatives: torch.Tensor,
    recommendation_labels: Sequence[CausalLabel],
    causal_labels: Mapping[int, CausalLabel],
    optimizer: AdamW,
    config: JointTrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    generator.train()
    totals = {
        "recommendation_bpr_loss": 0.0,
        "recommendation_kl_loss": 0.0,
        "generation_loss": 0.0,
        "generation_kl_loss": 0.0,
        "loss": 0.0,
    }
    counts = {
        "recommendation_updates": 0,
        "generation_updates": 0,
        "optimizer_updates": 0,
        "backward_updates": 0,
    }
    pending_gradients = 0
    optimizer.zero_grad()

    def maybe_step() -> None:
        nonlocal pending_gradients
        if pending_gradients <= 0:
            return
        clip_grad_norm_(
            [parameter for group in optimizer.param_groups for parameter in group["params"]],
            config.training.gradient_clip_norm,
        )
        optimizer.step()
        optimizer.zero_grad()
        pending_gradients = 0
        counts["optimizer_updates"] += 1

    def backward(loss: torch.Tensor) -> None:
        nonlocal pending_gradients
        (loss / config.training.gradient_accumulation_steps).backward()
        pending_gradients += 1
        counts["backward_updates"] += 1
        if pending_gradients >= config.training.gradient_accumulation_steps:
            maybe_step()

    for stage in config.training.alternating_update_order:
        if stage == "recommendation":
            for recommendation_batch in recommendation_loader:
                row_indices, users, positives = recommendation_batch
                users, positives = users.to(device), positives.to(device)
                negative_items = negatives[row_indices].to(device)
                bpr_loss = model.bpr_loss(users, positives, negative_items)
                label_values = [recommendation_labels[int(row_index)] for row_index in row_indices]
                target = signed_target_distribution(label_values, model.vocabulary_size, device)
                kl_loss = model.kl_loss(users, positives, target)
                loss = config.training.bpr_weight * bpr_loss + config.training.kl_weight * kl_loss
                backward(loss)
                totals["recommendation_bpr_loss"] += float(bpr_loss.item())
                totals["recommendation_kl_loss"] += float(kl_loss.item())
                totals["loss"] += float(loss.item())
                counts["recommendation_updates"] += 1
            continue

        if stage != "generation":
            raise RuntimeError(f"Unsupported alternating training stage {stage!r}.")

        for explanations in explanation_loader:
            exp_users = torch.tensor([value.user_index for value in explanations], dtype=torch.long, device=device)
            exp_items = torch.tensor([value.item_index for value in explanations], dtype=torch.long, device=device)
            preference, _, user_embeddings, item_embeddings = model.preference(exp_users, exp_items)
            prompts = [
                hard_prompt(generator.config.instruction, value.title, value.user_profile, value.item_profile)
                for value in explanations
            ]
            generation_loss = generator(
                preference, user_embeddings, item_embeddings, prompts, [value.explanation for value in explanations]
            ).loss
            label_values: list[CausalLabel] = []
            for value in explanations:
                label = causal_labels.get(value.pair_index)
                if label is None:
                    raise RuntimeError(
                        f"Alternating generation batch contains unlabeled pair_index {value.pair_index}."
                    )
                label_values.append(label)
            target = signed_target_distribution(label_values, model.vocabulary_size, device)
            kl_loss = model.kl_loss(exp_users, exp_items, target)
            loss = config.training.kl_weight * kl_loss + config.training.generation_weight * generation_loss
            backward(loss)
            totals["generation_kl_loss"] += float(kl_loss.item())
            totals["generation_loss"] += float(generation_loss.item())
            totals["loss"] += float(loss.item())
            counts["generation_updates"] += 1

    maybe_step()
    total_updates = counts["recommendation_updates"] + counts["generation_updates"]
    return {
        "recommendation_bpr_loss": totals["recommendation_bpr_loss"] / counts["recommendation_updates"]
        if counts["recommendation_updates"]
        else 0.0,
        "recommendation_kl_loss": totals["recommendation_kl_loss"] / counts["recommendation_updates"]
        if counts["recommendation_updates"]
        else 0.0,
        "generation_loss": totals["generation_loss"] / counts["generation_updates"]
        if counts["generation_updates"]
        else 0.0,
        "generation_kl_loss": totals["generation_kl_loss"] / counts["generation_updates"]
        if counts["generation_updates"]
        else 0.0,
        "loss": totals["loss"] / total_updates if total_updates else 0.0,
        **{key: float(value) for key, value in counts.items()},
    }


def checkpoint_payload(
    model: CausalJointModel, generator: FrozenSoftPromptLM | None, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "model": model.trainable_state_dict(),
        "prompt_generators": generator.prompt_state_dict() if generator is not None else None,
        "metadata": dict(metadata),
    }


def save_checkpoint(
    path: Path,
    model: CausalJointModel,
    generator: FrozenSoftPromptLM | None,
    metadata: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, generator, metadata), path)


def load_checkpoint(path: Path, model: CausalJointModel, generator: FrozenSoftPromptLM | None) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    model.load_trainable_state_dict(payload["model"])
    if generator is not None and payload.get("prompt_generators") is not None:
        generator.load_prompt_state_dict(payload["prompt_generators"])
    return dict(payload.get("metadata", {}))


def _json_safe_config(config: JointTrainingConfig) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value
    return convert(asdict(config))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_training(config: JointTrainingConfig) -> dict[str, Any]:
    seed_everything(config.training.seed)
    device = resolve_device(config.training.device)
    artifacts: LoadedArtifacts = load_artifacts(config)
    model = CausalJointModel(
        artifacts.user_embeddings, artifacts.item_embeddings, len(artifacts.vocabulary), config.model
    ).to(device)
    if not artifacts.explanation_train:
        raise RuntimeError("No mapped explanation training rows are available.")
    labeled_split = build_labeled_pair_split(
        artifacts.explanation_train,
        artifacts.causal_labels,
        validation_ratio=config.training.labeled_validation_ratio,
        seed=config.training.seed,
    )
    recommendation_training_pairs = examples_to_pair_tensor(labeled_split.train_examples)
    training_history = merge_history_with_examples(
        artifacts.recommendation.train_history,
        labeled_split.train_examples,
    )
    recommendation_validation_pairs, recommendation_validation_stats = examples_to_evaluation_pairs(
        labeled_split.validation_examples,
        training_history,
    )
    history = train_causal_warmup(
        model,
        labeled_split.train_labels,
        labeled_split.validation_labels,
        config,
        device,
    )

    generator = FrozenSoftPromptLM.from_pretrained(
        config.generation, config.model.embedding_dim, config.model.preference_dim, device=device
    )
    if not config.generation.load_in_4bit:
        generator = generator.to(device)
    generator.move_prompt_generators(device)
    candidate_pools = precompute_top_cf_unseen(
        artifacts.user_embeddings,
        artifacts.item_embeddings,
        training_history,
        config.negative_sampling.candidate_pool_size,
        config.negative_sampling.user_batch_size,
        users=recommendation_training_pairs[:, 0].tolist(),
    )
    recommendation_loader = DataLoader(
        RecommendationDataset(recommendation_training_pairs),
        batch_size=config.training.recommendation_batch_size,
        shuffle=True,
    )
    explanation_loader = DataLoader(
        ExplanationDataset(labeled_split.train_examples),
        batch_size=config.training.generation_batch_size,
        shuffle=True,
        collate_fn=_identity_collate,
    )
    optimizer = AdamW(
        [
            {
                "params": [parameter for parameter in model.parameters() if parameter.requires_grad],
                "lr": config.training.causal_learning_rate,
            },
            {"params": list(_prompt_parameters(generator)), "lr": config.training.prompt_learning_rate},
        ],
        weight_decay=config.training.weight_decay,
    )
    output_dir = config.paths.output_dir
    _write_json(
        output_dir / "run_config.json",
        {
            **_json_safe_config(config),
            "labeled_pair_split": {
                **labeled_split.stats,
                **recommendation_validation_stats,
                "recommendation_positive_source": "OMP-labeled trn.pkl subset",
                "recommendation_positive_count": int(recommendation_training_pairs.shape[0]),
                "history_source": "total_trn_new.csv plus OMP-labeled training positives",
                "alternating_update_order": list(config.training.alternating_update_order),
            },
        },
    )
    best = {"ndcg_at_20": -float("inf"), "generation_loss": float("inf"), "attribute_recall_at_5": -float("inf")}
    for epoch in range(config.training.joint_epochs):
        negatives, negative_stats = refresh_semihard_negatives(
            model,
            recommendation_training_pairs,
            candidate_pools,
            artifacts.item_attributes,
            artifacts.semantic_embeddings,
            config.negative_sampling.predicted_attribute_count,
            config.negative_sampling.similarity_threshold,
            config.training.recommendation_batch_size,
            device,
        )
        train_metrics = train_alternating_epoch(
            model,
            generator,
            recommendation_loader,
            explanation_loader,
            negatives,
            labeled_split.train_labels,
            artifacts.causal_labels,
            optimizer,
            config,
            device,
        )
        causal_metrics = evaluate_causal(
            model, labeled_split.validation_labels, config.training.causal_batch_size, device
        )
        recommendation_metrics = evaluate_recommendation(
            model,
            recommendation_validation_pairs,
            training_history,
            device,
        )
        recommendation_metrics = {**recommendation_metrics, **recommendation_validation_stats}
        generation_metrics = evaluate_generation_loss(
            model, generator, labeled_split.validation_examples, device
        )
        record = {
            "phase": "joint",
            "epoch": epoch + 1,
            "labeled_pair_split": labeled_split.stats,
            "alternating_update_order": list(config.training.alternating_update_order),
            "negative_sampling": negative_stats,
            "train": train_metrics,
            "causal_validation": causal_metrics,
            "recommendation_validation": recommendation_metrics,
            "generation_validation": generation_metrics,
        }
        history.append(record)
        metadata = {"epoch": epoch + 1, "metrics": record}
        save_checkpoint(output_dir / "latest.pt", model, generator, metadata)
        checks = (
            ("ndcg_at_20", recommendation_metrics["ndcg_at_20"], "best_ndcg20.pt", True),
            ("generation_loss", generation_metrics["generation_loss"], "best_generation_loss.pt", False),
            ("attribute_recall_at_5", causal_metrics["attribute_recall_at_5"], "best_causal_recall5.pt", True),
        )
        for metric, value, filename, maximize in checks:
            if (maximize and value > best[metric]) or (not maximize and value < best[metric]):
                best[metric] = float(value)
                save_checkpoint(output_dir / filename, model, generator, metadata)
        _write_json(output_dir / "training_history.json", history)
    return {"artifacts": artifacts.stats, "best": best, "history": history}
