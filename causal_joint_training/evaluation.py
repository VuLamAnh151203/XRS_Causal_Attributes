"""Evaluation helpers for causal labels, recommendation ranking, and generation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import torch

from .data import CausalLabel, ExplanationExample
from .model import CausalJointModel, FrozenSoftPromptLM, hard_prompt, signed_target_distribution


@dataclass
class CausalMetrics:
    kl_loss: float = 0.0
    attribute_recall_at_5: float = 0.0
    signed_channel_recall_at_5: float = 0.0
    row_count: int = 0


@torch.no_grad()
def evaluate_causal(
    model: CausalJointModel,
    labels: Sequence[CausalLabel],
    batch_size: int,
    device: torch.device,
    recall_k: int = 5,
) -> dict[str, float | int]:
    if not labels:
        return asdict(CausalMetrics())
    was_training = model.training
    model.eval()
    total_kl = 0.0
    attribute_recall = 0.0
    signed_recall = 0.0
    for start in range(0, len(labels), batch_size):
        batch = labels[start : start + batch_size]
        users = torch.tensor([label.user_index for label in batch], dtype=torch.long, device=device)
        items = torch.tensor([label.item_index for label in batch], dtype=torch.long, device=device)
        target = signed_target_distribution(batch, model.vocabulary_size, device)
        logits, probabilities, _ = model.causal_outputs(users, items)
        total_kl += float(model.kl_loss(users, items, target).item()) * len(batch)
        positive, negative = probabilities.split(model.vocabulary_size, dim=-1)
        predicted_attributes = (positive + negative).topk(min(recall_k, model.vocabulary_size), dim=-1).indices.cpu().tolist()
        predicted_channels = logits.topk(min(recall_k, 2 * model.vocabulary_size), dim=-1).indices.cpu().tolist()
        for label, attributes, channels in zip(batch, predicted_attributes, predicted_channels):
            true_attributes = set(label.attribute_indices)
            true_channels = {
                index if coefficient > 0 else model.vocabulary_size + index
                for index, coefficient in zip(label.attribute_indices, label.coefficients)
                if coefficient != 0
            }
            attribute_recall += len(true_attributes & set(attributes)) / len(true_attributes)
            signed_recall += len(true_channels & set(channels)) / len(true_channels)
    model.train(was_training)
    return asdict(
        CausalMetrics(
            kl_loss=total_kl / len(labels),
            attribute_recall_at_5=attribute_recall / len(labels),
            signed_channel_recall_at_5=signed_recall / len(labels),
            row_count=len(labels),
        )
    )


@torch.no_grad()
def evaluate_recommendation(
    model: CausalJointModel,
    evaluation_pairs: Mapping[int, set[int]],
    train_history: Mapping[int, set[int]],
    device: torch.device,
    recall_ks: tuple[int, ...] = (10, 20),
    ndcg_k: int = 20,
    item_batch_size: int = 512,
) -> dict[str, float | int]:
    was_training = model.training
    model.eval()
    recall_totals = {k: 0.0 for k in recall_ks}
    ndcg_total = 0.0
    evaluated = 0
    all_items = torch.arange(model.item_embeddings.num_embeddings, dtype=torch.long, device=device)
    max_k = min(max((*recall_ks, ndcg_k)), all_items.shape[0])
    for user, positives in evaluation_pairs.items():
        if not positives:
            continue
        score_chunks = []
        for start in range(0, all_items.shape[0], item_batch_size):
            items = all_items[start : start + item_batch_size]
            users = torch.full_like(items, user)
            score_chunks.append(model.score_pairs(users, items))
        scores = torch.cat(score_chunks)
        seen = train_history.get(user, set())
        if seen:
            scores[list(seen)] = -torch.inf
        ranking = torch.topk(scores, k=max_k).indices.cpu().tolist()
        for k in recall_ks:
            recall_totals[k] += len(set(ranking[:k]) & positives) / len(positives)
        dcg = sum(1.0 / math.log2(rank + 2) for rank, item in enumerate(ranking[:ndcg_k]) if item in positives)
        ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(positives), ndcg_k)))
        ndcg_total += dcg / ideal if ideal else 0.0
        evaluated += 1
    model.train(was_training)
    return {
        **{f"recall_at_{k}": recall_totals[k] / evaluated if evaluated else 0.0 for k in recall_ks},
        f"ndcg_at_{ndcg_k}": ndcg_total / evaluated if evaluated else 0.0,
        "evaluated_users": evaluated,
    }


@torch.no_grad()
def evaluate_generation_loss(
    model: CausalJointModel,
    generator: FrozenSoftPromptLM,
    examples: Sequence[ExplanationExample],
    device: torch.device,
) -> dict[str, float | int]:
    if not examples:
        return {"generation_loss": 0.0, "row_count": 0}
    model_was_training, generator_was_training = model.training, generator.training
    model.eval()
    generator.eval()
    total = 0.0
    for example in examples:
        users = torch.tensor([example.user_index], dtype=torch.long, device=device)
        items = torch.tensor([example.item_index], dtype=torch.long, device=device)
        preference, _, user_embeddings, item_embeddings = model.preference(users, items)
        prompt = hard_prompt(generator.config.instruction, example.title, example.user_profile, example.item_profile)
        output = generator(preference, user_embeddings, item_embeddings, [prompt], [example.explanation])
        total += float(output.loss.item())
    model.train(model_was_training)
    generator.train(generator_was_training)
    return {"generation_loss": total / len(examples), "row_count": len(examples)}


@torch.no_grad()
def generate_explanations(
    model: CausalJointModel,
    generator: FrozenSoftPromptLM,
    examples: Sequence[ExplanationExample],
    device: torch.device,
) -> list[dict[str, str | int]]:
    model_was_training, generator_was_training = model.training, generator.training
    model.eval()
    generator.eval()
    records: list[dict[str, str | int]] = []
    for example in examples:
        users = torch.tensor([example.user_index], dtype=torch.long, device=device)
        items = torch.tensor([example.item_index], dtype=torch.long, device=device)
        preference, _, user_embeddings, item_embeddings = model.preference(users, items)
        prompt = hard_prompt(generator.config.instruction, example.title, example.user_profile, example.item_profile)
        prediction = generator.generate(preference, user_embeddings, item_embeddings, [prompt])[0]
        records.append(
            {
                "pair_index": example.pair_index,
                "user_id": example.user_id,
                "user_index": example.user_index,
                "item_id": example.item_id,
                "item_index": example.item_index,
                "prediction": prediction,
                "reference": example.explanation,
            }
        )
    model.train(model_was_training)
    generator.train(generator_was_training)
    return records
