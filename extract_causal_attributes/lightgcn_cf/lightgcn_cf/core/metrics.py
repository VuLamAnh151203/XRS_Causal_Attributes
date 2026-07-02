"""Core full-ranking recommendation metrics."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch


def _discounted_gain(ranked_items: list[int], relevant_items: set[int], cutoff: int) -> float:
    return sum(
        1.0 / math.log2(rank + 2.0)
        for rank, item in enumerate(ranked_items[:cutoff])
        if item in relevant_items
    )


@torch.no_grad()
def evaluate(
    model,
    normalized_adjacency: torch.Tensor,
    validation_pairs: dict[int, set[int]],
    train_history: dict[int, set[int]] | None = None,
    recall_k: Iterable[int] = (10, 20),
    ndcg_k: int = 20,
    user_batch_size: int = 256,
) -> dict[str, float | int]:
    """Evaluate over every item absent from each user's training history."""

    train_history = train_history or {}
    cutoffs = tuple(sorted(set(int(cutoff) for cutoff in recall_k)))
    if not cutoffs or min(cutoffs) <= 0 or ndcg_k <= 0:
        raise ValueError("Metric cutoffs must be positive")
    if user_batch_size <= 0:
        raise ValueError("user_batch_size must be positive")

    was_training = model.training
    model.eval()
    user_embeddings, item_embeddings = model(normalized_adjacency)
    num_items = int(item_embeddings.shape[0])
    max_cutoff = max((*cutoffs, ndcg_k))
    recall_totals = {cutoff: 0.0 for cutoff in cutoffs}
    ndcg_total = 0.0
    users_evaluated = 0

    users = sorted(user for user, items in validation_pairs.items() if items)
    for start in range(0, len(users), user_batch_size):
        batch_users = users[start : start + user_batch_size]
        scores = user_embeddings[batch_users] @ item_embeddings.T
        for row, user_index in enumerate(batch_users):
            seen_items = train_history.get(user_index, set())
            if seen_items:
                seen = torch.tensor(sorted(seen_items), dtype=torch.long, device=scores.device)
                scores[row, seen] = -torch.inf
            candidate_count = num_items - len(seen_items)
            if candidate_count <= 0:
                continue
            ranked_items = (
                scores[row]
                .topk(min(max_cutoff, candidate_count))
                .indices.detach()
                .cpu()
                .tolist()
            )
            relevant_items = validation_pairs[user_index]
            for cutoff in cutoffs:
                hits = sum(item in relevant_items for item in ranked_items[:cutoff])
                recall_totals[cutoff] += hits / len(relevant_items)
            ideal_length = min(len(relevant_items), ndcg_k)
            ideal_gain = sum(
                1.0 / math.log2(rank + 2.0) for rank in range(ideal_length)
            )
            ndcg_total += _discounted_gain(ranked_items, relevant_items, ndcg_k) / ideal_gain
            users_evaluated += 1

    if was_training:
        model.train()
    metrics: dict[str, float | int] = {
        f"recall@{cutoff}": (
            recall_totals[cutoff] / users_evaluated if users_evaluated else 0.0
        )
        for cutoff in cutoffs
    }
    metrics[f"ndcg@{ndcg_k}"] = ndcg_total / users_evaluated if users_evaluated else 0.0
    metrics["users_evaluated"] = users_evaluated
    return metrics
