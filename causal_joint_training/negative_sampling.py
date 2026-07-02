"""Semi-hard negative mining with frozen CF candidates and semantic attribute filtering."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass

import torch

from .model import CausalJointModel


class NegativeSamplingError(ValueError):
    """Raised when no valid recommendation negative can be mined."""


@dataclass(frozen=True)
class NegativeSelection:
    item_index: int
    strategy: str
    similarity: float | None


@dataclass
class NegativeSamplingStats:
    threshold_match_count: int = 0
    least_similar_fallback_count: int = 0
    missing_attribute_fallback_count: int = 0


def precompute_top_cf_unseen(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    train_history: Mapping[int, set[int]],
    pool_size: int,
    user_batch_size: int,
    users: Iterable[int] | None = None,
) -> dict[int, tuple[int, ...]]:
    """Return highest-dot-product unseen items for each requested user."""

    if pool_size <= 0 or user_batch_size <= 0:
        raise NegativeSamplingError("Candidate pool and user batch sizes must be positive.")
    requested_users = sorted(set(range(user_embeddings.shape[0])) if users is None else set(users))
    pools: dict[int, tuple[int, ...]] = {}
    item_embeddings = item_embeddings.float()
    for start in range(0, len(requested_users), user_batch_size):
        user_ids = requested_users[start : start + user_batch_size]
        scores = user_embeddings[user_ids].float() @ item_embeddings.T
        for local_row, user in enumerate(user_ids):
            seen = train_history.get(user, set())
            if len(seen) >= item_embeddings.shape[0]:
                raise NegativeSamplingError(f"User index {user} has no unseen item.")
            if seen:
                scores[local_row, list(seen)] = -torch.inf
            count = min(pool_size, item_embeddings.shape[0] - len(seen))
            candidates = torch.topk(scores[local_row], k=count).indices.tolist()
            pools[user] = tuple(int(item) for item in candidates)
    return pools


def causal_attribute_similarity(
    predicted_attributes: Sequence[int],
    candidate_attributes: Sequence[int],
    semantic_embeddings: torch.Tensor,
) -> float:
    """Average maximum cosine similarity from predicted attributes to candidate attributes."""

    if not predicted_attributes or not candidate_attributes:
        raise NegativeSamplingError("Semantic similarity requires two non-empty attribute sets.")
    predicted = semantic_embeddings[list(predicted_attributes)]
    candidates = semantic_embeddings[list(candidate_attributes)]
    return float((predicted @ candidates.T).max(dim=1).values.mean().item())


def select_semihard_negative(
    candidates: Sequence[int],
    predicted_attributes: Sequence[int],
    item_attributes: Mapping[int, Sequence[int]],
    semantic_embeddings: torch.Tensor,
    similarity_threshold: float,
) -> NegativeSelection:
    """Choose the first high-CF dissimilar candidate, then apply documented fallbacks."""

    if not candidates:
        raise NegativeSamplingError("Cannot select a negative from an empty CF candidate pool.")
    if not predicted_attributes:
        return NegativeSelection(int(candidates[0]), "missing_attribute_fallback", None)

    measured: list[tuple[float, int]] = []
    for candidate in candidates:
        attributes = item_attributes.get(int(candidate), ())
        if not attributes:
            continue
        similarity = causal_attribute_similarity(predicted_attributes, attributes, semantic_embeddings)
        measured.append((similarity, int(candidate)))
        if similarity < similarity_threshold:
            return NegativeSelection(int(candidate), "threshold_match", similarity)
    if measured:
        similarity, candidate = min(measured, key=lambda value: value[0])
        return NegativeSelection(candidate, "least_similar_fallback", similarity)
    return NegativeSelection(int(candidates[0]), "missing_attribute_fallback", None)


@torch.no_grad()
def refresh_semihard_negatives(
    model: CausalJointModel,
    recommendation_pairs: torch.Tensor,
    candidate_pools: Mapping[int, Sequence[int]],
    item_attributes: Mapping[int, Sequence[int]],
    semantic_embeddings: torch.Tensor,
    predicted_attribute_count: int,
    similarity_threshold: float,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Refresh one detached semi-hard negative per recommendation pair."""

    model_was_training = model.training
    model.eval()
    negatives: list[int] = []
    stats = NegativeSamplingStats()
    for start in range(0, recommendation_pairs.shape[0], batch_size):
        batch = recommendation_pairs[start : start + batch_size]
        users = batch[:, 0].to(device)
        items = batch[:, 1].to(device)
        top_attributes = model.top_attribute_indices(users, items, predicted_attribute_count).cpu().tolist()
        for row, attributes in zip(batch.tolist(), top_attributes):
            user = int(row[0])
            selection = select_semihard_negative(
                candidate_pools[user],
                attributes,
                item_attributes,
                semantic_embeddings,
                similarity_threshold,
            )
            negatives.append(selection.item_index)
            if selection.strategy == "threshold_match":
                stats.threshold_match_count += 1
            elif selection.strategy == "least_similar_fallback":
                stats.least_similar_fallback_count += 1
            else:
                stats.missing_attribute_fallback_count += 1
    model.train(model_was_training)
    return torch.tensor(negatives, dtype=torch.long), asdict(stats)
