"""Direct exact edge-drop counterfactual propagation."""

from __future__ import annotations

from pathlib import Path

import torch

from ..core.artifacts import ArtifactPaths, load_mappings, load_tensor, load_user_history
from ..core.graph import build_normalized_adjacency, propagate, remove_user_item_edges


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _rank_items(
    user_embedding: torch.Tensor,
    item_embeddings: torch.Tensor,
    excluded_items: set[int],
    index_to_item: list[str],
    top_k: int,
) -> tuple[list[dict], dict[int, int], torch.Tensor]:
    scores = item_embeddings @ user_embedding
    candidate_mask = torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)
    if excluded_items:
        excluded = torch.tensor(sorted(excluded_items), dtype=torch.long, device=scores.device)
        candidate_mask[excluded] = False
    candidates = torch.arange(scores.shape[0], device=scores.device)[candidate_mask]
    candidate_scores = scores[candidate_mask]
    order = torch.argsort(candidate_scores, descending=True)
    ranked_items = candidates[order].detach().cpu().tolist()
    ranks = {int(item): rank for rank, item in enumerate(ranked_items, start=1)}
    recommendations = [
        {
            "rank": rank,
            "item_id": index_to_item[item],
            "score": float(scores[item].detach().cpu()),
        }
        for rank, item in enumerate(ranked_items[:top_k], start=1)
    ]
    return recommendations, ranks, scores


def run_edge_drop(
    artifact_dir: str | Path,
    user_id: str,
    drop_item_ids: list[str],
    num_layers: int,
    top_k: int = 20,
    device: str = "auto",
    save_full_embeddings: str | Path | None = None,
) -> dict:
    """Drop one user's history edges and exactly rerun sparse propagation."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    paths = ArtifactPaths(Path(artifact_dir).resolve())
    mappings = load_mappings(paths.mappings)
    train_pairs = load_tensor(paths.train_pairs).long()
    train_history = load_user_history(paths.user_history)
    user_key = str(user_id)
    if user_key not in mappings.user_to_index:
        raise ValueError(f"Unknown user ID: {user_id}")
    user_index = mappings.user_to_index[user_key]

    drop_indices: list[int] = []
    for item_id in drop_item_ids:
        item_key = str(item_id)
        if item_key not in mappings.item_to_index:
            raise ValueError(f"Unknown item ID: {item_id}")
        drop_indices.append(mappings.item_to_index[item_key])
    perturbed_pairs = remove_user_item_edges(train_pairs, user_index, drop_indices)

    resolved_device = _resolve_device(device)
    user_ego = load_tensor(paths.user_ego_embeddings).to(resolved_device)
    item_ego = load_tensor(paths.item_ego_embeddings).to(resolved_device)
    baseline_users = load_tensor(paths.user_baseline_embeddings).to(resolved_device)
    baseline_items = load_tensor(paths.item_baseline_embeddings).to(resolved_device)
    perturbed_adjacency = build_normalized_adjacency(
        mappings.num_users, mappings.num_items, perturbed_pairs, resolved_device
    )
    perturbed_all = propagate(
        torch.cat((user_ego, item_ego), dim=0), perturbed_adjacency, num_layers
    )
    perturbed_users, perturbed_items = torch.split(
        perturbed_all, (mappings.num_users, mappings.num_items), dim=0
    )

    excluded_items = train_history[user_index]
    baseline_top_k, baseline_ranks, baseline_scores = _rank_items(
        baseline_users[user_index],
        baseline_items,
        excluded_items,
        mappings.index_to_item,
        top_k,
    )
    perturbed_top_k, perturbed_ranks, perturbed_scores = _rank_items(
        perturbed_users[user_index],
        perturbed_items,
        excluded_items,
        mappings.index_to_item,
        top_k,
    )
    compared_items: list[str] = []
    for entry in (*baseline_top_k, *perturbed_top_k):
        if entry["item_id"] not in compared_items:
            compared_items.append(entry["item_id"])
    changes = []
    for item_id in compared_items:
        item_index = mappings.item_to_index[item_id]
        baseline_rank = baseline_ranks[item_index]
        perturbed_rank = perturbed_ranks[item_index]
        baseline_score = float(baseline_scores[item_index].detach().cpu())
        perturbed_score = float(perturbed_scores[item_index].detach().cpu())
        changes.append(
            {
                "item_id": item_id,
                "baseline_score": baseline_score,
                "perturbed_score": perturbed_score,
                "score_delta": perturbed_score - baseline_score,
                "baseline_rank": baseline_rank,
                "perturbed_rank": perturbed_rank,
                "rank_delta": perturbed_rank - baseline_rank,
            }
        )

    if save_full_embeddings is not None:
        prefix = Path(save_full_embeddings)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        torch.save(perturbed_users.detach().cpu(), prefix.with_name(prefix.name + "_users.pt"))
        torch.save(perturbed_items.detach().cpu(), prefix.with_name(prefix.name + "_items.pt"))

    return {
        "user_id": user_key,
        "dropped_item_ids": [str(item_id) for item_id in drop_item_ids],
        "perturbed_user_embedding": perturbed_users[user_index].detach().cpu().tolist(),
        "baseline_top_k": baseline_top_k,
        "perturbed_top_k": perturbed_top_k,
        "score_changes": changes,
        "rank_changes": [
            {
                "item_id": change["item_id"],
                "baseline_rank": change["baseline_rank"],
                "perturbed_rank": change["perturbed_rank"],
                "rank_delta": change["rank_delta"],
            }
            for change in changes
        ],
    }
