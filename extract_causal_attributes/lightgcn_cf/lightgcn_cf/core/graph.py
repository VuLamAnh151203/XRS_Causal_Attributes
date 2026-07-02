"""Core sparse graph construction and LightGCN propagation."""

from __future__ import annotations

import torch


def _as_pair_tensor(train_pairs: torch.Tensor | list[tuple[int, int]]) -> torch.Tensor:
    pairs = torch.as_tensor(train_pairs, dtype=torch.long)
    if pairs.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("train_pairs must have shape [num_edges, 2]")
    return torch.unique(pairs.cpu(), dim=0)


def build_normalized_adjacency(
    num_users: int,
    num_items: int,
    train_pairs: torch.Tensor | list[tuple[int, int]],
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build D^-1/2 A D^-1/2 for a deduplicated bipartite graph."""

    pairs = _as_pair_tensor(train_pairs)
    num_nodes = num_users + num_items
    if pairs.numel() == 0:
        indices = torch.empty((2, 0), dtype=torch.long, device=device)
        values = torch.empty((0,), dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(
            indices, values, (num_nodes, num_nodes), device=device
        ).coalesce()

    users = pairs[:, 0]
    items = pairs[:, 1]
    if users.min() < 0 or users.max() >= num_users:
        raise ValueError("train_pairs contains an out-of-range user index")
    if items.min() < 0 or items.max() >= num_items:
        raise ValueError("train_pairs contains an out-of-range item index")

    item_nodes = items + num_users
    sources = torch.cat((users, item_nodes))
    destinations = torch.cat((item_nodes, users))
    degrees = torch.bincount(sources, minlength=num_nodes).float()
    inverse_sqrt_degree = torch.zeros_like(degrees)
    nonzero = degrees > 0
    inverse_sqrt_degree[nonzero] = degrees[nonzero].pow(-0.5)
    values = inverse_sqrt_degree[sources] * inverse_sqrt_degree[destinations]
    indices = torch.stack((sources, destinations))
    return torch.sparse_coo_tensor(
        indices.to(device),
        values.to(device),
        (num_nodes, num_nodes),
        device=device,
    ).coalesce()


def propagate(
    ego_embeddings: torch.Tensor,
    normalized_adjacency: torch.Tensor,
    num_layers: int,
) -> torch.Tensor:
    """Run LightGCN sparse propagation and mean-pool layer representations."""

    if num_layers < 0:
        raise ValueError("num_layers must be non-negative")
    embeddings = [ego_embeddings]
    current = ego_embeddings
    for _ in range(num_layers):
        current = torch.sparse.mm(normalized_adjacency, current)
        embeddings.append(current)
    return torch.stack(embeddings, dim=0).mean(dim=0)


def remove_user_item_edges(
    train_pairs: torch.Tensor,
    user_index: int,
    item_indices: list[int],
) -> torch.Tensor:
    """Remove selected user-item pairs, rejecting edges absent from history."""

    pairs = _as_pair_tensor(train_pairs)
    unique_items = sorted(set(int(item) for item in item_indices))
    if not unique_items:
        raise ValueError("At least one item edge must be selected for removal")
    pair_set = {(int(user), int(item)) for user, item in pairs.tolist()}
    missing = [item for item in unique_items if (user_index, item) not in pair_set]
    if missing:
        raise ValueError(
            f"Cannot drop item indices {missing}: they are not history edges for user index "
            f"{user_index}"
        )
    drop_items = set(unique_items)
    keep_mask = [
        not (int(user) == user_index and int(item) in drop_items)
        for user, item in pairs.tolist()
    ]
    return pairs[torch.tensor(keep_mask, dtype=torch.bool)]
