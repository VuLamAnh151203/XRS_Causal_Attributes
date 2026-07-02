"""Direct exact local LightGCN edge-drop scoring."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..core.artifacts import ArtifactPaths, load_mappings, load_tensor, load_user_history
from ..core.data import IdMappings
from ..core.graph import build_normalized_adjacency, propagate, remove_user_item_edges


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _layer_embeddings(
    ego_embeddings: torch.Tensor,
    normalized_adjacency: torch.Tensor,
    num_layers: int,
) -> list[torch.Tensor]:
    if num_layers < 0:
        raise ValueError("num_layers must be non-negative")
    layers = [ego_embeddings]
    current = ego_embeddings
    for _ in range(num_layers):
        current = torch.sparse.mm(normalized_adjacency, current)
        layers.append(current)
    return layers


def _score_and_rank_from_scores(
    scores: torch.Tensor,
    target_item_index: int,
    excluded_items: set[int],
) -> tuple[float, int]:
    if target_item_index in excluded_items:
        raise ValueError(
            f"Target item index {target_item_index} is in the user's training history"
        )
    candidate_mask = torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)
    if excluded_items:
        excluded = torch.tensor(sorted(excluded_items), dtype=torch.long, device=scores.device)
        candidate_mask[excluded] = False
    candidates = torch.arange(scores.shape[0], device=scores.device)[candidate_mask]
    ranked_items = candidates[torch.argsort(scores[candidate_mask], descending=True)]
    matches = (ranked_items == int(target_item_index)).nonzero(as_tuple=False)
    if matches.numel() == 0:
        raise ValueError(f"Target item index {target_item_index} was not rankable")
    return float(scores[target_item_index].detach().cpu()), int(matches[0].item()) + 1


def _score_and_rank(
    user_embedding: torch.Tensor,
    item_embeddings: torch.Tensor,
    target_item_index: int,
    excluded_items: set[int],
) -> tuple[float, int]:
    return _score_and_rank_from_scores(
        item_embeddings @ user_embedding,
        target_item_index,
        excluded_items,
    )


def _score_drop_ratio(baseline_score: float, perturbed_score: float) -> float | None:
    if baseline_score == 0.0:
        return None
    return (baseline_score - perturbed_score) / baseline_score


@dataclass
class LocalPerturbationScorer:
    mappings: IdMappings
    train_pairs: torch.Tensor
    train_history: dict[int, set[int]]
    num_layers: int
    device: torch.device
    ego_embeddings: torch.Tensor
    baseline_layers: list[torch.Tensor]
    baseline_users: torch.Tensor
    baseline_items: torch.Tensor
    neighbors: list[set[int]]
    degrees: list[int]
    score_cache_max_entries: int = 100_000
    score_cache: OrderedDict[tuple[int, int, tuple[int, ...]], float] = field(
        default_factory=OrderedDict
    )
    score_cache_hit_count: int = 0
    score_cache_miss_count: int = 0

    @classmethod
    def from_artifacts(
        cls,
        artifact_dir: str | Path,
        num_layers: int,
        device: str = "auto",
    ) -> "LocalPerturbationScorer":
        paths = ArtifactPaths(Path(artifact_dir).resolve())
        mappings = load_mappings(paths.mappings)
        train_pairs = load_tensor(paths.train_pairs).long()
        train_history = load_user_history(paths.user_history)
        resolved_device = _resolve_device(device)
        user_ego = load_tensor(paths.user_ego_embeddings).to(resolved_device)
        item_ego = load_tensor(paths.item_ego_embeddings).to(resolved_device)
        ego_embeddings = torch.cat((user_ego, item_ego), dim=0)
        adjacency = build_normalized_adjacency(
            mappings.num_users,
            mappings.num_items,
            train_pairs,
            resolved_device,
        )
        baseline_layers = _layer_embeddings(ego_embeddings, adjacency, num_layers)
        baseline_all = torch.stack(baseline_layers, dim=0).mean(dim=0)
        baseline_users, baseline_items = torch.split(
            baseline_all,
            (mappings.num_users, mappings.num_items),
            dim=0,
        )
        neighbors, degrees = _build_neighbors(
            mappings.num_users,
            mappings.num_items,
            train_pairs,
        )
        return cls(
            mappings=mappings,
            train_pairs=train_pairs,
            train_history=train_history,
            num_layers=num_layers,
            device=resolved_device,
            ego_embeddings=ego_embeddings,
            baseline_layers=baseline_layers,
            baseline_users=baseline_users,
            baseline_items=baseline_items,
            neighbors=neighbors,
            degrees=degrees,
        )

    def baseline_score_and_rank(
        self,
        user_index: int,
        target_item_index: int,
        excluded_items: set[int],
    ) -> tuple[float, int]:
        return _score_and_rank(
            self.baseline_users[user_index],
            self.baseline_items,
            target_item_index,
            excluded_items,
        )

    def baseline_target_score(self, user_index: int, target_item_index: int) -> float:
        return float(
            (self.baseline_users[user_index] * self.baseline_items[target_item_index])
            .sum()
            .detach()
            .cpu()
        )

    def score_target_many(
        self,
        user_index: int,
        target_item_index: int,
        drop_item_index_groups: Iterable[Iterable[int]],
    ) -> tuple[float, ...]:
        keys = tuple(
            (user_index, target_item_index, tuple(sorted(set(int(item) for item in group))))
            for group in drop_item_index_groups
        )
        if not keys:
            return ()

        resolved: dict[tuple[int, int, tuple[int, ...]], float] = {}
        missing_keys: list[tuple[int, int, tuple[int, ...]]] = []
        for key in keys:
            if key in self.score_cache:
                self.score_cache_hit_count += 1
                value = self.score_cache.pop(key)
                self.score_cache[key] = value
                resolved[key] = value
            elif key in resolved:
                self.score_cache_hit_count += 1
            else:
                self.score_cache_miss_count += 1
                resolved[key] = 0.0
                missing_keys.append(key)

        if missing_keys:
            missing_scores = self._score_target_many_uncached(
                user_index,
                target_item_index,
                [key[2] for key in missing_keys],
            )
            for key, value in zip(missing_keys, missing_scores, strict=True):
                resolved[key] = value
                self.score_cache[key] = value
                if len(self.score_cache) > self.score_cache_max_entries:
                    self.score_cache.popitem(last=False)

        return tuple(resolved[key] for key in keys)

    def score_drop_many(
        self,
        user_index: int,
        target_item_index: int,
        drop_item_index_groups: Sequence[Iterable[int]],
        propagation_mode: str = "local-score",
    ) -> tuple[dict, ...]:
        if propagation_mode == "local-score":
            baseline_score = self.baseline_target_score(user_index, target_item_index)
            perturbed_scores = self.score_target_many(
                user_index,
                target_item_index,
                drop_item_index_groups,
            )
            return tuple(
                {
                    "baseline_score": baseline_score,
                    "ratios": _score_drop_ratio(baseline_score, perturbed_score),
                    "score_drop": baseline_score - perturbed_score,
                    "baseline_rank": None,
                    "perturbed_rank": None,
                    "rank_drop": None,
                }
                for perturbed_score in perturbed_scores
            )

        excluded_items = self.train_history.get(user_index, set())
        baseline_score, baseline_rank = self.baseline_score_and_rank(
            user_index,
            target_item_index,
            excluded_items,
        )
        return tuple(
            self.score_drop(
                user_index,
                target_item_index,
                list(drop_items),
                excluded_items,
                baseline_score,
                baseline_rank,
                propagation_mode,
            )
            for drop_items in drop_item_index_groups
        )

    def score_drop(
        self,
        user_index: int,
        target_item_index: int,
        drop_item_indices: list[int],
        excluded_items: set[int],
        baseline_score: float,
        baseline_rank: int,
        propagation_mode: str = "local-lhop",
    ) -> dict:
        if propagation_mode == "local-score":
            perturbed_score = self.score_target_many(
                user_index,
                target_item_index,
                (drop_item_indices,),
            )[0]
            return {
                "baseline_score": baseline_score,
                "ratios": _score_drop_ratio(baseline_score, perturbed_score),
                "score_drop": baseline_score - perturbed_score,
                "baseline_rank": None,
                "perturbed_rank": None,
                "rank_drop": None,
            }
        if propagation_mode == "local-lhop":
            perturbed_score, perturbed_rank = self.score_drop_local(
                user_index,
                target_item_index,
                drop_item_indices,
                excluded_items,
            )
        elif propagation_mode == "full":
            perturbed_score, perturbed_rank = self.score_drop_full(
                user_index,
                target_item_index,
                drop_item_indices,
                excluded_items,
            )
        else:
            raise ValueError("propagation_mode must be 'local-score', 'local-lhop', or 'full'")

        return {
            "baseline_score": baseline_score,
            "ratios": _score_drop_ratio(baseline_score, perturbed_score),
            "score_drop": baseline_score - perturbed_score,
            "baseline_rank": baseline_rank,
            "perturbed_rank": perturbed_rank,
            "rank_drop": perturbed_rank - baseline_rank,
        }

    def score_drop_full(
        self,
        user_index: int,
        target_item_index: int,
        drop_item_indices: list[int],
        excluded_items: set[int],
    ) -> tuple[float, int]:
        perturbed_pairs = remove_user_item_edges(
            self.train_pairs,
            user_index,
            drop_item_indices,
        )
        perturbed_adjacency = build_normalized_adjacency(
            self.mappings.num_users,
            self.mappings.num_items,
            perturbed_pairs,
            self.device,
        )
        perturbed_all = propagate(
            self.ego_embeddings,
            perturbed_adjacency,
            self.num_layers,
        )
        perturbed_users, perturbed_items = torch.split(
            perturbed_all,
            (self.mappings.num_users, self.mappings.num_items),
            dim=0,
        )
        return _score_and_rank(
            perturbed_users[user_index],
            perturbed_items,
            target_item_index,
            excluded_items,
        )

    def _score_target_many_uncached(
        self,
        user_index: int,
        target_item_index: int,
        drop_item_index_groups: Sequence[tuple[int, ...]],
    ) -> tuple[float, ...]:
        if not 0 <= user_index < self.mappings.num_users:
            raise ValueError(f"User index is out of range: {user_index}")
        if not 0 <= target_item_index < self.mappings.num_items:
            raise ValueError(f"Target item index is out of range: {target_item_index}")
        if not drop_item_index_groups:
            return ()

        user_history = self.train_history.get(user_index, set())
        removed_edge_groups: list[frozenset[tuple[int, int]]] = []
        for drop_items in drop_item_index_groups:
            unique_items = tuple(sorted(set(int(item) for item in drop_items)))
            missing = [item for item in unique_items if item not in user_history]
            if missing:
                raise ValueError(
                    f"Cannot drop item indices {missing}: they are not history edges for user "
                    f"index {user_index}"
                )
            removed_edge_groups.append(
                frozenset(
                    (user_index, self.mappings.num_users + item_index)
                    for item_index in unique_items
                )
            )

        batch_size = len(removed_edge_groups)
        degree_cache: dict[int, torch.Tensor] = {}
        removed_mask_cache: dict[tuple[int, int], torch.Tensor] = {}
        embedding_cache: dict[tuple[int, int], torch.Tensor] = {}

        def perturbed_degree(node: int) -> torch.Tensor:
            if node not in degree_cache:
                degree = torch.tensor(
                    [
                        self.degrees[node]
                        - sum(node == edge[0] or node == edge[1] for edge in removed_edges)
                        for removed_edges in removed_edge_groups
                    ],
                    dtype=self.ego_embeddings.dtype,
                    device=self.device,
                )
                if bool((degree < 0).any().item()):
                    raise ValueError(f"Perturbed degree became negative for node {node}")
                degree_cache[node] = degree
            return degree_cache[node]

        def canonical_edge(node: int, neighbor: int) -> tuple[int, int]:
            if node < self.mappings.num_users:
                return node, neighbor
            return neighbor, node

        def removed_mask(node: int, neighbor: int) -> torch.Tensor:
            edge = canonical_edge(node, neighbor)
            if edge not in removed_mask_cache:
                removed_mask_cache[edge] = torch.tensor(
                    [edge in removed_edges for removed_edges in removed_edge_groups],
                    dtype=torch.bool,
                    device=self.device,
                )
            return removed_mask_cache[edge]

        def layer_embedding(node: int, layer: int) -> torch.Tensor:
            cache_key = (node, layer)
            if cache_key in embedding_cache:
                return embedding_cache[cache_key]
            if layer == 0:
                embedding = self.ego_embeddings[node].unsqueeze(0).expand(batch_size, -1)
                embedding_cache[cache_key] = embedding
                return embedding

            node_degree = perturbed_degree(node)
            aggregated = torch.zeros(
                (batch_size, self.ego_embeddings.shape[1]),
                dtype=self.ego_embeddings.dtype,
                device=self.device,
            )
            for neighbor in self.neighbors[node]:
                neighbor_degree = perturbed_degree(neighbor)
                active = (
                    (node_degree > 0)
                    & (neighbor_degree > 0)
                    & ~removed_mask(node, neighbor)
                )
                normalization = torch.where(
                    active,
                    torch.sqrt(node_degree * neighbor_degree),
                    torch.ones_like(node_degree),
                )
                aggregated = aggregated + (
                    layer_embedding(neighbor, layer - 1)
                    / normalization.unsqueeze(1)
                    * active.unsqueeze(1)
                )
            embedding_cache[cache_key] = aggregated
            return aggregated

        user_node = user_index
        target_item_node = self.mappings.num_users + target_item_index
        user_embedding = torch.stack(
            [layer_embedding(user_node, layer) for layer in range(self.num_layers + 1)]
        ).mean(dim=0)
        target_embedding = torch.stack(
            [
                layer_embedding(target_item_node, layer)
                for layer in range(self.num_layers + 1)
            ]
        ).mean(dim=0)
        scores = (user_embedding * target_embedding).sum(dim=1)
        return tuple(float(score) for score in scores.detach().cpu().tolist())

    def score_drop_local(
        self,
        user_index: int,
        target_item_index: int,
        drop_item_indices: list[int],
        excluded_items: set[int],
    ) -> tuple[float, int]:
        if self.num_layers == 0:
            return _score_and_rank(
                self.baseline_users[user_index],
                self.baseline_items,
                target_item_index,
                excluded_items,
            )

        unique_items = sorted(set(int(item) for item in drop_item_indices))
        if not unique_items:
            raise ValueError("At least one item edge must be selected for removal")
        missing = [item for item in unique_items if item not in self.train_history[user_index]]
        if missing:
            raise ValueError(
                f"Cannot drop item indices {missing}: they are not history edges for user index "
                f"{user_index}"
            )

        user_node = user_index
        drop_item_nodes = {self.mappings.num_users + item for item in unique_items}
        dropped_edges = {(user_node, item_node) for item_node in drop_item_nodes}
        affected_by_layer = self._affected_nodes_by_layer({user_node, *drop_item_nodes})
        layer_maps = self._perturbed_layer_maps(
            affected_by_layer,
            user_node,
            drop_item_nodes,
            dropped_edges,
        )
        perturbed_final = self._perturbed_final_embeddings(affected_by_layer, layer_maps)
        perturbed_user = perturbed_final[user_node]

        scores = self.baseline_items @ perturbed_user
        for node, embedding in perturbed_final.items():
            if self.mappings.num_users <= node < self.mappings.num_users + self.mappings.num_items:
                item_index = node - self.mappings.num_users
                scores[item_index] = embedding @ perturbed_user
        return _score_and_rank_from_scores(scores, target_item_index, excluded_items)

    def _affected_nodes_by_layer(self, roots: set[int]) -> list[set[int]]:
        affected_by_layer = [set()]
        seen = set(roots)
        frontier = set(roots)
        for _ in range(self.num_layers):
            next_frontier: set[int] = set()
            for node in frontier:
                next_frontier.update(self.neighbors[node])
            next_frontier.difference_update(seen)
            seen.update(next_frontier)
            affected_by_layer.append(set(seen))
            frontier = next_frontier
        return affected_by_layer

    def _perturbed_layer_maps(
        self,
        affected_by_layer: list[set[int]],
        user_node: int,
        drop_item_nodes: set[int],
        dropped_edges: set[tuple[int, int]],
    ) -> list[dict[int, torch.Tensor]]:
        layer_maps: list[dict[int, torch.Tensor]] = [{}]
        previous: dict[int, torch.Tensor] = {}
        for layer in range(1, self.num_layers + 1):
            current: dict[int, torch.Tensor] = {}
            for node in affected_by_layer[layer]:
                current[node] = self._perturbed_layer_embedding(
                    node,
                    layer,
                    previous,
                    user_node,
                    drop_item_nodes,
                    dropped_edges,
                )
            layer_maps.append(current)
            previous = current
        return layer_maps

    def _perturbed_layer_embedding(
        self,
        node: int,
        layer: int,
        previous: dict[int, torch.Tensor],
        user_node: int,
        drop_item_nodes: set[int],
        dropped_edges: set[tuple[int, int]],
    ) -> torch.Tensor:
        degree = self._perturbed_degree(node, user_node, drop_item_nodes)
        result = torch.zeros(
            self.ego_embeddings.shape[1],
            dtype=self.ego_embeddings.dtype,
            device=self.device,
        )
        if degree <= 0:
            return result

        node_weight = degree ** -0.5
        for neighbor in self.neighbors[node]:
            if (node, neighbor) in dropped_edges or (neighbor, node) in dropped_edges:
                continue
            neighbor_degree = self._perturbed_degree(neighbor, user_node, drop_item_nodes)
            if neighbor_degree <= 0:
                continue
            source_embedding = previous.get(neighbor)
            if source_embedding is None:
                source_embedding = self.baseline_layers[layer - 1][neighbor]
            result = result + node_weight * (neighbor_degree ** -0.5) * source_embedding
        return result

    def _perturbed_degree(
        self,
        node: int,
        user_node: int,
        drop_item_nodes: set[int],
    ) -> int:
        if node == user_node:
            return self.degrees[node] - len(drop_item_nodes)
        if node in drop_item_nodes:
            return self.degrees[node] - 1
        return self.degrees[node]

    def _perturbed_final_embeddings(
        self,
        affected_by_layer: list[set[int]],
        layer_maps: list[dict[int, torch.Tensor]],
    ) -> dict[int, torch.Tensor]:
        affected_nodes = set().union(*affected_by_layer[1:]) if self.num_layers else set()
        final_embeddings: dict[int, torch.Tensor] = {}
        for node in affected_nodes:
            total = self.baseline_layers[0][node].clone()
            for layer in range(1, self.num_layers + 1):
                total = total + layer_maps[layer].get(node, self.baseline_layers[layer][node])
            final_embeddings[node] = total / float(self.num_layers + 1)
        return final_embeddings


def _build_neighbors(
    num_users: int,
    num_items: int,
    train_pairs: torch.Tensor,
) -> tuple[list[set[int]], list[int]]:
    num_nodes = num_users + num_items
    neighbors = [set() for _ in range(num_nodes)]
    pairs = torch.unique(train_pairs.cpu().long(), dim=0)
    for user_index, item_index in pairs.tolist():
        if user_index < 0 or user_index >= num_users:
            raise ValueError("train_pairs contains an out-of-range user index")
        if item_index < 0 or item_index >= num_items:
            raise ValueError("train_pairs contains an out-of-range item index")
        user_node = int(user_index)
        item_node = num_users + int(item_index)
        neighbors[user_node].add(item_node)
        neighbors[item_node].add(user_node)
    return neighbors, [len(node_neighbors) for node_neighbors in neighbors]
