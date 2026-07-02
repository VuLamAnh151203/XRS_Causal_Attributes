"""Direct pair-target metrics after cumulative attribute-support edge drops."""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..core.artifacts import ArtifactPaths, load_json, load_mappings, load_tensor, load_user_history
from ..core.data import IdMappings
from ..core.graph import propagate


@dataclass
class MetricAccumulator:
    recall_k: tuple[int, ...]
    ndcg_k: int
    count: int = 0
    recall_totals: dict[int, float] = field(default_factory=dict)
    ndcg_total: float = 0.0

    def __post_init__(self) -> None:
        self.recall_totals = {cutoff: 0.0 for cutoff in self.recall_k}

    def add_rank(self, rank: int) -> None:
        self.count += 1
        for cutoff in self.recall_k:
            self.recall_totals[cutoff] += 1.0 if rank <= cutoff else 0.0
        if rank <= self.ndcg_k:
            self.ndcg_total += 1.0 / math.log2(rank + 1.0)

    def as_dict(self) -> dict[str, float | int]:
        metrics: dict[str, float | int] = {
            f"recall@{cutoff}": (
                self.recall_totals[cutoff] / self.count if self.count else 0.0
            )
            for cutoff in self.recall_k
        }
        metrics[f"ndcg@{self.ndcg_k}"] = self.ndcg_total / self.count if self.count else 0.0
        metrics["pairs_evaluated"] = self.count
        return metrics


@dataclass
class CoverageCounter:
    pairs_seen: int = 0
    pairs_with_attributes: int = 0
    pairs_missing_support_mapping: int = 0
    pairs_unknown_user: int = 0
    pairs_unknown_target_item: int = 0
    pairs_target_seen_in_train: int = 0
    pairs_invalid_target: int = 0
    unknown_support_items: int = 0
    support_items_not_in_history: int = 0
    empty_drop_sets_by_m: dict[int, int] = field(default_factory=dict)
    valid_pairs_by_m: dict[int, int] = field(default_factory=dict)

    def as_dict(self, max_m: int) -> dict[str, Any]:
        return {
            "pairs_seen": self.pairs_seen,
            "pairs_with_attributes": self.pairs_with_attributes,
            "pairs_missing_support_mapping": self.pairs_missing_support_mapping,
            "pairs_unknown_user": self.pairs_unknown_user,
            "pairs_unknown_target_item": self.pairs_unknown_target_item,
            "pairs_target_seen_in_train": self.pairs_target_seen_in_train,
            "pairs_invalid_target": self.pairs_invalid_target,
            "unknown_support_items": self.unknown_support_items,
            "support_items_not_in_history": self.support_items_not_in_history,
            "empty_drop_sets_by_m": {
                str(m): self.empty_drop_sets_by_m.get(m, 0) for m in range(1, max_m + 1)
            },
            "valid_pairs_by_m": {
                str(m): self.valid_pairs_by_m.get(m, 0) for m in range(1, max_m + 1)
            },
        }


@dataclass(frozen=True)
class PerturbationTask:
    raw_user_id: str
    raw_target_item_id: str
    user_index: int
    target_item_index: int
    m: int
    drop_item_indices: tuple[int, ...]
    origin_rank: int


@dataclass(frozen=True)
class FixedDegreeAdjacencyBuilder:
    num_users: int
    num_items: int
    base_indices: torch.Tensor
    base_values: torch.Tensor
    edge_positions: dict[tuple[int, int], tuple[int, int]]

    @property
    def num_nodes(self) -> int:
        return self.num_users + self.num_items

    @classmethod
    def from_train_pairs(
        cls,
        num_users: int,
        num_items: int,
        train_pairs: torch.Tensor,
        device: torch.device,
    ) -> "FixedDegreeAdjacencyBuilder":
        pairs = torch.unique(train_pairs.cpu().long(), dim=0)
        if pairs.numel() == 0:
            base_indices = torch.empty((2, 0), dtype=torch.long, device=device)
            base_values = torch.empty((0,), dtype=torch.float32, device=device)
            return cls(num_users, num_items, base_indices, base_values, {})
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("train_pairs must have shape [num_edges, 2]")

        users = pairs[:, 0]
        items = pairs[:, 1]
        if users.min() < 0 or users.max() >= num_users:
            raise ValueError("train_pairs contains an out-of-range user index")
        if items.min() < 0 or items.max() >= num_items:
            raise ValueError("train_pairs contains an out-of-range item index")

        item_nodes = items + num_users
        sources = torch.cat((users, item_nodes))
        destinations = torch.cat((item_nodes, users))
        degrees = torch.bincount(sources, minlength=num_users + num_items).float()
        inverse_sqrt_degree = torch.zeros_like(degrees)
        nonzero = degrees > 0
        inverse_sqrt_degree[nonzero] = degrees[nonzero].pow(-0.5)
        values = inverse_sqrt_degree[sources] * inverse_sqrt_degree[destinations]
        indices = torch.stack((sources, destinations))

        edge_positions: dict[tuple[int, int], tuple[int, int]] = {}
        edge_count = int(pairs.shape[0])
        for position, (user_index, item_index) in enumerate(pairs.tolist()):
            edge_positions[(int(user_index), int(item_index))] = (
                position,
                position + edge_count,
            )

        return cls(
            num_users,
            num_items,
            indices.to(device),
            values.to(device),
            edge_positions,
        )

    def perturbed_components(
        self,
        user_index: int,
        drop_item_indices: tuple[int, ...],
        offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = torch.ones(
            self.base_values.shape[0],
            dtype=torch.bool,
            device=self.base_values.device,
        )
        missing: list[int] = []
        for item_index in drop_item_indices:
            positions = self.edge_positions.get((int(user_index), int(item_index)))
            if positions is None:
                missing.append(int(item_index))
                continue
            forward_position, reverse_position = positions
            mask[forward_position] = False
            mask[reverse_position] = False
        if missing:
            raise ValueError(
                f"Cannot drop item indices {missing}: they are not history edges for user "
                f"index {user_index}"
            )
        return (
            self.base_indices[:, mask] + offset * self.num_nodes,
            self.base_values[mask],
        )


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as input_file:
        return pickle.load(input_file)


def _save_json_atomic(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(destination.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
    temporary_path.replace(destination)


def _validate_cutoffs(recall_k: tuple[int, ...], ndcg_k: int, max_m: int) -> None:
    if not recall_k or min(recall_k) <= 0:
        raise ValueError("recall_k must contain positive cutoffs")
    if ndcg_k <= 0:
        raise ValueError("ndcg_k must be positive")
    if max_m <= 0:
        raise ValueError("max_m must be positive")


def _target_rank(
    user_embedding: torch.Tensor,
    item_embeddings: torch.Tensor,
    target_item_index: int,
    excluded_items: set[int],
) -> int | None:
    if target_item_index in excluded_items:
        return None
    scores = item_embeddings @ user_embedding
    candidate_mask = torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)
    if excluded_items:
        excluded = torch.tensor(sorted(excluded_items), dtype=torch.long, device=scores.device)
        candidate_mask[excluded] = False
    if not bool(candidate_mask[target_item_index].item()):
        return None
    target_score = scores[target_item_index]
    higher_scores = scores[candidate_mask] > target_score
    return int(higher_scores.sum().detach().cpu().item()) + 1


def _map_raw_pair(
    raw_pair: Any,
    mappings: IdMappings,
    coverage: CoverageCounter,
) -> tuple[str, str, int, int] | None:
    if not isinstance(raw_pair, tuple) or len(raw_pair) != 2:
        coverage.pairs_invalid_target += 1
        return None
    raw_user_id = str(raw_pair[0])
    raw_target_item_id = str(raw_pair[1])
    user_index = mappings.user_to_index.get(raw_user_id)
    if user_index is None:
        coverage.pairs_unknown_user += 1
        return None
    target_item_index = mappings.item_to_index.get(raw_target_item_id)
    if target_item_index is None:
        coverage.pairs_unknown_target_item += 1
        return None
    return raw_user_id, raw_target_item_id, user_index, target_item_index


def _map_support_items(
    raw_items: Any,
    mappings: IdMappings,
    train_history: dict[int, set[int]],
    user_index: int,
    coverage: CoverageCounter,
) -> list[int]:
    if raw_items is None:
        return []
    if not isinstance(raw_items, list):
        raise ValueError("attribute_items_mapping values must be lists of item IDs")

    mapped_items: list[int] = []
    seen: set[int] = set()
    user_history = train_history.get(user_index, set())
    for raw_item in raw_items:
        item_index = mappings.item_to_index.get(str(raw_item))
        if item_index is None:
            coverage.unknown_support_items += 1
            continue
        if item_index not in user_history:
            coverage.support_items_not_in_history += 1
            continue
        if item_index not in seen:
            mapped_items.append(item_index)
            seen.add(item_index)
    return mapped_items


def _planned_tasks(
    chosen_attributes: dict[Any, Any],
    attribute_supports: dict[Any, Any],
    mappings: IdMappings,
    train_history: dict[int, set[int]],
    baseline_users: torch.Tensor,
    baseline_items: torch.Tensor,
    recall_k: tuple[int, ...],
    ndcg_k: int,
    max_m: int,
) -> tuple[list[PerturbationTask], dict[str, dict[str, float | int]], CoverageCounter]:
    coverage = CoverageCounter()
    origin_accumulators = {
        str(m): MetricAccumulator(recall_k, ndcg_k) for m in range(1, max_m + 1)
    }
    tasks: list[PerturbationTask] = []

    for raw_pair, attributes in chosen_attributes.items():
        coverage.pairs_seen += 1
        mapped_pair = _map_raw_pair(raw_pair, mappings, coverage)
        if mapped_pair is None:
            continue
        raw_user_id, raw_target_item_id, user_index, target_item_index = mapped_pair
        excluded_items = train_history.get(user_index, set())
        if target_item_index in excluded_items:
            coverage.pairs_target_seen_in_train += 1
            continue

        if not isinstance(attributes, list):
            raise ValueError("new_chosen_sorted_attributes values must be lists")
        if not attributes:
            continue
        coverage.pairs_with_attributes += 1

        support_by_attribute = attribute_supports.get(raw_pair)
        if support_by_attribute is None:
            coverage.pairs_missing_support_mapping += 1
            continue
        if not isinstance(support_by_attribute, dict):
            raise ValueError("attribute_items_mapping values must be dictionaries")

        origin_rank = _target_rank(
            baseline_users[user_index],
            baseline_items,
            target_item_index,
            excluded_items,
        )
        if origin_rank is None:
            coverage.pairs_invalid_target += 1
            continue

        cumulative_drop: set[int] = set()
        max_pair_m = min(max_m, len(attributes))
        for m in range(1, max_m + 1):
            if m <= max_pair_m:
                raw_attribute = attributes[m - 1]
                if not isinstance(raw_attribute, (tuple, list)) or not raw_attribute:
                    raise ValueError("chosen attributes must be non-empty tuples/lists")
                attribute_name = str(raw_attribute[0])
                support_items = support_by_attribute.get(attribute_name)
                cumulative_drop.update(
                    _map_support_items(
                        support_items,
                        mappings,
                        train_history,
                        user_index,
                        coverage,
                    )
                )

            if not cumulative_drop:
                coverage.empty_drop_sets_by_m[m] = coverage.empty_drop_sets_by_m.get(m, 0) + 1
                continue

            drop_item_indices = tuple(sorted(cumulative_drop))
            coverage.valid_pairs_by_m[m] = coverage.valid_pairs_by_m.get(m, 0) + 1
            origin_accumulators[str(m)].add_rank(origin_rank)
            tasks.append(
                PerturbationTask(
                    raw_user_id=raw_user_id,
                    raw_target_item_id=raw_target_item_id,
                    user_index=user_index,
                    target_item_index=target_item_index,
                    m=m,
                    drop_item_indices=drop_item_indices,
                    origin_rank=origin_rank,
                )
            )

    return tasks, {m: accumulator.as_dict() for m, accumulator in origin_accumulators.items()}, coverage


@torch.no_grad()
def _run_perturbation_tasks(
    tasks: list[PerturbationTask],
    mappings: IdMappings,
    adjacency_builder: FixedDegreeAdjacencyBuilder,
    train_history: dict[int, set[int]],
    ego_embeddings: torch.Tensor,
    num_layers: int,
    recall_k: tuple[int, ...],
    ndcg_k: int,
    device: torch.device,
    perturbation_batch_size: int,
) -> dict[str, dict[str, float | int]]:
    if perturbation_batch_size <= 0:
        raise ValueError("perturbation_batch_size must be positive")

    accumulators = {
        str(m): MetricAccumulator(recall_k, ndcg_k)
        for m in sorted({task.m for task in tasks})
    }
    num_nodes = adjacency_builder.num_nodes

    from tqdm import tqdm

    for start in tqdm(
        range(0, len(tasks), perturbation_batch_size),
        desc="Running batched global top-M perturbations",
    ):
        batch_tasks = tasks[start : start + perturbation_batch_size]
        adjacency_indices: list[torch.Tensor] = []
        adjacency_values: list[torch.Tensor] = []
        for offset, task in enumerate(batch_tasks):
            indices, values = adjacency_builder.perturbed_components(
                task.user_index,
                task.drop_item_indices,
                offset,
            )
            adjacency_indices.append(indices)
            adjacency_values.append(values)

        batched_adjacency = torch.sparse_coo_tensor(
            torch.cat(adjacency_indices, dim=1),
            torch.cat(adjacency_values),
            (len(batch_tasks) * num_nodes, len(batch_tasks) * num_nodes),
            device=device,
        ).coalesce()
        perturbed_all = propagate(
            ego_embeddings.repeat(len(batch_tasks), 1),
            batched_adjacency,
            num_layers,
        )

        for offset, task in enumerate(batch_tasks):
            node_offset = offset * num_nodes
            perturbed_users = perturbed_all[
                node_offset : node_offset + mappings.num_users
            ]
            perturbed_items = perturbed_all[
                node_offset + mappings.num_users : node_offset + num_nodes
            ]
            rank = _target_rank(
                perturbed_users[task.user_index],
                perturbed_items,
                task.target_item_index,
                train_history.get(task.user_index, set()),
            )
            if rank is None:
                continue
            accumulators[str(task.m)].add_rank(rank)

    return {m: accumulator.as_dict() for m, accumulator in accumulators.items()}


def _metric_delta(
    origin_by_m: dict[str, dict[str, float | int]],
    perturbed_by_m: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float]]:
    deltas: dict[str, dict[str, float]] = {}
    for m, origin_metrics in origin_by_m.items():
        perturbed_metrics = perturbed_by_m.get(m, {})
        deltas[m] = {}
        for key, origin_value in origin_metrics.items():
            if key == "pairs_evaluated" or key not in perturbed_metrics:
                continue
            deltas[m][key] = float(perturbed_metrics[key]) - float(origin_value)
    return deltas


@torch.no_grad()
def run_top_m_attribute_perturbation_evaluation(
    artifact_dir: str | Path,
    chosen_attributes_pkl: str | Path,
    attribute_items_mapping_pkl: str | Path,
    num_layers: int,
    recall_k: tuple[int, ...] = (10, 20),
    ndcg_k: int = 20,
    max_m: int = 3,
    device: str = "auto",
    perturbation_batch_size: int = 4,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate pair-target metrics after cumulative top-M attribute drops."""

    recall_k = tuple(sorted(set(int(cutoff) for cutoff in recall_k)))
    ndcg_k = int(ndcg_k)
    max_m = int(max_m)
    _validate_cutoffs(recall_k, ndcg_k, max_m)

    paths = ArtifactPaths(Path(artifact_dir).resolve())
    mappings = load_mappings(paths.mappings)
    train_pairs = load_tensor(paths.train_pairs).long()
    train_history = load_user_history(paths.user_history)
    chosen_attributes = _load_pickle(chosen_attributes_pkl)
    attribute_supports = _load_pickle(attribute_items_mapping_pkl)
    if not isinstance(chosen_attributes, dict):
        raise ValueError("new_chosen_sorted_attributes.pkl must contain a dictionary")
    if not isinstance(attribute_supports, dict):
        raise ValueError("attribute_items_mapping.pkl must contain a dictionary")

    resolved_device = _resolve_device(device)
    baseline_users = load_tensor(paths.user_baseline_embeddings).to(resolved_device)
    baseline_items = load_tensor(paths.item_baseline_embeddings).to(resolved_device)
    user_ego = load_tensor(paths.user_ego_embeddings).to(resolved_device)
    item_ego = load_tensor(paths.item_ego_embeddings).to(resolved_device)
    ego_embeddings = torch.cat((user_ego, item_ego), dim=0)
    adjacency_builder = FixedDegreeAdjacencyBuilder.from_train_pairs(
        mappings.num_users,
        mappings.num_items,
        train_pairs,
        resolved_device,
    )

    tasks, origin_by_m, coverage = _planned_tasks(
        chosen_attributes,
        attribute_supports,
        mappings,
        train_history,
        baseline_users,
        baseline_items,
        recall_k,
        ndcg_k,
        max_m,
    )
    perturbed_by_m = _run_perturbation_tasks(
        tasks,
        mappings,
        adjacency_builder,
        train_history,
        ego_embeddings,
        num_layers,
        recall_k,
        ndcg_k,
        resolved_device,
        perturbation_batch_size,
    )
    for m in range(1, max_m + 1):
        perturbed_by_m.setdefault(str(m), MetricAccumulator(recall_k, ndcg_k).as_dict())

    reference_validation_metrics = (
        load_json(paths.validation_metrics) if paths.validation_metrics.exists() else None
    )
    result = {
        "origin_by_m": origin_by_m,
        "perturbed_by_m": perturbed_by_m,
        "delta_by_m": _metric_delta(origin_by_m, perturbed_by_m),
        "coverage": coverage.as_dict(max_m),
        "reference_validation_metrics": reference_validation_metrics,
        "metadata": {
            "artifact_dir": str(paths.base_dir),
            "chosen_attributes_pkl": str(Path(chosen_attributes_pkl)),
            "attribute_items_mapping_pkl": str(Path(attribute_items_mapping_pkl)),
            "num_layers": int(num_layers),
            "recall_k": list(recall_k),
            "ndcg_k": ndcg_k,
            "max_m": max_m,
            "device": str(resolved_device),
            "perturbation_mode": "global-batched",
            "normalization": "fixed-original-degree",
            "perturbation_batch_size": int(perturbation_batch_size),
            "embedding_source_for_perturbation": [
                str(paths.user_ego_embeddings),
                str(paths.item_ego_embeddings),
            ],
            "embedding_source_for_origin": [
                str(paths.user_baseline_embeddings),
                str(paths.item_baseline_embeddings),
            ],
        },
    }
    if save_path is not None:
        _save_json_atomic(save_path, result)
    return result
