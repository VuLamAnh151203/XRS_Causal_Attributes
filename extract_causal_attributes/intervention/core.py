"""Core utilities for building sparse LightGCN intervention artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
import time
from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from extract_causal_attributes.id_mappings import IdMappings, load_id_mappings


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_SCHEMA_VERSION = 2
INTERVENTION_GENERATION_VERSION = 2
Node = tuple[str, int]
Edge = tuple[int, int]


class InterventionError(ValueError):
    """Raised when intervention inputs or settings are invalid."""


class PairIdOutOfRangeError(InterventionError):
    """Raised when a support pair cannot be scored against the embedding tensors."""


@dataclass(frozen=True)
class InterventionConfig:
    attribute_support_path: Path
    vocabulary_path: Path
    user_history_path: Path
    id_mappings_path: Path
    user_ego_embeddings_path: Path
    item_ego_embeddings_path: Path
    lightgcn_config_path: Path
    output_dir: Path
    sparsity_level: int
    sample_multiplier: int
    max_history_drop_fraction: float
    subset_probability: float
    random_seed: int
    exhaustive_attribute_limit: int
    max_sampling_attempts: int
    device: str
    pairs_per_shard: int
    score_cache_max_entries: int

    @property
    def requested_intervention_count(self) -> int:
        return self.sparsity_level * self.sample_multiplier


@dataclass(frozen=True)
class Vocabulary:
    attributes: tuple[str, ...]
    attribute_to_index: dict[str, int]


@dataclass(frozen=True)
class BipartiteGraph:
    user_items: dict[int, tuple[int, ...]]
    item_users: dict[int, tuple[int, ...]]

    def has_edge(self, user_id: int, item_id: int) -> bool:
        return item_id in self.user_items.get(user_id, ())

    def neighbors(self, node: Node) -> tuple[Node, ...]:
        kind, node_id = node
        if kind == "u":
            return tuple(("i", item_id) for item_id in self.user_items.get(node_id, ()))
        if kind == "i":
            return tuple(("u", user_id) for user_id in self.item_users.get(node_id, ()))
        raise InterventionError(f"Unsupported graph node kind: {kind!r}")

    def degree(self, node: Node) -> int:
        kind, node_id = node
        if kind == "u":
            return len(self.user_items.get(node_id, ()))
        if kind == "i":
            return len(self.item_users.get(node_id, ()))
        raise InterventionError(f"Unsupported graph node kind: {kind!r}")


@dataclass(frozen=True)
class SampledIntervention:
    attributes: tuple[str, ...]
    attribute_indices: tuple[int, ...]
    removed_item_ids: tuple[int, ...]


@dataclass(frozen=True)
class InterventionRow:
    attribute_indices: tuple[int, ...]
    removed_item_ids: tuple[int, ...]
    y_h: float
    y_delta: float


@dataclass(frozen=True)
class PairResult:
    pair_index: int
    user_id: int
    user_index: int | None
    target_item_id: int
    target_item_index: int | None
    baseline_score: float | None
    eligible_history_count: int
    supported_target_attribute_count: int
    requested_intervention_count: int
    rows: tuple[InterventionRow, ...]
    skip_reason: str | None = None


@dataclass
class RunStats:
    processed_pair_count: int = 0
    generated_intervention_count: int = 0
    shortfall_pair_count: int = 0
    zero_row_pair_count: int = 0
    skipped_pair_count: int = 0
    invalid_pair_id_skip_count: int = 0
    written_shard_count: int = 0
    baseline_scoring_seconds: float = 0.0
    intervention_scoring_seconds: float = 0.0
    previous_score_cache_hit_count: int = 0
    previous_score_cache_miss_count: int = 0
    previous_full_time_seconds: float = 0.0


@dataclass(frozen=True)
class SupportRecord:
    pair_index: int
    user_id: int
    user_index: int | None
    target_item_id: int
    target_item_index: int | None
    supports: dict[str, tuple[int, ...]]


def _resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else REPO_ROOT / path


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise InterventionError(f"Configuration field {key!r} must be a mapping.")
    return value


def _require_string(payload: Mapping[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InterventionError(f"Configuration field {context}.{key} must be a non-empty string.")
    return value.strip()


def _positive_int(payload: Mapping[str, Any], key: str, context: str, default: int) -> int:
    value = payload.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise InterventionError(f"Configuration field {context}.{key} must be a positive integer.")
    return value


def _probability(payload: Mapping[str, Any], key: str, context: str, default: float) -> float:
    value = payload.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise InterventionError(f"Configuration field {context}.{key} must be numeric.")
    probability = float(value)
    if probability <= 0.0 or probability > 1.0:
        raise InterventionError(f"Configuration field {context}.{key} must be in (0, 1].")
    return probability


def load_config(path: Path) -> InterventionConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required. Install extract_causal_attributes/intervention/requirements.txt."
        ) from exc

    with path.open("r", encoding="utf-8") as input_file:
        payload = yaml.safe_load(input_file)
    if not isinstance(payload, Mapping):
        raise InterventionError("The root YAML configuration value must be a mapping.")

    paths = _require_mapping(payload, "paths")
    interventions = _require_mapping(payload, "interventions")
    runtime = _require_mapping(payload, "runtime")
    seed = interventions.get("random_seed", 42)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise InterventionError("Configuration field interventions.random_seed must be an integer.")

    return InterventionConfig(
        attribute_support_path=_resolve_repo_path(
            _require_string(paths, "attribute_support", "paths")
        ),
        vocabulary_path=_resolve_repo_path(_require_string(paths, "vocabulary", "paths")),
        user_history_path=_resolve_repo_path(_require_string(paths, "user_history", "paths")),
        id_mappings_path=_resolve_repo_path(_require_string(paths, "id_mappings", "paths")),
        user_ego_embeddings_path=_resolve_repo_path(
            _require_string(paths, "user_ego_embeddings", "paths")
        ),
        item_ego_embeddings_path=_resolve_repo_path(
            _require_string(paths, "item_ego_embeddings", "paths")
        ),
        lightgcn_config_path=_resolve_repo_path(
            _require_string(paths, "lightgcn_config", "paths")
        ),
        output_dir=_resolve_repo_path(_require_string(paths, "output_dir", "paths")),
        sparsity_level=_positive_int(interventions, "sparsity_level", "interventions", 5),
        sample_multiplier=_positive_int(interventions, "sample_multiplier", "interventions", 3),
        max_history_drop_fraction=_probability(
            interventions, "max_history_drop_fraction", "interventions", 0.50
        ),
        subset_probability=_probability(
            interventions, "subset_probability", "interventions", 0.50
        ),
        random_seed=seed,
        exhaustive_attribute_limit=_positive_int(
            interventions, "exhaustive_attribute_limit", "interventions", 20
        ),
        max_sampling_attempts=_positive_int(
            interventions, "max_sampling_attempts", "interventions", 10_000
        ),
        device=_require_string(runtime, "device", "runtime"),
        pairs_per_shard=_positive_int(runtime, "pairs_per_shard", "runtime", 1),
        score_cache_max_entries=_positive_int(
            runtime, "score_cache_max_entries", "runtime", 100_000
        ),
    )


def _coerce_int(value: Any, context: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InterventionError(f"{context} must be integer-compatible, got {value!r}.") from exc


def _unwrap_vocabulary(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        for wrapper_key in (
            "attribute_to_index",
            "attribute_to_idx",
            "attribute2id",
            "token_to_id",
            "stoi",
            "index_to_attribute",
            "id_to_attribute",
            "itos",
        ):
            if wrapper_key in payload and isinstance(payload[wrapper_key], Mapping):
                return payload[wrapper_key]
        for wrapper_key in ("vocabulary", "attributes", "attribute_list", "id_to_attribute", "itos"):
            if wrapper_key in payload and isinstance(payload[wrapper_key], Sequence):
                return payload[wrapper_key]
    return payload


def load_vocabulary(path: Path) -> Vocabulary:
    with path.open("r", encoding="utf-8") as input_file:
        payload = _unwrap_vocabulary(json.load(input_file))

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        attributes = list(payload)
        if not all(isinstance(attribute, str) and attribute for attribute in attributes):
            raise InterventionError("Vocabulary list entries must be non-empty strings.")
        if len(set(attributes)) != len(attributes):
            raise InterventionError("Vocabulary list contains duplicate attributes.")
        return Vocabulary(tuple(attributes), {attribute: index for index, attribute in enumerate(attributes)})

    if isinstance(payload, Mapping):
        index_to_attribute: dict[int, str] = {}
        if payload and all(isinstance(attribute, str) and attribute for attribute in payload.values()):
            for raw_index, attribute in payload.items():
                index = _coerce_int(raw_index, f"Vocabulary index for {attribute!r}")
                if index in index_to_attribute:
                    raise InterventionError(f"Vocabulary index {index} is assigned more than once.")
                index_to_attribute[index] = attribute
        else:
            for attribute, raw_index in payload.items():
                if not isinstance(attribute, str) or not attribute:
                    raise InterventionError("Vocabulary mapping keys must be non-empty strings.")
                index = _coerce_int(raw_index, f"Vocabulary index for {attribute!r}")
                if index in index_to_attribute:
                    raise InterventionError(f"Vocabulary index {index} is assigned more than once.")
                index_to_attribute[index] = attribute
        expected_indices = set(range(len(index_to_attribute)))
        if set(index_to_attribute) != expected_indices:
            raise InterventionError("Vocabulary mapping indices must be contiguous and start at zero.")
        attributes = tuple(index_to_attribute[index] for index in range(len(index_to_attribute)))
        return Vocabulary(attributes, {attribute: index for index, attribute in enumerate(attributes)})

    raise InterventionError("Vocabulary must be an ordered JSON list or attribute-to-index mapping.")


def load_user_history(path: Path) -> dict[int, tuple[int, ...]]:
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if isinstance(payload, Mapping) and set(payload) == {"user_history"}:
        payload = payload["user_history"]
    if not isinstance(payload, Mapping):
        raise InterventionError("User history must be a JSON object keyed by user ID.")

    user_items: dict[int, tuple[int, ...]] = {}
    for raw_user_id, raw_items in payload.items():
        user_id = _coerce_int(raw_user_id, "User-history key")
        if isinstance(raw_items, Mapping):
            history_keys = [key for key in ("history", "items", "item_ids") if key in raw_items]
            if len(history_keys) != 1:
                raise InterventionError(
                    f"History for user {user_id} must contain exactly one history list field."
                )
            raw_items = raw_items[history_keys[0]]
        if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes, bytearray)):
            raise InterventionError(f"History for user {user_id} must be a list of item IDs.")
        seen: set[int] = set()
        items: list[int] = []
        for raw_item_id in raw_items:
            item_id = _coerce_int(raw_item_id, f"History item for user {user_id}")
            if item_id not in seen:
                seen.add(item_id)
                items.append(item_id)
        user_items[user_id] = tuple(items)
    return user_items


def build_graph(user_items: Mapping[int, Sequence[int]]) -> BipartiteGraph:
    normalized_user_items: dict[int, tuple[int, ...]] = {}
    item_users: dict[int, list[int]] = defaultdict(list)
    for user_id, raw_items in user_items.items():
        seen: set[int] = set()
        items: list[int] = []
        for item_id in raw_items:
            if item_id not in seen:
                seen.add(item_id)
                items.append(item_id)
                item_users[item_id].append(user_id)
        normalized_user_items[user_id] = tuple(items)
    return BipartiteGraph(
        user_items=normalized_user_items,
        item_users={item_id: tuple(users) for item_id, users in item_users.items()},
    )


def _recursive_config_values(payload: Any, accepted_keys: set[str]) -> list[Any]:
    values: list[Any] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if str(key).lower() in accepted_keys:
                values.append(value)
            values.extend(_recursive_config_values(value, accepted_keys))
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for value in payload:
            values.extend(_recursive_config_values(value, accepted_keys))
    return values


def load_lightgcn_layer_count(path: Path) -> int:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read the LightGCN configuration.") from exc

    with path.open("r", encoding="utf-8") as input_file:
        payload = yaml.safe_load(input_file)
    candidates = _recursive_config_values(
        payload,
        {
            "n_layers",
            "n_layer",
            "num_layers",
            "num_layer",
            "layers",
            "lightgcn_layers",
            "num_lightgcn_layers",
            "propagation_layers",
            "layer_num",
        },
    )
    layers = {
        value for value in candidates if isinstance(value, int) and not isinstance(value, bool)
    }
    if not layers:
        raise InterventionError(
            "Could not find the LightGCN layer count in the training config. "
            "Expected one of: n_layers, num_layers, lightgcn_layers, num_lightgcn_layers, layer_num."
        )
    if len(layers) != 1:
        raise InterventionError(f"LightGCN config contains ambiguous layer counts: {sorted(layers)}")
    layer_count = layers.pop()
    if layer_count < 0:
        raise InterventionError("LightGCN layer count must be non-negative.")

    aggregation_values = _recursive_config_values(
        payload, {"aggregation", "layer_aggregation", "pooling"}
    )
    for value in aggregation_values:
        if isinstance(value, str) and value.lower() not in {"mean", "average", "avg"}:
            raise InterventionError(
                f"Unsupported LightGCN layer aggregation {value!r}; expected mean aggregation."
            )
    return layer_count


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required. Install extract_causal_attributes/intervention/requirements.txt."
        ) from exc
    return torch


def _find_tensor(payload: Any, context: str) -> Any:
    torch = _load_torch()
    if torch.is_tensor(payload):
        return payload
    if hasattr(payload, "weight") and torch.is_tensor(payload.weight):
        return payload.weight
    if isinstance(payload, Mapping):
        preferred_keys = (
            "weight",
            "embedding",
            "embeddings",
            "ego_embeddings",
            "user_ego_embeddings",
            "item_ego_embeddings",
        )
        for key in preferred_keys:
            if key in payload and torch.is_tensor(payload[key]):
                return payload[key]
        tensors = [value for value in payload.values() if torch.is_tensor(value)]
        if len(tensors) == 1:
            return tensors[0]
    raise InterventionError(f"{context} does not contain one identifiable embedding tensor.")


def load_embedding_tensor(path: Path) -> Any:
    torch = _load_torch()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    tensor = _find_tensor(payload, str(path)).detach().float()
    if tensor.ndim != 2:
        raise InterventionError(f"Embedding tensor {path} must be two-dimensional.")
    return tensor


def resolve_device(requested_device: str) -> str:
    torch = _load_torch()
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda"):
        match = re.fullmatch(r"cuda(?::(\d+))?", requested_device)
        if match is None:
            raise InterventionError(
                f"Requested CUDA device {requested_device!r} must use "
                "'cuda' or 'cuda:<non-negative index>'."
            )
        if not torch.cuda.is_available():
            raise InterventionError(f"Requested device {requested_device!r}, but CUDA is unavailable.")
        if match.group(1) is not None:
            cuda_index = int(match.group(1))
            device_count = torch.cuda.device_count()
            if cuda_index >= device_count:
                raise InterventionError(
                    f"Requested device {requested_device!r}, but only {device_count} CUDA "
                    f"device(s) are available. Use a CUDA index in [0, {device_count})."
                )
    return requested_device


class HybridLocalLightGCNScorer:
    """Compute exact root scores through lazy K-hop LightGCN dependencies."""

    def __init__(
        self,
        graph: BipartiteGraph,
        user_ego_embeddings: Any,
        item_ego_embeddings: Any,
        layer_count: int,
        device: str = "cpu",
    ) -> None:
        torch = _load_torch()
        if user_ego_embeddings.ndim != 2 or item_ego_embeddings.ndim != 2:
            raise InterventionError("User and item ego embeddings must be two-dimensional tensors.")
        if user_ego_embeddings.shape[1] != item_ego_embeddings.shape[1]:
            raise InterventionError("User and item ego embeddings must have the same dimension.")
        if layer_count < 0:
            raise InterventionError("LightGCN layer count must be non-negative.")

        self.graph = graph
        self.user_ego_embeddings = user_ego_embeddings.detach().float().to(device)
        self.item_ego_embeddings = item_ego_embeddings.detach().float().to(device)
        self.layer_count = layer_count
        self.device = device
        self._torch = torch
        self._zero = torch.zeros(
            self.user_ego_embeddings.shape[1],
            dtype=self.user_ego_embeddings.dtype,
            device=device,
        )
        self._validate_graph_ids()

    def _validate_graph_ids(self) -> None:
        user_count = self.user_ego_embeddings.shape[0]
        item_count = self.item_ego_embeddings.shape[0]
        invalid_users = [user_id for user_id in self.graph.user_items if not 0 <= user_id < user_count]
        invalid_items = [item_id for item_id in self.graph.item_users if not 0 <= item_id < item_count]
        if invalid_users:
            raise InterventionError(
                f"Graph user ID {invalid_users[0]} is outside user embedding rows [0, {user_count})."
            )
        if invalid_items:
            raise InterventionError(
                f"Graph item ID {invalid_items[0]} is outside item embedding rows [0, {item_count})."
            )

    def validate_pair_ids(self, user_id: int, target_item_id: int) -> None:
        if not 0 <= user_id < self.user_ego_embeddings.shape[0]:
            raise PairIdOutOfRangeError(f"Pair user ID {user_id} is outside user embedding rows.")
        if not 0 <= target_item_id < self.item_ego_embeddings.shape[0]:
            raise PairIdOutOfRangeError(
                f"Pair target item ID {target_item_id} is outside item embedding rows."
            )

    def score(self, user_id: int, target_item_id: int, removed_history_item_ids: Iterable[int]) -> float:
        return self.score_many(user_id, target_item_id, (removed_history_item_ids,))[0]

    def score_many(
        self,
        user_id: int,
        target_item_id: int,
        removed_history_item_id_groups: Iterable[Iterable[int]],
    ) -> tuple[float, ...]:
        self.validate_pair_ids(user_id, target_item_id)
        removed_edge_groups: list[frozenset[Edge]] = []
        for removed_history_item_ids in removed_history_item_id_groups:
            removed_edges: set[Edge] = set()
            if self.graph.has_edge(user_id, target_item_id):
                removed_edges.add((user_id, target_item_id))
            for item_id in removed_history_item_ids:
                if self.graph.has_edge(user_id, item_id):
                    removed_edges.add((user_id, item_id))
            removed_edge_groups.append(frozenset(removed_edges))
        return self.score_many_with_removed_edges(user_id, target_item_id, removed_edge_groups)

    def score_with_removed_edges(
        self, user_id: int, target_item_id: int, removed_edges: frozenset[Edge]
    ) -> float:
        return self.score_many_with_removed_edges(user_id, target_item_id, (removed_edges,))[0]

    def score_many_with_removed_edges(
        self,
        user_id: int,
        target_item_id: int,
        removed_edge_groups: Sequence[frozenset[Edge]],
    ) -> tuple[float, ...]:
        self.validate_pair_ids(user_id, target_item_id)
        if not removed_edge_groups:
            return ()
        batch_size = len(removed_edge_groups)
        valid_removed_edge_groups = tuple(
            frozenset(
                edge
                for edge in removed_edges
                if self.graph.has_edge(edge[0], edge[1])
            )
            for removed_edges in removed_edge_groups
        )
        degree_cache: dict[Node, Any] = {}
        removed_mask_cache: dict[Edge, Any] = {}
        embedding_cache: dict[tuple[Node, int], Any] = {}

        def perturbed_degree(node: Node) -> Any:
            if node not in degree_cache:
                degree = self._torch.tensor(
                    [
                        self.graph.degree(node)
                        - sum(
                            node == ("u", edge_user_id) or node == ("i", edge_item_id)
                            for edge_user_id, edge_item_id in removed_edges
                        )
                        for removed_edges in valid_removed_edge_groups
                    ],
                    dtype=self.user_ego_embeddings.dtype,
                    device=self.device,
                )
                if bool((degree < 0).any().item()):
                    raise InterventionError(f"Perturbed degree became negative for node {node}.")
                degree_cache[node] = degree
            return degree_cache[node]

        def removed_mask(node: Node, neighbor: Node) -> Any:
            if node[0] == "u":
                edge = (node[1], neighbor[1])
            else:
                edge = (neighbor[1], node[1])
            if edge not in removed_mask_cache:
                removed_mask_cache[edge] = self._torch.tensor(
                    [edge in removed_edges for removed_edges in valid_removed_edge_groups],
                    dtype=self._torch.bool,
                    device=self.device,
                )
            return removed_mask_cache[edge]

        def layer_embedding(node: Node, layer: int) -> Any:
            cache_key = (node, layer)
            if cache_key in embedding_cache:
                return embedding_cache[cache_key]
            if layer == 0:
                embedding = (
                    self.user_ego_embeddings[node[1]]
                    if node[0] == "u"
                    else self.item_ego_embeddings[node[1]]
                )
                embedding = embedding.unsqueeze(0).expand(batch_size, -1)
                embedding_cache[cache_key] = embedding
                return embedding

            node_degree = perturbed_degree(node)
            aggregated = self._torch.zeros(
                (batch_size, self._zero.shape[0]),
                dtype=self._zero.dtype,
                device=self.device,
            )
            for neighbor in self.graph.neighbors(node):
                neighbor_degree = perturbed_degree(neighbor)
                active = (
                    (node_degree > 0)
                    & (neighbor_degree > 0)
                    & ~removed_mask(node, neighbor)
                )
                normalization = self._torch.where(
                    active,
                    self._torch.sqrt(node_degree * neighbor_degree),
                    self._torch.ones_like(node_degree),
                )
                aggregated = aggregated + (
                    layer_embedding(neighbor, layer - 1)
                    / normalization.unsqueeze(1)
                    * active.unsqueeze(1)
                )
            embedding_cache[cache_key] = aggregated
            return aggregated

        user_node = ("u", user_id)
        item_node = ("i", target_item_id)
        user_embedding = self._torch.stack(
            [layer_embedding(user_node, layer) for layer in range(self.layer_count + 1)]
        ).mean(dim=0)
        item_embedding = self._torch.stack(
            [layer_embedding(item_node, layer) for layer in range(self.layer_count + 1)]
        ).mean(dim=0)
        scores = (user_embedding * item_embedding).sum(dim=1)
        return tuple(float(score) for score in scores.detach().cpu().tolist())


class ScoreCache:
    def __init__(self, scorer: HybridLocalLightGCNScorer, max_entries: int) -> None:
        self.scorer = scorer
        self.max_entries = max_entries
        self.values: OrderedDict[tuple[int, int, tuple[int, ...]], float] = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0

    def score(self, user_id: int, target_item_id: int, removed_item_ids: Iterable[int]) -> float:
        return self.score_many(user_id, target_item_id, (removed_item_ids,))[0]

    def score_many(
        self,
        user_id: int,
        target_item_id: int,
        removed_item_id_groups: Iterable[Iterable[int]],
    ) -> tuple[float, ...]:
        keys = tuple(
            (user_id, target_item_id, tuple(sorted(set(removed_item_ids))))
            for removed_item_ids in removed_item_id_groups
        )
        if not keys:
            return ()

        resolved: dict[tuple[int, int, tuple[int, ...]], float] = {}
        missing_keys: list[tuple[int, int, tuple[int, ...]]] = []
        for key in keys:
            if key in self.values:
                self.hit_count += 1
                value = self.values.pop(key)
                self.values[key] = value
                resolved[key] = value
            elif key in resolved:
                self.hit_count += 1
            else:
                self.miss_count += 1
                resolved[key] = 0.0
                missing_keys.append(key)

        if missing_keys:
            score_many = getattr(self.scorer, "score_many", None)
            if score_many is None:
                missing_scores = tuple(
                    self.scorer.score(user_id, target_item_id, key[2])
                    for key in missing_keys
                )
            else:
                missing_scores = score_many(
                    user_id,
                    target_item_id,
                    (key[2] for key in missing_keys),
                )
            for key, value in zip(missing_keys, missing_scores, strict=True):
                resolved[key] = value
                self.values[key] = value
                if len(self.values) > self.max_entries:
                    self.values.popitem(last=False)
        return tuple(resolved[key] for key in keys)

    def validate_pair_ids(self, user_id: int, target_item_id: int) -> None:
        self.scorer.validate_pair_ids(user_id, target_item_id)


def _pair_rng(seed: int, pair_index: int, user_id: int, target_item_id: int) -> random.Random:
    raw_seed = f"{seed}:{pair_index}:{user_id}:{target_item_id}".encode("utf-8")
    digest = hashlib.sha256(raw_seed).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _mask_sequence(attribute_count: int, rng: random.Random) -> Iterable[int]:
    total_masks = (1 << attribute_count) - 1
    if total_masks <= 0:
        return
    start = rng.randrange(total_masks)
    step = rng.randrange(1, total_masks + 1)
    while math.gcd(step, total_masks) != 1:
        step = rng.randrange(1, total_masks + 1)
    for offset in range(total_masks):
        yield 1 + ((start + offset * step) % total_masks)


def sample_interventions(
    supported_items_by_attribute: Mapping[str, Sequence[int]],
    eligible_history_item_ids: Sequence[int],
    vocabulary: Vocabulary,
    requested_count: int,
    max_history_drop_fraction: float,
    subset_probability: float,
    random_seed: int,
    pair_index: int,
    user_id: int,
    target_item_id: int,
    exhaustive_attribute_limit: int,
    max_sampling_attempts: int,
) -> tuple[SampledIntervention, ...]:
    eligible_history = set(eligible_history_item_ids)
    supported_items: dict[str, frozenset[int]] = {}
    for attribute, item_ids in supported_items_by_attribute.items():
        if attribute not in vocabulary.attribute_to_index:
            raise InterventionError(f"Support attribute {attribute!r} is missing from vocabulary.")
        filtered_items = frozenset(item_id for item_id in item_ids if item_id in eligible_history)
        if filtered_items:
            supported_items[attribute] = filtered_items

    attributes = sorted(supported_items, key=vocabulary.attribute_to_index.__getitem__)
    if not attributes or not eligible_history:
        return ()
    max_removed_item_count = math.floor(max_history_drop_fraction * len(eligible_history))
    if max_removed_item_count < 1:
        return ()

    rng = _pair_rng(random_seed, pair_index, user_id, target_item_id)
    accepted: list[SampledIntervention] = []
    seen_masks: set[int] = set()

    def consider(mask: int, *, allow_over_target: bool = False) -> None:
        if (
            mask == 0
            or mask in seen_masks
            or (not allow_over_target and len(accepted) >= requested_count)
        ):
            return
        seen_masks.add(mask)
        selected_attributes = tuple(
            attribute for offset, attribute in enumerate(attributes) if mask & (1 << offset)
        )
        removed_item_ids = tuple(
            sorted(
                {
                    item_id
                    for attribute in selected_attributes
                    for item_id in supported_items[attribute]
                }
            )
        )
        if not removed_item_ids or len(removed_item_ids) > max_removed_item_count:
            return
        accepted.append(
            SampledIntervention(
                attributes=selected_attributes,
                attribute_indices=tuple(
                    vocabulary.attribute_to_index[attribute] for attribute in selected_attributes
                ),
                removed_item_ids=removed_item_ids,
            )
        )

    for offset in range(len(attributes)):
        consider(1 << offset, allow_over_target=True)
    if len(accepted) >= requested_count:
        return tuple(accepted)

    def consider_subset(mask: int) -> None:
        if mask.bit_count() >= 2:
            consider(mask)

    if len(attributes) <= exhaustive_attribute_limit:
        for mask in _mask_sequence(len(attributes), rng):
            consider_subset(mask)
            if len(accepted) >= requested_count:
                break
    else:
        for _ in range(max_sampling_attempts):
            mask = 0
            for offset in range(len(attributes)):
                if rng.random() < subset_probability:
                    mask |= 1 << offset
            consider_subset(mask)
            if len(accepted) >= requested_count:
                break
    return tuple(accepted)


def _validated_mapping_index(
    payload: Mapping[str, Any],
    index_key: str,
    raw_id: int,
    expected_index: int | None,
    context: str,
) -> int | None:
    if index_key not in payload:
        raise InterventionError(f"{context} is missing {index_key}.")
    raw_index = payload.get(index_key)
    if raw_index is None:
        if expected_index is not None:
            raise InterventionError(f"{context} is missing {index_key}.")
        return None
    index = _coerce_int(raw_index, f"{context} {index_key}")
    if expected_index != index:
        raise InterventionError(
            f"{context} has inconsistent raw ID {raw_id} and {index_key} {index}."
        )
    return index


def _support_item_indices(
    raw_evidence: Any, context: str, id_mappings: IdMappings
) -> tuple[int, ...]:
    if not isinstance(raw_evidence, Sequence) or isinstance(raw_evidence, (str, bytes, bytearray)):
        raise InterventionError(f"{context} must be a list.")
    items: list[int] = []
    seen: set[int] = set()
    for evidence in raw_evidence:
        if not isinstance(evidence, Mapping):
            raise InterventionError(
                f"{context} contains stale evidence without raw and internal item IDs. "
                "Regenerate support artifacts from the beginning."
            )
        raw_item_id = _coerce_int(evidence.get("item_id"), f"{context} item ID")
        item_index = _validated_mapping_index(
            evidence,
            "item_index",
            raw_item_id,
            id_mappings.item_index(raw_item_id),
            f"{context} item {raw_item_id}",
        )
        if item_index is None:
            raise InterventionError(
                f"{context} item ID {raw_item_id} is absent from LightGCN mappings."
            )
        if item_index not in seen:
            seen.add(item_index)
            items.append(item_index)
    return tuple(items)


def parse_support_record(payload: Any, line_number: int, id_mappings: IdMappings) -> SupportRecord:
    if not isinstance(payload, Mapping):
        raise InterventionError(f"Support JSONL line {line_number} must be an object.")
    if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise InterventionError(
            f"Support JSONL line {line_number} is stale or unsupported. "
            "Regenerate schema-version-2 support artifacts from the beginning."
        )
    pair_index = _coerce_int(payload.get("pair_index", line_number - 1), "Support pair index")
    user_id = _coerce_int(payload.get("user_id"), f"Support line {line_number} user ID")
    user_index = _validated_mapping_index(
        payload,
        "user_index",
        user_id,
        id_mappings.user_index(user_id),
        f"Support line {line_number} user",
    )
    target_item_id = _coerce_int(
        payload.get("target_item_id"), f"Support line {line_number} target item ID"
    )
    target_item_index = _validated_mapping_index(
        payload,
        "target_item_index",
        target_item_id,
        id_mappings.item_index(target_item_id),
        f"Support line {line_number} target item",
    )
    raw_supports = payload.get("supported_items_by_attribute")
    if not isinstance(raw_supports, Mapping):
        raise InterventionError(
            f"Support JSONL line {line_number} supported_items_by_attribute must be an object."
        )
    supports: dict[str, tuple[int, ...]] = {}
    for attribute, evidence in raw_supports.items():
        if not isinstance(attribute, str) or not attribute:
            raise InterventionError(f"Support JSONL line {line_number} has an invalid attribute.")
        supports[attribute] = _support_item_indices(
            evidence, f"Support JSONL line {line_number} attribute {attribute!r}", id_mappings
        )
    return SupportRecord(
        pair_index=pair_index,
        user_id=user_id,
        user_index=user_index,
        target_item_id=target_item_id,
        target_item_index=target_item_index,
        supports=supports,
    )


def iter_support_records(path: Path, id_mappings: IdMappings) -> Iterable[SupportRecord]:
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if line.strip():
                yield parse_support_record(json.loads(line), line_number, id_mappings)


def count_support_records(path: Path) -> int:
    with path.open("r", encoding="utf-8") as input_file:
        return sum(1 for line in input_file if line.strip())


def _load_tqdm() -> Any:
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError(
            "tqdm is required. Install extract_causal_attributes/intervention/requirements.txt."
        ) from exc
    return tqdm


def build_pair_result(
    pair_index: int,
    user_id: int,
    user_index: int | None,
    target_item_id: int,
    target_item_index: int | None,
    supports: Mapping[str, Sequence[int]],
    graph: BipartiteGraph,
    vocabulary: Vocabulary,
    score_cache: ScoreCache,
    config: InterventionConfig,
    stats: RunStats,
) -> PairResult:
    if user_index is None or target_item_index is None:
        stats.processed_pair_count += 1
        stats.shortfall_pair_count += 1
        stats.zero_row_pair_count += 1
        stats.skipped_pair_count += 1
        stats.invalid_pair_id_skip_count += 1
        missing = "user" if user_index is None else "target item"
        return PairResult(
            pair_index=pair_index,
            user_id=user_id,
            user_index=user_index,
            target_item_id=target_item_id,
            target_item_index=target_item_index,
            baseline_score=None,
            eligible_history_count=0,
            supported_target_attribute_count=0,
            requested_intervention_count=config.requested_intervention_count,
            rows=(),
            skip_reason=f"Pair raw {missing} ID is absent from LightGCN mappings.",
        )

    eligible_history = tuple(
        item_id for item_id in graph.user_items.get(user_index, ()) if item_id != target_item_index
    )
    eligible_history_set = set(eligible_history)
    supported_attribute_count = sum(
        1 for values in supports.values() if any(item_id in eligible_history_set for item_id in values)
    )
    try:
        score_cache.validate_pair_ids(user_index, target_item_index)
    except PairIdOutOfRangeError as exc:
        stats.processed_pair_count += 1
        stats.shortfall_pair_count += 1
        stats.zero_row_pair_count += 1
        stats.skipped_pair_count += 1
        stats.invalid_pair_id_skip_count += 1
        return PairResult(
            pair_index=pair_index,
            user_id=user_id,
            user_index=user_index,
            target_item_id=target_item_id,
            target_item_index=target_item_index,
            baseline_score=None,
            eligible_history_count=len(eligible_history),
            supported_target_attribute_count=supported_attribute_count,
            requested_intervention_count=config.requested_intervention_count,
            rows=(),
            skip_reason=str(exc),
        )

    samples = sample_interventions(
        supported_items_by_attribute=supports,
        eligible_history_item_ids=eligible_history,
        vocabulary=vocabulary,
        requested_count=config.requested_intervention_count,
        max_history_drop_fraction=config.max_history_drop_fraction,
        subset_probability=config.subset_probability,
        random_seed=config.random_seed,
        pair_index=pair_index,
        user_id=user_index,
        target_item_id=target_item_index,
        exhaustive_attribute_limit=config.exhaustive_attribute_limit,
        max_sampling_attempts=config.max_sampling_attempts,
    )

    baseline_started = time.perf_counter()
    baseline_score = score_cache.score(user_index, target_item_index, ())
    stats.baseline_scoring_seconds += time.perf_counter() - baseline_started

    rows: list[InterventionRow] = []
    intervention_started = time.perf_counter()
    intervention_scores = score_cache.score_many(
        user_index,
        target_item_index,
        (sample.removed_item_ids for sample in samples),
    )
    for sample, y_h in zip(samples, intervention_scores, strict=True):
        rows.append(
            InterventionRow(
                attribute_indices=sample.attribute_indices,
                removed_item_ids=sample.removed_item_ids,
                y_h=y_h,
                y_delta=baseline_score - y_h,
            )
        )
    stats.intervention_scoring_seconds += time.perf_counter() - intervention_started

    stats.processed_pair_count += 1
    stats.generated_intervention_count += len(rows)
    if len(rows) < config.requested_intervention_count:
        stats.shortfall_pair_count += 1
    if not rows:
        stats.zero_row_pair_count += 1

    return PairResult(
        pair_index=pair_index,
        user_id=user_id,
        user_index=user_index,
        target_item_id=target_item_id,
        target_item_index=target_item_index,
        baseline_score=baseline_score,
        eligible_history_count=len(eligible_history),
        supported_target_attribute_count=supported_attribute_count,
        requested_intervention_count=config.requested_intervention_count,
        rows=tuple(rows),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            json.dump(payload, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _write_jsonl_atomic(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            for record in records:
                json.dump(record, output_file, ensure_ascii=False)
                output_file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _write_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".npz",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_shard(
    shard_path: Path,
    shard_relative_path: str,
    pair_results: Sequence[PairResult],
    vocabulary_size: int,
) -> list[dict[str, Any]]:
    a_indices: list[int] = []
    a_indptr = [0]
    y_delta: list[float] = []
    y_h: list[float] = []
    pair_indices: list[int] = []
    intervention_indices: list[int] = []
    removed_item_ids: list[int] = []
    removed_item_indptr = [0]
    removed_item_count: list[int] = []
    manifests: list[dict[str, Any]] = []

    for pair_result in pair_results:
        row_start = len(y_delta)
        for intervention_index, row in enumerate(pair_result.rows):
            a_indices.extend(row.attribute_indices)
            a_indptr.append(len(a_indices))
            y_delta.append(row.y_delta)
            y_h.append(row.y_h)
            pair_indices.append(pair_result.pair_index)
            intervention_indices.append(intervention_index)
            removed_item_ids.extend(row.removed_item_ids)
            removed_item_indptr.append(len(removed_item_ids))
            removed_item_count.append(len(row.removed_item_ids))
        row_end = len(y_delta)
        manifests.append(
            {
                "pair_index": pair_result.pair_index,
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "user_id": pair_result.user_id,
                "user_index": pair_result.user_index,
                "target_item_id": pair_result.target_item_id,
                "target_item_index": pair_result.target_item_index,
                "baseline_score": pair_result.baseline_score,
                "eligible_history_count": pair_result.eligible_history_count,
                "supported_target_attribute_count": pair_result.supported_target_attribute_count,
                "requested_intervention_count": pair_result.requested_intervention_count,
                "generated_intervention_count": len(pair_result.rows),
                "shard": shard_relative_path,
                "row_start": row_start,
                "row_end": row_end,
                "skip_reason": pair_result.skip_reason,
            }
        )

    _write_npz_atomic(
        shard_path,
        A_data=np.ones(len(a_indices), dtype=np.int8),
        A_indices=np.asarray(a_indices, dtype=np.int64),
        A_indptr=np.asarray(a_indptr, dtype=np.int64),
        A_shape=np.asarray([len(y_delta), vocabulary_size], dtype=np.int64),
        y_delta=np.asarray(y_delta, dtype=np.float32),
        y_h=np.asarray(y_h, dtype=np.float32),
        pair_index=np.asarray(pair_indices, dtype=np.int64),
        intervention_index=np.asarray(intervention_indices, dtype=np.int32),
        removed_item_ids=np.asarray(removed_item_ids, dtype=np.int64),
        removed_item_indptr=np.asarray(removed_item_indptr, dtype=np.int64),
        removed_item_count=np.asarray(removed_item_count, dtype=np.int32),
    )
    return manifests


def _config_json(config: InterventionConfig) -> dict[str, Any]:
    payload = asdict(config)
    return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}


def _source_checksums(config: InterventionConfig) -> dict[str, str]:
    return {
        "attribute_support": file_sha256(config.attribute_support_path),
        "vocabulary": file_sha256(config.vocabulary_path),
        "user_history": file_sha256(config.user_history_path),
        "id_mappings": file_sha256(config.id_mappings_path),
        "user_ego_embeddings": file_sha256(config.user_ego_embeddings_path),
        "item_ego_embeddings": file_sha256(config.item_ego_embeddings_path),
        "lightgcn_config": file_sha256(config.lightgcn_config_path),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def _safe_remove_output_dir(path: Path) -> None:
    resolved_path = path.resolve()
    if resolved_path == Path(resolved_path.anchor) or len(resolved_path.parts) < 3:
        raise InterventionError(f"Refusing to remove unsafe output directory: {resolved_path}")
    shutil.rmtree(resolved_path)


def _resume_configs_match(previous: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    previous_payload = json.loads(json.dumps(previous))
    expected_payload = json.loads(json.dumps(expected))
    for payload in (previous_payload, expected_payload):
        config = payload.get("config")
        if isinstance(config, dict):
            # Shard size changes checkpoint granularity only, not experiment semantics.
            config.pop("pairs_per_shard", None)
    return previous_payload == expected_payload


def _prepare_output(
    config: InterventionConfig,
    vocabulary: Vocabulary,
    source_checksums: Mapping[str, str],
    resume: bool,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], RunStats, int]:
    if resume and overwrite:
        raise InterventionError("--resume and --overwrite cannot be used together.")
    output_dir = config.output_dir
    manifest_path = output_dir / "manifest.jsonl"
    run_config_path = output_dir / "run_config.json"

    if output_dir.exists() and overwrite:
        _safe_remove_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "shards").mkdir(parents=True, exist_ok=True)

    expected_run_config = {
        "intervention_generation_version": INTERVENTION_GENERATION_VERSION,
        "config": _config_json(config),
        "source_checksums": dict(source_checksums),
    }
    if resume:
        if not run_config_path.exists():
            raise InterventionError("Cannot resume because run_config.json is missing.")
        with run_config_path.open("r", encoding="utf-8") as input_file:
            previous_run_config = json.load(input_file)
        if not _resume_configs_match(previous_run_config, expected_run_config):
            raise InterventionError("Cannot resume because inputs or experiment settings changed.")
        manifests = _load_jsonl(manifest_path)
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as input_file:
                summary = json.load(input_file)
            stats = RunStats(
                **{
                    key: (
                        summary.get("score_cache_hit_count", 0)
                        if key == "previous_score_cache_hit_count"
                        else summary.get("score_cache_miss_count", 0)
                        if key == "previous_score_cache_miss_count"
                        else summary.get("full_time_seconds", 0.0)
                        if key == "previous_full_time_seconds"
                        else summary.get(key, getattr(RunStats(), key))
                    )
                    for key in asdict(RunStats())
                }
            )
        else:
            stats = RunStats()
        return manifests, stats, len({record["shard"] for record in manifests})

    existing_artifacts = [
        path
        for path in (manifest_path, run_config_path, output_dir / "summary.json")
        if path.exists()
    ]
    if existing_artifacts:
        raise InterventionError(
            f"Output artifacts already exist under {output_dir}. Use --resume or --overwrite."
        )
    _write_json_atomic(run_config_path, expected_run_config)
    _write_json_atomic(
        output_dir / "vocabulary.json",
        {
            "attributes": list(vocabulary.attributes),
            "attribute_to_index": vocabulary.attribute_to_index,
            "source_checksum": source_checksums["vocabulary"],
        },
    )
    return [], RunStats(), 0


def _summary_payload(
    config: InterventionConfig,
    stats: RunStats,
    score_cache: ScoreCache,
    layer_count: int,
    device: str,
    source_checksums: Mapping[str, str],
    run_started: float,
) -> dict[str, Any]:
    public_stats = {
        key: value for key, value in asdict(stats).items() if not key.startswith("previous_")
    }
    full_time_seconds = stats.previous_full_time_seconds + (time.perf_counter() - run_started)
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        **public_stats,
        "full_time_seconds": full_time_seconds,
        "average_time_per_pair_seconds": (
            full_time_seconds / stats.processed_pair_count if stats.processed_pair_count else 0.0
        ),
        "requested_interventions_per_pair": config.requested_intervention_count,
        "lightgcn_layer_count": layer_count,
        "device": device,
        "score_cache_hit_count": stats.previous_score_cache_hit_count + score_cache.hit_count,
        "score_cache_miss_count": stats.previous_score_cache_miss_count + score_cache.miss_count,
        "source_checksums": dict(source_checksums),
    }


def generate_intervention_artifacts(
    config: InterventionConfig,
    limit: int | None = None,
    resume: bool = False,
    overwrite: bool = False,
    scorer_factory: Callable[[BipartiteGraph, Any, Any, int, str], HybridLocalLightGCNScorer]
    | None = None,
) -> dict[str, Any]:
    run_started = time.perf_counter()
    if limit is not None and limit < 0:
        raise InterventionError("--limit must be greater than or equal to zero.")

    vocabulary = load_vocabulary(config.vocabulary_path)
    id_mappings = load_id_mappings(config.id_mappings_path)
    user_history = load_user_history(config.user_history_path)
    graph = build_graph(user_history)
    user_embeddings = load_embedding_tensor(config.user_ego_embeddings_path)
    item_embeddings = load_embedding_tensor(config.item_ego_embeddings_path)
    layer_count = load_lightgcn_layer_count(config.lightgcn_config_path)
    device = resolve_device(config.device)
    factory = scorer_factory or HybridLocalLightGCNScorer
    scorer = factory(graph, user_embeddings, item_embeddings, layer_count, device)
    score_cache = ScoreCache(scorer, config.score_cache_max_entries)
    source_checksums = _source_checksums(config)
    source_record_count = count_support_records(config.attribute_support_path)
    target_source_rows = min(source_record_count, limit) if limit is not None else source_record_count
    tqdm = _load_tqdm()

    manifests, stats, shard_index = _prepare_output(
        config, vocabulary, source_checksums, resume, overwrite
    )
    processed_source_rows = len(manifests)
    chunk: list[PairResult] = []

    def flush_chunk() -> None:
        nonlocal manifests, shard_index, chunk
        if not chunk:
            return
        relative_path = f"shards/interventions_{shard_index:06d}.npz"
        shard_path = config.output_dir / relative_path
        new_manifests = write_shard(shard_path, relative_path, chunk, len(vocabulary.attributes))
        manifests.extend(new_manifests)
        _write_jsonl_atomic(config.output_dir / "manifest.jsonl", manifests)
        stats.written_shard_count += 1
        summary = _summary_payload(
            config, stats, score_cache, layer_count, device, source_checksums, run_started
        )
        _write_json_atomic(config.output_dir / "summary.json", summary)
        shard_index += 1
        chunk = []

    with tqdm(
        total=target_source_rows,
        initial=min(processed_source_rows, target_source_rows),
        desc="Building interventions",
        unit="pair",
        dynamic_ncols=True,
    ) as progress:
        for source_index, record in enumerate(
            iter_support_records(config.attribute_support_path, id_mappings)
        ):
            if source_index < processed_source_rows:
                continue
            if source_index >= target_source_rows:
                break
            chunk.append(
                build_pair_result(
                    pair_index=record.pair_index,
                    user_id=record.user_id,
                    user_index=record.user_index,
                    target_item_id=record.target_item_id,
                    target_item_index=record.target_item_index,
                    supports=record.supports,
                    graph=graph,
                    vocabulary=vocabulary,
                    score_cache=score_cache,
                    config=config,
                    stats=stats,
                )
            )
            progress.update(1)
            if len(chunk) >= config.pairs_per_shard:
                flush_chunk()
    flush_chunk()

    summary = _summary_payload(
        config, stats, score_cache, layer_count, device, source_checksums, run_started
    )
    _write_json_atomic(config.output_dir / "summary.json", summary)
    return summary
