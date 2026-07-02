"""Validated artifact loading for standalone causal joint training."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import JointTrainingConfig


ARTIFACT_SCHEMA_VERSION = 2


class ArtifactError(ValueError):
    """Raised when an input artifact is stale, malformed, or misaligned."""


def _raw_id(value: Any) -> str:
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            value = item_method()
        except ValueError:
            pass
    return str(value)


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise ArtifactError(f"{context} must be an integer, got {value!r}.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ArtifactError(f"{context} must be integer-compatible, got {value!r}.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ArtifactError(f"{context} must be an integer, got {value!r}.")
    return result


def _sequence(value: Any, context: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ArtifactError(f"{context} must be a list.")
    return value


@dataclass(frozen=True)
class IdMappings:
    user_to_index: dict[str, int]
    item_to_index: dict[str, int]
    index_to_user: tuple[str, ...]
    index_to_item: tuple[str, ...]

    @property
    def num_users(self) -> int:
        return len(self.index_to_user)

    @property
    def num_items(self) -> int:
        return len(self.index_to_item)

    def user_index(self, raw_id: Any) -> int | None:
        return self.user_to_index.get(_raw_id(raw_id))

    def item_index(self, raw_id: Any) -> int | None:
        return self.item_to_index.get(_raw_id(raw_id))


def _load_inverse_mapping(payload: Any, context: str) -> tuple[str, ...]:
    values = tuple(_raw_id(value) for value in _sequence(payload, context))
    if any(not value for value in values):
        raise ArtifactError(f"{context} contains a blank ID.")
    if len(values) != len(set(values)):
        raise ArtifactError(f"{context} contains duplicate IDs.")
    return values


def _load_forward_mapping(payload: Any, inverse: tuple[str, ...], context: str) -> dict[str, int]:
    if not isinstance(payload, Mapping):
        raise ArtifactError(f"{context} must be an object.")
    mapping = {_raw_id(raw_id): _integer(index, f"{context}[{raw_id!r}]") for raw_id, index in payload.items()}
    expected = {raw_id: index for index, raw_id in enumerate(inverse)}
    if mapping != expected:
        raise ArtifactError(f"{context} disagrees with its inverse ordered ID list.")
    return mapping


def load_id_mappings(path: Path) -> IdMappings:
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, Mapping):
        raise ArtifactError("ID mappings root must be an object.")
    index_to_user = _load_inverse_mapping(payload.get("index_to_user"), "index_to_user")
    index_to_item = _load_inverse_mapping(payload.get("index_to_item"), "index_to_item")
    return IdMappings(
        user_to_index=_load_forward_mapping(payload.get("user_to_index"), index_to_user, "user_to_index"),
        item_to_index=_load_forward_mapping(payload.get("item_to_index"), index_to_item, "item_to_index"),
        index_to_user=index_to_user,
        index_to_item=index_to_item,
    )


def load_frozen_embeddings(
    user_path: Path, item_path: Path, mappings: IdMappings, embedding_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    users = torch.load(user_path, map_location="cpu")
    items = torch.load(item_path, map_location="cpu")
    for value, label, rows in (
        (users, "User embeddings", mappings.num_users),
        (items, "Item embeddings", mappings.num_items),
    ):
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise ArtifactError(f"{label} must be a rank-2 torch tensor.")
        if tuple(value.shape) != (rows, embedding_dim):
            raise ArtifactError(
                f"{label} shape {tuple(value.shape)} does not match expected {(rows, embedding_dim)}."
            )
        if not torch.isfinite(value).all():
            raise ArtifactError(f"{label} contain non-finite values.")
    return users.detach().float(), items.detach().float()


def load_vocabulary(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if isinstance(payload, Mapping) and "attributes" in payload:
        payload = payload["attributes"]
    if isinstance(payload, Mapping):
        indexed: dict[int, str] = {}
        for raw_index, attribute in payload.items():
            index = _integer(raw_index, "Vocabulary index")
            if not isinstance(attribute, str) or not attribute:
                raise ArtifactError("Vocabulary attributes must be non-empty strings.")
            indexed[index] = attribute
        if set(indexed) != set(range(len(indexed))):
            raise ArtifactError("Vocabulary indices must be contiguous from zero.")
        attributes = tuple(indexed[index] for index in range(len(indexed)))
    else:
        attributes = tuple(_sequence(payload, "Vocabulary"))
        if not all(isinstance(attribute, str) and attribute for attribute in attributes):
            raise ArtifactError("Vocabulary attributes must be non-empty strings.")
    if len(attributes) != len(set(attributes)):
        raise ArtifactError("Vocabulary contains duplicate attributes.")
    return attributes


def load_item_attributes(
    path: Path, mappings: IdMappings, vocabulary_size: int
) -> dict[int, tuple[int, ...]]:
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, Mapping):
        raise ArtifactError("item_attribute_ids.json must be an object.")
    result: dict[int, tuple[int, ...]] = {}
    for raw_item_id, raw_attributes in payload.items():
        item_index = mappings.item_index(raw_item_id)
        if item_index is None:
            continue
        attributes = tuple(dict.fromkeys(_integer(value, f"Attributes for item {raw_item_id}") for value in _sequence(raw_attributes, f"Attributes for item {raw_item_id}")))
        if any(index < 0 or index >= vocabulary_size for index in attributes):
            raise ArtifactError(f"Attributes for item {raw_item_id} contain an out-of-range index.")
        if attributes:
            result[item_index] = attributes
    return result


def load_semantic_embeddings(path: Path, vocabulary: Sequence[str]) -> torch.Tensor:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) < {"phrases", "embeddings"}:
            raise ArtifactError("attribute_embeddings.npz must contain phrases and embeddings.")
        phrases = archive["phrases"]
        embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
    if phrases.ndim != 1 or embeddings.ndim != 2 or embeddings.shape[0] != phrases.shape[0]:
        raise ArtifactError("Semantic attribute embeddings have incompatible shapes.")
    by_phrase: dict[str, int] = {}
    for index, phrase in enumerate(phrases.tolist()):
        text = str(phrase)
        if text in by_phrase:
            raise ArtifactError(f"Semantic embeddings contain duplicate phrase {text!r}.")
        by_phrase[text] = index
    missing = [attribute for attribute in vocabulary if attribute not in by_phrase]
    if missing:
        raise ArtifactError(f"Semantic embeddings are missing canonical attribute {missing[0]!r}.")
    canonical = embeddings[[by_phrase[attribute] for attribute in vocabulary]]
    norms = np.linalg.norm(canonical, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.all(np.isfinite(norms)):
        raise ArtifactError("Canonical semantic embeddings contain invalid norms.")
    return torch.from_numpy(canonical / norms)


@dataclass(frozen=True)
class RecommendationData:
    train_pairs: torch.Tensor
    train_history: dict[int, set[int]]
    validation_pairs: dict[int, set[int]]
    test_pairs: dict[int, set[int]]
    stats: dict[str, int]


def _read_csv_pairs(path: Path) -> list[tuple[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if not reader.fieldnames or "user" not in reader.fieldnames or "item" not in reader.fieldnames:
            raise ArtifactError(f"{path} must contain user and item columns.")
        return [(str(row["user"]).strip(), str(row["item"]).strip()) for row in reader]


def _map_eval_pairs(
    pairs: Sequence[tuple[str, str]], mappings: IdMappings, history: Mapping[int, set[int]], prefix: str
) -> tuple[dict[int, set[int]], dict[str, int]]:
    mapped: dict[int, set[int]] = {}
    stats = {f"{prefix}_skipped_unknown_user": 0, f"{prefix}_skipped_unknown_item": 0, f"{prefix}_skipped_seen": 0}
    for raw_user, raw_item in pairs:
        user = mappings.user_index(raw_user)
        item = mappings.item_index(raw_item)
        if user is None:
            stats[f"{prefix}_skipped_unknown_user"] += 1
        elif item is None:
            stats[f"{prefix}_skipped_unknown_item"] += 1
        elif item in history[user]:
            stats[f"{prefix}_skipped_seen"] += 1
        else:
            mapped.setdefault(user, set()).add(item)
    return mapped, stats


def load_recommendation_data(
    train_path: Path, validation_path: Path, test_path: Path, mappings: IdMappings
) -> RecommendationData:
    raw_train = _read_csv_pairs(train_path)
    mapped_train: set[tuple[int, int]] = set()
    history = {user: set() for user in range(mappings.num_users)}
    for raw_user, raw_item in raw_train:
        user = mappings.user_index(raw_user)
        item = mappings.item_index(raw_item)
        if user is None or item is None:
            raise ArtifactError("Recommendation training CSV contains an ID absent from LightGCN mappings.")
        mapped_train.add((user, item))
        history[user].add(item)
    ordered = sorted(mapped_train)
    train_pairs = torch.tensor(ordered, dtype=torch.long) if ordered else torch.empty((0, 2), dtype=torch.long)
    validation, validation_stats = _map_eval_pairs(_read_csv_pairs(validation_path), mappings, history, "validation")
    test, test_stats = _map_eval_pairs(_read_csv_pairs(test_path), mappings, history, "test")
    return RecommendationData(
        train_pairs=train_pairs,
        train_history=history,
        validation_pairs=validation,
        test_pairs=test,
        stats={
            "recommendation_train_pairs": len(ordered),
            "recommendation_validation_users": len(validation),
            "recommendation_test_users": len(test),
            **validation_stats,
            **test_stats,
        },
    )


def load_profiles(path: Path, id_key: str) -> dict[str, str]:
    profiles: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping) or id_key not in row:
                raise ArtifactError(f"{path} line {line_number} must contain {id_key}.")
            raw_id = _raw_id(row[id_key])
            completion = row.get("completion")
            if not isinstance(completion, str):
                raise ArtifactError(f"{path} line {line_number} completion must be JSON text.")
            try:
                parsed = json.loads(completion, strict=False)
            except json.JSONDecodeError as exc:
                raise ArtifactError(f"{path} line {line_number} completion is invalid JSON.") from exc
            summary = parsed.get("summarization") if isinstance(parsed, Mapping) else None
            if not isinstance(summary, str) or not summary.strip():
                raise ArtifactError(f"{path} line {line_number} has no summarization.")
            if raw_id in profiles:
                raise ArtifactError(f"{path} contains duplicate ID {raw_id}.")
            profiles[raw_id] = summary.strip()
    return profiles


@dataclass(frozen=True)
class ExplanationExample:
    pair_index: int
    user_id: str
    user_index: int
    item_id: str
    item_index: int
    title: str
    user_profile: str
    item_profile: str
    explanation: str


def load_explanations(
    path: Path, mappings: IdMappings, user_profiles: Mapping[str, str], item_profiles: Mapping[str, str]
) -> tuple[list[ExplanationExample], dict[str, int]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to load explanation pickle files.") from exc
    frame = pd.read_pickle(path)
    required = {"uid", "iid", "title", "explanation"}
    if not required.issubset(frame.columns):
        raise ArtifactError(f"{path} must contain columns {sorted(required)}.")
    examples: list[ExplanationExample] = []
    stats = {"rows": len(frame), "skipped_unknown_user": 0, "skipped_unknown_item": 0}
    for pair_index, row in enumerate(frame.itertuples(index=False)):
        values = row._asdict()
        raw_user, raw_item = _raw_id(values["uid"]), _raw_id(values["iid"])
        user_index, item_index = mappings.user_index(raw_user), mappings.item_index(raw_item)
        if user_index is None:
            stats["skipped_unknown_user"] += 1
            continue
        if item_index is None:
            stats["skipped_unknown_item"] += 1
            continue
        if raw_user not in user_profiles or raw_item not in item_profiles:
            raise ArtifactError(f"{path} pair {(raw_user, raw_item)} is missing profile text.")
        examples.append(
            ExplanationExample(
                pair_index=pair_index,
                user_id=raw_user,
                user_index=user_index,
                item_id=raw_item,
                item_index=item_index,
                title=str(values["title"]),
                user_profile=user_profiles[raw_user],
                item_profile=item_profiles[raw_item],
                explanation=str(values["explanation"]),
            )
        )
    stats["usable_rows"] = len(examples)
    return examples, stats


@dataclass(frozen=True)
class CausalLabel:
    pair_index: int
    user_id: str
    user_index: int
    item_id: str
    item_index: int
    attribute_indices: tuple[int, ...]
    coefficients: tuple[float, ...]
    relative_residual: float


@dataclass
class OmpStats:
    manifest_rows: int = 0
    recovered_rows: int = 0
    accepted_rows: int = 0
    skipped_upstream_rows: int = 0
    skipped_quality_rows: int = 0
    skipped_empty_rows: int = 0


class _OmpShardCache:
    def __init__(self, root: Path, vocabulary_size: int) -> None:
        self.root = root.resolve()
        self.vocabulary_size = vocabulary_size
        self.values: dict[str, dict[str, np.ndarray]] = {}

    def get(self, relative_path: str) -> dict[str, np.ndarray]:
        if relative_path in self.values:
            return self.values[relative_path]
        path = (self.root / relative_path).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ArtifactError(f"OMP shard escapes its artifact directory: {relative_path!r}.") from exc
        with np.load(path, allow_pickle=False) as archive:
            values = {key: archive[key] for key in archive.files}
        required = {
            "coef_data", "coef_indices", "coef_indptr", "coef_shape", "pair_index",
            "user_index", "target_item_index",
        }
        if not required.issubset(values):
            missing = sorted(required - set(values))
            raise ArtifactError(
                f"OMP shard {relative_path} is stale or malformed; missing arrays {missing}. "
                "Regenerate schema-version-2 extraction artifacts."
            )
        if values["coef_shape"].shape != (2,):
            raise ArtifactError(f"OMP shard {relative_path} coef_shape must contain two values.")
        shape = values["coef_shape"].tolist()
        if shape[1] != self.vocabulary_size:
            raise ArtifactError(f"OMP shard {relative_path} vocabulary width does not match.")
        rows = int(shape[0])
        if (
            values["coef_indptr"].shape != (rows + 1,)
            or values["pair_index"].shape != (rows,)
            or values["user_index"].shape != (rows,)
            or values["target_item_index"].shape != (rows,)
        ):
            raise ArtifactError(f"OMP shard {relative_path} row arrays are inconsistent.")
        if (
            values["coef_data"].ndim != 1
            or values["coef_indices"].shape != values["coef_data"].shape
            or int(values["coef_indptr"][0]) != 0
            or int(values["coef_indptr"][-1]) != values["coef_data"].shape[0]
            or np.any(np.diff(values["coef_indptr"]) < 0)
        ):
            raise ArtifactError(f"OMP shard {relative_path} sparse coefficient arrays are inconsistent.")
        self.values[relative_path] = values
        return values


def _validate_manifest_index(raw_id: Any, index: Any, expected: int | None, context: str) -> int | None:
    if index is None:
        if expected is not None:
            raise ArtifactError(f"{context} is missing for mapped raw ID {raw_id!r}.")
        return None
    parsed = _integer(index, context)
    if expected != parsed:
        raise ArtifactError(f"{context} does not match raw ID {raw_id!r}.")
    return parsed


def load_omp_labels(
    omp_dir: Path, mappings: IdMappings, vocabulary: Sequence[str], max_relative_residual: float
) -> tuple[dict[int, CausalLabel], dict[str, int]]:
    omp_vocabulary_path = omp_dir / "vocabulary.json"
    if omp_vocabulary_path.exists():
        omp_vocabulary = load_vocabulary(omp_vocabulary_path)
        if tuple(vocabulary) != omp_vocabulary:
            raise ArtifactError("OMP vocabulary does not match canonical attribute vocabulary.")
    cache = _OmpShardCache(omp_dir, len(vocabulary))
    labels: dict[int, CausalLabel] = {}
    stats = OmpStats()
    manifest_path = omp_dir / "manifest.jsonl"
    with manifest_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            stats.manifest_rows += 1
            row = json.loads(line)
            if not isinstance(row, Mapping) or row.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
                raise ArtifactError(
                    f"OMP manifest line {line_number} is stale or unsupported. "
                    "Regenerate schema-version-2 support, intervention, and OMP artifacts."
                )
            pair_index = _integer(row.get("pair_index"), f"OMP manifest line {line_number} pair_index")
            raw_user, raw_item = _raw_id(row.get("user_id")), _raw_id(row.get("target_item_id"))
            user_index = _validate_manifest_index(
                raw_user, row.get("user_index"), mappings.user_index(raw_user), "OMP manifest user_index"
            )
            item_index = _validate_manifest_index(
                raw_item, row.get("target_item_index"), mappings.item_index(raw_item), "OMP manifest target_item_index"
            )
            if row.get("status") != "recovered":
                stats.skipped_upstream_rows += 1
                continue
            if user_index is None or item_index is None:
                raise ArtifactError(f"OMP manifest line {line_number} recovered row requires mapped indices.")
            stats.recovered_rows += 1
            diagnostics = row.get("diagnostics")
            residual = diagnostics.get("relative_residual") if isinstance(diagnostics, Mapping) else None
            if not isinstance(residual, (int, float)) or not math.isfinite(residual) or residual > max_relative_residual:
                stats.skipped_quality_rows += 1
                continue
            shard_path, vector_row = row.get("vector_shard"), row.get("vector_row")
            if not isinstance(shard_path, str):
                raise ArtifactError(f"OMP manifest line {line_number} has no vector shard.")
            vector_row = _integer(vector_row, f"OMP manifest line {line_number} vector_row")
            shard = cache.get(shard_path)
            if not 0 <= vector_row < shard["pair_index"].shape[0]:
                raise ArtifactError(f"OMP manifest line {line_number} vector_row is out of range.")
            if (
                int(shard["pair_index"][vector_row]) != pair_index
                or int(shard["user_index"][vector_row]) != user_index
                or int(shard["target_item_index"][vector_row]) != item_index
            ):
                raise ArtifactError(f"OMP manifest line {line_number} disagrees with its coefficient shard.")
            start, end = int(shard["coef_indptr"][vector_row]), int(shard["coef_indptr"][vector_row + 1])
            indices = tuple(int(value) for value in shard["coef_indices"][start:end])
            coefficients = tuple(float(value) for value in shard["coef_data"][start:end])
            if not indices or not any(value != 0.0 for value in coefficients):
                stats.skipped_empty_rows += 1
                continue
            if any(index < 0 or index >= len(vocabulary) for index in indices):
                raise ArtifactError(f"OMP manifest line {line_number} has out-of-range attribute index.")
            if not all(math.isfinite(value) for value in coefficients):
                raise ArtifactError(f"OMP manifest line {line_number} has non-finite coefficients.")
            if pair_index in labels:
                raise ArtifactError(f"OMP manifest contains duplicate pair_index {pair_index}.")
            labels[pair_index] = CausalLabel(
                pair_index, raw_user, user_index, raw_item, item_index, indices, coefficients, float(residual)
            )
            stats.accepted_rows += 1
    return labels, asdict(stats)


@dataclass(frozen=True)
class LoadedArtifacts:
    mappings: IdMappings
    user_embeddings: torch.Tensor
    item_embeddings: torch.Tensor
    vocabulary: tuple[str, ...]
    item_attributes: dict[int, tuple[int, ...]]
    semantic_embeddings: torch.Tensor
    recommendation: RecommendationData
    explanation_train: list[ExplanationExample]
    explanation_validation: list[ExplanationExample]
    explanation_test: list[ExplanationExample]
    causal_labels: dict[int, CausalLabel]
    stats: dict[str, Any] = field(default_factory=dict)


def load_artifacts(config: JointTrainingConfig) -> LoadedArtifacts:
    mappings = load_id_mappings(config.paths.id_mappings)
    user_embeddings, item_embeddings = load_frozen_embeddings(
        config.paths.user_embeddings, config.paths.item_embeddings, mappings, config.model.embedding_dim
    )
    vocabulary = load_vocabulary(config.paths.vocabulary)
    labels, omp_stats = load_omp_labels(
        config.paths.omp_dir, mappings, vocabulary, config.training.omp_max_relative_residual
    )
    item_attributes = load_item_attributes(config.paths.item_attribute_ids, mappings, len(vocabulary))
    semantic_embeddings = load_semantic_embeddings(config.paths.attribute_embeddings, vocabulary)
    recommendation = load_recommendation_data(
        config.paths.recommendation_train,
        config.paths.recommendation_validation,
        config.paths.recommendation_test,
        mappings,
    )
    user_profiles = load_profiles(config.paths.user_profiles, "uid")
    item_profiles = load_profiles(config.paths.item_profiles, "iid")
    explanation_train, explanation_train_stats = load_explanations(
        config.paths.explanation_train, mappings, user_profiles, item_profiles
    )
    explanation_validation, explanation_validation_stats = load_explanations(
        config.paths.explanation_validation, mappings, user_profiles, item_profiles
    )
    explanation_test, explanation_test_stats = load_explanations(
        config.paths.explanation_test, mappings, user_profiles, item_profiles
    )
    return LoadedArtifacts(
        mappings=mappings,
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        vocabulary=vocabulary,
        item_attributes=item_attributes,
        semantic_embeddings=semantic_embeddings,
        recommendation=recommendation,
        explanation_train=explanation_train,
        explanation_validation=explanation_validation,
        explanation_test=explanation_test,
        causal_labels=labels,
        stats={
            "num_users": mappings.num_users,
            "num_items": mappings.num_items,
            "vocabulary_size": len(vocabulary),
            "items_with_attributes": len(item_attributes),
            "recommendation": recommendation.stats,
            "explanation_train": explanation_train_stats,
            "explanation_validation": explanation_validation_stats,
            "explanation_test": explanation_test_stats,
            "omp": omp_stats,
        },
    )
