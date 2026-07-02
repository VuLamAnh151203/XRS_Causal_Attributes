"""Build semantic history-item support evidence for XRec training pairs."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from extract_causal_attributes.id_mappings import IdMappings, load_id_mappings


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "attribute_support_config.yaml"

USER_ID_KEYS = ("user_id", "user", "uid", "reviewerID")
ITEM_ID_KEYS = ("target_item_id", "item_id", "target_id", "item", "iid")
EXPLICIT_ATTRIBUTE_KEYS = ("explicit_attributes", "explicit")
IMPLICIT_ATTRIBUTE_KEYS = ("implicit_attributes", "implicit")
COMBINED_ATTRIBUTE_KEYS = ("attributes", "item_attributes")
HISTORY_KEYS = ("history", "items", "item_ids")


class SchemaError(ValueError):
    """Raised when an input artifact does not match a supported schema."""


class ConfigError(ValueError):
    """Raised when the generator configuration is invalid."""


@dataclass(frozen=True)
class TrainingPair:
    pair_index: int
    user_id: Any
    target_item_id: int


@dataclass(frozen=True)
class GeneratorConfig:
    training_pairs_path: Path
    item_attributes_path: Path
    user_history_path: Path
    id_mappings_path: Path
    output_path: Path
    summary_output_path: Path
    model_name: str
    threshold: float
    batch_size: int
    device: str
    comparison_chunk_size: int
    comparison_cache_max_entries: int


@dataclass
class GenerationStats:
    processed_pair_count: int = 0
    emitted_row_count: int = 0
    missing_history_count: int = 0
    missing_user_mapping_count: int = 0
    missing_target_item_mapping_count: int = 0
    missing_target_attribute_count: int = 0
    skipped_history_item_count: int = 0
    semantic_match_count: int = 0


class Embedder(Protocol):
    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        """Encode texts as a two-dimensional array."""


class Matcher(Protocol):
    def best_match(self, target_attribute: str, history_item_id: int) -> tuple[float, str] | None:
        """Return the best score and history attribute for an item."""


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required. Install dependencies from "
                "extract_causal_attributes/requirements-attribute-support.txt."
            ) from exc

        kwargs = {} if device == "auto" else {"device": device}
        self._model = SentenceTransformer(model_name, **kwargs)

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        return self._model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )


class EmbeddingIndex:
    def __init__(self, attributes: Sequence[str], embeddings: np.ndarray) -> None:
        attribute_list = list(attributes)
        matrix = np.asarray(embeddings, dtype=np.float32)

        if not attribute_list:
            self.attributes: list[str] = []
            self.attribute_to_index: dict[str, int] = {}
            self.embeddings = np.empty((0, 0), dtype=np.float32)
            return

        if matrix.ndim != 2 or matrix.shape[0] != len(attribute_list):
            raise ValueError(
                "The embedder must return a two-dimensional matrix with one row per attribute."
            )

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("The embedder returned at least one zero-length embedding.")

        self.attributes = attribute_list
        self.attribute_to_index = {
            attribute: index for index, attribute in enumerate(attribute_list)
        }
        self.embeddings = matrix / norms


class AttributeMatcher:
    def __init__(
        self,
        embedding_index: EmbeddingIndex,
        item_attributes: Mapping[int, list[str]],
        comparison_chunk_size: int,
        cache_max_entries: int,
    ) -> None:
        self._embedding_index = embedding_index
        self._item_attributes = item_attributes
        self._comparison_chunk_size = comparison_chunk_size
        self._cached_best_match = lru_cache(maxsize=cache_max_entries)(self._best_match)

    def best_match(self, target_attribute: str, history_item_id: int) -> tuple[float, str] | None:
        return self._cached_best_match(target_attribute, history_item_id)

    def _best_match(self, target_attribute: str, history_item_id: int) -> tuple[float, str] | None:
        history_attributes = self._item_attributes.get(history_item_id, [])
        if not history_attributes:
            return None

        try:
            target_index = self._embedding_index.attribute_to_index[target_attribute]
            history_indices = [
                self._embedding_index.attribute_to_index[attribute]
                for attribute in history_attributes
            ]
        except KeyError as exc:
            raise SchemaError(f"Missing embedding for attribute: {exc.args[0]!r}") from exc

        target_embedding = self._embedding_index.embeddings[target_index]
        best_score = -1.0
        best_attribute: str | None = None

        for start in range(0, len(history_indices), self._comparison_chunk_size):
            chunk_indices = history_indices[start : start + self._comparison_chunk_size]
            scores = self._embedding_index.embeddings[chunk_indices] @ target_embedding
            local_index = int(np.argmax(scores))
            score = float(scores[local_index])
            if score > best_score:
                best_score = score
                best_attribute = history_attributes[start + local_index]

        if best_attribute is None:
            return None
        return best_score, best_attribute


def _first_present_key(
    mapping: Mapping[Any, Any], keys: Sequence[str], context: str
) -> str | None:
    present_keys = [key for key in keys if key in mapping]
    if len(present_keys) > 1:
        raise SchemaError(f"{context} has ambiguous fields: {present_keys}")
    return present_keys[0] if present_keys else None


def _coerce_int(value: Any, context: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SchemaError(f"{context} must be integer-compatible, got {value!r}.") from exc


def _json_safe_scalar(value: Any) -> Any:
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return item_method()
        except ValueError:
            pass
    return value


def _normalize_attribute_values(value: Any, context: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_values: Iterable[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_values = value
    else:
        raise SchemaError(f"{context} must be a string or a list of strings.")

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        if not isinstance(raw_value, str):
            raise SchemaError(f"{context} contains a non-string attribute: {raw_value!r}.")
        attribute = raw_value.strip()
        if attribute and attribute not in seen:
            normalized.append(attribute)
            seen.add(attribute)
    return normalized


def _stable_concat(*groups: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group:
            if value not in seen:
                result.append(value)
                seen.add(value)
    return result


def _attributes_from_payload(payload: Any, context: str) -> list[str]:
    if isinstance(payload, Mapping):
        explicit_key = _first_present_key(payload, EXPLICIT_ATTRIBUTE_KEYS, context)
        implicit_key = _first_present_key(payload, IMPLICIT_ATTRIBUTE_KEYS, context)
        combined_key = _first_present_key(payload, COMBINED_ATTRIBUTE_KEYS, context)

        if combined_key is not None and (explicit_key is not None or implicit_key is not None):
            raise SchemaError(
                f"{context} must use either explicit/implicit fields or a combined attribute field."
            )
        if combined_key is not None:
            return _normalize_attribute_values(payload[combined_key], f"{context}.{combined_key}")
        if explicit_key is not None or implicit_key is not None:
            explicit = _normalize_attribute_values(
                payload.get(explicit_key) if explicit_key else None,
                f"{context}.{explicit_key or 'explicit_attributes'}",
            )
            implicit = _normalize_attribute_values(
                payload.get(implicit_key) if implicit_key else None,
                f"{context}.{implicit_key or 'implicit_attributes'}",
            )
            return _stable_concat(explicit, implicit)
        raise SchemaError(
            f"{context} must contain explicit/implicit attributes or a combined attribute field."
        )

    return _normalize_attribute_values(payload, context)


def load_item_attributes(path: Path) -> dict[int, list[str]]:
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    if isinstance(payload, Mapping) and set(payload) == {"items"}:
        payload = payload["items"]

    item_attributes: dict[int, list[str]] = {}
    if isinstance(payload, Mapping):
        for raw_item_id, attributes_payload in payload.items():
            item_id = _coerce_int(raw_item_id, "Item attribute dictionary key")
            item_attributes[item_id] = _attributes_from_payload(
                attributes_payload, f"Attributes for item {item_id}"
            )
        return item_attributes

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for record_index, record in enumerate(payload):
            if not isinstance(record, Mapping):
                raise SchemaError(f"Item attribute record {record_index} must be an object.")
            item_key = _first_present_key(record, ITEM_ID_KEYS, f"Item attribute record {record_index}")
            if item_key is None:
                raise SchemaError(f"Item attribute record {record_index} has no item ID field.")
            item_id = _coerce_int(record[item_key], f"Item attribute record {record_index}.{item_key}")
            item_attributes[item_id] = _attributes_from_payload(
                record, f"Item attribute record {record_index}"
            )
        return item_attributes

    raise SchemaError("Item attributes must be a top-level item mapping or a list of item records.")


def _history_items_from_payload(payload: Any, context: str) -> list[int]:
    if isinstance(payload, Mapping):
        history_key = _first_present_key(payload, HISTORY_KEYS, context)
        if history_key is None:
            raise SchemaError(f"{context} has no supported history field.")
        payload = payload[history_key]

    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
        raise SchemaError(f"{context} must be a list of item IDs.")

    items: list[int] = []
    seen: set[int] = set()
    for index, raw_item_id in enumerate(payload):
        item_id = _coerce_int(raw_item_id, f"{context}[{index}]")
        if item_id not in seen:
            items.append(item_id)
            seen.add(item_id)
    return items


def load_user_history(path: Path) -> dict[str, list[int]]:
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    if isinstance(payload, Mapping) and set(payload) == {"user_history"}:
        payload = payload["user_history"]

    histories: dict[str, list[int]] = {}
    if isinstance(payload, Mapping):
        for raw_user_id, history_payload in payload.items():
            histories[str(raw_user_id)] = _history_items_from_payload(
                history_payload, f"History for user {raw_user_id!r}"
            )
        return histories

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for record_index, record in enumerate(payload):
            if not isinstance(record, Mapping):
                raise SchemaError(f"User history record {record_index} must be an object.")
            user_key = _first_present_key(record, USER_ID_KEYS, f"User history record {record_index}")
            if user_key is None:
                raise SchemaError(f"User history record {record_index} has no user ID field.")
            histories[str(record[user_key])] = _history_items_from_payload(
                record, f"User history record {record_index}"
            )
        return histories

    raise SchemaError("User history must be a top-level user mapping or a list of user records.")


def convert_internal_user_histories_to_raw(
    user_histories: Mapping[str, list[int]], id_mappings: IdMappings
) -> dict[str, list[int]]:
    """Translate LightGCN history rows into the raw IDs used by attribute artifacts."""

    raw_histories: dict[str, list[int]] = {}
    for raw_user_index, internal_item_indices in user_histories.items():
        user_index = _coerce_int(raw_user_index, f"LightGCN history user index {raw_user_index!r}")
        try:
            raw_user_id = id_mappings.raw_user_id(user_index)
        except ValueError as exc:
            raise SchemaError(str(exc)) from exc
        raw_items: list[int] = []
        for item_index in internal_item_indices:
            try:
                raw_item_id = id_mappings.raw_item_id(item_index)
            except ValueError as exc:
                raise SchemaError(str(exc)) from exc
            raw_items.append(_coerce_int(raw_item_id, f"Raw item ID for LightGCN item index {item_index}"))
        raw_histories[raw_user_id] = raw_items
    return raw_histories


def _is_non_string_iterable(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray, Mapping))


def _pairs_from_column_mapping(payload: Mapping[Any, Any]) -> list[Any] | None:
    user_key = _first_present_key(payload, USER_ID_KEYS, "Training-pair column mapping")
    item_key = _first_present_key(payload, ITEM_ID_KEYS, "Training-pair column mapping")
    if user_key is None or item_key is None:
        return None

    users = payload[user_key]
    items = payload[item_key]
    if not _is_non_string_iterable(users) or not _is_non_string_iterable(items):
        return None

    user_list = list(users)
    item_list = list(items)
    if len(user_list) != len(item_list):
        raise SchemaError("Training-pair user and item columns have different lengths.")
    return [{user_key: user, item_key: item} for user, item in zip(user_list, item_list)]


def _records_from_training_payload(payload: Any) -> list[Any]:
    if hasattr(payload, "to_dict") and hasattr(payload, "columns"):
        columns = list(payload.columns)
        has_named_user = any(key in columns for key in USER_ID_KEYS)
        has_named_item = any(key in columns for key in ITEM_ID_KEYS)
        if not (has_named_user and has_named_item) and hasattr(payload, "itertuples"):
            return list(payload.itertuples(index=False, name=None))
        try:
            records = payload.to_dict(orient="records")
        except TypeError as exc:
            raise SchemaError("Could not convert the training-pair DataFrame to records.") from exc
        if not isinstance(records, list):
            raise SchemaError("Training-pair DataFrame conversion did not return records.")
        return records

    if isinstance(payload, Mapping):
        column_records = _pairs_from_column_mapping(payload)
        if column_records is not None:
            return column_records
        user_key = _first_present_key(payload, USER_ID_KEYS, "Training pair mapping")
        item_key = _first_present_key(payload, ITEM_ID_KEYS, "Training pair mapping")
        if user_key is not None and item_key is not None:
            return [payload]
        for wrapper_key in ("pairs", "records", "data"):
            if set(payload) == {wrapper_key}:
                return _records_from_training_payload(payload[wrapper_key])
        if all(isinstance(value, Mapping) for value in payload.values()):
            return list(payload.values())
        raise SchemaError("Unsupported training-pair mapping schema.")

    if _is_non_string_iterable(payload):
        return list(payload)

    raise SchemaError("Training pairs must be an iterable, mapping, or DataFrame.")


def parse_training_pairs(payload: Any, limit: int | None = None) -> list[TrainingPair]:
    records = _records_from_training_payload(payload)
    if limit is not None:
        records = records[:limit]

    pairs: list[TrainingPair] = []
    for pair_index, record in enumerate(records):
        if isinstance(record, Mapping):
            user_key = _first_present_key(record, USER_ID_KEYS, f"Training pair {pair_index}")
            item_key = _first_present_key(record, ITEM_ID_KEYS, f"Training pair {pair_index}")
            if user_key is None or item_key is None:
                raise SchemaError(f"Training pair {pair_index} has no supported user/item fields.")
            user_id = record[user_key]
            target_item_id = record[item_key]
        elif _is_non_string_iterable(record):
            values = list(record)
            if len(values) < 2:
                raise SchemaError(f"Training pair {pair_index} must have at least two values.")
            user_id, target_item_id = values[0], values[1]
        else:
            raise SchemaError(f"Training pair {pair_index} must be an object or tuple/list.")

        pairs.append(
            TrainingPair(
                pair_index=pair_index,
                user_id=_json_safe_scalar(user_id),
                target_item_id=_coerce_int(target_item_id, f"Training pair {pair_index} item ID"),
            )
        )
    return pairs


def load_training_pairs(path: Path, limit: int | None = None) -> list[TrainingPair]:
    with path.open("rb") as input_file:
        payload = pickle.load(input_file)
    return parse_training_pairs(payload, limit=limit)


def collect_unique_attributes(item_attributes: Mapping[int, Sequence[str]]) -> list[str]:
    attributes: list[str] = []
    seen: set[str] = set()
    for item_attribute_list in item_attributes.values():
        for attribute in item_attribute_list:
            if attribute not in seen:
                attributes.append(attribute)
                seen.add(attribute)
    return attributes


def build_support_record(
    pair: TrainingPair,
    user_histories: Mapping[str, list[int]],
    item_attributes: Mapping[int, list[str]],
    matcher: Matcher,
    threshold: float,
    stats: GenerationStats,
    id_mappings: IdMappings,
) -> dict[str, Any]:
    stats.processed_pair_count += 1
    user_index = id_mappings.user_index(pair.user_id)
    target_item_index = id_mappings.item_index(pair.target_item_id)
    if user_index is None:
        stats.missing_user_mapping_count += 1
    if target_item_index is None:
        stats.missing_target_item_mapping_count += 1

    history_item_ids = user_histories.get(str(pair.user_id), []) if user_index is not None else []
    if not history_item_ids:
        stats.missing_history_count += 1

    target_attributes = item_attributes.get(pair.target_item_id, [])
    if not target_attributes:
        stats.missing_target_attribute_count += 1

    candidate_items: list[tuple[int, int]] = []
    seen_item_ids: set[int] = set()
    for history_position, history_item_id in enumerate(history_item_ids):
        if history_item_id == pair.target_item_id or history_item_id in seen_item_ids:
            continue
        seen_item_ids.add(history_item_id)
        if not item_attributes.get(history_item_id):
            stats.skipped_history_item_count += 1
            continue
        candidate_items.append((history_item_id, history_position))

    supported_items_by_attribute: dict[str, list[dict[str, Any]]] = {}
    for target_attribute in target_attributes:
        supported_items: list[tuple[dict[str, Any], int, float]] = []
        for history_item_id, history_position in candidate_items:
            match = matcher.best_match(target_attribute, history_item_id)
            if match is None:
                continue
            score, matched_attribute = match
            if score > threshold:
                history_item_index = id_mappings.item_index(history_item_id)
                if history_item_index is None:
                    raise SchemaError(
                        f"History item raw ID {history_item_id} is absent from LightGCN mappings."
                    )
                supported_items.append(
                    (
                        {
                            "item_id": history_item_id,
                            "item_index": history_item_index,
                            "score": round(score, 6),
                            "matched_attribute": matched_attribute,
                        },
                        history_position,
                        score,
                    )
                )
                stats.semantic_match_count += 1

        supported_items.sort(key=lambda item: (-item[2], item[1]))
        supported_items_by_attribute[target_attribute] = [item[0] for item in supported_items]

    return {
        "schema_version": 2,
        "pair_index": pair.pair_index,
        "user_id": pair.user_id,
        "user_index": user_index,
        "target_item_id": pair.target_item_id,
        "target_item_index": target_item_index,
        "target_attributes": target_attributes,
        "supported_items_by_attribute": supported_items_by_attribute,
    }


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"Configuration field {key!r} must be a mapping.")
    return value


def _required_string(payload: Mapping[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Configuration field {context}.{key} must be a non-empty string.")
    return value.strip()


def _positive_int(payload: Mapping[str, Any], key: str, context: str, default: int) -> int:
    value = payload.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"Configuration field {context}.{key} must be a positive integer.")
    return value


def _resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path) -> GeneratorConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required. Install dependencies from "
            "extract_causal_attributes/requirements-attribute-support.txt."
        ) from exc

    with path.open("r", encoding="utf-8") as input_file:
        payload = yaml.safe_load(input_file)
    if not isinstance(payload, Mapping):
        raise ConfigError("The root YAML configuration value must be a mapping.")

    paths = _required_mapping(payload, "paths")
    similarity = _required_mapping(payload, "semantic_similarity")
    runtime = payload.get("runtime", {})
    if not isinstance(runtime, Mapping):
        raise ConfigError("Configuration field 'runtime' must be a mapping.")

    threshold = similarity.get("threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ConfigError("Configuration field semantic_similarity.threshold must be numeric.")
    threshold = float(threshold)
    if threshold < -1.0 or threshold > 1.0:
        raise ConfigError("Configuration field semantic_similarity.threshold must be in [-1, 1].")

    return GeneratorConfig(
        training_pairs_path=_resolve_repo_path(_required_string(paths, "training_pairs", "paths")),
        item_attributes_path=_resolve_repo_path(_required_string(paths, "item_attributes", "paths")),
        user_history_path=_resolve_repo_path(_required_string(paths, "user_history", "paths")),
        id_mappings_path=_resolve_repo_path(_required_string(paths, "id_mappings", "paths")),
        output_path=_resolve_repo_path(_required_string(paths, "output", "paths")),
        summary_output_path=_resolve_repo_path(_required_string(paths, "summary_output", "paths")),
        model_name=_required_string(similarity, "model_name", "semantic_similarity"),
        threshold=threshold,
        batch_size=_positive_int(similarity, "batch_size", "semantic_similarity", 128),
        device=_required_string(similarity, "device", "semantic_similarity"),
        comparison_chunk_size=_positive_int(runtime, "comparison_chunk_size", "runtime", 4096),
        comparison_cache_max_entries=_positive_int(
            runtime, "comparison_cache_max_entries", "runtime", 250_000
        ),
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
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


def generate_artifact(
    config: GeneratorConfig,
    limit: int | None = None,
    overwrite: bool = False,
    embedder: Embedder | None = None,
) -> dict[str, Any]:
    if limit is not None and limit < 0:
        raise ValueError("--limit must be greater than or equal to zero.")
    for output_path in (config.output_path, config.summary_output_path):
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")

    item_attributes = load_item_attributes(config.item_attributes_path)
    id_mappings = load_id_mappings(config.id_mappings_path)
    user_histories = convert_internal_user_histories_to_raw(
        load_user_history(config.user_history_path), id_mappings
    )
    training_pairs = load_training_pairs(config.training_pairs_path, limit=limit)
    unique_attributes = collect_unique_attributes(item_attributes)

    if unique_attributes:
        actual_embedder = embedder or SentenceTransformerEmbedder(config.model_name, config.device)
        embeddings = actual_embedder.encode(unique_attributes, config.batch_size)
    else:
        embeddings = np.empty((0, 0), dtype=np.float32)

    matcher = AttributeMatcher(
        embedding_index=EmbeddingIndex(unique_attributes, embeddings),
        item_attributes=item_attributes,
        comparison_chunk_size=config.comparison_chunk_size,
        cache_max_entries=config.comparison_cache_max_entries,
    )
    stats = GenerationStats()

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=config.output_path.parent,
            prefix=f".{config.output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_output_path = Path(output_file.name)
            from tqdm import tqdm
            for pair in tqdm(training_pairs, desc="Processing training pairs"):
                record = build_support_record(
                    pair=pair,
                    user_histories=user_histories,
                    item_attributes=item_attributes,
                    matcher=matcher,
                    threshold=config.threshold,
                    stats=stats,
                    id_mappings=id_mappings,
                )
                json.dump(record, output_file, ensure_ascii=False)
                output_file.write("\n")
                stats.emitted_row_count += 1
        os.replace(temporary_output_path, config.output_path)
    finally:
        if temporary_output_path is not None and temporary_output_path.exists():
            temporary_output_path.unlink()

    summary = {
        "schema_version": 2,
        "model_name": config.model_name,
        "threshold": config.threshold,
        "source_paths": {
            "training_pairs": str(config.training_pairs_path),
            "item_attributes": str(config.item_attributes_path),
            "user_history": str(config.user_history_path),
            "id_mappings": str(config.id_mappings_path),
            "output": str(config.output_path),
        },
        "unique_attribute_count": len(unique_attributes),
        **asdict(stats),
    }
    _write_json_atomic(config.summary_output_path, summary)
    return summary


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML configuration path. Defaults to {DEFAULT_CONFIG_PATH}.",
    )
    parser.add_argument("--limit", type=int, help="Process only the first N training pairs.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing JSONL artifact and summary.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        summary = generate_artifact(config, limit=args.limit, overwrite=args.overwrite)
    except (ConfigError, FileExistsError, OSError, RuntimeError, SchemaError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
