"""Validated raw-ID and LightGCN internal-index conversions."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class IdMappingError(ValueError):
    """Raised when LightGCN ID mappings are malformed or inconsistent."""


def _raw_id(value: Any) -> str:
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            value = item_method()
        except ValueError:
            pass
    return str(value)


def _coerce_index(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise IdMappingError(f"{context} must be an integer, got {value!r}.")
    try:
        index = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise IdMappingError(f"{context} must be integer-compatible, got {value!r}.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise IdMappingError(f"{context} must be an integer, got {value!r}.")
    return index


def _validate_index_to_raw(payload: Any, context: str) -> tuple[str, ...]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
        raise IdMappingError(f"{context} must be a list of raw IDs.")
    values = tuple(_raw_id(value) for value in payload)
    if any(not value for value in values):
        raise IdMappingError(f"{context} contains an empty raw ID.")
    if len(set(values)) != len(values):
        raise IdMappingError(f"{context} contains duplicate raw IDs.")
    return values


def _validate_raw_to_index(
    payload: Any, index_to_raw: tuple[str, ...], context: str
) -> dict[str, int]:
    if not isinstance(payload, Mapping):
        raise IdMappingError(f"{context} must be an object.")
    mapping: dict[str, int] = {}
    seen_indices: set[int] = set()
    for raw_id, raw_index in payload.items():
        normalized_raw_id = _raw_id(raw_id)
        index = _coerce_index(raw_index, f"{context}[{normalized_raw_id!r}]")
        if index < 0 or index >= len(index_to_raw):
            raise IdMappingError(
                f"{context}[{normalized_raw_id!r}] index {index} is out of range "
                f"for {len(index_to_raw)} IDs."
            )
        if normalized_raw_id in mapping:
            raise IdMappingError(f"{context} contains duplicate raw ID {normalized_raw_id!r}.")
        if index in seen_indices:
            raise IdMappingError(f"{context} contains duplicate internal index {index}.")
        mapping[normalized_raw_id] = index
        seen_indices.add(index)

    expected = {raw_id: index for index, raw_id in enumerate(index_to_raw)}
    if mapping != expected:
        raise IdMappingError(f"{context} disagrees with its inverse ordered ID list.")
    return mapping


def _validate_internal_index(index: Any, size: int, context: str) -> int:
    normalized = _coerce_index(index, context)
    if normalized < 0 or normalized >= size:
        raise IdMappingError(f"{context} {normalized} is out of range for {size} IDs.")
    return normalized


@dataclass(frozen=True)
class IdMappings:
    """Bidirectional mappings between persisted raw IDs and compact LightGCN rows."""

    user_to_index: dict[str, int]
    item_to_index: dict[str, int]
    index_to_user: tuple[str, ...]
    index_to_item: tuple[str, ...]

    def user_index(self, raw_user_id: Any) -> int | None:
        return self.user_to_index.get(_raw_id(raw_user_id))

    def item_index(self, raw_item_id: Any) -> int | None:
        return self.item_to_index.get(_raw_id(raw_item_id))

    def raw_user_id(self, user_index: Any) -> str:
        index = _validate_internal_index(user_index, len(self.index_to_user), "User index")
        return self.index_to_user[index]

    def raw_item_id(self, item_index: Any) -> str:
        index = _validate_internal_index(item_index, len(self.index_to_item), "Item index")
        return self.index_to_item[index]


def load_id_mappings(path: Path) -> IdMappings:
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, Mapping):
        raise IdMappingError("ID mappings root must be an object.")

    index_to_user = _validate_index_to_raw(payload.get("index_to_user"), "index_to_user")
    index_to_item = _validate_index_to_raw(payload.get("index_to_item"), "index_to_item")
    return IdMappings(
        user_to_index=_validate_raw_to_index(
            payload.get("user_to_index"), index_to_user, "user_to_index"
        ),
        item_to_index=_validate_raw_to_index(
            payload.get("item_to_index"), index_to_item, "item_to_index"
        ),
        index_to_user=index_to_user,
        index_to_item=index_to_item,
    )
