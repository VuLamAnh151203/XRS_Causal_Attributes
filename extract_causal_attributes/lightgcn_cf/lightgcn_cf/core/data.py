"""Core CSV loading, ID mapping, and implicit-feedback sampling utilities."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


USER_COLUMN_CANDIDATES = (
    "user_id",
    "userid",
    "user",
    "uid",
    "reviewerid",
    "reviewer_id",
)
ITEM_COLUMN_CANDIDATES = (
    "item_id",
    "itemid",
    "item",
    "iid",
    "asin",
    "product_id",
)


@dataclass(frozen=True)
class IdMappings:
    """Bidirectional mapping between raw CSV IDs and contiguous indices."""

    user_to_index: dict[str, int]
    item_to_index: dict[str, int]
    index_to_user: list[str]
    index_to_item: list[str]

    @property
    def num_users(self) -> int:
        return len(self.index_to_user)

    @property
    def num_items(self) -> int:
        return len(self.index_to_item)

    def to_dict(self) -> dict:
        return {
            "user_to_index": self.user_to_index,
            "item_to_index": self.item_to_index,
            "index_to_user": self.index_to_user,
            "index_to_item": self.index_to_item,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "IdMappings":
        return cls(
            user_to_index={str(key): int(index) for key, index in value["user_to_index"].items()},
            item_to_index={str(key): int(index) for key, index in value["item_to_index"].items()},
            index_to_user=[str(value) for value in value["index_to_user"]],
            index_to_item=[str(value) for value in value["index_to_item"]],
        )


@dataclass(frozen=True)
class RawInteractions:
    pairs: list[tuple[str, str]]
    user_column: str
    item_column: str
    row_count: int


@dataclass
class DatasetBundle:
    mappings: IdMappings
    train_pairs: torch.Tensor
    train_history: dict[int, set[int]]
    validation_pairs: dict[int, set[int]]
    summary: dict


def _resolve_column(
    fieldnames: Iterable[str],
    explicit_name: str | None,
    candidates: tuple[str, ...],
    label: str,
) -> str:
    names = [name for name in fieldnames if name is not None]
    by_lower = {name.lower(): name for name in names}
    if explicit_name:
        resolved = by_lower.get(explicit_name.lower())
        if resolved is None:
            raise ValueError(
                f"Configured {label} column {explicit_name!r} was not found. "
                f"Available columns: {names}"
            )
        return resolved
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    raise ValueError(
        f"Could not detect the {label} column. Available columns: {names}. "
        f"Set {label}_column explicitly in config.yaml."
    )


def load_raw_interactions(
    csv_path: str | Path,
    user_column: str | None = None,
    item_column: str | None = None,
) -> RawInteractions:
    """Read raw user-item CSV pairs while preserving IDs as strings."""

    path = Path(csv_path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV file has no header: {path}")
        resolved_user_column = _resolve_column(
            reader.fieldnames, user_column, USER_COLUMN_CANDIDATES, "user"
        )
        resolved_item_column = _resolve_column(
            reader.fieldnames, item_column, ITEM_COLUMN_CANDIDATES, "item"
        )
        pairs: list[tuple[str, str]] = []
        row_count = 0
        for row in reader:
            row_count += 1
            user_id = str(row[resolved_user_column]).strip()
            item_id = str(row[resolved_item_column]).strip()
            if not user_id or not item_id:
                raise ValueError(
                    f"Blank user or item ID at data row {row_count + 1} in {path}"
                )
            pairs.append((user_id, item_id))
    return RawInteractions(
        pairs=pairs,
        user_column=resolved_user_column,
        item_column=resolved_item_column,
        row_count=row_count,
    )


def _build_train_data(
    raw_pairs: Iterable[tuple[str, str]],
) -> tuple[IdMappings, torch.Tensor, dict[int, set[int]]]:
    user_to_index: dict[str, int] = {}
    item_to_index: dict[str, int] = {}
    unique_pairs: set[tuple[int, int]] = set()

    for user_id, item_id in raw_pairs:
        user_index = user_to_index.setdefault(user_id, len(user_to_index))
        item_index = item_to_index.setdefault(item_id, len(item_to_index))
        unique_pairs.add((user_index, item_index))

    index_to_user = [""] * len(user_to_index)
    index_to_item = [""] * len(item_to_index)
    for raw_id, index in user_to_index.items():
        index_to_user[index] = raw_id
    for raw_id, index in item_to_index.items():
        index_to_item[index] = raw_id

    sorted_pairs = sorted(unique_pairs)
    train_pairs = torch.tensor(sorted_pairs, dtype=torch.long)
    if train_pairs.numel() == 0:
        train_pairs = torch.empty((0, 2), dtype=torch.long)
    train_history = {index: set() for index in range(len(user_to_index))}
    for user_index, item_index in sorted_pairs:
        train_history[user_index].add(item_index)

    return (
        IdMappings(user_to_index, item_to_index, index_to_user, index_to_item),
        train_pairs,
        train_history,
    )


def map_validation_pairs(
    raw_pairs: Iterable[tuple[str, str]],
    mappings: IdMappings,
    train_history: dict[int, set[int]],
) -> tuple[dict[int, set[int]], dict[str, int]]:
    """Map validation rows and skip cold-start or already-seen interactions."""

    validation_pairs: dict[int, set[int]] = {}
    stats = {
        "validation_rows_skipped_unknown_user": 0,
        "validation_rows_skipped_unknown_item": 0,
        "validation_rows_skipped_seen_in_train": 0,
    }
    for user_id, item_id in raw_pairs:
        user_index = mappings.user_to_index.get(user_id)
        if user_index is None:
            stats["validation_rows_skipped_unknown_user"] += 1
            continue
        item_index = mappings.item_to_index.get(item_id)
        if item_index is None:
            stats["validation_rows_skipped_unknown_item"] += 1
            continue
        if item_index in train_history[user_index]:
            stats["validation_rows_skipped_seen_in_train"] += 1
            continue
        validation_pairs.setdefault(user_index, set()).add(item_index)
    return validation_pairs, stats


def load_dataset(
    train_csv: str | Path,
    validation_csv: str | Path,
    user_column: str | None = None,
    item_column: str | None = None,
) -> DatasetBundle:
    """Load training and validation data with mappings learned from train only."""

    train = load_raw_interactions(train_csv, user_column, item_column)
    validation = load_raw_interactions(
        validation_csv, train.user_column, train.item_column
    )
    mappings, train_pairs, train_history = _build_train_data(train.pairs)
    validation_pairs, validation_stats = map_validation_pairs(
        validation.pairs, mappings, train_history
    )
    summary = {
        "user_column": train.user_column,
        "item_column": train.item_column,
        "train_csv_rows": train.row_count,
        "train_unique_pairs": int(train_pairs.shape[0]),
        "validation_csv_rows": validation.row_count,
        "validation_users_evaluable": len(validation_pairs),
        "num_users": mappings.num_users,
        "num_items": mappings.num_items,
        **validation_stats,
    }
    return DatasetBundle(
        mappings=mappings,
        train_pairs=train_pairs,
        train_history=train_history,
        validation_pairs=validation_pairs,
        summary=summary,
    )


def load_validation_for_mappings(
    validation_csv: str | Path,
    mappings: IdMappings,
    train_history: dict[int, set[int]],
    user_column: str | None = None,
    item_column: str | None = None,
) -> tuple[dict[int, set[int]], dict]:
    raw = load_raw_interactions(validation_csv, user_column, item_column)
    validation_pairs, stats = map_validation_pairs(raw.pairs, mappings, train_history)
    return validation_pairs, {
        "user_column": raw.user_column,
        "item_column": raw.item_column,
        "validation_csv_rows": raw.row_count,
        "validation_users_evaluable": len(validation_pairs),
        **stats,
    }


def sample_negative_items(
    users: torch.Tensor,
    num_items: int,
    train_history: dict[int, set[int]],
    rng: random.Random,
) -> torch.Tensor:
    """Sample one unseen item for every user in a CPU tensor."""

    negatives: list[int] = []
    for user_index in users.tolist():
        seen_items = train_history[int(user_index)]
        if len(seen_items) >= num_items:
            raise ValueError(f"User index {user_index} has no negative items to sample")
        candidate = rng.randrange(num_items)
        while candidate in seen_items:
            candidate = rng.randrange(num_items)
        negatives.append(candidate)
    return torch.tensor(negatives, dtype=torch.long)
