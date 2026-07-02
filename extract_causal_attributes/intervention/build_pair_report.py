"""Join intervention and OMP artifacts into an inspectable per-pair report."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTERVENTION_DIR = Path(__file__).resolve().parent / "artifacts" / "amazon"
DEFAULT_OMP_DIR = Path(__file__).resolve().parent / "omp" / "artifacts" / "amazon"
DEFAULT_ITEM_ATTRIBUTES_PATH = (
    REPO_ROOT / "attribute_pipeline" / "outputs" / "amazon" / "item_attributes_im_ex.json"
)
DEFAULT_TRAINING_PATH = REPO_ROOT / "XRec" / "data" / "amazon" / "trn.pkl"
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "amazon"
    / "pair_intervention_omp_report.jsonl"
)

USER_KEYS = ("user_id", "user", "uid", "reviewerID")
ITEM_KEYS = ("target_item_id", "item_id", "target_id", "item", "iid", "asin")
EXPLICIT_KEYS = ("explicit_attributes", "explicit")
IMPLICIT_KEYS = ("implicit_attributes", "implicit")
COMBINED_ATTRIBUTE_KEYS = ("attributes", "item_attributes")
EXPLANATION_KEYS = (
    "explanation",
    "explanations",
    "exp",
    "template",
    "review",
    "reviewText",
    "text",
    "sentence",
    "reason",
    "target_text",
)


class PairReportError(ValueError):
    """Raised when source artifacts cannot be joined safely."""


@dataclass(frozen=True)
class TrainingRecord:
    pair_index: int
    user_id: Any
    target_item_id: int
    explanation: Any
    raw_record: Any


class ShardCache:
    def __init__(self, root: Path, max_entries: int = 8) -> None:
        self.root = root
        self.max_entries = max_entries
        self._cache: dict[str, dict[str, np.ndarray]] = {}
        self._order: list[str] = []

    def load(self, relative_path: str) -> dict[str, np.ndarray]:
        if relative_path in self._cache:
            self._order.remove(relative_path)
            self._order.append(relative_path)
            return self._cache[relative_path]

        path = _safe_relative_path(self.root, relative_path)
        with np.load(path, allow_pickle=False) as archive:
            payload = {key: archive[key].copy() for key in archive.files}
        self._cache[relative_path] = payload
        self._order.append(relative_path)
        if len(self._order) > self.max_entries:
            oldest = self._order.pop(0)
            del self._cache[oldest]
        return payload


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _safe_relative_path(root: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise PairReportError("Artifact shard path must be a non-empty string.")
    root_resolved = root.resolve()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root_resolved)
    except ValueError as exc:
        raise PairReportError(f"Shard path escapes artifact directory: {relative_path!r}") from exc
    return path


def _coerce_int(value: Any, context: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PairReportError(f"{context} must be integer-compatible, got {value!r}.") from exc


def _json_safe(value: Any) -> Any:
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_safe(item_method())
        except ValueError:
            pass
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _first_present_key(mapping: Mapping[Any, Any], keys: Sequence[str], context: str) -> str | None:
    present = [key for key in keys if key in mapping]
    if len(present) > 1:
        raise PairReportError(f"{context} has ambiguous fields: {present}")
    return present[0] if present else None


def _normalize_attributes(value: Any, context: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Iterable[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        raise PairReportError(f"{context} must be a string or list of strings.")

    result: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        if not isinstance(raw_value, str):
            raise PairReportError(f"{context} contains a non-string attribute: {raw_value!r}.")
        attribute = raw_value.strip()
        if attribute and attribute not in seen:
            result.append(attribute)
            seen.add(attribute)
    return result


def _merge_attributes(*groups: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for attribute in group:
            if attribute not in seen:
                result.append(attribute)
                seen.add(attribute)
    return result


def _attributes_from_payload(payload: Any, context: str) -> dict[str, list[str]]:
    if isinstance(payload, Mapping):
        explicit_key = _first_present_key(payload, EXPLICIT_KEYS, context)
        implicit_key = _first_present_key(payload, IMPLICIT_KEYS, context)
        combined_key = _first_present_key(payload, COMBINED_ATTRIBUTE_KEYS, context)
        if combined_key and (explicit_key or implicit_key):
            raise PairReportError(
                f"{context} must use either explicit/implicit fields or a combined attribute field."
            )
        if combined_key:
            combined = _normalize_attributes(payload[combined_key], f"{context}.{combined_key}")
            return {"explicit": combined, "implicit": [], "combined": combined}

        explicit = _normalize_attributes(
            payload.get(explicit_key) if explicit_key else None,
            f"{context}.{explicit_key or 'explicit_attributes'}",
        )
        implicit = _normalize_attributes(
            payload.get(implicit_key) if implicit_key else None,
            f"{context}.{implicit_key or 'implicit_attributes'}",
        )
        return {"explicit": explicit, "implicit": implicit, "combined": _merge_attributes(explicit, implicit)}

    combined = _normalize_attributes(payload, context)
    return {"explicit": combined, "implicit": [], "combined": combined}


def load_item_attributes(path: Path) -> dict[int, dict[str, list[str]]]:
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if isinstance(payload, Mapping) and set(payload) == {"items"}:
        payload = payload["items"]

    item_attributes: dict[int, dict[str, list[str]]] = {}
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
                raise PairReportError(f"Item attribute record {record_index} must be an object.")
            item_key = _first_present_key(record, ITEM_KEYS, f"Item attribute record {record_index}")
            if item_key is None:
                raise PairReportError(f"Item attribute record {record_index} has no item ID field.")
            item_id = _coerce_int(record[item_key], f"Item attribute record {record_index}.{item_key}")
            item_attributes[item_id] = _attributes_from_payload(
                record, f"Item attribute record {record_index}"
            )
        return item_attributes

    raise PairReportError("Item attributes must be a top-level mapping or a list of records.")


def load_vocabulary(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if isinstance(payload, Mapping) and "attributes" in payload:
        payload = payload["attributes"]
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
        raise PairReportError("Vocabulary must contain an ordered attributes list.")
    vocabulary = list(payload)
    if not all(isinstance(attribute, str) for attribute in vocabulary):
        raise PairReportError("Vocabulary entries must be strings.")
    return vocabulary


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise PairReportError(f"JSONL line {line_number} in {path} must be an object.")
            records.append(payload)
    return records


def _is_non_string_iterable(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray, Mapping))


def _records_from_training_payload(payload: Any) -> list[Any]:
    if hasattr(payload, "to_dict") and hasattr(payload, "columns"):
        try:
            records = payload.to_dict(orient="records")
        except TypeError as exc:
            raise PairReportError("Could not convert training DataFrame to records.") from exc
        if not isinstance(records, list):
            raise PairReportError("Training DataFrame conversion did not return records.")
        return records

    if isinstance(payload, Mapping):
        for wrapper_key in ("pairs", "records", "data"):
            if set(payload) == {wrapper_key}:
                return _records_from_training_payload(payload[wrapper_key])
        user_key = _first_present_key(payload, USER_KEYS, "Training mapping")
        item_key = _first_present_key(payload, ITEM_KEYS, "Training mapping")
        if user_key is not None and item_key is not None:
            users = payload[user_key]
            items = payload[item_key]
            if _is_non_string_iterable(users) and _is_non_string_iterable(items):
                user_list = list(users)
                item_list = list(items)
                if len(user_list) != len(item_list):
                    raise PairReportError("Training user and item columns have different lengths.")
                return [{user_key: user, item_key: item} for user, item in zip(user_list, item_list)]
            return [payload]
        if all(isinstance(value, Mapping) for value in payload.values()):
            return list(payload.values())
        if all(
            isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
            for value in payload.values()
        ):
            return list(payload.values())
        raise PairReportError("Unsupported training-pickle mapping schema.")

    if _is_non_string_iterable(payload):
        return list(payload)

    raise PairReportError("Training pickle must contain a DataFrame, mapping, or iterable records.")


def _extract_explanation(record: Any) -> Any:
    if isinstance(record, Mapping):
        for key in EXPLANATION_KEYS:
            if key in record:
                return _json_safe(record[key])
        return None
    if isinstance(record, Sequence) and not isinstance(record, (str, bytes, bytearray)):
        for value in record[2:]:
            if isinstance(value, (str, Mapping, list, tuple)):
                return _json_safe(value)
    return None


def load_training_records(path: Path) -> dict[int, TrainingRecord]:
    with path.open("rb") as input_file:
        payload = pickle.load(input_file)
    records = _records_from_training_payload(payload)

    training_by_pair: dict[int, TrainingRecord] = {}
    for pair_index, record in enumerate(records):
        if isinstance(record, Mapping):
            user_key = _first_present_key(record, USER_KEYS, f"Training record {pair_index}")
            item_key = _first_present_key(record, ITEM_KEYS, f"Training record {pair_index}")
            if user_key is None or item_key is None:
                raise PairReportError(f"Training record {pair_index} has no user/item fields.")
            user_id = record[user_key]
            target_item_id = record[item_key]
        elif isinstance(record, Sequence) and not isinstance(record, (str, bytes, bytearray)):
            if len(record) < 2:
                raise PairReportError(f"Training record {pair_index} must contain user and item IDs.")
            user_id, target_item_id = record[0], record[1]
        else:
            raise PairReportError(f"Training record {pair_index} must be a mapping or sequence.")

        training_by_pair[pair_index] = TrainingRecord(
            pair_index=pair_index,
            user_id=_json_safe(user_id),
            target_item_id=_coerce_int(target_item_id, f"Training record {pair_index} item ID"),
            explanation=_extract_explanation(record),
            raw_record=_json_safe(record),
        )
    return training_by_pair


def extract_interventions(
    record: Mapping[str, Any], shard_cache: ShardCache, vocabulary: Sequence[str]
) -> list[dict[str, Any]]:
    if record.get("skip_reason"):
        return []
    row_start = _coerce_int(record.get("row_start"), "Intervention row_start")
    row_end = _coerce_int(record.get("row_end"), "Intervention row_end")
    if row_end == row_start:
        return []

    shard = shard_cache.load(str(record["shard"]))
    required = {
        "A_data",
        "A_indices",
        "A_indptr",
        "A_shape",
        "y_delta",
        "y_h",
        "removed_item_ids",
        "removed_item_indptr",
    }
    missing = sorted(required - set(shard))
    if missing:
        raise PairReportError(f"Intervention shard {record['shard']} is missing arrays: {missing}")

    a_indices_all = shard["A_indices"]
    a_indptr_all = shard["A_indptr"]
    removed_ids = shard["removed_item_ids"]
    removed_indptr = shard["removed_item_indptr"]
    y_delta = shard["y_delta"]
    y_h = shard["y_h"]

    interventions: list[dict[str, Any]] = []
    for shard_row in range(row_start, row_end):
        attr_start = int(a_indptr_all[shard_row])
        attr_end = int(a_indptr_all[shard_row + 1])
        attribute_indices = [int(index) for index in a_indices_all[attr_start:attr_end]]
        item_start = int(removed_indptr[shard_row])
        item_end = int(removed_indptr[shard_row + 1])
        interventions.append(
            {
                "intervention_index": shard_row - row_start,
                "attribute_indices": attribute_indices,
                "attributes": [vocabulary[index] for index in attribute_indices],
                "removed_item_ids": [int(item_id) for item_id in removed_ids[item_start:item_end]],
                "removed_item_count": int(item_end - item_start),
                "y_h": float(y_h[shard_row]),
                "y_delta": float(y_delta[shard_row]),
            }
        )
    return interventions


def extract_sparse_attributes(
    omp_record: Mapping[str, Any] | None, omp_cache: ShardCache, vocabulary: Sequence[str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if omp_record is None:
        return {"status": "missing"}, []
    if omp_record.get("status") != "recovered":
        return {
            "status": omp_record.get("status", "skipped"),
            "skip_reason": omp_record.get("skip_reason"),
            "skip_details": omp_record.get("skip_details"),
        }, []

    shard = omp_cache.load(str(omp_record["vector_shard"]))
    required = {"coef_data", "coef_indices", "coef_indptr"}
    missing = sorted(required - set(shard))
    if missing:
        raise PairReportError(f"OMP shard {omp_record['vector_shard']} is missing arrays: {missing}")
    row = _coerce_int(omp_record.get("vector_row"), "OMP vector_row")
    indptr = shard["coef_indptr"]
    start = int(indptr[row])
    end = int(indptr[row + 1])
    indices = shard["coef_indices"][start:end]
    coefficients = shard["coef_data"][start:end]
    sparse_attributes = [
        {
            "attribute_index": int(index),
            "attribute": vocabulary[int(index)],
            "coefficient": float(coefficient),
        }
        for index, coefficient in zip(indices, coefficients)
    ]
    return {
        "status": "recovered",
        "selected_attribute_count": omp_record.get("selected_attribute_count"),
        "zero_signal": omp_record.get("zero_signal"),
        "diagnostics": omp_record.get("diagnostics"),
    }, sparse_attributes


def _write_jsonl_atomic(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
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


def build_pair_reports(
    intervention_dir: Path,
    omp_dir: Path,
    item_attributes_path: Path,
    training_path: Path,
    output_path: Path,
    limit: int | None = None,
    include_training_record: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")

    vocabulary = load_vocabulary(intervention_dir / "vocabulary.json")
    item_attributes = load_item_attributes(item_attributes_path)
    training_records = load_training_records(training_path)
    intervention_records = load_jsonl(intervention_dir / "manifest.jsonl")
    omp_records = load_jsonl(omp_dir / "manifest.jsonl")
    omp_by_pair = {record["pair_index"]: record for record in omp_records}
    intervention_cache = ShardCache(intervention_dir)
    omp_cache = ShardCache(omp_dir)

    summary = {
        "processed_pair_count": 0,
        "emitted_pair_count": 0,
        "missing_training_record_count": 0,
        "missing_target_attributes_count": 0,
        "intervention_row_count": 0,
        "omp_recovered_pair_count": 0,
        "omp_skipped_or_missing_pair_count": 0,
    }

    def iter_reports() -> Iterable[dict[str, Any]]:
        for source_index, intervention_record in enumerate(intervention_records):
            if limit is not None and source_index >= limit:
                break
            pair_index = _coerce_int(intervention_record.get("pair_index"), "Pair index")
            target_item_id = _coerce_int(
                intervention_record.get("target_item_id"), f"Pair {pair_index} target item ID"
            )
            training_record = training_records.get(pair_index)
            attributes = item_attributes.get(
                target_item_id, {"explicit": [], "implicit": [], "combined": []}
            )
            interventions = extract_interventions(intervention_record, intervention_cache, vocabulary)
            omp_status, sparse_attributes = extract_sparse_attributes(
                omp_by_pair.get(pair_index), omp_cache, vocabulary
            )

            summary["processed_pair_count"] += 1
            summary["emitted_pair_count"] += 1
            summary["intervention_row_count"] += len(interventions)
            if training_record is None:
                summary["missing_training_record_count"] += 1
            if not attributes["combined"]:
                summary["missing_target_attributes_count"] += 1
            if omp_status.get("status") == "recovered":
                summary["omp_recovered_pair_count"] += 1
            else:
                summary["omp_skipped_or_missing_pair_count"] += 1

            report = {
                "pair_index": pair_index,
                "user_id": intervention_record.get("user_id"),
                "target_item_id": target_item_id,
                "baseline_score": intervention_record.get("baseline_score"),
                "target_attributes": attributes,
                "intervention_metadata": {
                    "eligible_history_count": intervention_record.get("eligible_history_count"),
                    "supported_target_attribute_count": intervention_record.get(
                        "supported_target_attribute_count"
                    ),
                    "requested_intervention_count": intervention_record.get(
                        "requested_intervention_count"
                    ),
                    "generated_intervention_count": intervention_record.get(
                        "generated_intervention_count"
                    ),
                    "skip_reason": intervention_record.get("skip_reason"),
                },
                "interventions": interventions,
                "omp": omp_status,
                "sparse_attributes": sparse_attributes,
                "xrec_explanation": training_record.explanation if training_record else None,
            }
            if include_training_record:
                report["xrec_training_record"] = training_record.raw_record if training_record else None
            yield report

    _write_jsonl_atomic(output_path, iter_reports())
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    _write_json_atomic(summary_path, summary)
    return {**summary, "output": str(output_path), "summary_output": str(summary_path)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intervention-dir", type=Path, default=DEFAULT_INTERVENTION_DIR)
    parser.add_argument("--omp-dir", type=Path, default=DEFAULT_OMP_DIR)
    parser.add_argument("--item-attributes", type=Path, default=DEFAULT_ITEM_ATTRIBUTES_PATH)
    parser.add_argument("--training-pairs", type=Path, default=DEFAULT_TRAINING_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include-training-record", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        summary = build_pair_reports(
            intervention_dir=_resolve_path(args.intervention_dir),
            omp_dir=_resolve_path(args.omp_dir),
            item_attributes_path=_resolve_path(args.item_attributes),
            training_path=_resolve_path(args.training_pairs),
            output_path=_resolve_path(args.output),
            limit=args.limit,
            include_training_record=args.include_training_record,
            overwrite=args.overwrite,
        )
    except (FileExistsError, KeyError, OSError, PairReportError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
