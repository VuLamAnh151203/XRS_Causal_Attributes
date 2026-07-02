"""OMP recovery and artifact persistence for intervention matrices."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_SCHEMA_VERSION = 2


class OmpError(ValueError):
    """Raised when OMP inputs, settings, or artifacts are invalid."""


@dataclass(frozen=True)
class OmpConfig:
    intervention_dir: Path
    output_dir: Path
    sparsity_level: int
    minimum_intervention_multiplier: int
    require_strictly_more_than_minimum: bool
    correlation_tolerance: float
    residual_tolerance: float
    pairs_per_shard: int
    source_shard_cache_max_entries: int

    @property
    def minimum_intervention_count(self) -> int:
        return self.sparsity_level * self.minimum_intervention_multiplier


@dataclass(frozen=True)
class Vocabulary:
    attributes: tuple[str, ...]
    attribute_to_index: dict[str, int]


@dataclass(frozen=True)
class InterventionSlice:
    matrix: np.ndarray
    y_delta: np.ndarray
    active_columns: tuple[int, ...]
    unique_intervention_row_count: int
    unique_removed_item_signature_count: int


@dataclass(frozen=True)
class OmpFit:
    global_indices: tuple[int, ...]
    coefficients: tuple[float, ...]
    residual_norm: float
    relative_residual: float
    reconstruction_r_squared: float | None
    zero_signal: bool


@dataclass(frozen=True)
class RecoveryResult:
    pair_index: int
    user_id: int
    user_index: int | None
    target_item_id: int
    target_item_index: int | None
    status: str
    global_indices: tuple[int, ...] = ()
    coefficients: tuple[float, ...] = ()
    zero_signal: bool = False
    diagnostics: dict[str, Any] | None = None
    skip_reason: str | None = None
    skip_details: dict[str, Any] | None = None


@dataclass
class RunStats:
    processed_pair_count: int = 0
    recovered_pair_count: int = 0
    skipped_pair_count: int = 0
    zero_signal_pair_count: int = 0
    written_shard_count: int = 0
    selected_attribute_total: int = 0
    relative_residual_total: float = 0.0
    relative_residual_count: int = 0
    reconstruction_r_squared_total: float = 0.0
    reconstruction_r_squared_count: int = 0
    skip_count_by_reason: dict[str, int] = field(default_factory=dict)
    previous_full_time_seconds: float = 0.0


def _resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else REPO_ROOT / path


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise OmpError(f"Configuration field {key!r} must be a mapping.")
    return value


def _require_string(payload: Mapping[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise OmpError(f"Configuration field {context}.{key} must be a non-empty string.")
    return value.strip()


def _positive_int(payload: Mapping[str, Any], key: str, context: str, default: int) -> int:
    value = payload.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise OmpError(f"Configuration field {context}.{key} must be a positive integer.")
    return value


def _nonnegative_float(
    payload: Mapping[str, Any], key: str, context: str, default: float
) -> float:
    value = payload.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise OmpError(f"Configuration field {context}.{key} must be non-negative.")
    return float(value)


def load_config(path: Path) -> OmpConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required. Install extract_causal_attributes/intervention/omp/requirements.txt."
        ) from exc

    with path.open("r", encoding="utf-8") as input_file:
        payload = yaml.safe_load(input_file)
    if not isinstance(payload, Mapping):
        raise OmpError("The root YAML configuration value must be a mapping.")

    paths = _require_mapping(payload, "paths")
    omp = _require_mapping(payload, "omp")
    runtime = _require_mapping(payload, "runtime")
    strict = omp.get("require_strictly_more_than_minimum", True)
    if not isinstance(strict, bool):
        raise OmpError("Configuration field omp.require_strictly_more_than_minimum must be boolean.")

    return OmpConfig(
        intervention_dir=_resolve_repo_path(_require_string(paths, "intervention_dir", "paths")),
        output_dir=_resolve_repo_path(_require_string(paths, "output_dir", "paths")),
        sparsity_level=_positive_int(omp, "sparsity_level", "omp", 5),
        minimum_intervention_multiplier=_positive_int(
            omp, "minimum_intervention_multiplier", "omp", 2
        ),
        require_strictly_more_than_minimum=strict,
        correlation_tolerance=_nonnegative_float(
            omp, "correlation_tolerance", "omp", 1.0e-12
        ),
        residual_tolerance=_nonnegative_float(omp, "residual_tolerance", "omp", 1.0e-10),
        pairs_per_shard=_positive_int(runtime, "pairs_per_shard", "runtime", 100),
        source_shard_cache_max_entries=_positive_int(
            runtime, "source_shard_cache_max_entries", "runtime", 8
        ),
    )


def _load_tqdm() -> Any:
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError(
            "tqdm is required. Install extract_causal_attributes/intervention/omp/requirements.txt."
        ) from exc
    return tqdm


def _coerce_int(value: Any, context: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OmpError(f"{context} must be integer-compatible, got {value!r}.") from exc


def _optional_int(value: Any, context: str) -> int | None:
    return None if value is None else _coerce_int(value, context)


def _record_identity(
    record: Mapping[str, Any], context: str = "Intervention manifest"
) -> tuple[int, int, int | None, int, int | None]:
    if record.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise OmpError(
            f"{context} is stale or unsupported. Regenerate schema-version-2 intervention "
            "artifacts from the beginning."
        )
    for key in ("user_index", "target_item_index"):
        if key not in record:
            raise OmpError(
                f"{context} is missing {key}. Regenerate schema-version-2 intervention "
                "artifacts from the beginning."
            )
    return (
        _coerce_int(record.get("pair_index"), f"{context} pair_index"),
        _coerce_int(record.get("user_id"), f"{context} user_id"),
        _optional_int(record.get("user_index"), f"{context} user_index"),
        _coerce_int(record.get("target_item_id"), f"{context} target_item_id"),
        _optional_int(record.get("target_item_index"), f"{context} target_item_index"),
    )


def load_vocabulary(path: Path) -> Vocabulary:
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if isinstance(payload, Mapping) and "attributes" in payload:
        attributes_payload = payload["attributes"]
    else:
        attributes_payload = payload
    if not isinstance(attributes_payload, Sequence) or isinstance(
        attributes_payload, (str, bytes, bytearray)
    ):
        raise OmpError("Resolved intervention vocabulary must contain an ordered attributes list.")
    attributes = tuple(attributes_payload)
    if not all(isinstance(attribute, str) and attribute for attribute in attributes):
        raise OmpError("Vocabulary attributes must be non-empty strings.")
    if len(set(attributes)) != len(attributes):
        raise OmpError("Vocabulary contains duplicate attributes.")

    attribute_to_index = {attribute: index for index, attribute in enumerate(attributes)}
    if isinstance(payload, Mapping) and "attribute_to_index" in payload:
        supplied_mapping = payload["attribute_to_index"]
        if not isinstance(supplied_mapping, Mapping):
            raise OmpError("Vocabulary attribute_to_index must be an object.")
        normalized_mapping = {
            str(attribute): _coerce_int(index, f"Vocabulary index for {attribute!r}")
            for attribute, index in supplied_mapping.items()
        }
        if normalized_mapping != attribute_to_index:
            raise OmpError("Vocabulary attributes list and attribute_to_index mapping disagree.")
    return Vocabulary(attributes, attribute_to_index)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise OmpError(f"JSONL line {line_number} in {path} must be an object.")
            records.append(payload)
    return records


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_source_path(root: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise OmpError("Intervention manifest shard path must be a non-empty string.")
    root_resolved = root.resolve()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root_resolved)
    except ValueError as exc:
        raise OmpError(f"Intervention shard escapes source directory: {relative_path!r}") from exc
    return path


def _required_arrays(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    required = {
        "A_data",
        "A_indices",
        "A_indptr",
        "A_shape",
        "y_delta",
        "removed_item_ids",
        "removed_item_indptr",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise OmpError(f"Intervention shard {path} is missing arrays: {missing}")


class InterventionShardCache:
    def __init__(self, intervention_dir: Path, max_entries: int) -> None:
        self.intervention_dir = intervention_dir
        self.max_entries = max_entries
        self.values: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def load(self, relative_path: str) -> dict[str, np.ndarray]:
        if relative_path in self.values:
            value = self.values.pop(relative_path)
            self.values[relative_path] = value
            return value
        path = _safe_source_path(self.intervention_dir, relative_path)
        with np.load(path, allow_pickle=False) as archive:
            value = {key: archive[key].copy() for key in archive.files}
        _required_arrays(path, value)
        self.values[relative_path] = value
        if len(self.values) > self.max_entries:
            self.values.popitem(last=False)
        return value


def _validate_shard(path_label: str, shard: Mapping[str, np.ndarray]) -> tuple[int, int]:
    shape = np.asarray(shard["A_shape"])
    if shape.shape != (2,):
        raise OmpError(f"Intervention shard {path_label} A_shape must contain two values.")
    row_count, vocabulary_size = (int(shape[0]), int(shape[1]))
    if row_count < 0 or vocabulary_size < 0:
        raise OmpError(f"Intervention shard {path_label} A_shape cannot be negative.")

    indptr = np.asarray(shard["A_indptr"])
    indices = np.asarray(shard["A_indices"])
    data = np.asarray(shard["A_data"])
    y_delta = np.asarray(shard["y_delta"])
    removed_indptr = np.asarray(shard["removed_item_indptr"])
    removed_ids = np.asarray(shard["removed_item_ids"])
    if indptr.ndim != 1 or len(indptr) != row_count + 1:
        raise OmpError(f"Intervention shard {path_label} has invalid A_indptr.")
    if removed_indptr.ndim != 1 or len(removed_indptr) != row_count + 1:
        raise OmpError(f"Intervention shard {path_label} has invalid removed_item_indptr.")
    if indices.ndim != 1 or data.ndim != 1 or len(indices) != len(data):
        raise OmpError(f"Intervention shard {path_label} has invalid CSR arrays.")
    if y_delta.ndim != 1 or len(y_delta) != row_count:
        raise OmpError(f"Intervention shard {path_label} has invalid y_delta.")
    if removed_ids.ndim != 1:
        raise OmpError(f"Intervention shard {path_label} has invalid removed_item_ids.")
    if (
        indptr[0] != 0
        or indptr[-1] != len(indices)
        or np.any(indptr[1:] < indptr[:-1])
        or removed_indptr[0] != 0
        or removed_indptr[-1] != len(removed_ids)
        or np.any(removed_indptr[1:] < removed_indptr[:-1])
    ):
        raise OmpError(f"Intervention shard {path_label} has invalid CSR offsets.")
    if len(indices) and (np.any(indices < 0) or np.any(indices >= vocabulary_size)):
        raise OmpError(f"Intervention shard {path_label} has out-of-range attribute indices.")
    if len(data) and not np.all(data == 1):
        raise OmpError(f"Intervention shard {path_label} A_data must be binary ones.")
    return row_count, vocabulary_size


def load_intervention_slice(
    record: Mapping[str, Any],
    shard_cache: InterventionShardCache,
    expected_vocabulary_size: int,
) -> InterventionSlice:
    relative_path = record.get("shard")
    shard = shard_cache.load(relative_path)
    shard_row_count, vocabulary_size = _validate_shard(relative_path, shard)
    if vocabulary_size != expected_vocabulary_size:
        raise OmpError(
            f"Intervention shard {relative_path} vocabulary width {vocabulary_size} "
            f"does not match resolved vocabulary width {expected_vocabulary_size}."
        )

    row_start = _coerce_int(record.get("row_start"), "Intervention manifest row_start")
    row_end = _coerce_int(record.get("row_end"), "Intervention manifest row_end")
    if row_start < 0 or row_end < row_start or row_end > shard_row_count:
        raise OmpError(f"Intervention manifest has invalid row range [{row_start}, {row_end}).")
    expected_row_count = record.get("generated_intervention_count")
    if expected_row_count is not None and _coerce_int(
        expected_row_count, "Intervention manifest generated_intervention_count"
    ) != row_end - row_start:
        raise OmpError("Intervention manifest generated row count does not match its shard slice.")

    indptr = np.asarray(shard["A_indptr"], dtype=np.int64)
    indices = np.asarray(shard["A_indices"], dtype=np.int64)
    data = np.asarray(shard["A_data"], dtype=np.float64)
    active_indices = indices[int(indptr[row_start]) : int(indptr[row_end])]
    active_columns = tuple(sorted({int(index) for index in active_indices}))
    local_lookup = {column: offset for offset, column in enumerate(active_columns)}
    matrix = np.zeros((row_end - row_start, len(active_columns)), dtype=np.float64)
    for local_row, shard_row in enumerate(range(row_start, row_end)):
        start, end = int(indptr[shard_row]), int(indptr[shard_row + 1])
        row_indices = indices[start:end]
        if len(set(int(index) for index in row_indices)) != len(row_indices):
            raise OmpError(f"Intervention shard {relative_path} contains duplicate row indices.")
        for global_index, value in zip(row_indices, data[start:end]):
            matrix[local_row, local_lookup[int(global_index)]] = float(value)

    removed_indptr = np.asarray(shard["removed_item_indptr"], dtype=np.int64)
    removed_ids = np.asarray(shard["removed_item_ids"], dtype=np.int64)
    removed_signatures = {
        tuple(
            sorted(
                int(item_id)
                for item_id in removed_ids[
                    int(removed_indptr[shard_row]) : int(removed_indptr[shard_row + 1])
                ]
            )
        )
        for shard_row in range(row_start, row_end)
    }
    unique_row_count = int(np.unique(matrix, axis=0).shape[0]) if len(matrix) else 0
    return InterventionSlice(
        matrix=matrix,
        y_delta=np.asarray(shard["y_delta"][row_start:row_end], dtype=np.float64),
        active_columns=active_columns,
        unique_intervention_row_count=unique_row_count,
        unique_removed_item_signature_count=len(removed_signatures),
    )


def _reconstruction_diagnostics(
    matrix: np.ndarray, outcomes: np.ndarray, coefficients: np.ndarray
) -> tuple[float, float, float | None]:
    residual = outcomes - matrix @ coefficients
    residual_norm = float(np.linalg.norm(residual))
    outcome_norm = float(np.linalg.norm(outcomes))
    relative_residual = residual_norm / outcome_norm if outcome_norm else 0.0
    centered = outcomes - float(np.mean(outcomes)) if len(outcomes) else outcomes
    total_variance = float(np.dot(centered, centered))
    r_squared = None
    if total_variance > 0:
        r_squared = 1.0 - float(np.dot(residual, residual)) / total_variance
    return residual_norm, relative_residual, r_squared


def signed_omp(
    matrix: np.ndarray,
    outcomes: np.ndarray,
    global_columns: Sequence[int],
    sparsity_level: int,
    correlation_tolerance: float,
    residual_tolerance: float,
) -> OmpFit | None:
    matrix = np.asarray(matrix, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    if matrix.ndim != 2 or outcomes.ndim != 1 or matrix.shape[0] != len(outcomes):
        raise OmpError("OMP matrix and outcomes have incompatible shapes.")
    if matrix.shape[1] != len(global_columns):
        raise OmpError("OMP global-column mapping has the wrong length.")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(outcomes)):
        raise OmpError("OMP inputs must be finite.")

    outcome_norm = float(np.linalg.norm(outcomes))
    if outcome_norm <= residual_tolerance:
        return OmpFit((), (), 0.0, 0.0, None, True)

    residual = outcomes.copy()
    selected: list[int] = []
    coefficients = np.empty(0, dtype=np.float64)
    for _ in range(min(sparsity_level, matrix.shape[1])):
        previous_rank = int(np.linalg.matrix_rank(matrix[:, selected])) if selected else 0
        candidates: list[tuple[float, int, int]] = []
        for local_index, global_index in enumerate(global_columns):
            if local_index in selected:
                continue
            column = matrix[:, local_index]
            column_norm = float(np.linalg.norm(column))
            if column_norm == 0:
                continue
            normalized_correlation = abs(float(np.dot(column, residual))) / column_norm
            candidates.append((-normalized_correlation, int(global_index), local_index))
        candidates.sort()

        selected_candidate: int | None = None
        selected_correlation = 0.0
        for negative_correlation, _, local_index in candidates:
            correlation = -negative_correlation
            if correlation <= correlation_tolerance:
                break
            trial_columns = selected + [local_index]
            if int(np.linalg.matrix_rank(matrix[:, trial_columns])) > previous_rank:
                selected_candidate = local_index
                selected_correlation = correlation
                break
        if selected_candidate is None or selected_correlation <= correlation_tolerance:
            return None

        selected.append(selected_candidate)
        coefficients = np.linalg.lstsq(matrix[:, selected], outcomes, rcond=None)[0]
        residual = outcomes - matrix[:, selected] @ coefficients
        if float(np.linalg.norm(residual)) / outcome_norm <= residual_tolerance:
            break

    full_coefficients = np.zeros(matrix.shape[1], dtype=np.float64)
    if selected:
        full_coefficients[selected] = coefficients
    residual_norm, relative_residual, r_squared = _reconstruction_diagnostics(
        matrix, outcomes, full_coefficients
    )
    ordered = sorted(
        ((int(global_columns[local_index]), float(full_coefficients[local_index])) for local_index in selected),
        key=lambda item: item[0],
    )
    return OmpFit(
        global_indices=tuple(item[0] for item in ordered),
        coefficients=tuple(item[1] for item in ordered),
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        reconstruction_r_squared=r_squared,
        zero_signal=False,
    )


def _skip_result(
    record: Mapping[str, Any], reason: str, details: Mapping[str, Any] | None = None
) -> RecoveryResult:
    pair_index, user_id, user_index, target_item_id, target_item_index = _record_identity(record)
    return RecoveryResult(
        pair_index=pair_index,
        user_id=user_id,
        user_index=user_index,
        target_item_id=target_item_id,
        target_item_index=target_item_index,
        status="skipped",
        skip_reason=reason,
        skip_details=dict(details or {}),
    )


def _meets_minimum(observed: int, minimum: int, strictly_more: bool) -> bool:
    return observed > minimum if strictly_more else observed >= minimum


def recover_pair(
    record: Mapping[str, Any],
    shard_cache: InterventionShardCache,
    vocabulary_size: int,
    config: OmpConfig,
) -> RecoveryResult:
    pair_index, user_id, user_index, target_item_id, target_item_index = _record_identity(record)
    upstream_reason = record.get("skip_reason")
    if upstream_reason:
        return _skip_result(record, "upstream_skip", {"upstream_reason": upstream_reason})

    intervention_slice = load_intervention_slice(record, shard_cache, vocabulary_size)
    matrix = intervention_slice.matrix
    outcomes = intervention_slice.y_delta
    intervention_count = matrix.shape[0]
    active_attribute_count = matrix.shape[1]
    required = config.minimum_intervention_count
    required_label = f"> {required}" if config.require_strictly_more_than_minimum else f">= {required}"
    if not np.all(np.isfinite(outcomes)):
        return _skip_result(record, "invalid_nonfinite_outcomes")
    if not _meets_minimum(intervention_count, required, config.require_strictly_more_than_minimum):
        return _skip_result(
            record,
            "insufficient_interventions",
            {"observed": intervention_count, "required": required_label},
        )
    if not _meets_minimum(
        intervention_slice.unique_intervention_row_count,
        required,
        config.require_strictly_more_than_minimum,
    ):
        return _skip_result(
            record,
            "insufficient_unique_rows",
            {
                "observed": intervention_slice.unique_intervention_row_count,
                "required": required_label,
            },
        )
    if active_attribute_count < config.sparsity_level:
        return _skip_result(
            record,
            "insufficient_active_attributes",
            {"observed": active_attribute_count, "required": f">= {config.sparsity_level}"},
        )
    matrix_rank = int(np.linalg.matrix_rank(matrix))
    if matrix_rank < config.sparsity_level:
        return _skip_result(
            record,
            "insufficient_matrix_rank",
            {"observed": matrix_rank, "required": f">= {config.sparsity_level}"},
        )

    fit = signed_omp(
        matrix=matrix,
        outcomes=outcomes,
        global_columns=intervention_slice.active_columns,
        sparsity_level=config.sparsity_level,
        correlation_tolerance=config.correlation_tolerance,
        residual_tolerance=config.residual_tolerance,
    )
    if fit is None:
        return _skip_result(record, "omp_stalled")
    diagnostics = {
        "intervention_count": intervention_count,
        "unique_intervention_row_count": intervention_slice.unique_intervention_row_count,
        "unique_removed_item_signature_count": intervention_slice.unique_removed_item_signature_count,
        "active_attribute_count": active_attribute_count,
        "matrix_rank": matrix_rank,
        "residual_norm": fit.residual_norm,
        "relative_residual": fit.relative_residual,
        "reconstruction_r_squared": fit.reconstruction_r_squared,
    }
    return RecoveryResult(
        pair_index=pair_index,
        user_id=user_id,
        user_index=user_index,
        target_item_id=target_item_id,
        target_item_index=target_item_index,
        status="recovered",
        global_indices=fit.global_indices,
        coefficients=fit.coefficients,
        zero_signal=fit.zero_signal,
        diagnostics=diagnostics,
    )


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


def write_coefficient_shard(
    path: Path, results: Sequence[RecoveryResult], vocabulary_size: int
) -> None:
    coefficients: list[float] = []
    indices: list[int] = []
    indptr = [0]
    pair_indices: list[int] = []
    user_indices: list[int] = []
    target_item_indices: list[int] = []
    for result in results:
        if result.status != "recovered":
            raise OmpError("Coefficient shards may contain only recovered results.")
        if result.user_index is None or result.target_item_index is None:
            raise OmpError("Recovered coefficient rows require internal user and target item indices.")
        coefficients.extend(result.coefficients)
        indices.extend(result.global_indices)
        indptr.append(len(coefficients))
        pair_indices.append(result.pair_index)
        user_indices.append(result.user_index)
        target_item_indices.append(result.target_item_index)
    _write_npz_atomic(
        path,
        coef_data=np.asarray(coefficients, dtype=np.float32),
        coef_indices=np.asarray(indices, dtype=np.int64),
        coef_indptr=np.asarray(indptr, dtype=np.int64),
        coef_shape=np.asarray([len(results), vocabulary_size], dtype=np.int64),
        pair_index=np.asarray(pair_indices, dtype=np.int64),
        user_index=np.asarray(user_indices, dtype=np.int64),
        target_item_index=np.asarray(target_item_indices, dtype=np.int64),
    )


def _manifest_record(
    result: RecoveryResult, vector_shard: str | None, vector_row: int | None
) -> dict[str, Any]:
    if result.status == "recovered":
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "pair_index": result.pair_index,
            "user_id": result.user_id,
            "user_index": result.user_index,
            "target_item_id": result.target_item_id,
            "target_item_index": result.target_item_index,
            "status": "recovered",
            "vector_shard": vector_shard,
            "vector_row": vector_row,
            "selected_attribute_count": len(result.global_indices),
            "zero_signal": result.zero_signal,
            "diagnostics": result.diagnostics,
        }
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "pair_index": result.pair_index,
        "user_id": result.user_id,
        "user_index": result.user_index,
        "target_item_id": result.target_item_id,
        "target_item_index": result.target_item_index,
        "status": "skipped",
        "vector_shard": None,
        "vector_row": None,
        "skip_reason": result.skip_reason,
        "skip_details": result.skip_details,
    }


def _safe_remove_output_dir(path: Path) -> None:
    resolved_path = path.resolve()
    if resolved_path == Path(resolved_path.anchor) or len(resolved_path.parts) < 3:
        raise OmpError(f"Refusing to remove unsafe output directory: {resolved_path}")
    shutil.rmtree(resolved_path)


def _config_json(config: OmpConfig) -> dict[str, Any]:
    payload = asdict(config)
    return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}


def _source_checksums(
    config: OmpConfig, source_records: Sequence[Mapping[str, Any]], tqdm: Any
) -> dict[str, Any]:
    manifest_path = config.intervention_dir / "manifest.jsonl"
    vocabulary_path = config.intervention_dir / "vocabulary.json"
    run_config_path = config.intervention_dir / "run_config.json"
    shard_paths = sorted(
        {
            str(record["shard"])
            for record in source_records
            if isinstance(record.get("shard"), str) and record["shard"]
        }
    )
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "manifest": file_sha256(manifest_path),
        "vocabulary": file_sha256(vocabulary_path),
        "run_config": file_sha256(run_config_path),
        "shards": {
            relative_path: file_sha256(_safe_source_path(config.intervention_dir, relative_path))
            for relative_path in tqdm(
                shard_paths,
                desc="Hashing intervention shards",
                unit="shard",
                dynamic_ncols=True,
                leave=False,
            )
        },
    }


def _resume_configs_match(previous: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    previous_payload = json.loads(json.dumps(previous))
    expected_payload = json.loads(json.dumps(expected))
    for payload in (previous_payload, expected_payload):
        config = payload.get("config")
        if isinstance(config, dict):
            config.pop("pairs_per_shard", None)
    return previous_payload == expected_payload


def _stats_from_manifests(manifests: Sequence[Mapping[str, Any]], summary_path: Path) -> RunStats:
    previous_full_time_seconds = 0.0
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as input_file:
            previous_summary = json.load(input_file)
        previous_full_time_seconds = float(previous_summary.get("full_time_seconds", 0.0))

    stats = RunStats(previous_full_time_seconds=previous_full_time_seconds)
    vector_shards: set[str] = set()
    for record in manifests:
        stats.processed_pair_count += 1
        if record.get("status") == "skipped":
            stats.skipped_pair_count += 1
            reason = str(record.get("skip_reason") or "unknown")
            stats.skip_count_by_reason[reason] = stats.skip_count_by_reason.get(reason, 0) + 1
            continue

        stats.recovered_pair_count += 1
        stats.selected_attribute_total += _coerce_int(
            record.get("selected_attribute_count", 0), "OMP manifest selected_attribute_count"
        )
        if record.get("zero_signal") is True:
            stats.zero_signal_pair_count += 1
        if isinstance(record.get("vector_shard"), str):
            vector_shards.add(record["vector_shard"])
        diagnostics = record.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            continue
        relative_residual = diagnostics.get("relative_residual")
        if isinstance(relative_residual, (int, float)) and math.isfinite(relative_residual):
            stats.relative_residual_total += float(relative_residual)
            stats.relative_residual_count += 1
        r_squared = diagnostics.get("reconstruction_r_squared")
        if isinstance(r_squared, (int, float)) and math.isfinite(r_squared):
            stats.reconstruction_r_squared_total += float(r_squared)
            stats.reconstruction_r_squared_count += 1
    stats.written_shard_count = len(vector_shards)
    return stats


def _prepare_output(
    config: OmpConfig,
    vocabulary: Vocabulary,
    source_checksums: Mapping[str, Any],
    resume: bool,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], RunStats, int]:
    if resume and overwrite:
        raise OmpError("--resume and --overwrite cannot be used together.")
    output_dir = config.output_dir
    manifest_path = output_dir / "manifest.jsonl"
    run_config_path = output_dir / "run_config.json"
    expected_run_config = {"config": _config_json(config), "source_checksums": source_checksums}

    if output_dir.exists() and overwrite:
        _safe_remove_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "shards").mkdir(parents=True, exist_ok=True)

    if resume:
        if not run_config_path.exists():
            raise OmpError("Cannot resume because run_config.json is missing.")
        with run_config_path.open("r", encoding="utf-8") as input_file:
            previous_run_config = json.load(input_file)
        if not _resume_configs_match(previous_run_config, expected_run_config):
            raise OmpError("Cannot resume because source artifacts or OMP settings changed.")
        manifests = load_jsonl(manifest_path) if manifest_path.exists() else []
        stats = _stats_from_manifests(manifests, output_dir / "summary.json")
        shard_paths = {
            record["vector_shard"]
            for record in manifests
            if isinstance(record.get("vector_shard"), str)
        }
        return manifests, stats, len(shard_paths)

    existing_artifacts = [
        path
        for path in (manifest_path, run_config_path, output_dir / "summary.json")
        if path.exists()
    ]
    if existing_artifacts:
        raise OmpError(f"OMP outputs already exist under {output_dir}. Use --resume or --overwrite.")
    _write_json_atomic(run_config_path, expected_run_config)
    _write_json_atomic(
        output_dir / "vocabulary.json",
        {
            "attributes": list(vocabulary.attributes),
            "attribute_to_index": vocabulary.attribute_to_index,
            "source_checksum": source_checksums["vocabulary"],
        },
    )
    _write_jsonl_atomic(manifest_path, [])
    return [], RunStats(), 0


def _record_result(stats: RunStats, result: RecoveryResult) -> None:
    stats.processed_pair_count += 1
    if result.status == "skipped":
        stats.skipped_pair_count += 1
        reason = result.skip_reason or "unknown"
        stats.skip_count_by_reason[reason] = stats.skip_count_by_reason.get(reason, 0) + 1
        return
    stats.recovered_pair_count += 1
    stats.selected_attribute_total += len(result.global_indices)
    if result.zero_signal:
        stats.zero_signal_pair_count += 1
    diagnostics = result.diagnostics or {}
    relative_residual = diagnostics.get("relative_residual")
    if isinstance(relative_residual, (int, float)) and math.isfinite(relative_residual):
        stats.relative_residual_total += float(relative_residual)
        stats.relative_residual_count += 1
    r_squared = diagnostics.get("reconstruction_r_squared")
    if isinstance(r_squared, (int, float)) and math.isfinite(r_squared):
        stats.reconstruction_r_squared_total += float(r_squared)
        stats.reconstruction_r_squared_count += 1


def _summary_payload(stats: RunStats, run_started: float) -> dict[str, Any]:
    full_time_seconds = stats.previous_full_time_seconds + (time.perf_counter() - run_started)
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "processed_pair_count": stats.processed_pair_count,
        "recovered_pair_count": stats.recovered_pair_count,
        "skipped_pair_count": stats.skipped_pair_count,
        "zero_signal_pair_count": stats.zero_signal_pair_count,
        "skip_count_by_reason": stats.skip_count_by_reason,
        "written_shard_count": stats.written_shard_count,
        "selected_attribute_total": stats.selected_attribute_total,
        "relative_residual_total": stats.relative_residual_total,
        "relative_residual_count": stats.relative_residual_count,
        "reconstruction_r_squared_total": stats.reconstruction_r_squared_total,
        "reconstruction_r_squared_count": stats.reconstruction_r_squared_count,
        "average_selected_attribute_count": (
            stats.selected_attribute_total / stats.recovered_pair_count
            if stats.recovered_pair_count
            else 0.0
        ),
        "average_relative_residual": (
            stats.relative_residual_total / stats.relative_residual_count
            if stats.relative_residual_count
            else 0.0
        ),
        "average_reconstruction_r_squared": (
            stats.reconstruction_r_squared_total / stats.reconstruction_r_squared_count
            if stats.reconstruction_r_squared_count
            else None
        ),
        "full_time_seconds": full_time_seconds,
        "average_time_per_pair_seconds": (
            full_time_seconds / stats.processed_pair_count if stats.processed_pair_count else 0.0
        ),
    }


def generate_omp_artifacts(
    config: OmpConfig,
    limit: int | None = None,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    run_started = time.perf_counter()
    if limit is not None and limit < 0:
        raise OmpError("--limit must be greater than or equal to zero.")
    tqdm = _load_tqdm()
    source_manifest_path = config.intervention_dir / "manifest.jsonl"
    vocabulary = load_vocabulary(config.intervention_dir / "vocabulary.json")
    source_records = load_jsonl(source_manifest_path)
    for record_index, record in enumerate(source_records, start=1):
        _record_identity(record, f"Intervention manifest line {record_index}")
    source_checksums = _source_checksums(config, source_records, tqdm)
    manifests, stats, shard_index = _prepare_output(
        config, vocabulary, source_checksums, resume, overwrite
    )
    processed_source_rows = len(manifests)
    target_source_rows = min(len(source_records), limit) if limit is not None else len(source_records)
    shard_cache = InterventionShardCache(
        config.intervention_dir, config.source_shard_cache_max_entries
    )
    chunk: list[RecoveryResult] = []

    def flush_chunk() -> None:
        nonlocal chunk, shard_index, manifests
        if not chunk:
            return
        recovered = [result for result in chunk if result.status == "recovered"]
        relative_path: str | None = None
        if recovered:
            relative_path = f"shards/omp_vectors_{shard_index:06d}.npz"
            write_coefficient_shard(
                config.output_dir / relative_path,
                recovered,
                len(vocabulary.attributes),
            )
            stats.written_shard_count += 1
            shard_index += 1
        vector_row = 0
        for result in chunk:
            if result.status == "recovered":
                manifests.append(_manifest_record(result, relative_path, vector_row))
                vector_row += 1
            else:
                manifests.append(_manifest_record(result, None, None))
        _write_jsonl_atomic(config.output_dir / "manifest.jsonl", manifests)
        _write_json_atomic(config.output_dir / "summary.json", _summary_payload(stats, run_started))
        chunk = []

    with tqdm(
        total=target_source_rows,
        initial=min(processed_source_rows, target_source_rows),
        desc="Running OMP recovery",
        unit="pair",
        dynamic_ncols=True,
    ) as progress:
        for source_index, record in enumerate(source_records):
            if source_index < processed_source_rows:
                continue
            if source_index >= target_source_rows:
                break
            result = recover_pair(record, shard_cache, len(vocabulary.attributes), config)
            _record_result(stats, result)
            chunk.append(result)
            progress.update(1)
            if len(chunk) >= config.pairs_per_shard:
                flush_chunk()
    flush_chunk()

    summary = _summary_payload(stats, run_started)
    _write_json_atomic(config.output_dir / "summary.json", summary)
    return summary
