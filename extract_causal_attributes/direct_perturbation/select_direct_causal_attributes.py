"""Select causal attributes directly from perturbation score drops.

This script does not run OMP. It reads direct attribute-drop scores, keeps the
top positive score drops for each user-item pair, and can export the result in
an OMP-compatible artifact layout so the existing joint trainer can load it.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from extract_causal_attributes.id_mappings import load_id_mappings  # noqa: E402


ARTIFACT_SCHEMA_VERSION = 2
DEFAULT_SCORES = Path(
    "extract_causal_attributes/direct_perturbation/artifacts/amazon/direct_attribute_drop_effects.json"
)
DEFAULT_OUTPUT = Path(
    "extract_causal_attributes/direct_perturbation/artifacts/amazon/direct_causal_attributes.jsonl"
)
DEFAULT_SUMMARY = Path(
    "extract_causal_attributes/direct_perturbation/artifacts/amazon/summary.json"
)
DEFAULT_COMPAT_DIR = Path(
    "extract_causal_attributes/direct_perturbation/artifacts/amazon/direct_omp_compatible"
)


@dataclass(frozen=True)
class SelectedAttribute:
    rank: int
    attr_id: int
    attr_name: str
    coefficient: float
    score_drop: float
    rank_drop: int | None
    baseline_score: float | None
    perturbed_score: float | None
    baseline_rank: int | None
    perturbed_rank: int | None
    ratios: float | None


@dataclass(frozen=True)
class PairSelection:
    pair_index: int
    user_id: str
    user_index: int | None
    target_item_id: str
    target_item_index: int | None
    candidate_attribute_count: int
    scored_attribute_count: int
    positive_attribute_count: int
    selected_attributes: tuple[SelectedAttribute, ...]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    temporary_path.replace(path)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as output_file:
        temporary_path = Path(output_file.name)
        for row in rows:
            json.dump(row, output_file, ensure_ascii=False)
            output_file.write("\n")
    temporary_path.replace(path)


def _write_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".npz",
        delete=False,
    ) as output_file:
        temporary_path = Path(output_file.name)
    np.savez_compressed(temporary_path, **arrays)
    temporary_path.replace(path)


def _load_vocabulary(path: Path) -> tuple[str, ...]:
    payload = _load_json(path)
    if isinstance(payload, Mapping):
        return tuple(str(payload[str(index)]) for index in range(len(payload)))
    if isinstance(payload, list):
        return tuple(str(value) for value in payload)
    raise ValueError(f"Vocabulary must be a JSON object or list: {path}")


def _parse_pair_key(pair_key: str) -> tuple[str, str]:
    parts = [part.strip() for part in pair_key.split(",")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Pair key must be 'user_id,item_id': {pair_key!r}")
    return parts[0], parts[1]


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _score_sort_key(attribute: Mapping[str, Any]) -> tuple[float, int, str]:
    score_drop = _finite_float(attribute.get("score_drop"))
    rank_drop = _optional_int(attribute.get("rank_drop"))
    return (
        score_drop if score_drop is not None else -math.inf,
        rank_drop if rank_drop is not None else -10**12,
        str(attribute.get("attr_name", "")),
    )


def _select_pair_attributes(
    pair_key: str,
    pair_index: int,
    raw_attributes: Any,
    vocabulary_index: Mapping[str, int],
    id_mappings: Any,
    top_k: int,
    min_score_drop: float,
) -> PairSelection:
    if not isinstance(raw_attributes, list):
        raise ValueError(f"Direct score value for {pair_key!r} must be a list.")

    user_id, target_item_id = _parse_pair_key(pair_key)
    user_index = id_mappings.user_index(user_id)
    target_item_index = id_mappings.item_index(target_item_id)
    candidates: list[Mapping[str, Any]] = []
    scored_count = 0
    positive_count = 0

    for raw_attribute in raw_attributes:
        if not isinstance(raw_attribute, Mapping):
            raise ValueError(f"Attribute entry for {pair_key!r} must be an object.")
        score_drop = _finite_float(raw_attribute.get("score_drop"))
        if score_drop is None:
            continue
        scored_count += 1
        if score_drop > 0:
            positive_count += 1
        attr_name = str(raw_attribute.get("attr_name", ""))
        if not attr_name or attr_name not in vocabulary_index:
            continue
        if score_drop <= min_score_drop:
            continue
        candidates.append(raw_attribute)

    ordered = sorted(candidates, key=_score_sort_key, reverse=True)[:top_k]
    selected: list[SelectedAttribute] = []
    for rank, attribute in enumerate(ordered, start=1):
        attr_name = str(attribute["attr_name"])
        score_drop = float(attribute["score_drop"])
        selected.append(
            SelectedAttribute(
                rank=rank,
                attr_id=int(vocabulary_index[attr_name]),
                attr_name=attr_name,
                coefficient=score_drop,
                score_drop=score_drop,
                rank_drop=_optional_int(attribute.get("rank_drop")),
                baseline_score=_finite_float(attribute.get("baseline_score")),
                perturbed_score=_finite_float(attribute.get("perturbed_score")),
                baseline_rank=_optional_int(attribute.get("baseline_rank")),
                perturbed_rank=_optional_int(attribute.get("perturbed_rank")),
                ratios=_finite_float(attribute.get("ratios")),
            )
        )

    return PairSelection(
        pair_index=pair_index,
        user_id=user_id,
        user_index=user_index,
        target_item_id=target_item_id,
        target_item_index=target_item_index,
        candidate_attribute_count=len(raw_attributes),
        scored_attribute_count=scored_count,
        positive_attribute_count=positive_count,
        selected_attributes=tuple(selected),
    )


def _selection_record(selection: PairSelection) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pair_index": selection.pair_index,
        "user_id": selection.user_id,
        "user_index": selection.user_index,
        "target_item_id": selection.target_item_id,
        "target_item_index": selection.target_item_index,
        "candidate_attribute_count": selection.candidate_attribute_count,
        "scored_attribute_count": selection.scored_attribute_count,
        "positive_attribute_count": selection.positive_attribute_count,
        "selected_attribute_count": len(selection.selected_attributes),
        "selected_attributes": [asdict(attribute) for attribute in selection.selected_attributes],
    }


def _manifest_record(
    selection: PairSelection, shard_path: str | None, vector_row: int | None
) -> dict[str, Any]:
    if (
        selection.user_index is None
        or selection.target_item_index is None
        or not selection.selected_attributes
    ):
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "pair_index": selection.pair_index,
            "user_id": selection.user_id,
            "user_index": selection.user_index,
            "target_item_id": selection.target_item_id,
            "target_item_index": selection.target_item_index,
            "status": "skipped",
            "vector_shard": None,
            "vector_row": None,
            "skip_reason": "no_direct_selected_attributes",
            "skip_details": {
                "selected_attribute_count": len(selection.selected_attributes),
                "mapped_user": selection.user_index is not None,
                "mapped_target_item": selection.target_item_index is not None,
            },
        }

    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "pair_index": selection.pair_index,
        "user_id": selection.user_id,
        "user_index": selection.user_index,
        "target_item_id": selection.target_item_id,
        "target_item_index": selection.target_item_index,
        "status": "recovered",
        "vector_shard": shard_path,
        "vector_row": vector_row,
        "selected_attribute_count": len(selection.selected_attributes),
        "zero_signal": False,
        "diagnostics": {
            "source": "direct_perturbation",
            "relative_residual": 0.0,
            "scored_attribute_count": selection.scored_attribute_count,
            "positive_attribute_count": selection.positive_attribute_count,
        },
    }


def _write_omp_compatible_artifacts(
    output_dir: Path,
    selections: Sequence[PairSelection],
    vocabulary: Sequence[str],
    overwrite: bool,
) -> dict[str, Any]:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    (output_dir / "shards").mkdir(parents=True, exist_ok=True)

    recovered = [
        selection
        for selection in selections
        if selection.user_index is not None
        and selection.target_item_index is not None
        and selection.selected_attributes
    ]
    shard_relative = "shards/direct_vectors_000000.npz"
    coef_data: list[float] = []
    coef_indices: list[int] = []
    coef_indptr = [0]
    pair_indices: list[int] = []
    user_indices: list[int] = []
    target_item_indices: list[int] = []

    for selection in recovered:
        coef_data.extend(attribute.coefficient for attribute in selection.selected_attributes)
        coef_indices.extend(attribute.attr_id for attribute in selection.selected_attributes)
        coef_indptr.append(len(coef_data))
        pair_indices.append(selection.pair_index)
        user_indices.append(int(selection.user_index))
        target_item_indices.append(int(selection.target_item_index))

    _write_npz_atomic(
        output_dir / shard_relative,
        coef_data=np.asarray(coef_data, dtype=np.float32),
        coef_indices=np.asarray(coef_indices, dtype=np.int64),
        coef_indptr=np.asarray(coef_indptr, dtype=np.int64),
        coef_shape=np.asarray([len(recovered), len(vocabulary)], dtype=np.int64),
        pair_index=np.asarray(pair_indices, dtype=np.int64),
        user_index=np.asarray(user_indices, dtype=np.int64),
        target_item_index=np.asarray(target_item_indices, dtype=np.int64),
    )

    vector_rows = {selection.pair_index: row for row, selection in enumerate(recovered)}
    manifest = [
        _manifest_record(
            selection,
            shard_relative if selection.pair_index in vector_rows else None,
            vector_rows.get(selection.pair_index),
        )
        for selection in selections
    ]
    _write_jsonl_atomic(output_dir / "manifest.jsonl", manifest)
    _write_json_atomic(
        output_dir / "vocabulary.json",
        {str(index): attribute for index, attribute in enumerate(vocabulary)},
    )
    run_config = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "mode": "direct_perturbation",
        "coefficient_source": "score_drop",
    }
    _write_json_atomic(output_dir / "run_config.json", run_config)
    summary = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "mode": "direct_perturbation",
        "pair_count": len(selections),
        "recovered_pair_count": len(recovered),
        "skipped_pair_count": len(selections) - len(recovered),
        "selected_attribute_total": sum(len(selection.selected_attributes) for selection in recovered),
        "written_shard_count": 1,
    }
    _write_json_atomic(output_dir / "summary.json", summary)
    return summary


def select_direct_causal_attributes(
    scores_path: Path,
    vocabulary_path: Path,
    id_mappings_path: Path,
    output_path: Path,
    summary_path: Path,
    compat_output_dir: Path | None,
    top_k: int,
    min_score_drop: float,
    overwrite: bool,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("--top-k must be positive.")
    scores = _load_json(scores_path)
    if not isinstance(scores, Mapping):
        raise ValueError("Direct score file must be a JSON object keyed by user-item pair.")
    vocabulary = _load_vocabulary(vocabulary_path)
    vocabulary_index = {attribute: index for index, attribute in enumerate(vocabulary)}
    id_mappings = load_id_mappings(id_mappings_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f"Summary already exists: {summary_path}")

    selections = [
        _select_pair_attributes(
            str(pair_key),
            pair_index,
            raw_attributes,
            vocabulary_index,
            id_mappings,
            top_k,
            min_score_drop,
        )
        for pair_index, (pair_key, raw_attributes) in enumerate(scores.items())
    ]
    records = [_selection_record(selection) for selection in selections]
    _write_jsonl_atomic(output_path, records)

    summary = {
        "schema_version": 1,
        "mode": "direct_perturbation",
        "scores": str(scores_path),
        "vocabulary": str(vocabulary_path),
        "id_mappings": str(id_mappings_path),
        "output": str(output_path),
        "top_k": top_k,
        "min_score_drop": min_score_drop,
        "pair_count": len(selections),
        "selected_pair_count": sum(1 for selection in selections if selection.selected_attributes),
        "selected_attribute_total": sum(len(selection.selected_attributes) for selection in selections),
        "scored_attribute_total": sum(selection.scored_attribute_count for selection in selections),
        "positive_attribute_total": sum(selection.positive_attribute_count for selection in selections),
    }

    if compat_output_dir is not None:
        summary["omp_compatible"] = _write_omp_compatible_artifacts(
            compat_output_dir,
            selections,
            vocabulary,
            overwrite,
        )
        summary["omp_compatible_dir"] = str(compat_output_dir)

    _write_json_atomic(summary_path, summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=Path("attribute_pipeline/outputs/amazon/vocabulary.json"),
    )
    parser.add_argument(
        "--id-mappings",
        type=Path,
        default=Path("extract_causal_attributes/lightgcn_cf/artifacts/amazon/id_mappings.json"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--omp-compatible-output-dir",
        type=Path,
        default=DEFAULT_COMPAT_DIR,
        help="Write direct labels in the OMP artifact layout consumed by causal_joint_training.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score-drop", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        summary = select_direct_causal_attributes(
            scores_path=args.scores,
            vocabulary_path=args.vocabulary,
            id_mappings_path=args.id_mappings,
            output_path=args.output,
            summary_path=args.summary_output,
            compat_output_dir=args.omp_compatible_output_dir,
            top_k=args.top_k,
            min_score_drop=args.min_score_drop,
            overwrite=args.overwrite,
        )
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
