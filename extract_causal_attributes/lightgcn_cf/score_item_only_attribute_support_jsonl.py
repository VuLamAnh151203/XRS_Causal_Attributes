"""Score item-only LightGCN attribute drops from support JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lightgcn_cf.artifacts import artifact_paths, load_config
from lightgcn_cf.attribute_perturbation import (
    NULL_METRICS,
    load_attribute_supports,
    load_candidate_attributes,
)
from lightgcn_cf.local_perturbation import LocalPerturbationScorer


EXTRACT_CAUSAL_ATTRIBUTES_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUPPORT_JSONL = (
    EXTRACT_CAUSAL_ATTRIBUTES_ROOT
    / "artifacts"
    / "amazon"
    / "tst_attribute_support.jsonl"
)
PROPAGATION_MODES = ("local-score", "local-lhop", "full")


def derive_candidate_attributes_from_support_jsonl(
    support_jsonl: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    """Build candidate attribute records directly from support JSONL records."""

    candidates: dict[str, list[dict[str, Any]]] = {}
    with Path(support_jsonl).open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                pair_key = f"{record['user_id']},{record['target_item_id']}"
            except KeyError as exc:
                raise ValueError(
                    f"Support record at line {line_number} is missing {exc.args[0]!r}"
                ) from exc
            if pair_key in candidates:
                raise ValueError(f"Duplicate support record for pair {pair_key!r}")

            target_attributes = record.get("target_attributes")
            if target_attributes is None:
                supported = record.get("supported_items_by_attribute", {})
                if not isinstance(supported, dict):
                    raise ValueError(
                        f"Support record for {pair_key!r} has invalid "
                        "supported_items_by_attribute"
                    )
                attribute_names = list(supported)
            elif isinstance(target_attributes, list):
                attribute_names = [str(attribute) for attribute in target_attributes]
            else:
                raise ValueError(
                    f"Support record for {pair_key!r} has invalid target_attributes"
                )

            candidates[pair_key] = [
                {"attr_name": attribute_name}
                for attribute_name in attribute_names
                if attribute_name
            ]

    return candidates


def run_item_only_attribute_support_jsonl(
    config: dict,
    support_jsonl: str | Path,
    candidate_json: str | Path | None = None,
    num_layers: int | None = None,
    device: str | None = None,
    save_path: str | Path | None = None,
    save_steps: int = 1,
    resume: bool = True,
    propagation_mode: str = "local-score",
) -> dict[str, list[dict]]:
    """Score attribute drops against the artifact directory in item-only config."""

    if save_steps <= 0:
        raise ValueError("save_steps must be positive")
    if propagation_mode not in PROPAGATION_MODES:
        raise ValueError("propagation_mode must be 'local-score', 'local-lhop', or 'full'")

    paths = artifact_paths(config)
    resolved_num_layers = int(
        num_layers if num_layers is not None else config.get("num_layers", 3)
    )
    resolved_device = str(device if device is not None else config.get("device", "auto"))
    candidates = (
        load_candidate_attributes(candidate_json)
        if candidate_json is not None
        else derive_candidate_attributes_from_support_jsonl(support_jsonl)
    )
    supports = load_attribute_supports(support_jsonl)
    scorer = LocalPerturbationScorer.from_artifacts(
        paths.base_dir,
        resolved_num_layers,
        resolved_device,
    )

    output_path = Path(save_path) if save_path is not None else None
    output = _load_resume_output(output_path, candidates) if resume else {}
    baseline_cache: dict[tuple[int, int], tuple[float, int]] = {}
    completed_since_save = 0

    from tqdm import tqdm

    pending_pairs = [pair_key for pair_key in candidates if pair_key not in output]
    for pair_key in tqdm(pending_pairs, desc="Processing item-only attributes"):
        attributes = candidates[pair_key]
        pair_results = _score_pair_attributes(
            scorer,
            pair_key,
            attributes,
            supports,
            baseline_cache,
            propagation_mode,
        )
        output[pair_key] = pair_results
        completed_since_save += 1
        if output_path is not None and completed_since_save >= save_steps:
            _save_json_atomic(output_path, _ordered_output(output, candidates))
            completed_since_save = 0

    final_output = _ordered_output(output, candidates)
    if output_path is not None:
        _save_json_atomic(output_path, final_output)
    return final_output


def _score_pair_attributes(
    scorer: LocalPerturbationScorer,
    pair_key: str,
    attributes: list[dict[str, Any]],
    supports: dict[str, dict[str, list[dict[str, Any]]]],
    baseline_cache: dict[tuple[int, int], tuple[float, int]],
    propagation_mode: str,
) -> list[dict]:
    user_id, target_item_id = _parse_pair_key(pair_key)
    if (
        user_id not in scorer.mappings.user_to_index
        or target_item_id not in scorer.mappings.item_to_index
    ):
        return [_attribute_result(attribute) for attribute in attributes]

    user_index = scorer.mappings.user_to_index[user_id]
    target_item_index = scorer.mappings.item_to_index[target_item_id]
    support_by_attribute = supports.get(f"{user_id},{target_item_id}", {})

    if propagation_mode == "local-score":
        return _score_pair_attributes_local_score(
            scorer,
            user_index,
            target_item_index,
            attributes,
            support_by_attribute,
        )

    excluded_items = scorer.train_history.get(user_index, set())
    pair_results: list[dict] = []
    for attribute in attributes:
        attr_name = str(attribute["attr_name"])
        support_items = support_by_attribute.get(attr_name)
        if not support_items:
            pair_results.append(_attribute_result(attribute))
            continue

        drop_indices = _support_item_indices(support_items, scorer.mappings.item_to_index)
        if not drop_indices:
            pair_results.append(_attribute_result(attribute))
            continue

        baseline_key = (user_index, target_item_index)
        if baseline_key not in baseline_cache:
            baseline_cache[baseline_key] = scorer.baseline_score_and_rank(
                user_index,
                target_item_index,
                excluded_items,
            )
        baseline_score, baseline_rank = baseline_cache[baseline_key]
        metrics = scorer.score_drop(
            user_index,
            target_item_index,
            drop_indices,
            excluded_items,
            baseline_score,
            baseline_rank,
            propagation_mode,
        )
        pair_results.append(_attribute_result(attribute, metrics))
    return pair_results


def _score_pair_attributes_local_score(
    scorer: LocalPerturbationScorer,
    user_index: int,
    target_item_index: int,
    attributes: list[dict[str, Any]],
    support_by_attribute: dict[str, list[dict[str, Any]]],
) -> list[dict]:
    pair_results = [_attribute_result(attribute) for attribute in attributes]
    positions: list[int] = []
    drop_groups: list[list[int]] = []

    for position, attribute in enumerate(attributes):
        attr_name = str(attribute["attr_name"])
        support_items = support_by_attribute.get(attr_name)
        if not support_items:
            continue
        drop_indices = _support_item_indices(support_items, scorer.mappings.item_to_index)
        if not drop_indices:
            continue
        positions.append(position)
        drop_groups.append(drop_indices)

    if not drop_groups:
        return pair_results

    metrics_by_position = scorer.score_drop_many(
        user_index,
        target_item_index,
        drop_groups,
        propagation_mode="local-score",
    )
    for position, metrics in zip(positions, metrics_by_position, strict=True):
        pair_results[position] = _attribute_result(attributes[position], metrics)
    return pair_results


def _parse_pair_key(pair_key: str) -> tuple[str, str]:
    parts = [part.strip() for part in pair_key.split(",")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Candidate pair key must be 'user_id,item_id': {pair_key!r}")
    return parts[0], parts[1]


def _support_item_indices(
    support_items: list[dict[str, Any]],
    item_to_index: dict[str, int],
) -> list[int]:
    indices: list[int] = []
    seen: set[int] = set()
    for item in support_items:
        if not isinstance(item, dict):
            raise ValueError(f"Support item must be an object, got {item!r}")
        if "item_index" in item:
            item_index = int(item["item_index"])
        elif "item_id" in item:
            item_key = str(item["item_id"])
            if item_key not in item_to_index:
                raise ValueError(f"Unknown support item ID: {item_key}")
            item_index = item_to_index[item_key]
        else:
            raise ValueError(f"Support item has neither item_index nor item_id: {item!r}")
        if item_index not in seen:
            indices.append(item_index)
            seen.add(item_index)
    return indices


def _attribute_result(
    attribute: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> dict:
    result = {
        "attr_id": attribute.get("attr_id"),
        "attr_name": attribute["attr_name"],
        "baseline_score": None,
        "ratios": None,
    }
    result.update(NULL_METRICS if metrics is None else metrics)
    return result


def _load_resume_output(
    output_path: Path | None,
    candidates: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict]]:
    if output_path is None or not output_path.exists():
        return {}
    with output_path.open("r", encoding="utf-8") as input_file:
        raw_output = json.load(input_file)
    if not isinstance(raw_output, dict):
        raise ValueError(f"Existing output is not a JSON object: {output_path}")
    return {
        pair_key: raw_output[pair_key]
        for pair_key in candidates
        if pair_key in raw_output and isinstance(raw_output[pair_key], list)
    }


def _ordered_output(
    output: dict[str, list[dict]],
    candidates: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict]]:
    return {pair_key: output[pair_key] for pair_key in candidates if pair_key in output}


def _save_json_atomic(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(destination.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
    temporary_path.replace(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config_item_only.yaml"),
        help="Path to the item-only LightGCN run configuration.",
    )
    parser.add_argument(
        "--support-jsonl",
        type=Path,
        default=DEFAULT_SUPPORT_JSONL,
        help="Attribute support JSONL keyed by user_id and target_item_id.",
    )
    parser.add_argument(
        "--candidate-json",
        type=Path,
        help="Optional candidate attribute JSON. If omitted, candidates are derived from support JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path. Defaults to the item-only artifact directory.",
    )
    parser.add_argument(
        "--save-steps",
        "--save_steps",
        type=int,
        default=1,
        dest="save_steps",
        help="Save after this many newly completed user-target pairs.",
    )
    parser.add_argument(
        "--no-resume",
        "--no_resume",
        action="store_true",
        dest="no_resume",
        help="Do not load and skip completed pairs from an existing output JSON.",
    )
    parser.add_argument(
        "--propagation-mode",
        "--propagation_mode",
        choices=PROPAGATION_MODES,
        default="local-score",
        dest="propagation_mode",
        help="Use fast local target-score, exact local L-hop rank, or full-graph propagation.",
    )
    parser.add_argument(
        "--device",
        help="Device override. Defaults to the device from config_item_only.yaml.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        help="LightGCN propagation layer override. Defaults to config_item_only.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = artifact_paths(config)
    output_path = (
        args.output
        or paths.base_dir / "item_only_tst_attribute_drop_effects.json"
    )
    result = run_item_only_attribute_support_jsonl(
        config,
        args.support_jsonl,
        args.candidate_json,
        args.num_layers,
        args.device,
        output_path,
        args.save_steps,
        not args.no_resume,
        args.propagation_mode,
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "pair_count": len(result),
                "attribute_count": sum(len(attributes) for attributes in result.values()),
                "support_jsonl": str(args.support_jsonl),
                "candidate_source": (
                    str(args.candidate_json)
                    if args.candidate_json is not None
                    else "derived_from_support_jsonl"
                ),
                "propagation_mode": args.propagation_mode,
                "save_steps": args.save_steps,
                "resume": not args.no_resume,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
