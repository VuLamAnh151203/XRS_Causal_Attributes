"""Evaluate item-only LightGCN pair metrics after top-M attribute perturbations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lightgcn_cf.artifacts import artifact_paths, load_config
from lightgcn_cf.pair_metric_perturbation import (
    run_top_m_attribute_perturbation_evaluation,
)


EXTRACT_CAUSAL_ATTRIBUTES_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ATTRIBUTE_ITEMS_MAPPING = (
    EXTRACT_CAUSAL_ATTRIBUTES_ROOT
    / "artifacts"
    / "amazon"
    / "attribute_items_mapping.pkl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config_item_only.yaml"),
        help="Path to the item-only LightGCN configuration.",
    )
    parser.add_argument(
        "--chosen-attributes",
        type=Path,
        help="Pickle mapping raw (user_id, item_id) to sorted chosen attributes.",
    )
    parser.add_argument(
        "--attribute-items-mapping",
        type=Path,
        default=DEFAULT_ATTRIBUTE_ITEMS_MAPPING,
        help="Pickle mapping raw (user_id, item_id) and attribute names to support item IDs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path. Defaults to the item-only artifact directory.",
    )
    parser.add_argument(
        "--max-m",
        type=int,
        default=3,
        help="Evaluate cumulative top-M attribute drops up to this M.",
    )
    parser.add_argument(
        "--perturbation-batch-size",
        type=int,
        default=8,
        help="Number of fixed-degree perturbed graphs to propagate together.",
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
    chosen_attributes = args.chosen_attributes or (
        paths.base_dir / "new_chosen_sorted_attributes.pkl"
    )
    output_path = args.output or (
        paths.base_dir / "top3_cumulative_attribute_perturbation_metrics.json"
    )
    result = run_top_m_attribute_perturbation_evaluation(
        paths.base_dir,
        chosen_attributes,
        args.attribute_items_mapping,
        int(args.num_layers if args.num_layers is not None else config.get("num_layers", 3)),
        tuple(int(value) for value in config.get("recall_k", (10, 20))),
        int(config.get("ndcg_k", 20)),
        args.max_m,
        str(args.device if args.device is not None else config.get("device", "auto")),
        args.perturbation_batch_size,
        output_path,
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "origin_by_m": result["origin_by_m"],
                "perturbed_by_m": result["perturbed_by_m"],
                "delta_by_m": result["delta_by_m"],
                "coverage": result["coverage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
