"""Run exact LightGCN propagation after dropping selected user-history edges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lightgcn_cf.artifacts import artifact_paths, load_config, save_json
from lightgcn_cf.perturbation import run_edge_drop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="Path to the YAML run configuration",
    )
    parser.add_argument("--user-id", required=True, help="Raw CSV user ID")
    parser.add_argument(
        "--drop-item-id",
        action="append",
        required=True,
        help="Raw CSV history item ID. Repeat this argument to remove multiple edges.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON path. The result is always printed to stdout.",
    )
    parser.add_argument(
        "--save-full-embeddings",
        type=Path,
        help="Optional filename prefix for all perturbed user and item embeddings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    result = run_edge_drop(
        artifact_paths(config).base_dir,
        args.user_id,
        args.drop_item_id,
        int(config.get("num_layers", 3)),
        args.top_k,
        str(config.get("device", "auto")),
        args.save_full_embeddings,
    )
    if args.output:
        save_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

