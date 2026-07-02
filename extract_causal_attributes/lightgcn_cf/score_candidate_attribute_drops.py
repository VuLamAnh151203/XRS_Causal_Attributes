"""Score candidate attributes by exact LightGCN support-edge removal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lightgcn_cf.artifacts import artifact_paths, load_config, save_json
from lightgcn_cf.attribute_perturbation import run_candidate_attribute_drops


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="Path to the LightGCN run configuration.",
    )
    parser.add_argument(
        "--candidate-json",
        type=Path,
        default=REPO_ROOT / "candidate_tst_attributes.json",
        help="Candidate attribute JSON keyed by 'user_id,item_id'.",
    )
    parser.add_argument(
        "--support-jsonl",
        type=Path,
        default=REPO_ROOT
        / "extract_causal_attributes"
        / "artifacts"
        / "amazon"
        / "tst_attribute_support.jsonl",
        help="Attribute support JSONL produced for the same candidate pairs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path. Defaults to the LightGCN artifact directory.",
    )
    parser.add_argument(
        "--device",
        help="Device override. Defaults to the device from config.yaml.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        help="LightGCN propagation layer override. Defaults to config.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = artifact_paths(config)
    output_path = args.output or paths.base_dir / "candidate_attribute_drop_effects.json"
    result = run_candidate_attribute_drops(
        paths.base_dir,
        args.candidate_json,
        args.support_jsonl,
        int(args.num_layers if args.num_layers is not None else config.get("num_layers", 3)),
        str(args.device if args.device is not None else config.get("device", "auto")),
    )
    save_json(output_path, result)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "pair_count": len(result),
                "attribute_count": sum(len(attributes) for attributes in result.values()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
