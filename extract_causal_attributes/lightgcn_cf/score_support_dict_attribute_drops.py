"""Score support-dictionary attributes by exact LightGCN edge removal."""

from __future__ import annotations

import argparse
from pathlib import Path

from lightgcn_cf.artifacts import artifact_paths, load_config
from lightgcn_cf.attribute_perturbation import run_support_dict_attribute_drops


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
        "--support-pkl",
        type=Path,
        default=REPO_ROOT / "support_dict_tst.pkl",
        help="Support dictionary pickle keyed by internal (user_index, item_index).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output pickle path. Defaults to the LightGCN artifact directory.",
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
    parser.add_argument(
        "--save-every-pairs",
        type=int,
        default=1,
        help="Continuously save the output pickle after this many user-item pairs.",
    )
    parser.add_argument(
        "--perturbation-batch-size",
        type=int,
        default=8,
        help="Number of edge-drop graphs to propagate together. Use 1 for sequential mode.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="CPU worker processes that prebuild perturbation batches. Use 4 to overlap CPU and GPU work.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = artifact_paths(config)
    output_path = args.output or paths.base_dir / "support_dict_tst_drop_effects.pkl"
    result = run_support_dict_attribute_drops(
        paths.base_dir,
        args.support_pkl,
        int(args.num_layers if args.num_layers is not None else config.get("num_layers", 3)),
        str(args.device if args.device is not None else config.get("device", "auto")),
        output_path,
        args.save_every_pairs,
        args.perturbation_batch_size,
        args.num_workers,
    )
    print(
        {
            "output": str(output_path),
            "pair_count": len(result),
            "attribute_count": sum(len(attributes) for attributes in result.values()),
        }
    )


if __name__ == "__main__":
    main()
