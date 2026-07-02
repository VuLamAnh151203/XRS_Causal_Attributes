"""Recover sparse global attribute vectors from LightGCN intervention artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from extract_causal_attributes.intervention.omp.core import (  # noqa: E402
    OmpError,
    generate_omp_artifacts,
    load_config,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "omp_config.yaml"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--limit", type=int, help="Process only the first N intervention pairs.")
    parser.add_argument("--resume", action="store_true", help="Resume a compatible interrupted run.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing OMP outputs and start a fresh run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        summary = generate_omp_artifacts(
            config=config,
            limit=args.limit,
            resume=args.resume,
            overwrite=args.overwrite,
        )
    except (OmpError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
