"""Run causal warm-up followed by standalone joint recommendation and explanation training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .config import DEFAULT_CONFIG_PATH, ConfigError, load_config
from .data import ArtifactError
from .model import ModelError
from .negative_sampling import NegativeSamplingError
from .training import run_training


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = run_training(load_config(args.config))
    except (ArtifactError, ConfigError, ModelError, NegativeSamplingError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary["best"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
