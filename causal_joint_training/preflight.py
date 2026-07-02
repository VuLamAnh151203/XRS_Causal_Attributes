"""Validate standalone trainer artifacts without loading or downloading an LLM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .config import DEFAULT_CONFIG_PATH, ConfigError, load_config
from .data import ArtifactError, load_artifacts
from .training import build_labeled_pair_split, examples_to_evaluation_pairs, merge_history_with_examples


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        artifacts = load_artifacts(config)
        labeled_split = build_labeled_pair_split(
            artifacts.explanation_train,
            artifacts.causal_labels,
            validation_ratio=config.training.labeled_validation_ratio,
            seed=config.training.seed,
        )
        training_history = merge_history_with_examples(
            artifacts.recommendation.train_history,
            labeled_split.train_examples,
        )
        _, validation_stats = examples_to_evaluation_pairs(
            labeled_split.validation_examples,
            training_history,
        )
    except (ArtifactError, ConfigError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                **artifacts.stats,
                "labeled_pair_split": {**labeled_split.stats, **validation_stats},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
