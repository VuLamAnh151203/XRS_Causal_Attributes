"""Evaluate a saved standalone causal joint trainer checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .config import DEFAULT_CONFIG_PATH, ConfigError, load_config
from .data import ArtifactError, load_artifacts
from .evaluation import evaluate_causal, evaluate_generation_loss, evaluate_recommendation
from .model import CausalJointModel, FrozenSoftPromptLM, ModelError
from .training import (
    build_labeled_pair_split,
    examples_to_evaluation_pairs,
    load_checkpoint,
    merge_history_with_examples,
    resolve_device,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, help="Defaults to paths.output_dir/latest.pt.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        artifacts = load_artifacts(config)
        device = resolve_device(config.training.device)
        model = CausalJointModel(
            artifacts.user_embeddings, artifacts.item_embeddings, len(artifacts.vocabulary), config.model
        ).to(device)
        generator = FrozenSoftPromptLM.from_pretrained(
            config.generation, config.model.embedding_dim, config.model.preference_dim, device=device
        )
        if not config.generation.load_in_4bit:
            generator = generator.to(device)
        generator.move_prompt_generators(device)
        load_checkpoint(args.checkpoint or config.paths.output_dir / "latest.pt", model, generator)
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
        recommendation_validation_pairs, recommendation_validation_stats = examples_to_evaluation_pairs(
            labeled_split.validation_examples,
            training_history,
        )
        metrics = {
            "causal_validation": evaluate_causal(
                model, labeled_split.validation_labels, config.training.causal_batch_size, device
            ),
            "recommendation_validation": evaluate_recommendation(
                model,
                recommendation_validation_pairs,
                training_history,
                device,
            ),
            "generation_validation": evaluate_generation_loss(
                model, generator, labeled_split.validation_examples, device
            ),
            "labeled_pair_split": labeled_split.stats,
        }
        metrics["recommendation_validation"].update(recommendation_validation_stats)
    except (ArtifactError, ConfigError, ModelError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
