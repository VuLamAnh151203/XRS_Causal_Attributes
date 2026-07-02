"""Generate explanations from a saved standalone causal joint trainer checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .config import DEFAULT_CONFIG_PATH, ConfigError, load_config
from .data import ArtifactError, load_artifacts
from .evaluation import generate_explanations
from .model import CausalJointModel, FrozenSoftPromptLM, ModelError
from .training import load_checkpoint, resolve_device


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, help="Defaults to paths.output_dir/latest.pt.")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--limit", type=int)
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
        examples = (
            artifacts.explanation_validation if args.split == "validation" else artifacts.explanation_test
        )
        if args.limit is not None:
            if args.limit < 0:
                raise ValueError("--limit must be non-negative.")
            examples = examples[: args.limit]
        records = generate_explanations(model, generator, examples, device)
        output_path = config.paths.output_dir / "generated_explanations.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            for record in records:
                json.dump(record, output_file, ensure_ascii=False)
                output_file.write("\n")
    except (ArtifactError, ConfigError, ModelError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(output_path), "rows": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
