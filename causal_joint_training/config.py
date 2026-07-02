"""Configuration loading for the standalone trainer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


class ConfigError(ValueError):
    """Raised when standalone trainer configuration is invalid."""


@dataclass(frozen=True)
class PathsConfig:
    id_mappings: Path
    user_embeddings: Path
    item_embeddings: Path
    recommendation_train: Path
    recommendation_validation: Path
    recommendation_test: Path
    explanation_train: Path
    explanation_validation: Path
    explanation_test: Path
    user_profiles: Path
    item_profiles: Path
    omp_dir: Path
    vocabulary: Path
    item_attribute_ids: Path
    attribute_embeddings: Path
    output_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    embedding_dim: int = 64
    extractor_hidden_dim: int = 256
    preference_hidden_dim: int = 128
    preference_dim: int = 64


@dataclass(frozen=True)
class NegativeSamplingConfig:
    candidate_pool_size: int = 100
    predicted_attribute_count: int = 5
    similarity_threshold: float = 0.5
    user_batch_size: int = 128


@dataclass(frozen=True)
class GenerationConfig:
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    local_files_only: bool = False
    load_in_4bit: bool = True
    compute_dtype: str = "float16"
    causal_prompt_tokens: int = 4
    user_prompt_tokens: int = 2
    item_prompt_tokens: int = 2
    prompt_hidden_dim: int = 256
    max_text_tokens: int = 512
    max_new_tokens: int = 128
    instruction: str = "Explain why the user would buy the book within 50 words."


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    device: str = "auto"
    omp_max_relative_residual: float = 0.5
    labeled_validation_ratio: float = 0.1
    causal_warmup_epochs: int = 5
    joint_epochs: int = 10
    recommendation_batch_size: int = 256
    causal_batch_size: int = 128
    generation_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    gradient_clip_norm: float = 1.0
    causal_learning_rate: float = 1.0e-3
    prompt_learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-6
    bpr_weight: float = 1.0
    kl_weight: float = 1.0
    generation_weight: float = 0.1
    evaluation_user_batch_size: int = 128
    alternating_update_order: tuple[str, ...] = ("recommendation", "generation")


@dataclass(frozen=True)
class JointTrainingConfig:
    paths: PathsConfig
    model: ModelConfig
    negative_sampling: NegativeSamplingConfig
    generation: GenerationConfig
    training: TrainingConfig


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"Configuration field {key!r} must be a mapping.")
    return value


def _path(value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Configuration field {context} must be a non-empty path.")
    path = Path(value.strip())
    return path if path.is_absolute() else REPO_ROOT / path


def _integer(payload: Mapping[str, Any], key: str, default: int, *, positive: bool = True) -> int:
    value = payload.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or (positive and value <= 0):
        label = "positive integer" if positive else "integer"
        raise ConfigError(f"Configuration field {key} must be a {label}.")
    return value


def _number(payload: Mapping[str, Any], key: str, default: float, *, nonnegative: bool = True) -> float:
    value = payload.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"Configuration field {key} must be numeric.")
    value = float(value)
    if nonnegative and value < 0:
        raise ConfigError(f"Configuration field {key} must be non-negative.")
    return value


def _string(payload: Mapping[str, Any], key: str, default: str) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Configuration field {key} must be a non-empty string.")
    return value.strip()


def _boolean(payload: Mapping[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"Configuration field {key} must be boolean.")
    return value


def _string_sequence(
    payload: Mapping[str, Any],
    key: str,
    default: tuple[str, ...],
    *,
    allowed: set[str] | None = None,
) -> tuple[str, ...]:
    value = payload.get(key, default)
    if isinstance(value, tuple):
        values = value
    elif isinstance(value, list):
        values = tuple(value)
    else:
        raise ConfigError(f"Configuration field {key} must be a list of strings.")
    if not values or not all(isinstance(item, str) and item.strip() for item in values):
        raise ConfigError(f"Configuration field {key} must contain at least one non-empty string.")
    normalized = tuple(item.strip() for item in values)
    if allowed is not None:
        invalid = sorted(set(normalized) - allowed)
        if invalid:
            raise ConfigError(f"Configuration field {key} contains unsupported stages: {invalid}.")
    return normalized


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> JointTrainingConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load causal_joint_training configuration.") from exc

    with path.open("r", encoding="utf-8") as input_file:
        payload = yaml.safe_load(input_file)
    if not isinstance(payload, Mapping):
        raise ConfigError("Configuration root must be a mapping.")

    paths = _mapping(payload, "paths")
    model = _mapping(payload, "model")
    negative = _mapping(payload, "negative_sampling")
    generation = _mapping(payload, "generation")
    training = _mapping(payload, "training")
    required_paths = (
        "id_mappings", "user_embeddings", "item_embeddings", "recommendation_train",
        "recommendation_validation", "recommendation_test", "explanation_train",
        "explanation_validation", "explanation_test", "user_profiles", "item_profiles",
        "omp_dir", "vocabulary", "item_attribute_ids", "attribute_embeddings", "output_dir",
    )
    resolved_paths = {key: _path(paths.get(key), f"paths.{key}") for key in required_paths}
    threshold = _number(negative, "similarity_threshold", 0.5)
    if threshold < -1.0 or threshold > 1.0:
        raise ConfigError("negative_sampling.similarity_threshold must be in [-1, 1].")
    labeled_validation_ratio = _number(training, "labeled_validation_ratio", 0.1)
    if labeled_validation_ratio <= 0.0 or labeled_validation_ratio >= 1.0:
        raise ConfigError("training.labeled_validation_ratio must be greater than 0 and less than 1.")

    return JointTrainingConfig(
        paths=PathsConfig(**resolved_paths),
        model=ModelConfig(
            embedding_dim=_integer(model, "embedding_dim", 64),
            extractor_hidden_dim=_integer(model, "extractor_hidden_dim", 256),
            preference_hidden_dim=_integer(model, "preference_hidden_dim", 128),
            preference_dim=_integer(model, "preference_dim", 64),
        ),
        negative_sampling=NegativeSamplingConfig(
            candidate_pool_size=_integer(negative, "candidate_pool_size", 100),
            predicted_attribute_count=_integer(negative, "predicted_attribute_count", 5),
            similarity_threshold=threshold,
            user_batch_size=_integer(negative, "user_batch_size", 128),
        ),
        generation=GenerationConfig(
            model_name=_string(generation, "model_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
            local_files_only=_boolean(generation, "local_files_only", False),
            load_in_4bit=_boolean(generation, "load_in_4bit", True),
            compute_dtype=_string(generation, "compute_dtype", "float16"),
            causal_prompt_tokens=_integer(generation, "causal_prompt_tokens", 4),
            user_prompt_tokens=_integer(generation, "user_prompt_tokens", 2),
            item_prompt_tokens=_integer(generation, "item_prompt_tokens", 2),
            prompt_hidden_dim=_integer(generation, "prompt_hidden_dim", 256),
            max_text_tokens=_integer(generation, "max_text_tokens", 512),
            max_new_tokens=_integer(generation, "max_new_tokens", 128),
            instruction=_string(
                generation, "instruction", "Explain why the user would buy the book within 50 words."
            ),
        ),
        training=TrainingConfig(
            seed=_integer(training, "seed", 42, positive=False),
            device=_string(training, "device", "auto"),
            omp_max_relative_residual=_number(training, "omp_max_relative_residual", 0.5),
            labeled_validation_ratio=labeled_validation_ratio,
            causal_warmup_epochs=_integer(training, "causal_warmup_epochs", 5),
            joint_epochs=_integer(training, "joint_epochs", 10),
            recommendation_batch_size=_integer(training, "recommendation_batch_size", 256),
            causal_batch_size=_integer(training, "causal_batch_size", 128),
            generation_batch_size=_integer(training, "generation_batch_size", 1),
            gradient_accumulation_steps=_integer(training, "gradient_accumulation_steps", 8),
            gradient_clip_norm=_number(training, "gradient_clip_norm", 1.0),
            causal_learning_rate=_number(training, "causal_learning_rate", 1.0e-3),
            prompt_learning_rate=_number(training, "prompt_learning_rate", 1.0e-4),
            weight_decay=_number(training, "weight_decay", 1.0e-6),
            bpr_weight=_number(training, "bpr_weight", 1.0),
            kl_weight=_number(training, "kl_weight", 1.0),
            generation_weight=_number(training, "generation_weight", 0.1),
            evaluation_user_batch_size=_integer(training, "evaluation_user_batch_size", 128),
            alternating_update_order=_string_sequence(
                training,
                "alternating_update_order",
                ("recommendation", "generation"),
                allowed={"recommendation", "generation"},
            ),
        ),
    )
