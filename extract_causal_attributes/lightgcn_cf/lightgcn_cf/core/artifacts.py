"""Core configuration and artifact serialization helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

try:
    import yaml
except ImportError as error:  # pragma: no cover - exercised only in incomplete environments
    raise ImportError("PyYAML is required. Install dependencies with: pip install -r requirements.txt") from error

from .data import IdMappings


@dataclass(frozen=True)
class ArtifactPaths:
    base_dir: Path

    @property
    def checkpoint(self) -> Path:
        return self.base_dir / "best_checkpoint.pt"

    @property
    def mappings(self) -> Path:
        return self.base_dir / "id_mappings.json"

    @property
    def train_pairs(self) -> Path:
        return self.base_dir / "train_pairs.pt"

    @property
    def user_history(self) -> Path:
        return self.base_dir / "user_history.json"

    @property
    def run_config(self) -> Path:
        return self.base_dir / "run_config.yaml"

    @property
    def data_summary(self) -> Path:
        return self.base_dir / "data_summary.json"

    @property
    def validation_metrics(self) -> Path:
        return self.base_dir / "validation_metrics.json"

    @property
    def training_history(self) -> Path:
        return self.base_dir / "training_history.json"

    @property
    def user_ego_embeddings(self) -> Path:
        return self.base_dir / "user_ego_embeddings.pt"

    @property
    def item_ego_embeddings(self) -> Path:
        return self.base_dir / "item_ego_embeddings.pt"

    @property
    def user_baseline_embeddings(self) -> Path:
        return self.base_dir / "user_baseline_embeddings.pt"

    @property
    def item_baseline_embeddings(self) -> Path:
        return self.base_dir / "item_baseline_embeddings.pt"


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_path"] = str(path)
    return config


def resolve_config_path(config: dict, key: str) -> Path:
    path = Path(str(config[key]))
    if not path.is_absolute():
        path = Path(config["_config_path"]).parent / path
    return path.resolve()


def artifact_paths(config: dict) -> ArtifactPaths:
    return ArtifactPaths(resolve_config_path(config, "artifact_dir"))


def public_config(config: dict) -> dict:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def save_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_yaml(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)


def save_mappings(path: str | Path, mappings: IdMappings) -> None:
    save_json(path, mappings.to_dict())


def load_mappings(path: str | Path) -> IdMappings:
    return IdMappings.from_dict(load_json(path))


def save_user_history(path: str | Path, history: dict[int, set[int]]) -> None:
    save_json(path, {str(user): sorted(items) for user, items in history.items()})


def load_user_history(path: str | Path) -> dict[int, set[int]]:
    raw = load_json(path)
    return {int(user): {int(item) for item in items} for user, items in raw.items()}


def load_tensor(path: str | Path) -> torch.Tensor:
    return torch.load(Path(path), map_location="cpu")
