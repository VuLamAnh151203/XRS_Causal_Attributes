"""Evaluate the saved LightGCN checkpoint against validation interactions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lightgcn_cf.artifacts import (
    artifact_paths,
    load_config,
    load_mappings,
    load_tensor,
    load_user_history,
    resolve_config_path,
)
from lightgcn_cf.data import load_validation_for_mappings
from lightgcn_cf.graph import build_normalized_adjacency
from lightgcn_cf.metrics import evaluate
from lightgcn_cf.model import LightGCN
from train import resolve_device


def run_evaluation(config: dict) -> dict:
    device = resolve_device(str(config.get("device", "auto")))
    paths = artifact_paths(config)
    mappings = load_mappings(paths.mappings)
    train_pairs = load_tensor(paths.train_pairs).long()
    train_history = load_user_history(paths.user_history)
    validation_pairs, validation_summary = load_validation_for_mappings(
        resolve_config_path(config, "validation_csv"),
        mappings,
        train_history,
        config.get("user_column"),
        config.get("item_column"),
    )
    adjacency = build_normalized_adjacency(
        mappings.num_users, mappings.num_items, train_pairs, device
    )
    model = LightGCN(
        mappings.num_users,
        mappings.num_items,
        int(config.get("embedding_dim", 64)),
        int(config.get("num_layers", 3)),
    ).to(device)
    checkpoint = torch.load(paths.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate(
        model,
        adjacency,
        validation_pairs,
        train_history,
        tuple(int(value) for value in config.get("recall_k", (10, 20))),
        int(config.get("ndcg_k", 20)),
        int(config.get("evaluation_user_batch_size", 256)),
    )
    return {**metrics, "validation_data": validation_summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="Path to the YAML run configuration",
    )
    parser.add_argument(
        "--split",
        choices=("val",),
        default="val",
        help="Evaluation split. Only validation is supported to keep test data untouched.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(run_evaluation(load_config(args.config)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

