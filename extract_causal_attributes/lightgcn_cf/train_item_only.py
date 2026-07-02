"""Train ItemOnlyLightGCN and export graph-independent ego embeddings."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from lightgcn_cf.artifacts import (
    artifact_paths,
    load_config,
    public_config,
    resolve_config_path,
    save_json,
    save_mappings,
    save_user_history,
    save_yaml,
)
from lightgcn_cf.data import load_dataset, sample_negative_items
from lightgcn_cf.graph import build_normalized_adjacency
from lightgcn_cf.item_only_model import ItemOnlyLightGCN
from lightgcn_cf.metrics import evaluate
from train import resolve_device, set_seed


MODEL_VARIANT = "item_only_fixed_random_users"


def run_item_only_training(config: dict) -> dict:
    """Train with frozen random user embeddings and trainable item embeddings."""

    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = resolve_device(str(config.get("device", "auto")))
    paths = artifact_paths(config)
    paths.base_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        resolve_config_path(config, "train_csv"),
        resolve_config_path(config, "validation_csv"),
        config.get("user_column"),
        config.get("item_column"),
    )
    if dataset.train_pairs.shape[0] == 0:
        raise ValueError("Training data contains no user-item interactions")
    if dataset.mappings.num_items < 2:
        raise ValueError("LightGCN BPR training requires at least two distinct items")

    saved_config = public_config(config)
    saved_config["model_variant"] = MODEL_VARIANT
    saved_config["user_column"] = dataset.summary["user_column"]
    saved_config["item_column"] = dataset.summary["item_column"]
    save_yaml(paths.run_config, saved_config)
    save_json(paths.data_summary, dataset.summary)
    save_mappings(paths.mappings, dataset.mappings)
    save_user_history(paths.user_history, dataset.train_history)
    torch.save(dataset.train_pairs.cpu(), paths.train_pairs)

    adjacency = build_normalized_adjacency(
        dataset.mappings.num_users,
        dataset.mappings.num_items,
        dataset.train_pairs,
        device,
    )
    model = ItemOnlyLightGCN(
        dataset.mappings.num_users,
        dataset.mappings.num_items,
        int(config.get("embedding_dim", 64)),
        int(config.get("num_layers", 3)),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.item_embedding.parameters(),
        lr=float(config.get("learning_rate", 0.001)),
    )
    rng = random.Random(seed)
    batch_size = int(config.get("batch_size", 2048))
    epochs = int(config.get("epochs", 300))
    evaluation_every = int(config.get("evaluation_every", 10))
    patience = int(config.get("early_stopping_patience", 20))
    recall_k = tuple(int(value) for value in config.get("recall_k", (10, 20)))
    ndcg_k = int(config.get("ndcg_k", 20))
    metric_user_batch_size = int(config.get("evaluation_user_batch_size", 256))
    l2_regularization = float(config.get("l2_regularization", 0.0001))
    if min(batch_size, epochs, evaluation_every, patience) <= 0:
        raise ValueError("batch_size, epochs, evaluation_every, and patience must be positive")

    eligible_mask = torch.tensor(
        [
            len(dataset.train_history[int(user)]) < dataset.mappings.num_items
            for user in dataset.train_pairs[:, 0].tolist()
        ],
        dtype=torch.bool,
    )
    training_pairs = dataset.train_pairs[eligible_mask]
    if training_pairs.shape[0] == 0:
        raise ValueError("No train pairs have an available negative item for BPR sampling")
    dataset.summary["train_pairs_skipped_no_negative_item"] = int(
        dataset.train_pairs.shape[0] - training_pairs.shape[0]
    )
    save_json(paths.data_summary, dataset.summary)

    best_score = (-float("inf"), -float("inf"))
    best_metrics: dict = {}
    best_epoch = 0
    evaluations_without_improvement = 0
    training_history: list[dict] = []
    from tqdm import tqdm

    for epoch in tqdm(range(1, epochs + 1), desc="Training item-only"):
        model.train()
        permutation = torch.randperm(training_pairs.shape[0])
        epoch_loss = 0.0
        epoch_ranking_loss = 0.0
        epoch_regularization = 0.0
        batches = 0
        for start in range(0, training_pairs.shape[0], batch_size):
            batch = training_pairs[permutation[start : start + batch_size]]
            users_cpu = batch[:, 0]
            positives_cpu = batch[:, 1]
            negatives_cpu = sample_negative_items(
                users_cpu, dataset.mappings.num_items, dataset.train_history, rng
            )
            optimizer.zero_grad()
            loss, ranking_loss, regularization_loss = model.bpr_loss(
                adjacency,
                users_cpu.to(device),
                positives_cpu.to(device),
                negatives_cpu.to(device),
                l2_regularization,
            )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            epoch_ranking_loss += float(ranking_loss.cpu())
            epoch_regularization += float(regularization_loss.cpu())
            batches += 1

        record = {
            "epoch": epoch,
            "loss": epoch_loss / batches,
            "ranking_loss": epoch_ranking_loss / batches,
            "regularization_loss": epoch_regularization / batches,
        }
        if epoch % evaluation_every == 0:
            metrics = evaluate(
                model,
                adjacency,
                dataset.validation_pairs,
                dataset.train_history,
                recall_k,
                ndcg_k,
                metric_user_batch_size,
            )
            record.update(metrics)
            score = (
                float(metrics[f"ndcg@{ndcg_k}"]),
                float(metrics.get("recall@20", metrics[f"recall@{max(recall_k)}"])),
            )
            if score > best_score:
                best_score = score
                best_metrics = metrics
                best_epoch = epoch
                evaluations_without_improvement = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "metrics": metrics,
                        "model_variant": MODEL_VARIANT,
                        "model_state_dict": model.state_dict(),
                    },
                    paths.checkpoint,
                )
            else:
                evaluations_without_improvement += 1
        training_history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if evaluations_without_improvement >= patience:
            break

    if not paths.checkpoint.exists():
        raise RuntimeError("Training completed without producing a checkpoint")
    checkpoint = torch.load(paths.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        user_ego = model.user_embedding.weight.detach().cpu()
        item_ego = model.item_embedding.weight.detach().cpu()
        baseline_users, baseline_items = model(adjacency)
        torch.save(user_ego, paths.user_ego_embeddings)
        torch.save(item_ego, paths.item_ego_embeddings)
        torch.save(baseline_users.detach().cpu(), paths.user_baseline_embeddings)
        torch.save(baseline_items.detach().cpu(), paths.item_baseline_embeddings)

    result = {
        "best_epoch": best_epoch,
        **best_metrics,
        "device": str(device),
        "model_variant": MODEL_VARIANT,
        "artifacts": str(paths.base_dir),
    }
    save_json(paths.training_history, training_history)
    save_json(paths.validation_metrics, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config_item_only.yaml"),
        help="Path to the YAML run configuration",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_item_only_training(load_config(args.config))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
