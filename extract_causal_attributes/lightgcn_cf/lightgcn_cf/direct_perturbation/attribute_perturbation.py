"""Direct batch attribute-level edge-drop counterfactual scoring."""

from __future__ import annotations

import json
import pickle
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import torch

from ..core.artifacts import ArtifactPaths, load_mappings, load_tensor, load_user_history
from ..core.graph import build_normalized_adjacency, propagate


NULL_METRICS = {
    "score_drop": None,
    "baseline_rank": None,
    "perturbed_rank": None,
    "rank_drop": None,
}

_WORKER_TRAIN_PAIRS: torch.Tensor | None = None
_WORKER_TRAIN_HISTORY: dict[int, set[int]] | None = None
_WORKER_NUM_USERS: int | None = None
_WORKER_NUM_ITEMS: int | None = None


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _parse_pair_key(pair_key: str) -> tuple[str, str]:
    parts = [part.strip() for part in pair_key.split(",")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Candidate pair key must be 'user_id,item_id': {pair_key!r}")
    return parts[0], parts[1]


def load_candidate_attributes(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    with Path(path).open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, dict):
        raise ValueError("Candidate attributes must be a JSON object keyed by user-item pair")
    candidates: dict[str, list[dict[str, Any]]] = {}
    for pair_key, attributes in payload.items():
        if not isinstance(attributes, list):
            raise ValueError(f"Candidate value for {pair_key!r} must be a list")
        normalized_attributes: list[dict[str, Any]] = []
        for index, attribute in enumerate(attributes):
            if not isinstance(attribute, dict):
                raise ValueError(f"Candidate attribute {pair_key!r}[{index}] must be an object")
            if "attr_name" not in attribute:
                raise ValueError(f"Candidate attribute {pair_key!r}[{index}] has no attr_name")
            normalized_attributes.append(attribute)
        candidates[str(pair_key)] = normalized_attributes
    return candidates


def load_attribute_supports(path: str | Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    supports: dict[str, dict[str, list[dict[str, Any]]]] = {}
    with Path(path).open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                pair_key = f"{record['user_id']},{record['target_item_id']}"
            except KeyError as exc:
                raise ValueError(
                    f"Support record at line {line_number} is missing {exc.args[0]!r}"
                ) from exc
            if pair_key in supports:
                raise ValueError(f"Duplicate support record for pair {pair_key!r}")
            attribute_supports = record.get("supported_items_by_attribute", {})
            if not isinstance(attribute_supports, dict):
                raise ValueError(
                    f"Support record for {pair_key!r} has invalid supported_items_by_attribute"
                )
            supports[pair_key] = attribute_supports
    return supports


def load_support_dict(path: str | Path) -> dict[tuple[int, int], dict[int, dict[str, Any]]]:
    with Path(path).open("rb") as input_file:
        payload = pickle.load(input_file)
    if not isinstance(payload, dict):
        raise ValueError("Support pickle must contain a dictionary")

    support_dict: dict[tuple[int, int], dict[int, dict[str, Any]]] = {}
    for pair_key, attributes in payload.items():
        if not isinstance(pair_key, tuple) or len(pair_key) != 2:
            raise ValueError(f"Support pickle pair key must be (user_index, item_index): {pair_key!r}")
        if not isinstance(attributes, dict):
            raise ValueError(f"Support pickle value for {pair_key!r} must be a dictionary")
        user_index, item_index = (int(pair_key[0]), int(pair_key[1]))
        normalized_attributes: dict[int, dict[str, Any]] = {}
        for attr_id, record in attributes.items():
            if not isinstance(record, dict):
                raise ValueError(f"Support record for {pair_key!r}/{attr_id!r} must be a dictionary")
            if "attr_name" not in record:
                raise ValueError(f"Support record for {pair_key!r}/{attr_id!r} has no attr_name")
            normalized_attributes[int(attr_id)] = record
        support_dict[(user_index, item_index)] = normalized_attributes
    return support_dict


def _minimal_result(attribute: dict[str, Any], metrics: dict[str, Any] | None = None) -> dict:
    result = {
        "attr_id": attribute.get("attr_id"),
        "attr_name": attribute["attr_name"],
    }
    result.update(NULL_METRICS if metrics is None else metrics)
    return result


def _minimal_null_results(attributes: list[dict[str, Any]]) -> list[dict]:
    return [_minimal_result(attribute) for attribute in attributes]


def _support_dict_result(record: dict[str, Any], metrics: dict[str, Any] | None = None) -> dict:
    result = {
        "attr_name": record["attr_name"],
        "candidate_score": record.get("candidate_score"),
    }
    result.update(NULL_METRICS if metrics is None else metrics)
    return result


def _save_pickle(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(destination.name + ".tmp")
    with temporary_path.open("wb") as output_file:
        pickle.dump(value, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(destination)


def _support_item_indices(
    support_items: list[dict[str, Any]],
    item_to_index: dict[str, int],
) -> list[int]:
    indices: list[int] = []
    seen: set[int] = set()
    for item in support_items:
        if not isinstance(item, dict):
            raise ValueError(f"Support item must be an object, got {item!r}")
        if "item_index" in item:
            item_index = int(item["item_index"])
        elif "item_id" in item:
            item_key = str(item["item_id"])
            if item_key not in item_to_index:
                raise ValueError(f"Unknown support item ID: {item_key}")
            item_index = item_to_index[item_key]
        else:
            raise ValueError(f"Support item has neither item_index nor item_id: {item!r}")
        if item_index not in seen:
            indices.append(item_index)
            seen.add(item_index)
    return indices


def _internal_support_item_indices(support_items: Any, num_items: int) -> list[int]:
    if support_items is None:
        return []
    if not isinstance(support_items, list):
        raise ValueError("support_items must be a list of internal item indices")

    indices: list[int] = []
    seen: set[int] = set()
    for raw_item in support_items:
        item_index = int(raw_item)
        if item_index < 0 or item_index >= num_items:
            raise ValueError(f"Support item index is out of range: {item_index}")
        if item_index not in seen:
            indices.append(item_index)
            seen.add(item_index)
    return indices


def _drop_user_edges(
    train_pairs: torch.Tensor,
    train_history: dict[int, set[int]],
    user_index: int,
    item_indices: list[int],
) -> torch.Tensor:
    unique_items = sorted(set(int(item) for item in item_indices))
    if not unique_items:
        raise ValueError("At least one item edge must be selected for removal")
    user_history = train_history.get(user_index, set())
    missing = [item for item in unique_items if item not in user_history]
    if missing:
        raise ValueError(
            f"Cannot drop item indices {missing}: they are not history edges for user index "
            f"{user_index}"
        )
    drop_items = torch.tensor(unique_items, dtype=train_pairs.dtype)
    drop_mask = train_pairs[:, 0].eq(user_index) & torch.isin(train_pairs[:, 1], drop_items)
    return train_pairs[~drop_mask.cpu()]


def _score_and_rank(
    user_embedding: torch.Tensor,
    item_embeddings: torch.Tensor,
    target_item_index: int,
    excluded_items: set[int],
) -> tuple[float, int]:
    if target_item_index in excluded_items:
        raise ValueError(
            f"Target item index {target_item_index} is in the user's training history"
        )
    scores = item_embeddings @ user_embedding
    candidate_mask = torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)
    if excluded_items:
        excluded = torch.tensor(sorted(excluded_items), dtype=torch.long, device=scores.device)
        candidate_mask[excluded] = False
    candidates = torch.arange(scores.shape[0], device=scores.device)[candidate_mask]
    candidate_scores = scores[candidate_mask]
    ranked_items = candidates[torch.argsort(candidate_scores, descending=True)]
    matches = (ranked_items == int(target_item_index)).nonzero(as_tuple=False)
    if matches.numel() == 0:
        raise ValueError(f"Target item index {target_item_index} was not rankable")
    rank = int(matches[0].item()) + 1
    score = float(scores[target_item_index].detach().cpu())
    return score, rank


def _build_perturbation_batch_from_data(
    train_pairs: torch.Tensor,
    train_history: dict[int, set[int]],
    num_users: int,
    num_items: int,
    batch_tasks: list[tuple[int, int, tuple[int, ...], float, int]],
) -> tuple[list[tuple[int, int, tuple[int, ...], float, int]], torch.Tensor, torch.Tensor]:
    num_nodes = num_users + num_items
    adjacency_indices: list[torch.Tensor] = []
    adjacency_values: list[torch.Tensor] = []
    for offset, (user_index, _target_item_index, drop_set, _score, _rank) in enumerate(
        batch_tasks
    ):
        perturbed_pairs = _drop_user_edges(
            train_pairs, train_history, user_index, list(drop_set)
        )
        adjacency = build_normalized_adjacency(
            num_users,
            num_items,
            perturbed_pairs,
        )
        adjacency_indices.append(adjacency.indices() + offset * num_nodes)
        adjacency_values.append(adjacency.values())

    return batch_tasks, torch.cat(adjacency_indices, dim=1), torch.cat(adjacency_values)


def _init_perturbation_batch_worker(
    train_pairs: torch.Tensor,
    train_history: dict[int, set[int]],
    num_users: int,
    num_items: int,
) -> None:
    global _WORKER_TRAIN_PAIRS
    global _WORKER_TRAIN_HISTORY
    global _WORKER_NUM_USERS
    global _WORKER_NUM_ITEMS
    _WORKER_TRAIN_PAIRS = train_pairs
    _WORKER_TRAIN_HISTORY = train_history
    _WORKER_NUM_USERS = num_users
    _WORKER_NUM_ITEMS = num_items


def _build_perturbation_batch_worker(
    batch_tasks: list[tuple[int, int, tuple[int, ...], float, int]]
) -> tuple[list[tuple[int, int, tuple[int, ...], float, int]], torch.Tensor, torch.Tensor]:
    if (
        _WORKER_TRAIN_PAIRS is None
        or _WORKER_TRAIN_HISTORY is None
        or _WORKER_NUM_USERS is None
        or _WORKER_NUM_ITEMS is None
    ):
        raise RuntimeError("Perturbation batch worker was not initialized")
    return _build_perturbation_batch_from_data(
        _WORKER_TRAIN_PAIRS,
        _WORKER_TRAIN_HISTORY,
        _WORKER_NUM_USERS,
        _WORKER_NUM_ITEMS,
        batch_tasks,
    )


def _batched_support_dict_metrics(
    train_pairs: torch.Tensor,
    train_history: dict[int, set[int]],
    mappings: Any,
    ego_embeddings: torch.Tensor,
    user_index: int,
    target_item_index: int,
    excluded_items: set[int],
    drop_sets: list[tuple[int, ...]],
    baseline_score: float,
    baseline_rank: int,
    num_layers: int,
    device: torch.device,
    perturbation_batch_size: int,
) -> dict[tuple[int, ...], dict[str, Any]]:
    num_nodes = mappings.num_users + mappings.num_items
    metrics_by_drop_set: dict[tuple[int, ...], dict[str, Any]] = {}

    for start in range(0, len(drop_sets), perturbation_batch_size):
        batch_drop_sets = drop_sets[start : start + perturbation_batch_size]
        adjacency_indices: list[torch.Tensor] = []
        adjacency_values: list[torch.Tensor] = []
        for offset, drop_set in enumerate(batch_drop_sets):
            perturbed_pairs = _drop_user_edges(
                train_pairs, train_history, user_index, list(drop_set)
            )
            adjacency = build_normalized_adjacency(
                mappings.num_users,
                mappings.num_items,
                perturbed_pairs,
                device,
            )
            adjacency_indices.append(adjacency.indices() + offset * num_nodes)
            adjacency_values.append(adjacency.values())

        batched_adjacency = torch.sparse_coo_tensor(
            torch.cat(adjacency_indices, dim=1),
            torch.cat(adjacency_values),
            (len(batch_drop_sets) * num_nodes, len(batch_drop_sets) * num_nodes),
            device=device,
        ).coalesce()
        batched_ego = ego_embeddings.repeat(len(batch_drop_sets), 1)
        perturbed_all = propagate(batched_ego, batched_adjacency, num_layers)

        for offset, drop_set in enumerate(batch_drop_sets):
            node_offset = offset * num_nodes
            perturbed_score, perturbed_rank = _score_and_rank(
                perturbed_all[node_offset + user_index],
                perturbed_all[
                    node_offset + mappings.num_users : node_offset
                    + mappings.num_users
                    + mappings.num_items
                ],
                target_item_index,
                excluded_items,
            )
            metrics_by_drop_set[drop_set] = {
                "score_drop": baseline_score - perturbed_score,
                "baseline_rank": baseline_rank,
                "perturbed_rank": perturbed_rank,
                "rank_drop": perturbed_rank - baseline_rank,
            }

    return metrics_by_drop_set


def _batched_support_dict_task_metrics(
    train_pairs: torch.Tensor,
    train_history: dict[int, set[int]],
    mappings: Any,
    ego_embeddings: torch.Tensor,
    tasks: list[tuple[int, int, tuple[int, ...], float, int]],
    num_layers: int,
    device: torch.device,
    perturbation_batch_size: int,
    num_workers: int = 0,
    on_batch_metrics: Any | None = None,
) -> dict[tuple[int, int, tuple[int, ...]], dict[str, Any]]:
    num_nodes = mappings.num_users + mappings.num_items
    metrics_by_task: dict[tuple[int, int, tuple[int, ...]], dict[str, Any]] = {}

    from tqdm import tqdm

    task_batches = [
        tasks[start : start + perturbation_batch_size]
        for start in range(0, len(tasks), perturbation_batch_size)
    ]
    if not task_batches:
        return metrics_by_task

    def run_prepared_batch(
        batch_tasks: list[tuple[int, int, tuple[int, ...], float, int]],
        adjacency_indices: torch.Tensor,
        adjacency_values: torch.Tensor,
    ) -> None:
        batched_adjacency = torch.sparse_coo_tensor(
            adjacency_indices.to(device),
            adjacency_values.to(device),
            (len(batch_tasks) * num_nodes, len(batch_tasks) * num_nodes),
            device=device,
        ).coalesce()
        batched_ego = ego_embeddings.repeat(len(batch_tasks), 1)
        perturbed_all = propagate(batched_ego, batched_adjacency, num_layers)

        batch_metrics: dict[tuple[int, int, tuple[int, ...]], dict[str, Any]] = {}
        for offset, (
            user_index,
            target_item_index,
            drop_set,
            baseline_score,
            baseline_rank,
        ) in enumerate(batch_tasks):
            node_offset = offset * num_nodes
            excluded_items = train_history.get(user_index, set())
            perturbed_score, perturbed_rank = _score_and_rank(
                perturbed_all[node_offset + user_index],
                perturbed_all[
                    node_offset + mappings.num_users : node_offset
                    + mappings.num_users
                    + mappings.num_items
                ],
                target_item_index,
                excluded_items,
            )
            metrics_by_task[(user_index, target_item_index, drop_set)] = {
                "score_drop": baseline_score - perturbed_score,
                "baseline_rank": baseline_rank,
                "perturbed_rank": perturbed_rank,
                "rank_drop": perturbed_rank - baseline_rank,
            }
            batch_metrics[(user_index, target_item_index, drop_set)] = metrics_by_task[
                (user_index, target_item_index, drop_set)
            ]
        if on_batch_metrics is not None:
            on_batch_metrics(batch_metrics)

    if num_workers <= 0:
        for batch_tasks in tqdm(task_batches, desc="Running perturbation batches"):
            prepared_tasks, adjacency_indices, adjacency_values = (
                _build_perturbation_batch_from_data(
                    train_pairs,
                    train_history,
                    mappings.num_users,
                    mappings.num_items,
                    batch_tasks,
                )
            )
            run_prepared_batch(prepared_tasks, adjacency_indices, adjacency_values)
        return metrics_by_task

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_perturbation_batch_worker,
        initargs=(
            train_pairs.cpu(),
            train_history,
            mappings.num_users,
            mappings.num_items,
        ),
    ) as executor:
        next_batch_index = 0
        pending = {}
        max_pending = min(num_workers, len(task_batches))
        for _ in range(max_pending):
            future = executor.submit(
                _build_perturbation_batch_worker, task_batches[next_batch_index]
            )
            pending[future] = next_batch_index
            next_batch_index += 1

        with tqdm(total=len(task_batches), desc="Running perturbation batches") as progress:
            while pending:
                done, _pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    prepared_tasks, adjacency_indices, adjacency_values = future.result()
                    run_prepared_batch(prepared_tasks, adjacency_indices, adjacency_values)
                    progress.update(1)

                    if next_batch_index < len(task_batches):
                        next_future = executor.submit(
                            _build_perturbation_batch_worker,
                            task_batches[next_batch_index],
                        )
                        pending[next_future] = next_batch_index
                        next_batch_index += 1

    return metrics_by_task


@torch.no_grad()
def run_candidate_attribute_drops(
    artifact_dir: str | Path,
    candidate_json: str | Path,
    support_jsonl: str | Path,
    num_layers: int,
    device: str = "auto",
) -> dict[str, list[dict]]:
    """Score target drops after removing each candidate attribute's support items."""

    paths = ArtifactPaths(Path(artifact_dir).resolve())
    mappings = load_mappings(paths.mappings)
    train_pairs = load_tensor(paths.train_pairs).long()
    train_history = load_user_history(paths.user_history)
    candidates = load_candidate_attributes(candidate_json)
    supports = load_attribute_supports(support_jsonl)

    resolved_device = _resolve_device(device)
    user_ego = load_tensor(paths.user_ego_embeddings).to(resolved_device)
    item_ego = load_tensor(paths.item_ego_embeddings).to(resolved_device)
    ego_embeddings = torch.cat((user_ego, item_ego), dim=0)
    baseline_adjacency = build_normalized_adjacency(
        mappings.num_users,
        mappings.num_items,
        train_pairs,
        resolved_device,
    )
    baseline_all = propagate(ego_embeddings, baseline_adjacency, num_layers)
    baseline_users, baseline_items = torch.split(
        baseline_all, (mappings.num_users, mappings.num_items), dim=0
    )

    baseline_cache: dict[tuple[int, int], tuple[float, int]] = {}
    perturbation_cache: dict[tuple[int, int, tuple[int, ...]], dict[str, Any]] = {}
    output: dict[str, list[dict]] = {}

    from tqdm import tqdm 
    for pair_key, attributes in tqdm(list(candidates.items()), desc="Processing candidate attributes"):
        user_id, target_item_id = _parse_pair_key(pair_key)
        canonical_pair_key = f"{user_id},{target_item_id}"
        if user_id not in mappings.user_to_index or target_item_id not in mappings.item_to_index:
            output[pair_key] = _minimal_null_results(attributes)
            continue
        user_index = mappings.user_to_index[user_id]
        target_item_index = mappings.item_to_index[target_item_id]
        excluded_items = train_history.get(user_index, set())
        support_by_attribute = supports.get(canonical_pair_key, {})

        pair_results: list[dict] = []
        for attribute in attributes:
            attr_name = str(attribute["attr_name"])
            support_items = support_by_attribute.get(attr_name)
            if not support_items:
                pair_results.append(_minimal_result(attribute))
                continue

            drop_indices = _support_item_indices(support_items, mappings.item_to_index)
            if not drop_indices:
                pair_results.append(_minimal_result(attribute))
                continue

            baseline_key = (user_index, target_item_index)
            if baseline_key not in baseline_cache:
                baseline_cache[baseline_key] = _score_and_rank(
                    baseline_users[user_index],
                    baseline_items,
                    target_item_index,
                    excluded_items,
                )
            baseline_score, baseline_rank = baseline_cache[baseline_key]

            perturbation_key = (user_index, target_item_index, tuple(sorted(set(drop_indices))))
            metrics = perturbation_cache.get(perturbation_key)
            if metrics is None:
                perturbed_pairs = _drop_user_edges(
                    train_pairs, train_history, user_index, drop_indices
                )
                perturbed_adjacency = build_normalized_adjacency(
                    mappings.num_users,
                    mappings.num_items,
                    perturbed_pairs,
                    resolved_device,
                )
                perturbed_all = propagate(ego_embeddings, perturbed_adjacency, num_layers)
                perturbed_users, perturbed_items = torch.split(
                    perturbed_all, (mappings.num_users, mappings.num_items), dim=0
                )
                perturbed_score, perturbed_rank = _score_and_rank(
                    perturbed_users[user_index],
                    perturbed_items,
                    target_item_index,
                    excluded_items,
                )
                metrics = {
                    "score_drop": baseline_score - perturbed_score,
                    "baseline_rank": baseline_rank,
                    "perturbed_rank": perturbed_rank,
                    "rank_drop": perturbed_rank - baseline_rank,
                }
                perturbation_cache[perturbation_key] = metrics
            pair_results.append(_minimal_result(attribute, metrics))
        output[pair_key] = pair_results

    return output


@torch.no_grad()
def run_support_dict_attribute_drops(
    artifact_dir: str | Path,
    support_pkl: str | Path,
    num_layers: int,
    device: str = "auto",
    save_path: str | Path | None = None,
    save_every_pairs: int = 1,
    perturbation_batch_size: int = 8,
    num_workers: int = 0,
) -> dict[tuple[int, int], dict[int, dict]]:
    """Score target drops from an internal-ID support dictionary pickle."""

    if save_every_pairs <= 0:
        raise ValueError("save_every_pairs must be positive")
    if perturbation_batch_size <= 0:
        raise ValueError("perturbation_batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    paths = ArtifactPaths(Path(artifact_dir).resolve())
    mappings = load_mappings(paths.mappings)
    train_pairs = load_tensor(paths.train_pairs).long()
    train_history = load_user_history(paths.user_history)
    support_dict = load_support_dict(support_pkl)

    resolved_device = _resolve_device(device)
    user_ego = load_tensor(paths.user_ego_embeddings).to(resolved_device)
    item_ego = load_tensor(paths.item_ego_embeddings).to(resolved_device)
    ego_embeddings = torch.cat((user_ego, item_ego), dim=0)
    baseline_adjacency = build_normalized_adjacency(
        mappings.num_users,
        mappings.num_items,
        train_pairs,
        resolved_device,
    )
    baseline_all = propagate(ego_embeddings, baseline_adjacency, num_layers)
    baseline_users, baseline_items = torch.split(
        baseline_all, (mappings.num_users, mappings.num_items), dim=0
    )

    baseline_cache: dict[tuple[int, int], tuple[float, int]] = {}
    output: dict[tuple[int, int], dict[int, dict]] = {}
    completed_output: dict[tuple[int, int], dict[int, dict]] = {}
    pair_remaining: dict[tuple[int, int], int] = {}
    task_records: dict[
        tuple[int, int, tuple[int, ...]], list[tuple[tuple[int, int], int, dict[str, Any]]]
    ] = {}
    tasks_by_pair: dict[tuple[int, int], list[tuple[int, int, tuple[int, ...], float, int]]] = {}
    planned_task_keys: set[tuple[int, int, tuple[int, ...]]] = set()

    from tqdm import tqdm

    for (user_index, target_item_index), attributes in tqdm(
        list(support_dict.items()), desc="Planning support dictionary"
    ):
        if user_index < 0 or user_index >= mappings.num_users:
            raise ValueError(f"User index is out of range: {user_index}")
        if target_item_index < 0 or target_item_index >= mappings.num_items:
            raise ValueError(f"Target item index is out of range: {target_item_index}")

        pair_key = (user_index, target_item_index)
        excluded_items = train_history.get(user_index, set())
        pair_results: dict[int, dict] = {}
        pair_remaining[pair_key] = 0

        for attr_id, record in attributes.items():
            drop_indices = _internal_support_item_indices(
                record.get("support_items", []), mappings.num_items
            )
            if not drop_indices:
                pair_results[attr_id] = _support_dict_result(record)
                continue

            baseline_key = (user_index, target_item_index)
            if baseline_key not in baseline_cache:
                baseline_cache[baseline_key] = _score_and_rank(
                    baseline_users[user_index],
                    baseline_items,
                    target_item_index,
                    excluded_items,
                )
            baseline_score, baseline_rank = baseline_cache[baseline_key]

            drop_set = tuple(sorted(set(drop_indices)))
            task_key = (user_index, target_item_index, drop_set)
            task_records.setdefault(task_key, []).append((pair_key, attr_id, record))
            pair_remaining[pair_key] += 1
            if task_key not in planned_task_keys:
                tasks_by_pair.setdefault(pair_key, []).append(
                    (
                        user_index,
                        target_item_index,
                        drop_set,
                        baseline_score,
                        baseline_rank,
                    )
                )
                planned_task_keys.add(task_key)

        output[pair_key] = pair_results
        if pair_remaining[pair_key] == 0:
            completed_output[pair_key] = pair_results

    pair_order = list(support_dict.keys())
    tasks: list[tuple[int, int, tuple[int, ...], float, int]] = []
    while any(tasks_by_pair.get(pair_key) for pair_key in pair_order):
        for pair_key in pair_order:
            pair_tasks = tasks_by_pair.get(pair_key)
            if pair_tasks:
                tasks.append(pair_tasks.pop(0))

    completed_since_save = len(completed_output)
    if save_path is not None and completed_since_save >= save_every_pairs:
        _save_pickle(save_path, completed_output)
        completed_since_save = 0

    def apply_batch_metrics(
        batch_metrics: dict[tuple[int, int, tuple[int, ...]], dict[str, Any]]
    ) -> None:
        nonlocal completed_since_save
        for task_key, metrics in batch_metrics.items():
            completed_pairs: set[tuple[int, int]] = set()
            for pair_key, attr_id, record in task_records[task_key]:
                output[pair_key][attr_id] = _support_dict_result(record, metrics)
                pair_remaining[pair_key] -= 1
                if pair_remaining[pair_key] == 0:
                    completed_pairs.add(pair_key)

            for pair_key in completed_pairs:
                completed_output[pair_key] = output[pair_key]
                completed_since_save += 1
        if save_path is not None and completed_since_save >= save_every_pairs:
            _save_pickle(save_path, completed_output)
            completed_since_save = 0

    _batched_support_dict_task_metrics(
        train_pairs,
        train_history,
        mappings,
        ego_embeddings,
        tasks,
        num_layers,
        resolved_device,
        perturbation_batch_size,
        num_workers,
        apply_batch_metrics,
    )

    if set(completed_output) != set(output):
        missing = sorted(set(output) - set(completed_output))[:5]
        raise RuntimeError(f"Internal error: unfinished pairs remain, examples: {missing}")

    if save_path is not None:
        _save_pickle(save_path, completed_output)

    return output
