from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .errors import PipelineError
from .io_utils import read_json, read_jsonl, write_json


DEFAULT_THRESHOLD = 0.85


def cluster_attributes(
    output_dir: Path,
    threshold: float | None = None,
) -> None:
    try:
        similarity_threshold = (
            threshold
            if threshold is not None
            else float(os.environ.get("ATTRIBUTE_SIMILARITY_THRESHOLD", DEFAULT_THRESHOLD))
        )
    except ValueError as exc:
        raise PipelineError("ATTRIBUTE_SIMILARITY_THRESHOLD must be a number.") from exc
    if not -1.0 <= similarity_threshold <= 1.0:
        raise PipelineError("--threshold must be between -1 and 1.")

    normalized_path = output_dir / "normalized_item_attributes.jsonl"
    embeddings_path = output_dir / "attribute_embeddings.npz"
    frequencies_path = output_dir / "attribute_frequencies.json"
    _require_file(normalized_path, "Run the extract stage first.")
    _require_file(frequencies_path, "Run the extract stage first.")
    _require_file(embeddings_path, "Run the embed stage first.")

    try:
        import numpy as np
        from scipy import sparse
    except ImportError as exc:
        raise PipelineError(
            "The 'numpy' and 'scipy' packages are required for clustering. "
            "Install attribute_pipeline/requirements.txt first."
        ) from exc

    with np.load(embeddings_path, allow_pickle=False) as archive:
        phrases = [str(value) for value in archive["phrases"].tolist()]
        vectors = np.asarray(archive["embeddings"], dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(phrases):
        raise PipelineError(f"Invalid embeddings artifact: {embeddings_path}")
    if not phrases:
        raise PipelineError("No embedded attributes are available to cluster.")

    frequencies = read_json(frequencies_path)
    if set(phrases) != set(frequencies):
        raise PipelineError(
            "Embedding phrases and normalized attribute frequencies do not match. "
            "Run the embed stage again."
        )

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise PipelineError("Embedding artifact contains a zero-length vector.")
    vectors = vectors / norms

    components = _hierarchical_complete_link_clusters(vectors, similarity_threshold)
    cluster_records, phrase_to_canonical = _canonicalize(
        phrases, vectors, components, frequencies
    )
    vocabulary_phrases = sorted({record["canonical"] for record in cluster_records})
    canonical_to_id = {phrase: index for index, phrase in enumerate(vocabulary_phrases)}
    phrase_to_id = {
        phrase: canonical_to_id[canonical]
        for phrase, canonical in phrase_to_canonical.items()
    }

    normalized_records = list(read_jsonl(normalized_path))
    matrix_rows = sorted(record["iid"] for record in normalized_records)
    records_by_iid = {record["iid"]: record for record in normalized_records}
    item_attribute_ids: dict[str, list[int]] = {}
    item_attributes: dict[str, list[str]] = {}
    row_indexes: list[int] = []
    column_indexes: list[int] = []
    for row_index, iid in enumerate(matrix_rows):
        ids = sorted(
            {
                phrase_to_id[phrase]
                for phrase in records_by_iid[iid]["attributes"]
                if phrase in phrase_to_id
            }
        )
        item_attribute_ids[str(iid)] = ids
        item_attributes[str(iid)] = [vocabulary_phrases[attribute_id] for attribute_id in ids]
        row_indexes.extend([row_index] * len(ids))
        column_indexes.extend(ids)

    matrix = sparse.csr_matrix(
        (
            np.ones(len(row_indexes), dtype=np.int8),
            (row_indexes, column_indexes),
        ),
        shape=(len(matrix_rows), len(vocabulary_phrases)),
        dtype=np.int8,
    )
    sparse.save_npz(output_dir / "item_attribute_matrix.npz", matrix)

    write_json(output_dir / "clusters.json", cluster_records)
    write_json(
        output_dir / "vocabulary.json",
        {str(index): phrase for index, phrase in enumerate(vocabulary_phrases)},
    )
    write_json(output_dir / "item_attribute_ids.json", item_attribute_ids)
    write_json(output_dir / "item_attributes.json", item_attributes)
    write_json(output_dir / "matrix_rows.json", matrix_rows)
    write_json(
        output_dir / "matrix_columns.json",
        {str(index): phrase for index, phrase in enumerate(vocabulary_phrases)},
    )
    print(
        f"Clustered {len(phrases)} normalized phrases into "
        f"{len(vocabulary_phrases)} canonical attributes."
    )


def _hierarchical_complete_link_clusters(vectors: Any, threshold: float) -> list[list[int]]:
    import numpy as np

    if vectors.shape[0] == 1:
        return [[0]]
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
    except ImportError as exc:
        raise PipelineError(
            "The 'scipy' package is required for hierarchical clustering. "
            "Install attribute_pipeline/requirements.txt first."
        ) from exc

    size = vectors.shape[0]
    distance_count = size * (size - 1) // 2
    condensed_distances = np.empty(distance_count, dtype=np.float32)
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    print(f"Computing {distance_count:,} pairwise cosine distances...")
    row_iterator = range(size - 1)
    if tqdm is not None:
        row_iterator = tqdm(
            row_iterator,
            total=size - 1,
            desc="Pairwise cosine distances",
            unit="row",
            dynamic_ncols=True,
        )
    offset = 0
    for row_index in row_iterator:
        row_distances = 1.0 - (vectors[row_index + 1 :] @ vectors[row_index])
        next_offset = offset + len(row_distances)
        condensed_distances[offset:next_offset] = row_distances
        offset = next_offset

    print(
        "Running hierarchical complete-link clustering. "
        "SciPy does not expose progress callbacks for this phase...",
        flush=True,
    )
    linkage_matrix = linkage(condensed_distances, method="complete")
    print("Hierarchical linkage complete. Assigning final clusters...", flush=True)
    labels = fcluster(linkage_matrix, t=1.0 - threshold, criterion="distance")
    grouped: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        grouped.setdefault(int(label), []).append(index)
    return sorted(grouped.values(), key=lambda component: min(component))


def _canonicalize(
    phrases: list[str],
    vectors: Any,
    components: list[list[int]],
    frequencies: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    import numpy as np

    records: list[dict[str, Any]] = []
    phrase_to_canonical: dict[str, str] = {}
    for component in components:
        centroid = vectors[component].mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm:
            centroid = centroid / centroid_norm
        candidates: list[tuple[str, int, float]] = []
        for index in component:
            phrase = phrases[index]
            centroid_similarity = float(vectors[index] @ centroid)
            candidates.append((phrase, int(frequencies[phrase]), centroid_similarity))
        candidates.sort(key=lambda value: (-value[1], -value[2], value[0]))
        canonical = candidates[0][0]
        members = [
            {
                "phrase": phrase,
                "frequency": frequency,
                "centroid_similarity": round(centroid_similarity, 8),
            }
            for phrase, frequency, centroid_similarity in sorted(candidates)
        ]
        records.append({"canonical": canonical, "members": members})
        for phrase, _, _ in candidates:
            phrase_to_canonical[phrase] = canonical
    records.sort(key=lambda record: record["canonical"])
    return records, phrase_to_canonical


def _require_file(path: Path, hint: str) -> None:
    if not path.is_file():
        raise PipelineError(f"Missing prerequisite artifact: {path}. {hint}")
