from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator

from .embed import DEFAULT_BGE_MODEL
from .errors import PipelineError
from .extract import load_source_items
from .io_utils import read_json, write_json, write_jsonl


def build_implicit_attributes(
    input_path: Path,
    output_dir: Path,
    top_k: int = 10,
    batch_size: int = 64,
    model_name: str | None = None,
    model_path: Path | None = None,
) -> None:
    if top_k <= 0:
        raise PipelineError("--top-k must be greater than zero.")
    if batch_size <= 0:
        raise PipelineError("--batch-size must be greater than zero.")
    if not input_path.is_file():
        raise PipelineError(f"Input JSONL file does not exist: {input_path}")

    explicit_path = output_dir / "item_attributes.json"
    vocabulary_path = output_dir / "vocabulary.json"
    embeddings_path = output_dir / "attribute_embeddings.npz"
    metadata_path = output_dir / "embedding_metadata.json"
    _require_file(explicit_path, "Run the cluster stage first.")
    _require_file(vocabulary_path, "Run the cluster stage first.")
    _require_file(embeddings_path, "Run the embed stage first.")
    _require_file(metadata_path, "Run the embed stage first.")

    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise PipelineError(
            "The 'numpy' and 'sentence-transformers' packages are required for "
            "implicit attribute ranking. Install attribute_pipeline/requirements.txt first."
        ) from exc

    explicit_by_iid = _load_explicit_attributes(explicit_path)
    vocabulary = _load_vocabulary(vocabulary_path)
    if not vocabulary:
        raise PipelineError("Canonical vocabulary is empty. Run the cluster stage again.")
    vocabulary_index = {phrase: index for index, phrase in enumerate(vocabulary)}
    stale_explicit = sorted(
        {
            phrase
            for attributes in explicit_by_iid.values()
            for phrase in attributes
            if phrase not in vocabulary_index
        }
    )
    if stale_explicit:
        raise PipelineError(
            "Explicit item attributes and canonical vocabulary do not match. "
            "Run the cluster stage again."
        )
    canonical_vectors = _load_canonical_vectors(embeddings_path, vocabulary)
    source_items, source_issues = load_source_items(input_path)
    descriptions = {str(item.iid): item.text for item in source_items}
    issues = list(source_issues)

    configured_path = model_path or _environment_path("BGE_MODEL_PATH")
    if configured_path is not None and not configured_path.exists():
        raise PipelineError(f"Configured BGE_MODEL_PATH does not exist: {configured_path}")
    metadata = read_json(metadata_path)
    if not isinstance(metadata, dict):
        raise PipelineError(f"Expected an embedding metadata object in {metadata_path}.")
    model_reference = str(
        configured_path
        or model_name
        or os.environ.get("BGE_MODEL")
        or metadata.get("model")
        or DEFAULT_BGE_MODEL
    )
    model = SentenceTransformer(model_reference)

    output: dict[str, dict[str, list[str]]] = {}
    rankable_iids: list[str] = []
    for iid in _sort_iids(explicit_by_iid):
        explicit = explicit_by_iid[iid]
        output[iid] = {"explicit": explicit, "implicit": []}
        if iid not in descriptions:
            issues.append({"iid": _coerce_iid(iid), "error": "missing usable item description"})
        else:
            rankable_iids.append(iid)

    for iid_batch in _progress(_batches(rankable_iids, batch_size), len(rankable_iids), batch_size):
        texts = [descriptions[iid] for iid in iid_batch]
        item_vectors = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores = np.asarray(item_vectors, dtype=np.float32) @ canonical_vectors.T
        for row_index, iid in enumerate(iid_batch):
            explicit = explicit_by_iid[iid]
            explicit_indexes = {
                vocabulary_index[phrase] for phrase in explicit if phrase in vocabulary_index
            }
            output[iid]["implicit"] = _top_non_explicit(
                scores[row_index],
                vocabulary,
                explicit_indexes,
                top_k,
            )

    write_json(output_dir / "item_attributes_im_ex.json", output)
    write_jsonl(output_dir / "implicit_issues.jsonl", issues)
    print(
        f"Wrote explicit and top-{top_k} implicit attributes for "
        f"{len(output)} items with {model_reference}."
    )


def _load_explicit_attributes(path: Path) -> dict[str, list[str]]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise PipelineError(f"Expected an item-to-attributes object in {path}.")
    result: dict[str, list[str]] = {}
    for iid, attributes in value.items():
        if not isinstance(attributes, list) or not all(
            isinstance(attribute, str) for attribute in attributes
        ):
            raise PipelineError(f"Expected a list of strings for iid {iid} in {path}.")
        result[str(iid)] = attributes
    return result


def _load_vocabulary(path: Path) -> list[str]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise PipelineError(f"Expected an ID-to-attribute object in {path}.")
    try:
        ordered_ids = sorted((int(attribute_id), attribute) for attribute_id, attribute in value.items())
    except (TypeError, ValueError) as exc:
        raise PipelineError(f"Vocabulary IDs in {path} must be integers.") from exc
    if [attribute_id for attribute_id, _ in ordered_ids] != list(range(len(ordered_ids))):
        raise PipelineError(f"Vocabulary IDs in {path} must be contiguous from zero.")
    if not all(isinstance(attribute, str) for _, attribute in ordered_ids):
        raise PipelineError(f"Vocabulary values in {path} must be strings.")
    return [attribute for _, attribute in ordered_ids]


def _load_canonical_vectors(path: Path, vocabulary: list[str]) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise PipelineError("The 'numpy' package is required for implicit attribute ranking.") from exc
    with np.load(path, allow_pickle=False) as archive:
        phrases = [str(value) for value in archive["phrases"].tolist()]
        vectors = np.asarray(archive["embeddings"], dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(phrases):
        raise PipelineError(f"Invalid embeddings artifact: {path}")
    phrase_to_index = {phrase: index for index, phrase in enumerate(phrases)}
    missing = sorted(set(vocabulary) - set(phrase_to_index))
    if missing:
        raise PipelineError(
            "Canonical vocabulary and embedded phrases do not match. Run the embed stage again."
        )
    canonical_vectors = vectors[[phrase_to_index[phrase] for phrase in vocabulary]]
    norms = np.linalg.norm(canonical_vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise PipelineError("Embedding artifact contains a zero-length canonical vector.")
    return canonical_vectors / norms


def _top_non_explicit(
    scores: Any,
    vocabulary: list[str],
    explicit_indexes: set[int],
    top_k: int,
) -> list[str]:
    import numpy as np

    eligible_count = len(vocabulary) - len(explicit_indexes)
    count = min(top_k, eligible_count)
    if count <= 0:
        return []
    filtered_scores = np.asarray(scores, dtype=np.float32).copy()
    if explicit_indexes:
        filtered_scores[list(explicit_indexes)] = -np.inf
    if count == len(vocabulary):
        candidate_indexes = np.arange(len(vocabulary))
    else:
        candidate_indexes = np.argpartition(filtered_scores, -count)[-count:]
    ordered_indexes = sorted(
        candidate_indexes,
        key=lambda index: (-float(filtered_scores[index]), vocabulary[index]),
    )
    return [vocabulary[index] for index in ordered_indexes[:count]]


def _batches(values: list[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _progress(batches: Iterator[list[str]], total_items: int, batch_size: int) -> Iterator[list[str]]:
    try:
        from tqdm import tqdm
    except ImportError:
        yield from batches
        return
    total_batches = (total_items + batch_size - 1) // batch_size
    yield from tqdm(batches, total=total_batches, desc="Ranking implicit attributes", unit="batch")


def _sort_iids(values: dict[str, Any]) -> list[str]:
    return sorted(values, key=lambda value: (0, int(value)) if value.isdigit() else (1, value))


def _coerce_iid(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def _environment_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def _require_file(path: Path, hint: str) -> None:
    if not path.is_file():
        raise PipelineError(f"Missing prerequisite artifact: {path}. {hint}")
