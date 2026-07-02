from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import PipelineError
from .io_utils import read_json, read_jsonl, write_json


DEFAULT_BGE_MODEL = "BAAI/bge-small-en-v1.5"


def embed_attributes(
    output_dir: Path,
    batch_size: int,
    model_name: str | None = None,
    model_path: Path | None = None,
) -> None:
    if batch_size <= 0:
        raise PipelineError("--batch-size must be greater than zero.")
    normalized_path = output_dir / "normalized_item_attributes.jsonl"
    frequencies_path = output_dir / "attribute_frequencies.json"
    _require_file(normalized_path, "Run the extract stage first.")
    _require_file(frequencies_path, "Run the extract stage first.")

    frequencies = read_json(frequencies_path)
    if not isinstance(frequencies, dict):
        raise PipelineError(f"Expected an attribute frequency object in {frequencies_path}.")
    phrases = sorted(frequencies)
    if not phrases:
        raise PipelineError("No normalized attributes are available to embed.")

    configured_path = model_path or _environment_path("BGE_MODEL_PATH")
    if configured_path is not None and not configured_path.exists():
        raise PipelineError(f"Configured BGE_MODEL_PATH does not exist: {configured_path}")
    model_reference = str(
        configured_path
        or model_name
        or os.environ.get("BGE_MODEL")
        or DEFAULT_BGE_MODEL
    )
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise PipelineError(
            "The 'numpy' and 'sentence-transformers' packages are required for embedding. "
            "Install attribute_pipeline/requirements.txt first."
        ) from exc

    model = SentenceTransformer(model_reference)
    vectors = model.encode(
        phrases,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    np.savez_compressed(
        output_dir / "attribute_embeddings.npz",
        phrases=np.asarray(phrases),
        embeddings=np.asarray(vectors, dtype=np.float32),
    )
    write_json(
        output_dir / "embedding_metadata.json",
        {
            "model": model_reference,
            "attribute_count": len(phrases),
            "batch_size": batch_size,
            "normalized_embeddings": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"Embedded {len(phrases)} normalized attributes with {model_reference}.")


def _environment_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def _require_file(path: Path, hint: str) -> None:
    if not path.is_file():
        raise PipelineError(f"Missing prerequisite artifact: {path}. {hint}")
