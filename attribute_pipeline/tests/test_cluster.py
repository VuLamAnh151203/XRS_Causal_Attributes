from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from item_attribute_pipeline.cluster import (
    _hierarchical_complete_link_clusters,
    cluster_attributes,
)
from item_attribute_pipeline.errors import PipelineError
from item_attribute_pipeline.io_utils import read_json


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, values: list[object]) -> None:
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def test_cluster_builds_canonical_mapping_and_sparse_matrix(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    phrases = np.asarray(["family conflict", "strong female lead", "strong heroine"])
    vectors = np.asarray(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.99, 0.01],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(output / "attribute_embeddings.npz", phrases=phrases, embeddings=vectors)
    write_json(
        output / "attribute_frequencies.json",
        {"family conflict": 1, "strong female lead": 2, "strong heroine": 1},
    )
    write_jsonl(
        output / "normalized_item_attributes.jsonl",
        [
            {"iid": 10, "attributes": ["strong heroine", "family conflict"]},
            {"iid": 20, "attributes": ["strong female lead"]},
        ],
    )

    cluster_attributes(output, threshold=0.85)

    assert read_json(output / "vocabulary.json") == {
        "0": "family conflict",
        "1": "strong female lead",
    }
    assert read_json(output / "item_attribute_ids.json") == {"10": [0, 1], "20": [1]}
    assert read_json(output / "item_attributes.json") == {
        "10": ["family conflict", "strong female lead"],
        "20": ["strong female lead"],
    }
    matrix = sparse.load_npz(output / "item_attribute_matrix.npz")
    assert matrix.shape == (2, 2)
    assert matrix.toarray().tolist() == [[1, 1], [0, 1]]


def test_cluster_rejects_stale_embeddings(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    np.savez_compressed(
        output / "attribute_embeddings.npz",
        phrases=np.asarray(["existing phrase"]),
        embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
    )
    write_json(
        output / "attribute_frequencies.json",
        {"existing phrase": 1, "new phrase": 1},
    )
    write_jsonl(
        output / "normalized_item_attributes.jsonl",
        [{"iid": 1, "attributes": ["existing phrase", "new phrase"]}],
    )

    with pytest.raises(PipelineError, match="Run the embed stage again"):
        cluster_attributes(output)


def test_hierarchical_complete_link_breaks_transitive_similarity_chain() -> None:
    vectors = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.28, 0.96],
        ],
        dtype=np.float32,
    )
    components = _hierarchical_complete_link_clusters(vectors, threshold=0.75)

    assert components == [[0, 1], [2]]
