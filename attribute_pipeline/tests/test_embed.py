from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from item_attribute_pipeline.embed import embed_attributes
from item_attribute_pipeline.io_utils import read_json


def test_embed_writes_vectors_and_metadata_with_mocked_bge(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "normalized_item_attributes.jsonl").write_text(
        json.dumps({"iid": 1, "attributes": ["family conflict", "strong heroine"]}) + "\n",
        encoding="utf-8",
    )
    (output / "attribute_frequencies.json").write_text(
        json.dumps({"strong heroine": 1, "family conflict": 1}),
        encoding="utf-8",
    )
    loaded_models: list[str] = []

    class FakeSentenceTransformer:
        def __init__(self, model_reference: str) -> None:
            loaded_models.append(model_reference)

        def encode(self, phrases, **kwargs):
            assert phrases == ["family conflict", "strong heroine"]
            assert kwargs["normalize_embeddings"] is True
            return np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    embed_attributes(output, batch_size=2, model_name="test-bge")

    assert loaded_models == ["test-bge"]
    with np.load(output / "attribute_embeddings.npz", allow_pickle=False) as archive:
        assert archive["phrases"].tolist() == ["family conflict", "strong heroine"]
        assert archive["embeddings"].shape == (2, 2)
    assert read_json(output / "embedding_metadata.json")["model"] == "test-bge"
