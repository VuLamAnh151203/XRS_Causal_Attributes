from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from item_attribute_pipeline.implicit import build_implicit_attributes
from item_attribute_pipeline.io_utils import read_json


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_implicit_attributes_exclude_explicit_matches(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "output"
    output.mkdir()
    source = tmp_path / "item_profile.json"
    source.write_text(
        json.dumps(
            {
                "iid": 7,
                "completion": json.dumps({"reasoning": "A tense zombie survival narrative."}),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(output / "item_attributes.json", {"7": ["zombie fiction"]})
    write_json(
        output / "vocabulary.json",
        {"0": "romance", "1": "survival thriller", "2": "zombie fiction"},
    )
    write_json(output / "embedding_metadata.json", {"model": "test-bge"})
    np.savez_compressed(
        output / "attribute_embeddings.npz",
        phrases=np.asarray(["romance", "survival thriller", "zombie fiction"]),
        embeddings=np.asarray([[0.0, 1.0], [0.8, 0.2], [1.0, 0.0]], dtype=np.float32),
    )

    class FakeSentenceTransformer:
        def __init__(self, model_reference: str) -> None:
            assert model_reference == "test-bge"

        def encode(self, texts, **kwargs):
            assert texts == ["A tense zombie survival narrative."]
            assert kwargs["normalize_embeddings"] is True
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    build_implicit_attributes(source, output, top_k=2, batch_size=1)

    assert read_json(output / "item_attributes_im_ex.json") == {
        "7": {
            "explicit": ["zombie fiction"],
            "implicit": ["survival thriller", "romance"],
        }
    }
