from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from item_attribute_pipeline.cli import main


def test_embed_fails_clearly_without_extract_artifacts(tmp_path: Path, capsys) -> None:
    result = main(["embed", "--output", str(tmp_path)])

    assert result == 2
    assert "Run the extract stage first" in capsys.readouterr().err


def test_cluster_fails_clearly_without_embedding_artifact(tmp_path: Path, capsys) -> None:
    (tmp_path / "normalized_item_attributes.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "attribute_frequencies.json").write_text("{}", encoding="utf-8")

    result = main(["cluster", "--output", str(tmp_path)])

    assert result == 2
    assert "Run the embed stage first" in capsys.readouterr().err
