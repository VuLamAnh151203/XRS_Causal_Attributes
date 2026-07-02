from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from item_attribute_pipeline.errors import PipelineError
from item_attribute_pipeline.extract import (
    _FallbackProgress,
    _extract_with_retries,
    extract_and_normalize,
    load_source_items,
    validate_attributes,
)
from item_attribute_pipeline.io_utils import read_json, read_jsonl
from item_attribute_pipeline.normalize import normalize_phrase


VALID_ATTRIBUTES = [
    "Historical Fiction",
    "Slow-Burn Romance",
    " strong heroine ",
    "family-centered drama",
    "happy ending",
]


class FakeExtractor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def extract(self, text: str) -> list[str]:
        self.calls.append(text)
        return VALID_ATTRIBUTES


def write_source(path: Path) -> None:
    records = [
        {
            "iid": 0,
            "completion": json.dumps(
                {
                    "summarization": "Fallback text.",
                    "reasoning": "Primary reasoning text.",
                }
            ),
        },
        {
            "iid": 1,
            "completion": json.dumps(
                {"summarization": "Summary fallback text.", "reasoning": " "}
            ),
        },
        {"iid": 2, "completion": json.dumps({"reasoning": "", "summarization": ""})},
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_load_source_items_decodes_completion_and_uses_summary_fallback(tmp_path: Path) -> None:
    source = tmp_path / "item_profile.json"
    write_source(source)

    items, issues = load_source_items(source)

    assert [(item.iid, item.source_field, item.text) for item in items] == [
        (0, "reasoning", "Primary reasoning text."),
        (1, "summarization", "Summary fallback text."),
    ]
    assert issues == [
        {"iid": 1, "line": 2, "warning": "reasoning missing; used summarization"},
        {"iid": 2, "line": 3, "error": "missing reasoning and summarization"},
    ]


def test_duplicate_iid_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "item_profile.json"
    record = {"iid": 4, "completion": json.dumps({"reasoning": "text"})}
    source.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(PipelineError, match="Duplicate iid 4"):
        load_source_items(source)


def test_malformed_outer_json_is_reported_and_skipped(tmp_path: Path) -> None:
    source = tmp_path / "item_profile.json"
    source.write_text(
        "{bad json}\n"
        + json.dumps({"iid": 8, "completion": json.dumps({"reasoning": "usable"})})
        + "\n",
        encoding="utf-8",
    )

    items, issues = load_source_items(source)

    assert [item.iid for item in items] == [8]
    assert issues[0]["line"] == 1
    assert issues[0]["error"].startswith("invalid source JSON:")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Slow-Burn Romance", "slow burn romance"),
        (" family-centered drama ", "family centered drama"),
        ("War Novel", "war"),
        ("great story", None),
        ("interesting plot", None),
    ],
)
def test_normalize_phrase(raw: str, expected: str | None) -> None:
    assert normalize_phrase(raw) == expected


def test_extract_normalize_writes_stage_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "item_profile.json"
    output = tmp_path / "output"
    write_source(source)
    extractor = FakeExtractor()

    extract_and_normalize(
        input_path=source,
        output_dir=output,
        limit=None,
        resume=False,
        max_retries=1,
        extractor=extractor,
    )

    raw_records = list(read_jsonl(output / "raw_item_attributes.jsonl"))
    normalized_records = list(read_jsonl(output / "normalized_item_attributes.jsonl"))
    frequencies = read_json(output / "attribute_frequencies.json")
    issues = list(read_jsonl(output / "issues.jsonl"))

    assert [record["iid"] for record in raw_records] == [0, 1]
    assert normalized_records[0]["attributes"] == [
        "historical fiction",
        "slow burn romance",
        "strong heroine",
        "family centered drama",
        "happy ending",
    ]
    assert frequencies["historical fiction"] == 2
    assert issues == [
        {"iid": 1, "line": 2, "warning": "reasoning missing; used summarization"},
        {"error": "missing reasoning and summarization", "iid": 2, "line": 3},
    ]


def test_resume_reuses_raw_item_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "item_profile.json"
    output = tmp_path / "output"
    write_source(source)
    extractor = FakeExtractor()

    extract_and_normalize(source, output, limit=1, resume=False, max_retries=1, extractor=extractor)
    extract_and_normalize(source, output, limit=None, resume=True, max_retries=1, extractor=extractor)

    assert extractor.calls == ["Primary reasoning text.", "Summary fallback text."]


def test_workers_call_api_in_parallel(tmp_path: Path) -> None:
    class ParallelExtractor(FakeExtractor):
        def __init__(self) -> None:
            super().__init__()
            self.barrier = threading.Barrier(2, timeout=2)
            self.lock = threading.Lock()

        def extract(self, text: str) -> list[str]:
            with self.lock:
                self.calls.append(text)
            self.barrier.wait()
            return VALID_ATTRIBUTES

    source = tmp_path / "item_profile.json"
    output = tmp_path / "output"
    write_source(source)
    extractor = ParallelExtractor()

    extract_and_normalize(
        source,
        output,
        limit=None,
        resume=False,
        max_retries=1,
        extractor=extractor,
        workers=2,
    )

    assert sorted(extractor.calls) == ["Primary reasoning text.", "Summary fallback text."]
    assert sorted(record["iid"] for record in read_jsonl(output / "raw_item_attributes.jsonl")) == [
        0,
        1,
    ]


def test_resume_retries_api_failure(tmp_path: Path) -> None:
    class FailOnceExtractor(FakeExtractor):
        def extract(self, text: str) -> list[str]:
            self.calls.append(text)
            if len(self.calls) == 1:
                raise RuntimeError("temporary API failure")
            return VALID_ATTRIBUTES

    source = tmp_path / "item_profile.json"
    output = tmp_path / "output"
    write_source(source)
    extractor = FailOnceExtractor()

    extract_and_normalize(
        source,
        output,
        limit=None,
        resume=False,
        max_retries=1,
        extractor=extractor,
        allow_partial=True,
    )
    extract_and_normalize(
        source,
        output,
        limit=None,
        resume=True,
        max_retries=1,
        extractor=extractor,
    )

    assert extractor.calls == [
        "Primary reasoning text.",
        "Summary fallback text.",
        "Primary reasoning text.",
    ]
    assert [record["iid"] for record in read_jsonl(output / "raw_item_attributes.jsonl")] == [0, 1]


def test_validate_attributes_requires_five_to_ten_short_phrases() -> None:
    with pytest.raises(ValueError, match="5-10"):
        validate_attributes({"attributes": ["only one"]})
    with pytest.raises(ValueError, match="1-5 words"):
        validate_attributes(
            {"attributes": ["one", "two", "three", "four", "this phrase has six total words"]}
        )


def test_extraction_retries_transient_errors(monkeypatch) -> None:
    class RetryExtractor:
        def __init__(self) -> None:
            self.calls = 0

        def extract(self, text: str) -> list[str]:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary API failure")
            return VALID_ATTRIBUTES

    extractor = RetryExtractor()
    monkeypatch.setattr("item_attribute_pipeline.extract.time.sleep", lambda _: None)

    assert _extract_with_retries(extractor, "text", max_retries=2) == VALID_ATTRIBUTES
    assert extractor.calls == 2


def test_fallback_progress_reports_completed_items(capsys) -> None:
    with _FallbackProgress(total=3) as progress:
        progress.update()
        progress.update(2)

    assert capsys.readouterr().out.splitlines() == [
        "Extraction progress: 1/3",
        "Extraction progress: 3/3",
    ]


def test_exhausted_extraction_errors_preserve_artifacts_and_stop_run(
    tmp_path: Path, monkeypatch
) -> None:
    class FailingExtractor:
        def extract(self, text: str) -> list[str]:
            raise RuntimeError("API unavailable")

    source = tmp_path / "item_profile.json"
    output = tmp_path / "output"
    source.write_text(
        json.dumps({"iid": 1, "completion": json.dumps({"reasoning": "usable text"})}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("item_attribute_pipeline.extract.time.sleep", lambda _: None)

    with pytest.raises(PipelineError, match="rerun with --resume"):
        extract_and_normalize(
            source,
            output,
            limit=None,
            resume=False,
            max_retries=2,
            extractor=FailingExtractor(),
        )

    assert (output / "raw_item_attributes.jsonl").is_file()
    assert "extraction failed: API unavailable" in (output / "issues.jsonl").read_text(
        encoding="utf-8"
    )
