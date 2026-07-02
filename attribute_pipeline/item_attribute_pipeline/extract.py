from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .errors import PipelineError
from .io_utils import append_jsonl, read_jsonl, write_json, write_jsonl
from .normalize import normalize_item_records


SYSTEM_PROMPT = """Extract reusable recommendation attributes from the item text.

Rules:
- Return 5-10 short attributes.
- Each attribute must be a noun phrase, 1-5 words.
- Attributes can describe genre, theme, trope, character type, setting, mood, style, or plot element.
- Do not include item title, author name, or full sentences.
- Avoid generic words such as book, story, novel, reader.
- Return a JSON object only, using this exact shape:
{"attributes": ["historical romance", "slow-burn romance", "strong heroine"]}
"""
WORD_PATTERN = re.compile(r"\b[\w]+(?:[-'][\w]+)?\b", flags=re.UNICODE)


@dataclass(frozen=True)
class SourceItem:
    iid: int
    text: str
    source_field: str


class ExtractionError(RuntimeError):
    """Raised after an item's extraction retries are exhausted."""


class _FallbackProgress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.completed = 0

    def __enter__(self) -> "_FallbackProgress":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def update(self, count: int = 1) -> None:
        self.completed += count
        print(f"Extraction progress: {self.completed}/{self.total}", flush=True)


class DeepSeekExtractor:
    def __init__(self) -> None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise PipelineError("DEEPSEEK_API_KEY must be set before running extraction.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise PipelineError(
                "The 'openai' package is required for extraction. "
                "Install attribute_pipeline/requirements.txt first."
            ) from exc
        self.model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def extract(self, text: str) -> list[str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Item text:\n{text}"},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=300,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("DeepSeek returned empty content")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"DeepSeek returned invalid JSON: {exc}") from exc
        return validate_attributes(payload)


def validate_attributes(payload: Any) -> list[str]:
    if not isinstance(payload, dict) or set(payload) != {"attributes"}:
        raise ValueError("expected a JSON object containing only 'attributes'")
    attributes = payload["attributes"]
    if not isinstance(attributes, list) or not 5 <= len(attributes) <= 10:
        raise ValueError("'attributes' must be a list containing 5-10 phrases")
    cleaned: list[str] = []
    for attribute in attributes:
        if not isinstance(attribute, str):
            raise ValueError("each attribute must be a string")
        phrase = attribute.strip()
        if not phrase or not 1 <= len(WORD_PATTERN.findall(phrase)) <= 5:
            raise ValueError("each attribute must contain 1-5 words")
        cleaned.append(phrase)
    return cleaned


def load_source_items(input_path: Path) -> tuple[list[SourceItem], list[dict[str, Any]]]:
    items: list[SourceItem] = []
    issues: list[dict[str, Any]] = []
    seen_iids: set[int] = set()
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append({"line": line_number, "error": f"invalid source JSON: {exc}"})
                continue
            if not isinstance(record, dict):
                issues.append({"line": line_number, "error": "source line must be a JSON object"})
                continue
            iid = record.get("iid")
            if not isinstance(iid, int):
                issues.append({"line": line_number, "error": "missing or non-integer iid"})
                continue
            if iid in seen_iids:
                raise PipelineError(f"Duplicate iid {iid} in {input_path}.")
            seen_iids.add(iid)
            completion = record.get("completion")
            try:
                payload = json.loads(completion) if isinstance(completion, str) else completion
            except json.JSONDecodeError as exc:
                issues.append(
                    {"iid": iid, "line": line_number, "error": f"invalid completion JSON: {exc}"}
                )
                continue
            if not isinstance(payload, dict):
                issues.append(
                    {"iid": iid, "line": line_number, "error": "completion must decode to an object"}
                )
                continue
            source_field, text = _select_text(payload)
            if text is None:
                issues.append(
                    {"iid": iid, "line": line_number, "error": "missing reasoning and summarization"}
                )
                continue
            if source_field == "summarization":
                issues.append(
                    {
                        "iid": iid,
                        "line": line_number,
                        "warning": "reasoning missing; used summarization",
                    }
                )
            items.append(SourceItem(iid=iid, text=text, source_field=source_field))
    return items, issues


def _select_text(payload: dict[str, Any]) -> tuple[str, str | None]:
    for field in ("reasoning", "summarization"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return field, value.strip()
    return "", None


def extract_and_normalize(
    input_path: Path,
    output_dir: Path,
    limit: int | None,
    resume: bool,
    max_retries: int,
    extractor: Any | None = None,
    allow_partial: bool = False,
    workers: int = 1,
) -> None:
    if limit is not None and limit <= 0:
        raise PipelineError("--limit must be greater than zero.")
    if workers <= 0:
        raise PipelineError("--workers must be greater than zero.")
    if max_retries <= 0:
        raise PipelineError("--max-retries must be greater than zero.")
    if not input_path.is_file():
        raise PipelineError(f"Input JSONL file does not exist: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_item_attributes.jsonl"
    issues_path = output_dir / "issues.jsonl"
    source_items, issues = load_source_items(input_path)
    if limit is not None:
        source_items = source_items[:limit]

    completed = _load_checkpoint(raw_path) if resume else {}
    if not resume:
        _reset_checkpoint(raw_path)
    pending_items = [item for item in source_items if item.iid not in completed]
    active_extractor = extractor
    records = dict(completed)
    if pending_items and active_extractor is None:
        active_extractor = DeepSeekExtractor()
    failed_extractions = _extract_pending_items(
        pending_items=pending_items,
        extractor=active_extractor,
        max_retries=max_retries,
        workers=workers,
        records=records,
        raw_path=raw_path,
        issues=issues,
    )

    ordered_records = _sort_by_iid(records.values())
    normalized_records, frequencies = normalize_item_records(ordered_records)
    write_jsonl(output_dir / "normalized_item_attributes.jsonl", normalized_records)
    write_json(
        output_dir / "attribute_frequencies.json",
        dict(sorted(frequencies.items())),
    )
    write_jsonl(issues_path, issues)
    print(f"Extracted attributes for {len(ordered_records)} items.")
    print(
        f"Attempted {len(pending_items)} unfinished items "
        f"with up to {workers} concurrent API requests."
    )
    print(f"Recorded {len(issues)} source or extraction issues.")
    if failed_extractions and not allow_partial:
        raise PipelineError(
            f"Extraction failed for {failed_extractions} items. "
            "Fix the API issue and rerun with --resume, or pass --allow-partial intentionally."
        )


def _extract_pending_items(
    pending_items: list[SourceItem],
    extractor: Any,
    max_retries: int,
    workers: int,
    records: dict[int, dict[str, Any]],
    raw_path: Path,
    issues: list[dict[str, Any]],
) -> int:
    if not pending_items:
        return 0
    pending_iterator = iter(pending_items)
    failed_extractions = 0
    with _progress_bar(len(pending_items)) as progress, ThreadPoolExecutor(
        max_workers=workers
    ) as executor:
        futures: dict[Future[list[str]], SourceItem] = {}
        for item in _take(pending_iterator, workers):
            futures[executor.submit(_extract_with_retries, extractor, item.text, max_retries)] = item
        while futures:
            completed_futures, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed_futures:
                item = futures.pop(future)
                try:
                    attributes = future.result()
                except ExtractionError as exc:
                    issues.append({"iid": item.iid, "error": f"extraction failed: {exc}"})
                    failed_extractions += 1
                else:
                    records[item.iid] = {
                        "iid": item.iid,
                        "source_field": item.source_field,
                        "attributes": attributes,
                    }
                    try:
                        append_jsonl(raw_path, records[item.iid])
                    except PermissionError as exc:
                        raise PipelineError(
                            f"Cannot append extraction checkpoint {raw_path}. "
                            "Close other pipeline runs or programs locking the file, "
                            "then rerun with --resume."
                        ) from exc
                progress.update(1)
                next_item = next(pending_iterator, None)
                if next_item is not None:
                    futures[
                        executor.submit(
                            _extract_with_retries,
                            extractor,
                            next_item.text,
                            max_retries,
                        )
                    ] = next_item
    return failed_extractions


def _progress_bar(total: int) -> Any:
    try:
        from tqdm import tqdm
    except ImportError:
        return _FallbackProgress(total)
    return tqdm(total=total, desc="Extracting attributes", unit="item", dynamic_ncols=True)


def _take(items: Iterator[SourceItem], count: int) -> Iterator[SourceItem]:
    for _ in range(count):
        item = next(items, None)
        if item is None:
            return
        yield item


def _reset_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text("", encoding="utf-8")
    except PermissionError as exc:
        raise PipelineError(
            f"Cannot reset extraction checkpoint {path}. "
            "Close other pipeline runs or programs locking the file, then retry."
        ) from exc


def _extract_with_retries(extractor: Any, text: str, max_retries: int) -> list[str]:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return extractor.extract(text)
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max_retries:
                time.sleep(2**attempt)
    assert last_error is not None
    raise ExtractionError(str(last_error)) from last_error


def _load_checkpoint(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[int, dict[str, Any]] = {}
    try:
        for record in read_jsonl(path):
            iid = record.get("iid")
            if not isinstance(iid, int) or not isinstance(record.get("attributes"), list):
                raise PipelineError(f"Invalid extraction checkpoint record in {path}.")
            records[iid] = record
    except ValueError as exc:
        raise PipelineError(str(exc)) from exc
    return records


def _sort_by_iid(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda record: record["iid"])
