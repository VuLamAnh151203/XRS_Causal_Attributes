from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Iterable


GENERIC_TOKENS = {"author", "book", "chapter", "novel", "reader", "readers", "story"}
BAD_PHRASES = {
    "author",
    "book",
    "chapter",
    "great story",
    "interesting plot",
    "novel",
    "reader",
    "readers",
    "story",
}
DASHES = re.compile(r"[\-\u058a\u05be\u1400\u1806\u2010-\u2015\u2e17\u2e1a\u2e3a-\u2e3b\u2e40\u301c\u3030\u30a0\ufe31-\ufe32\ufe58\ufe63\uff0d]")
SPACES = re.compile(r"\s+")


def normalize_phrase(phrase: str) -> str | None:
    value = unicodedata.normalize("NFKC", phrase).lower().strip()
    value = DASHES.sub(" ", value)
    value = "".join(
        character if character.isalnum() or character.isspace() else " "
        for character in value
    )
    value = SPACES.sub(" ", value).strip()
    if not value or value in BAD_PHRASES:
        return None
    tokens = [token for token in value.split() if token not in GENERIC_TOKENS]
    value = " ".join(tokens)
    if not value or value in BAD_PHRASES:
        return None
    return value


def normalize_item_records(
    raw_records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    normalized_records: list[dict[str, Any]] = []
    frequencies: Counter[str] = Counter()
    for record in raw_records:
        normalized: list[str] = []
        seen: set[str] = set()
        mapping: list[dict[str, str | None]] = []
        for raw_phrase in record["attributes"]:
            phrase = normalize_phrase(raw_phrase)
            mapping.append({"raw": raw_phrase, "normalized": phrase})
            if phrase is not None and phrase not in seen:
                normalized.append(phrase)
                seen.add(phrase)
        frequencies.update(normalized)
        normalized_records.append(
            {
                "iid": record["iid"],
                "attributes": normalized,
                "raw_to_normalized": mapping,
            }
        )
    return normalized_records, frequencies
