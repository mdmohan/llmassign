"""Meaning-preserving text normalization and document filters."""

from __future__ import annotations

import re
import unicodedata


_BLANK_LINES = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")


def normalize_text(text: str) -> str:
    """Normalize encoding and line endings without changing line indentation."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    return _BLANK_LINES.sub("\n\n", normalized).strip()


def normalized_paragraphs(text: str) -> list[str]:
    """Return non-empty paragraphs normalized for duplicate comparison."""
    return [" ".join(part.split()) for part in re.split(r"\n\s*\n", text) if part.strip()]


def repetition_ratio(text: str) -> float:
    """Return the fraction of paragraphs duplicating an earlier paragraph."""
    paragraphs = normalized_paragraphs(text)
    if not paragraphs:
        return 0.0
    duplicate_count = len(paragraphs) - len(set(paragraphs))
    return duplicate_count / len(paragraphs)


def fails_length_filter(text: str, minimum_characters: int = 50) -> bool:
    return len(text) < minimum_characters


def fails_repetition_filter(text: str, maximum_ratio: float = 0.30) -> bool:
    return repetition_ratio(text) > maximum_ratio
