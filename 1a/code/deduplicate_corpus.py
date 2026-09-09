"""Combine cleaned text files and conservatively deduplicate prose paragraphs.

This stage is intended to run after document/content cleaning and before
tokenization or a train/test split. It removes exact duplicates after Unicode,
case, and whitespace normalization. Short blocks, command examples, tables,
and procedural lists are deliberately preserved because identical-looking
technical fragments may be meaningful in more than one document context.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from cli_parsers import build_corpus_dedup_parser


_LIST_ITEM = re.compile(r"^\s*(?:[-*+•]|\d+[.)]|[A-Za-z][.)])\s+")
_CLI_LINE = re.compile(
    r"^\s*(?:"
    r"(?:switch|router|leaf|spine|n[3579]k)[\w.-]*(?:\([^)]*\))*[#>]"
    r"|configure(?:\s+terminal)?\b|conf\s+t\b|show\s+\S+"
    r"|interface\s+\S+|router\s+(?:bgp|ospf)\b|feature\s+\S+"
    r"|vlan\s+\d+\b|vrf(?:\s+context)?\s+\S+|ip\s+route\b"
    r"|no\s+shutdown\s*$|shutdown\s*$|switchport\b|exit\s*$|end\s*$"
    r")",
    re.IGNORECASE,
)


def _normalize_paragraph(text: str) -> str:
    """Normalize harmless presentation differences before exact comparison."""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def _paragraph_kind(paragraph: str, minimum_chars: int) -> str:
    """Classify blocks that should not participate in prose deduplication."""
    normalized = _normalize_paragraph(paragraph)
    if len(normalized) < minimum_chars:
        return "short"

    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    if not lines:
        return "empty"
    if "```" in paragraph:
        return "fenced_code"

    table_lines = sum(
        (line.count("|") >= 2) or (line.count("\t") >= 2)
        for line in lines
    )
    if table_lines >= 2 and table_lines / len(lines) >= 0.5:
        return "table"

    list_lines = sum(bool(_LIST_ITEM.match(line)) for line in lines)
    if list_lines >= 2 and list_lines / len(lines) >= 0.4:
        return "list"

    cli_lines = sum(bool(_CLI_LINE.match(line)) for line in lines)
    if cli_lines >= 2 and cli_lines / len(lines) >= 0.4:
        return "cli"
    return "prose"


def _text_files(
    input_dir: Path,
    *,
    recursive: bool,
    excluded_paths: set[Path] | None = None,
) -> list[Path]:
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    excluded = {path.expanduser().resolve() for path in excluded_paths or set()}
    candidates = input_dir.rglob("*.txt") if recursive else input_dir.glob("*.txt")
    files = sorted(
        path.resolve()
        for path in candidates
        if path.is_file() and path.resolve() not in excluded
    )
    if not files:
        raise ValueError(f"No .txt files found under: {input_dir}")
    return files


def _process_corpus(
    input_dir: str | Path,
    *,
    recursive: bool = True,
    minimum_paragraph_chars: int = 150,
    max_occurrences: int = 1,
    excluded_paths: set[Path] | None = None,
) -> tuple[str, dict]:
    """Return combined text and a detailed paragraph-level audit."""
    if minimum_paragraph_chars < 1:
        raise ValueError("minimum_paragraph_chars must be at least 1")
    if max_occurrences < 1:
        raise ValueError("max_occurrences must be at least 1")

    resolved_input = Path(input_dir).expanduser().resolve()
    files = _text_files(
        resolved_input,
        recursive=recursive,
        excluded_paths=excluded_paths,
    )

    occurrence_counts: Counter[str] = Counter()
    retained_origins: dict[str, dict] = {}
    duplicate_records = []
    combined_documents = []
    kind_counts: Counter[str] = Counter()
    blocks_before = 0
    blocks_after = 0
    characters_before = 0
    documents_with_content = 0

    for source_path in files:
        try:
            raw_text = source_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Text file is not valid UTF-8: {source_path}") from exc

        relative_source = str(source_path.relative_to(resolved_input))
        characters_before += len(raw_text)
        retained_blocks = []
        paragraphs = [
            block.strip()
            for block in re.split(r"\n\s*\n+", raw_text.replace("\r\n", "\n"))
            if block.strip()
        ]
        blocks_before += len(paragraphs)

        for block_index, paragraph in enumerate(paragraphs, start=1):
            kind = _paragraph_kind(paragraph, minimum_paragraph_chars)
            kind_counts[kind] += 1
            if kind != "prose":
                retained_blocks.append(paragraph)
                continue

            normalized = _normalize_paragraph(paragraph)
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            occurrence_counts[digest] += 1
            if occurrence_counts[digest] <= max_occurrences:
                retained_origins.setdefault(
                    digest,
                    {
                        "source_file": relative_source,
                        "paragraph_index": block_index,
                    },
                )
                retained_blocks.append(paragraph)
                continue

            duplicate_records.append(
                {
                    "sha256": digest,
                    "removed_source_file": relative_source,
                    "removed_paragraph_index": block_index,
                    "retained_original": retained_origins[digest],
                    "characters_removed": len(paragraph),
                    "preview": re.sub(r"\s+", " ", paragraph)[:180],
                }
            )

        if retained_blocks:
            document_text = "\n\n".join(retained_blocks)
            combined_documents.append(document_text)
            documents_with_content += 1
            blocks_after += len(retained_blocks)

    combined_text = "\n\n".join(combined_documents).strip()
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(resolved_input),
        "configuration": {
            "recursive": recursive,
            "minimum_paragraph_chars": minimum_paragraph_chars,
            "max_occurrences": max_occurrences,
            "normalization": "Unicode NFKC, case folding, and whitespace collapse",
            "preserved_block_types": [
                "short",
                "fenced_code",
                "table",
                "list",
                "cli",
            ],
        },
        "summary": {
            "files_read": len(files),
            "documents_with_retained_content": documents_with_content,
            "paragraphs_before": blocks_before,
            "paragraphs_after": blocks_after,
            "duplicate_paragraphs_removed": len(duplicate_records),
            "duplicate_paragraph_ratio": (
                len(duplicate_records) / blocks_before if blocks_before else 0.0
            ),
            "characters_before": characters_before,
            "characters_after": len(combined_text),
            "duplicate_characters_removed": sum(
                record["characters_removed"] for record in duplicate_records
            ),
            "block_types": dict(sorted(kind_counts.items())),
        },
        "removed_duplicates": duplicate_records,
    }
    return combined_text, report


def combine_and_deduplicate_text(
    input_dir: str | Path,
    *,
    recursive: bool = True,
    minimum_paragraph_chars: int = 150,
    max_occurrences: int = 1,
) -> str:
    """Combine a text corpus and return conservatively deduplicated text.

    This is the stable callable entry point intended for later use by the
    preparation pipeline. It does not write files or mutate the source corpus.
    """
    combined_text, _ = _process_corpus(
        input_dir,
        recursive=recursive,
        minimum_paragraph_chars=minimum_paragraph_chars,
        max_occurrences=max_occurrences,
    )
    return combined_text


def main() -> int:
    parser = build_corpus_dedup_parser()
    args = parser.parse_args()
    output_file = args.output_file.expanduser().resolve()
    audit_report = (
        args.audit_report.expanduser().resolve()
        if args.audit_report is not None
        else None
    )
    excluded_paths = {output_file}
    if audit_report is not None:
        excluded_paths.add(audit_report)

    try:
        combined_text, report = _process_corpus(
            args.input_dir,
            recursive=args.recursive,
            minimum_paragraph_chars=args.minimum_paragraph_chars,
            max_occurrences=args.max_occurrences,
            excluded_paths=excluded_paths,
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(combined_text + "\n", encoding="utf-8")
        report["output_file"] = str(output_file)

        if audit_report is not None:
            audit_report.parent.mkdir(parents=True, exist_ok=True)
            audit_report.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")

    summary = report["summary"]
    print("\n=== Corpus Paragraph Deduplication ===")
    print(f"Files read:                    {summary['files_read']:,}")
    print(f"Paragraphs before:             {summary['paragraphs_before']:,}")
    print(f"Paragraphs after:              {summary['paragraphs_after']:,}")
    print(
        "Duplicate paragraphs removed: "
        f"{summary['duplicate_paragraphs_removed']:,} "
        f"({100 * summary['duplicate_paragraph_ratio']:.2f}%)"
    )
    print(
        "Duplicate characters removed: "
        f"{summary['duplicate_characters_removed']:,}"
    )
    print(f"Combined text saved to:        {output_file}")
    if audit_report is not None:
        print(f"Audit report saved to:         {audit_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
