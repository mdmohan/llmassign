"""Conservatively remove machine artifacts from an already-cleaned corpus.

This is an isolated experiment: source files are never modified. Every input
``.txt`` file is written under a separate output directory, preserving its
relative path, and a JSON audit records exactly what changed.

The cleaner targets artifacts highlighted by the tokenizer audit—URLs,
Base64/certificate material, long hashes, fingerprints, packed lists, and
obvious PDF word-joining. It deliberately preserves CLI, product identifiers,
protocol terminology, tables, and ordinary technical prose.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)]\(https?://[^)]+\)", re.IGNORECASE)
FINGERPRINT_RE = re.compile(
    r"(?i)(?:fingerprint\s*[=:]?\s*)?(?:[0-9a-f]{2}:){7,}[0-9a-f]{2}"
)
UUID_RE = re.compile(
    r"(?i)(?<![0-9a-f])[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}(?![0-9a-f])"
)
LONG_HEX_RE = re.compile(r"(?i)(?<![0-9a-f])[0-9a-f]{40,}(?![0-9a-f])")
BASE64_RE = re.compile(
    r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{64,}={0,2}(?![A-Za-z0-9+/=])"
)
PEM_BEGIN_RE = re.compile(r"^-{5}BEGIN [A-Z0-9 ]+-{5}$")
PEM_END_RE = re.compile(r"^-{5}END [A-Z0-9 ]+-{5}$")
PACKED_LIST_RE = re.compile(r"\S{60,}")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
TABLE_LABEL_RE = re.compile(r"\b(Table|Figure|Example)(\d+):(?=\S)")
CAMEL_BOUNDARY_RE = re.compile(
    r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Post-clean an existing text corpus into a separate directory "
            "without modifying source files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Existing directory containing cleaned .txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Separate destination for post-cleaned .txt files.",
    )
    parser.add_argument(
        "--audit-report",
        type=Path,
        default=None,
        help="JSON audit path; defaults to post_cleaning_audit.json in output.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Process .txt files recursively and preserve relative paths.",
    )
    parser.add_argument(
        "--remove-urls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove bare URLs while preserving Markdown link labels.",
    )
    parser.add_argument(
        "--remove-encoded-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove PEM blocks, Base64 payloads, fingerprints, and long hashes.",
    )
    parser.add_argument(
        "--split-packed-lists",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Insert spaces after delimiters in very long packed lists.",
    )
    parser.add_argument(
        "--repair-joined-headings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Repair high-confidence CamelCase joining in long heading-like tokens.",
    )
    parser.add_argument(
        "--drop-suspicious-lines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop extremely long lines containing almost no whitespace.",
    )
    parser.add_argument(
        "--min-retained-chars",
        type=int,
        default=50,
        help="Retain the original if post-cleaning leaves fewer characters.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    return parser


def _inside(path: Path, possible_parent: Path) -> bool:
    try:
        path.relative_to(possible_parent)
        return True
    except ValueError:
        return False


def _validate_paths(args) -> tuple[Path, Path, Path, list[Path]]:
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if output_dir == input_dir or _inside(output_dir, input_dir):
        raise ValueError(
            "--output-dir must be separate from and outside --input-dir"
        )
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise ValueError(
            f"Output directory is not empty: {output_dir}. Use --overwrite "
            "to update an existing experimental output."
        )

    iterator = input_dir.rglob("*.txt") if args.recursive else input_dir.glob("*.txt")
    files = sorted(path for path in iterator if path.is_file())
    if not files:
        raise ValueError(f"No .txt files found in: {input_dir}")

    audit_report = (
        args.audit_report.expanduser().resolve()
        if args.audit_report is not None
        else output_dir / "post_cleaning_audit.json"
    )
    return input_dir, output_dir, audit_report, files


def _sample(kind: str, value: str) -> dict:
    compact = re.sub(r"\s+", " ", value).strip()
    return {
        "type": kind,
        "preview": compact[:157] + "..." if len(compact) > 160 else compact,
    }


def _remove_urls(line: str, actions: Counter, samples: list[dict]) -> str:
    def preserve_markdown_label(match):
        actions["markdown_urls_removed"] += 1
        if len(samples) < 30:
            samples.append(_sample("markdown_url", match.group(0)))
        return match.group(1)

    line = MARKDOWN_LINK_RE.sub(preserve_markdown_label, line)
    matches = list(URL_RE.finditer(line))
    if not matches:
        return line

    url_characters = sum(len(match.group(0)) for match in matches)
    if url_characters / max(1, len(line.strip())) >= 0.60:
        actions["url_dominated_lines_removed"] += 1
        if len(samples) < 30:
            samples.append(_sample("url_dominated_line", line))
        return ""

    def remove_bare_url(match):
        actions["inline_urls_removed"] += 1
        value = match.group(0)
        trailing = ""
        while value and value[-1] in ".,;:!?)]}":
            trailing = value[-1] + trailing
            value = value[:-1]
        if len(samples) < 30:
            samples.append(_sample("inline_url", value))
        return trailing

    return URL_RE.sub(remove_bare_url, line)


def _remove_encoded_tokens(
    line: str,
    actions: Counter,
    samples: list[dict],
) -> str:
    patterns = (
        (FINGERPRINT_RE, "fingerprints_removed"),
        (UUID_RE, "uuids_removed"),
        (LONG_HEX_RE, "long_hex_values_removed"),
        (BASE64_RE, "base64_values_removed"),
    )
    for pattern, action_name in patterns:
        matches = list(pattern.finditer(line))
        if not matches:
            continue
        for match in matches:
            actions[action_name] += 1
            if len(samples) < 30:
                samples.append(_sample(action_name, match.group(0)))
        line = pattern.sub("", line)
    return line


def _split_packed_token(token: str) -> tuple[str, bool]:
    delimiter_count = token.count(",") + token.count(";")
    if delimiter_count < 3:
        return token, False
    result = re.sub(r",(?=\S)", ", ", token)
    result = re.sub(r";(?=\S)", "; ", result)
    return result, result != token


def _split_packed_lists(line: str, actions: Counter) -> str:
    def replace(match):
        replacement, changed = _split_packed_token(match.group(0))
        if changed:
            actions["packed_lists_split"] += 1
        return replacement

    return PACKED_LIST_RE.sub(replace, line)


def _repair_joined_headings(line: str, actions: Counter) -> str:
    repaired = TABLE_LABEL_RE.sub(r"\1 \2: ", line)
    if repaired != line:
        actions["table_labels_repaired"] += 1
    line = repaired

    def repair_token(match):
        token = match.group(0)
        if len(token) < 30 or not token.isalpha():
            return token
        boundaries = len(CAMEL_BOUNDARY_RE.findall(token))
        if boundaries < 2:
            return token
        replacement = CAMEL_BOUNDARY_RE.sub(" ", token)
        if replacement != token:
            actions["joined_heading_tokens_repaired"] += 1
        return replacement

    return re.sub(r"[A-Za-z]{30,}", repair_token, line)


def _suspicious_unbroken_line(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 180 or len(stripped.split()) > 2:
        return False
    # Long prose has spaces. Lines this long without them are typically encoded
    # payloads, damaged URLs, or PDF columns fused into a single token.
    return max((len(token) for token in stripped.split()), default=0) >= 160


def clean_text(text: str, args) -> tuple[str, Counter, list[dict]]:
    """Apply conservative line-based cleanup and return an auditable result."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_RE.sub("", text)
    actions = Counter()
    samples = []
    cleaned_lines = []
    inside_pem_block = False

    for original_line in text.split("\n"):
        line = original_line.rstrip()
        stripped = line.strip()

        if args.remove_encoded_data and PEM_BEGIN_RE.match(stripped):
            inside_pem_block = True
            actions["pem_blocks_removed"] += 1
            if len(samples) < 30:
                samples.append(_sample("pem_block", stripped))
            continue
        if inside_pem_block:
            actions["pem_payload_lines_removed"] += 1
            if PEM_END_RE.match(stripped):
                inside_pem_block = False
            continue

        if args.remove_urls:
            line = _remove_urls(line, actions, samples)
        if args.remove_encoded_data:
            line = _remove_encoded_tokens(line, actions, samples)
        if args.split_packed_lists:
            line = _split_packed_lists(line, actions)
        if args.repair_joined_headings:
            line = _repair_joined_headings(line, actions)

        # Preserve leading spaces and internal alignment: they carry meaning in
        # CLI examples, configuration blocks, and extracted tables.
        line = line.rstrip()
        if args.drop_suspicious_lines and _suspicious_unbroken_line(line):
            actions["suspicious_unbroken_lines_removed"] += 1
            if len(samples) < 30:
                samples.append(_sample("suspicious_unbroken_line", line))
            continue
        cleaned_lines.append(line)

    # More than two blank lines carries no linguistic information and creates
    # unnecessary whitespace tokens during causal language-model training.
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if cleaned:
        cleaned += "\n"
    return cleaned, actions, samples


def _text_metrics(text: str) -> dict:
    return {
        "characters": len(text),
        "lines": len(text.splitlines()),
        "words": len(text.split()),
        "urls": len(URL_RE.findall(text)),
        "long_hex_values": len(LONG_HEX_RE.findall(text)),
        "base64_values": len(BASE64_RE.findall(text)),
    }


def run(args) -> dict:
    input_dir, output_dir, audit_report, files = _validate_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_actions = Counter()
    records = []

    print(f"Post-cleaning {len(files):,} files")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    for index, source_path in enumerate(files, start=1):
        relative_path = source_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        original = source_path.read_text(encoding="utf-8", errors="replace")
        cleaned, actions, samples = clean_text(original, args)

        retained_original = False
        if len(cleaned.strip()) < args.min_retained_chars:
            cleaned = original
            retained_original = True
            actions["original_retained_below_minimum"] += 1

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(cleaned, encoding="utf-8")
        aggregate_actions.update(actions)
        before = _text_metrics(original)
        after = _text_metrics(cleaned)
        records.append(
            {
                "source_file": str(source_path),
                "output_file": str(output_path),
                "relative_file": str(relative_path),
                "changed": cleaned != original,
                "retained_original": retained_original,
                "before": before,
                "after": after,
                "characters_removed": before["characters"] - after["characters"],
                "actions": dict(actions),
                "removed_samples": samples,
            }
        )
        if index % 25 == 0 or index == len(files):
            print(f"  processed {index:,}/{len(files):,} files")

    total_before = sum(record["before"]["characters"] for record in records)
    total_after = sum(record["after"]["characters"] for record in records)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": {
            "argv": [sys.executable, *sys.argv],
            "reconstructed_full_command": shlex.join([sys.executable, *sys.argv]),
            "working_directory": str(Path.cwd()),
        },
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "summary": {
            "input_files": len(files),
            "output_files": len(records),
            "changed_files": sum(record["changed"] for record in records),
            "unchanged_files": sum(not record["changed"] for record in records),
            "originals_retained_for_safety": sum(
                record["retained_original"] for record in records
            ),
            "characters_before": total_before,
            "characters_after": total_after,
            "characters_removed": total_before - total_after,
            "percent_characters_removed": (
                (total_before - total_after) / total_before * 100
                if total_before
                else 0.0
            ),
            "actions": dict(aggregate_actions),
        },
        "files": records,
    }
    audit_report.parent.mkdir(parents=True, exist_ok=True)
    audit_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = report["summary"]
    print("\n=== Post-cleaning summary ===")
    print(f"Changed files:       {summary['changed_files']:,}/{len(files):,}")
    print(f"Characters removed:  {summary['characters_removed']:,}")
    print(f"Corpus reduction:    {summary['percent_characters_removed']:.2f}%")
    print(
        "Originals retained:  "
        f"{summary['originals_retained_for_safety']:,}"
    )
    for name, count in sorted(aggregate_actions.items()):
        print(f"  {name}: {count:,}")
    print(f"Audit report:        {audit_report}")
    return report


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.min_retained_chars < 0:
        parser.error("--min-retained-chars must be zero or greater")
    try:
        run(args)
    except KeyboardInterrupt:
        parser.exit(130, "\nPost-cleaning interrupted by user.\n")
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
