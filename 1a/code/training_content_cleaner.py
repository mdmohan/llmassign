"""Block-aware cleanup for text that has already passed corpus quality gates.

This stage intentionally runs after ``TextCleaner.filter_corpus``.  It removes
training-irrelevant document furniture while preserving prose, configuration
examples, command output, and useful tables.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


_PAGE_LINE_PATTERNS = (
    re.compile(r"^.*\[\s*Page\s+\d+\s*\]\s*$", re.IGNORECASE),
    re.compile(r"^\s*Page\s+\d+(?:\s+of\s+\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*©.*?Page\s+\d+(?:\s+of\s+\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*RFC\s+\d+\s+.+(?:19|20)\d{2}\s*$", re.IGNORECASE),
)

_RUNNING_LINE_HINTS = re.compile(
    r"(?:configuration guide|command reference|data sheet|cisco public|"
    r"standards track|request for comments|rfc\s+\d+|page\s+\d+)",
    re.IGNORECASE,
)

_BOILERPLATE_PHRASES = (
    "the specifications and information regarding the products",
    "all printed copies and duplicate soft copies are considered uncontrolled",
    "cisco and the cisco logo are trademarks",
    "the documentation set for this product strives to use bias-free language",
    "to provide feedback about cisco technical documentation",
    "use the cisco bug search tool",
    "obtaining documentation and submitting a service request",
    "information about cisco's products, technologies, and network solutions",
    "this document is provided on an as is basis",
)

_COMMAND_START = re.compile(
    r"^\s*(?:configure(?:\s+terminal)?|conf\s+t|"
    r"interface\s+(?:ethernet|loopback|vlan|port-channel|mgmt|fc|nve)\S*|"
    r"router\s+(?:bgp|ospf)\b|feature\s+\S+|vlan\s+\d+\b|"
    r"vrf(?:\s+context)?\s+\S+|show\s+\S+|debug\s+\S+|"
    r"no\s+shutdown\s*$|shutdown\s*$|switchport(?:\s+\S+)?|"
    r"channel-group\s+\d+|ip\s+route\s+|ipv6\s+route\s+|copy\s+|"
    r"terminal\s+|end\s*$|exit\s*$|logging\s+|ntp\s+|snmp-server\s+)",
    re.IGNORECASE,
)

_DEVICE_PROMPT = re.compile(
    r"^\s*(?:switch|router|leaf|spine|n[3579]k)[\w.-]*"
    r"(?:\([^)]*\))*[#>]",
    re.IGNORECASE,
)

_BULLET = re.compile(r"^\s*(?:[-*•‣▪◦]|\d+[.)]|[a-z][.)])\s+")
_DOT_LEADER = re.compile(r"\.{2,}\s*(?:\d+|[ivxlcdm]+)?\s*$", re.IGNORECASE)
_PAGE_ONLY_CELL = re.compile(
    r"^(?:page\s*)?(?:\d+|[ivxlcdm]+)(?:\s*(?:-|of)\s*\d+)?$",
    re.IGNORECASE,
)

_EXTRACTION_GLYPHS = {
    "\ue023": "fi",
    "\ue025": "fl",
    "\ue026": "ff",
    "\ue027": "ffi",
    "\ue029": "ffl",
    "\uf0a7": "•",
    "\uf0b7": "•",
    "\uf0e0": "→",
    "\uf044": "Δ",
}


def _preview(text: str, limit: int = 180) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _normalized_line(line: str) -> str:
    normalized = re.sub(r"\d+", "<n>", line.casefold())
    return re.sub(r"\s+", " ", normalized).strip()


def _normalized_block(block: str) -> str:
    return re.sub(r"\s+", " ", block).strip().casefold()


class TrainingContentCleaner:
    """Remove high-confidence non-content blocks from retained documents."""

    def __init__(
        self,
        min_content_chars: int = 200,
        min_alphabetic_ratio: float = 0.25,
        max_cli_lines: int = 160,
        max_samples_per_reason: int = 3,
    ) -> None:
        self.min_content_chars = min_content_chars
        self.min_alphabetic_ratio = min_alphabetic_ratio
        self.max_cli_lines = max_cli_lines
        self.max_samples_per_reason = max_samples_per_reason

    @staticmethod
    def _document_type(text: str) -> str:
        beginning = text[:12000]
        if re.search(r"Request for Comments:\s*\d+", beginning, re.IGNORECASE):
            return "rfc"
        if re.search(r"\b(?:Cisco|Nexus|NX-OS)\b", beginning, re.IGNORECASE):
            return "cisco"
        if re.search(r"\bJuniper Networks?\b", beginning, re.IGNORECASE):
            return "juniper"
        return "generic"

    @staticmethod
    def _record_removal(
        stats: dict,
        reason: str,
        text: str,
        units: int = 1,
    ) -> None:
        entry = stats["removed_by_reason"].setdefault(
            reason, {"blocks": 0, "characters": 0}
        )
        entry["blocks"] += units
        entry["characters"] += len(text)
        samples = stats["removal_samples"]
        if (
            text.strip()
            and sum(sample["reason"] == reason for sample in samples)
            < stats["max_samples_per_reason"]
        ):
            samples.append({"reason": reason, "preview": _preview(text)})

    def _remove_line_ranges(
        self,
        lines: list[str],
        ranges: list[tuple[int, int, str]],
        stats: dict,
    ) -> list[str]:
        removed = set()
        for start, end, reason in ranges:
            valid = [index for index in range(max(0, start), min(end, len(lines)))]
            fresh = [index for index in valid if index not in removed]
            if fresh:
                text = "\n".join(lines[index] for index in fresh)
                self._record_removal(stats, reason, text)
                removed.update(fresh)
        return [line for index, line in enumerate(lines) if index not in removed]

    def _remove_front_matter(
        self, text: str, document_type: str, stats: dict
    ) -> str:
        lines = text.splitlines()
        ranges: list[tuple[int, int, str]] = []

        if document_type == "rfc":
            status = next(
                (
                    i
                    for i, line in enumerate(lines[:300])
                    if line.strip().casefold() == "status of this memo"
                ),
                None,
            )
            abstract = next(
                (
                    i
                    for i, line in enumerate(lines[:500])
                    if line.strip().casefold() == "abstract"
                ),
                None,
            )
            if status is not None and abstract is not None and status < abstract:
                ranges.append((status, abstract, "rfc_status_and_legal"))

            toc = next(
                (
                    i
                    for i, line in enumerate(lines[:1800])
                    if line.strip().casefold() == "table of contents"
                ),
                None,
            )
            if toc is not None:
                body = next(
                    (
                        i
                        for i in range(toc + 1, len(lines))
                        if re.fullmatch(
                            r"\s*1\.\s+Introduction\s*", lines[i], re.IGNORECASE
                        )
                    ),
                    None,
                )
                if body is not None and body - toc >= 5:
                    ranges.append((toc, body, "table_of_contents"))

        if document_type == "cisco":
            legal_start = next(
                (
                    i
                    for i, line in enumerate(lines[:500])
                    if "THE SPECIFICATIONS AND INFORMATION REGARDING THE PRODUCTS"
                    in line.upper()
                ),
                None,
            )
            if legal_start is not None:
                legal_end = next(
                    (
                        i + 1
                        for i in range(legal_start, min(len(lines), legal_start + 250))
                        if "ALL RIGHTS RESERVED" in lines[i].upper()
                    ),
                    min(len(lines), legal_start + 80),
                )
                ranges.append((legal_start, legal_end, "legal_boilerplate"))

            headquarters = next(
                (
                    i
                    for i, line in enumerate(lines[:300])
                    if line.strip().casefold() == "americas headquarters"
                ),
                None,
            )
            if headquarters is not None:
                end = headquarters + 1
                while end < min(len(lines), headquarters + 25):
                    if not lines[end].strip() and end > headquarters + 5:
                        break
                    end += 1
                ranges.append((headquarters, end, "contact_block"))

            toc = next(
                (
                    i
                    for i, line in enumerate(lines[:2500])
                    if re.fullmatch(
                        r"\s*(?:C\s+O\s+N\s+T\s+E\s+N\s+T\s+S|Contents)\s*",
                        line,
                        re.IGNORECASE,
                    )
                ),
                None,
            )
            if toc is not None:
                candidates = [
                    i
                    for i in range(toc + 1, len(lines))
                    if re.fullmatch(
                        r"\s*(?:Preface|Introduction|Overview|Background Information|"
                        r"Product Overview|C\s*H\s*A\s*P\s*T\s*E\s*R\s+1)\s*",
                        lines[i],
                        re.IGNORECASE,
                    )
                ]
                # A TOC entry is normally followed by another short entry; the
                # real body heading is followed by sentence-like prose. This
                # works for both compact technical notes and long manuals.
                body = next(
                    (
                        i
                        for i in candidates
                        if i - toc >= 3
                        and any(
                            len(lines[j].strip()) >= 70
                            or (
                                len(lines[j].strip()) >= 35
                                and lines[j].strip().endswith((".", ":"))
                            )
                            for j in range(i + 1, min(len(lines), i + 5))
                        )
                    ),
                    None,
                )
                if body is not None:
                    ranges.append((toc, body, "table_of_contents"))

        if not ranges:
            return text
        return "\n".join(self._remove_line_ranges(lines, ranges, stats))

    def _remove_running_lines(self, text: str, stats: dict) -> str:
        lines = text.splitlines()
        counts = Counter(
            _normalized_line(line)
            for line in lines
            if 8 <= len(line.strip()) <= 220
        )
        repeated = {
            line
            for line, count in counts.items()
            if count >= 3 and _RUNNING_LINE_HINTS.search(line)
        }

        retained: list[str] = []
        for line in lines:
            stripped = line.strip()
            reason = None
            if any(pattern.fullmatch(stripped) for pattern in _PAGE_LINE_PATTERNS):
                reason = "page_header_or_footer"
            elif _normalized_line(line) in repeated:
                reason = "repeated_running_header_or_footer"
            elif re.fullmatch(r"©\s*\d{4}.*", stripped, re.IGNORECASE):
                reason = "copyright_line"

            if reason:
                self._record_removal(stats, reason, line)
            else:
                retained.append(line)
        return "\n".join(retained)

    @staticmethod
    def _is_toc_block(lines: list[str]) -> bool:
        non_empty = [line.strip() for line in lines if line.strip()]
        if len(non_empty) < 3:
            return False
        heading = non_empty[0].casefold() in {"contents", "table of contents"}
        dot_leaders = sum(bool(_DOT_LEADER.search(line)) for line in non_empty)
        return (
            heading and dot_leaders >= 2
        ) or (len(non_empty) >= 5 and dot_leaders / len(non_empty) >= 0.6)

    @staticmethod
    def _is_boilerplate_block(block: str, lines: list[str]) -> bool:
        normalized = _normalized_block(block)
        if any(phrase in normalized for phrase in _BOILERPLATE_PHRASES):
            return True

        non_empty = [line for line in lines if line.strip()]
        if len(non_empty) < 3:
            return False
        contact_lines = sum(
            bool(
                re.search(
                    r"(?:https?://|www\.|\S+@\S+|"
                    r"\b(?:tel|telephone|fax)\s*[:.]?\s*\+?\d)",
                    line,
                    re.IGNORECASE,
                )
            )
            for line in non_empty
        )
        return contact_lines / len(non_empty) > 0.5

    @staticmethod
    def _is_table(lines: list[str]) -> bool:
        non_empty = [line for line in lines if line.strip()]
        return bool(non_empty) and sum(
            line.lstrip().startswith("|") for line in non_empty
        ) / len(non_empty) >= 0.5

    @staticmethod
    def _is_cli(lines: list[str]) -> bool:
        non_empty = [line for line in lines if line.strip()]
        if not non_empty:
            return False
        prompts = sum(bool(_DEVICE_PROMPT.match(line)) for line in non_empty)
        commands = sum(
            len(line.strip()) <= 160
            and not line.rstrip().endswith(".")
            and bool(_COMMAND_START.match(line))
            for line in non_empty
        )
        command_ratio = (prompts + commands) / len(non_empty)
        return (
            command_ratio >= 0.35
            or (prompts >= 3 and command_ratio >= 0.10)
            or (commands >= 4 and command_ratio >= 0.20)
        )

    @staticmethod
    def _is_ascii_diagram(lines: list[str]) -> bool:
        non_empty = [line for line in lines if line.strip()]
        if len(non_empty) < 4:
            return False
        characters = "".join(non_empty)
        if not characters:
            return False
        symbols = sum(char in "|+-=<>/\\_*" for char in characters)
        letters = sum(char.isalpha() for char in characters)
        return symbols / len(characters) > 0.28 and letters / len(characters) < 0.35

    @staticmethod
    def _is_garbled_columns(lines: list[str]) -> bool:
        non_empty = [line.strip() for line in lines if line.strip()]
        if len(non_empty) < 8:
            return False
        word_counts = [len(re.findall(r"\b\w+\b", line)) for line in non_empty]
        average_words = sum(word_counts) / len(word_counts)
        terminal = sum(line.endswith((".", ":", ";", "?", "!")) for line in non_empty)
        mid_bullets = sum(bool(re.search(r"\S\s+[•▪‣]\s+\S", line)) for line in non_empty)
        wide_gaps = sum(bool(re.search(r"\S\s{5,}\S", line)) for line in non_empty)
        return (
            3 <= average_words <= 12
            and terminal / len(non_empty) < 0.15
            and (mid_bullets >= 2 or wide_gaps / len(non_empty) >= 0.5)
        )

    @staticmethod
    def _normalize_table(lines: list[str]) -> tuple[str | None, bool]:
        rows: list[list[str]] = []
        for line in lines:
            if not line.lstrip().startswith("|"):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if cells:
                rows.append(cells)
        if len(rows) < 2:
            return None, False

        width = max(len(row) for row in rows)
        rows = [row + [""] * (width - len(row)) for row in rows]
        keep_columns = [
            index for index in range(width) if any(row[index] for row in rows)
        ]
        rows = [[row[index] for index in keep_columns] for row in rows]
        if not rows or not rows[0]:
            return None, False

        meaningful = [
            row
            for row in rows
            if any(cell for cell in row)
            and not all(
                not cell
                or re.fullmatch(r":?-{2,}:?", cell)
                or _PAGE_ONLY_CELL.fullmatch(cell)
                for cell in row
            )
        ]
        content_cells = [
            cell
            for row in rows
            for cell in row
            if cell
            and not re.fullmatch(r":?-{2,}:?", cell)
            and not _PAGE_ONLY_CELL.fullmatch(cell)
            and cell.casefold() not in {"log in", "login"}
        ]
        non_empty_cells = sum(bool(cell) for row in rows for cell in row)
        total_cells = len(rows) * len(rows[0])
        alpha_cells = sum(bool(re.search(r"[A-Za-z]", cell)) for row in rows for cell in row)
        density = non_empty_cells / total_cells
        if (len(rows[0]) == 1 or density < 0.5) and content_cells:
            # Some HTML-derived Cisco PDFs place an entire page in one cell of
            # a mostly empty layout grid. This is prose disguised as a table,
            # not a low-information table. Recover the populated cells in
            # reading order instead of rejecting the guide.
            flattened = "\n\n".join(content_cells)
            visible = sum(not char.isspace() for char in flattened)
            alpha = sum(char.isalpha() for char in flattened)
            if len(flattened) >= 200 and alpha / max(1, visible) >= 0.25:
                return flattened, True
        if len(meaningful) < 2 or density < 0.5 or alpha_cells == 0:
            return None, False

        # Normalize separator rows and ensure that a Markdown separator exists.
        normalized_rows = [
            [re.sub(r"\s+", " ", cell).strip() for cell in row] for row in rows
        ]
        separator = ["---"] * len(normalized_rows[0])
        if len(normalized_rows) > 1 and all(
            re.fullmatch(r":?-{2,}:?", cell or "") for cell in normalized_rows[1]
        ):
            normalized_rows[1] = separator
        else:
            normalized_rows.insert(1, separator)
        return "\n".join(
            "| " + " | ".join(row) + " |" for row in normalized_rows
        ), False

    @staticmethod
    def _normalize_spaced_heading(line: str) -> str:
        match = re.fullmatch(
            r"\s*((?:[A-Za-z]\s+){2,}[A-Za-z])(?:\s+(\d+))?\s*", line
        )
        if not match:
            return line.strip()
        word = re.sub(r"\s+", "", match.group(1)).title()
        return f"{word} {match.group(2)}" if match.group(2) else word

    @staticmethod
    def _looks_like_heading(line: str) -> bool:
        stripped = line.strip()
        words = re.findall(r"[A-Za-z0-9][\w/-]*", stripped)
        if not stripped or len(stripped) > 110 or not 1 <= len(words) <= 14:
            return False
        if stripped.endswith((".", ",", ";", "?", "!")):
            return False
        if re.match(r"^(?:CHAPTER|Chapter|Appendix|Preface|Overview)\b", stripped):
            return True
        alpha = [char for char in stripped if char.isalpha()]
        if alpha and all(char.isupper() for char in alpha):
            return True
        title_words = sum(word[:1].isupper() for word in words)
        return len(words) <= 8 and title_words / len(words) >= 0.75

    def _reflow_prose(self, lines: list[str]) -> tuple[str, str]:
        output: list[str] = []
        paragraph: list[str] = []
        heading_count = 0

        def flush() -> None:
            if not paragraph:
                return
            joined = paragraph[0]
            for continuation in paragraph[1:]:
                if joined.endswith("-") and continuation[:1].islower():
                    joined = joined[:-1] + continuation
                else:
                    joined += " " + continuation
            output.append(re.sub(r"\s+", " ", joined).strip())
            paragraph.clear()

        for original in lines:
            line = self._normalize_spaced_heading(original)
            if not line:
                flush()
            elif _BULLET.match(line):
                flush()
                output.append(re.sub(r"\s+", " ", line).strip())
            elif self._looks_like_heading(line):
                flush()
                output.append(line)
                heading_count += 1
            else:
                paragraph.append(line.strip())
        flush()
        block_type = "heading" if output and heading_count == len(output) else "prose"
        return "\n\n".join(part for part in output if part), block_type

    def clean_document(self, text: str, source_file: str = "") -> tuple[str, dict]:
        raw_text = text or ""
        stats = {
            "source_file": source_file,
            "raw_characters": len(raw_text),
            "clean_characters": 0,
            "removed_characters": 0,
            "removed_percentage": 0.0,
            "removed_by_reason": {},
            "kept_blocks": defaultdict(int),
            "removal_samples": [],
            "normalized_glyphs": {},
            "max_samples_per_reason": self.max_samples_per_reason,
        }
        document_type = self._document_type(raw_text)
        glyph_counts = Counter(char for char in raw_text if char in _EXTRACTION_GLYPHS)
        if glyph_counts:
            stats["normalized_glyphs"] = {
                f"U+{ord(char):04X}": count
                for char, count in sorted(glyph_counts.items(), key=lambda item: ord(item[0]))
            }
        text = raw_text.translate(str.maketrans(_EXTRACTION_GLYPHS))
        text = self._remove_front_matter(text, document_type, stats)
        text = self._remove_running_lines(text, stats)

        cleaned_blocks: list[str] = []
        seen_blocks: set[str] = set()
        for block in re.split(r"\n\s*\n+", text):
            block = block.strip()
            if not block:
                continue
            lines = block.splitlines()

            if self._is_toc_block(lines):
                self._record_removal(stats, "table_of_contents", block)
                continue
            if self._is_boilerplate_block(block, lines):
                self._record_removal(stats, "boilerplate", block)
                continue
            if self._is_garbled_columns(lines):
                self._record_removal(stats, "garbled_columns", block)
                continue

            if self._is_table(lines):
                normalized, is_layout_table = self._normalize_table(lines)
                if normalized is None:
                    self._record_removal(stats, "low_information_table", block)
                    continue
                if is_layout_table:
                    cleaned, _ = self._reflow_prose(normalized.splitlines())
                    block_type = "layout_table_text"
                else:
                    cleaned = normalized
                    block_type = "table"
            elif self._is_ascii_diagram(lines):
                self._record_removal(stats, "ascii_diagram", block)
                continue
            elif document_type != "rfc" and self._is_cli(lines):
                non_empty = [line.rstrip() for line in lines if line.strip()]
                if len(non_empty) > self.max_cli_lines:
                    omitted = "\n".join(non_empty[self.max_cli_lines :])
                    self._record_removal(stats, "oversized_cli_tail", omitted)
                    non_empty = non_empty[: self.max_cli_lines]
                cleaned = "```text\n" + "\n".join(non_empty) + "\n```"
                block_type = "cli"
            else:
                cleaned, block_type = self._reflow_prose(lines)

            normalized_key = _normalized_block(cleaned)
            if not normalized_key:
                continue
            if len(normalized_key) >= 120 and normalized_key in seen_blocks:
                self._record_removal(stats, "duplicate_block", block)
                continue
            if len(normalized_key) >= 120:
                seen_blocks.add(normalized_key)
            cleaned_blocks.append(cleaned)
            stats["kept_blocks"][block_type] += 1

        cleaned_text = re.sub(r"\n{3,}", "\n\n", "\n\n".join(cleaned_blocks)).strip()
        alpha_count = sum(char.isalpha() for char in cleaned_text)
        visible_count = sum(not char.isspace() for char in cleaned_text)
        alphabetic_ratio = alpha_count / max(1, visible_count)

        stats["clean_characters"] = len(cleaned_text)
        stats["removed_characters"] = max(0, len(raw_text) - len(cleaned_text))
        stats["removed_percentage"] = round(
            100 * stats["removed_characters"] / max(1, len(raw_text)), 2
        )
        stats["alphabetic_ratio"] = round(alphabetic_ratio, 4)
        stats["document_type"] = document_type
        stats["kept_blocks"] = dict(stats["kept_blocks"])
        del stats["max_samples_per_reason"]
        return cleaned_text, stats

    @staticmethod
    def _cleaned_filename(data: dict, source_file: str, used: set[str]) -> str:
        candidate = data.get("output_filename") or source_file
        stem = Path(str(candidate)).stem or "document"
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "document"
        filename = f"{stem}.txt"
        if filename in used:
            digest = hashlib.sha256(source_file.encode("utf-8")).hexdigest()[:8]
            filename = f"{stem}_{digest}.txt"
        used.add(filename)
        return filename

    def clean_corpus(
        self,
        documents: dict,
        audit_report_path: str | Path,
        cleaned_text_dir: str | Path | None = None,
        upstream_rejected: dict | None = None,
    ) -> tuple[dict, dict]:
        """Clean retained documents and write a consolidated final audit."""
        upstream_rejected = upstream_rejected or {}
        retained: dict = {}
        rejected: dict = {}
        retained_records: list[dict] = []
        rejected_records: list[dict] = []
        used_names: set[str] = set()
        output_dir = Path(cleaned_text_dir) if cleaned_text_dir is not None else None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        characters_before = 0
        characters_after = 0
        aggregate_reasons: Counter[str] = Counter()
        aggregate_reason_chars: Counter[str] = Counter()

        for key, original_data in documents.items():
            data = dict(original_data)
            source_file = str(key)
            output_value = data.get("output_filename")
            output_filename = str(output_value) if output_value else None
            text_value = data.get("extracted_text") or ""
            text = text_value if isinstance(text_value, str) else str(text_value)
            cleaned, details = self.clean_document(text, source_file)
            characters_before += details["raw_characters"]
            characters_after += details["clean_characters"]
            for reason, values in details["removed_by_reason"].items():
                aggregate_reasons[reason] += values["blocks"]
                aggregate_reason_chars[reason] += values["characters"]

            clean_name = self._cleaned_filename(data, source_file, used_names)
            rejection_reason = None
            if len(cleaned) < self.min_content_chars:
                rejection_reason = "insufficient_content_after_cleaning"
            elif details["alphabetic_ratio"] < self.min_alphabetic_ratio:
                rejection_reason = "low_alphabetic_content_after_cleaning"

            common = {
                "source_file": source_file,
                "output_filename": output_filename,
                "document_type": details["document_type"],
                "details": details,
            }
            if rejection_reason:
                record = {
                    **common,
                    "cleaned_output_filename": None,
                    "status": "rejected",
                    "reason": rejection_reason,
                    "rejection_stage": "content_cleaning",
                    "retained_original": None,
                }
                rejected[key] = record
                rejected_records.append(record)
                continue

            cleaned_output = None
            if output_dir is not None:
                destination = output_dir / clean_name
                destination.write_text(cleaned + "\n", encoding="utf-8")
                cleaned_output = str(destination)
            data["extracted_text"] = cleaned
            if cleaned_output is not None:
                data["cleaned_output_filename"] = cleaned_output
            retained[key] = data
            retained_records.append(
                {
                    **common,
                    "cleaned_output_filename": cleaned_output,
                    "status": "retained",
                }
            )

        report_path = Path(audit_report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        upstream_rejected_records = [
            {**dict(record), "rejection_stage": "document_quality_gates"}
            for record in upstream_rejected.values()
        ]
        all_rejected_records = upstream_rejected_records + rejected_records
        report = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "documents_received": len(documents) + len(upstream_rejected),
                "documents_entering_content_cleaning": len(documents),
                "documents_retained": len(retained),
                "documents_rejected_by_gates": len(upstream_rejected),
                "documents_rejected_after_content_cleaning": len(rejected),
                "documents_rejected": len(all_rejected_records),
                "characters_before": characters_before,
                "characters_after": characters_after,
                "removed_characters": max(0, characters_before - characters_after),
                "removed_percentage": round(
                    100 * max(0, characters_before - characters_after)
                    / max(1, characters_before),
                    2,
                ),
                "removals_by_reason": {
                    reason: {
                        "blocks": aggregate_reasons[reason],
                        "characters": aggregate_reason_chars[reason],
                    }
                    for reason in sorted(aggregate_reasons)
                },
            },
            "configuration": {
                "min_content_chars": self.min_content_chars,
                "min_alphabetic_ratio": self.min_alphabetic_ratio,
                "max_cli_lines": self.max_cli_lines,
                "cleaned_text_dir": str(output_dir) if output_dir else None,
            },
            "retained_documents": retained_records,
            # This final list is intentionally consolidated. Upstream duplicate
            # records retain their retained_original lineage from TextCleaner.
            "rejected_documents": all_rejected_records,
        }
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return retained, rejected
