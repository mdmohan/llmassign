import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

from datasketch import MinHash, MinHashLSH
from langdetect import DetectorFactory, LangDetectException, detect


# langdetect is non-deterministic unless its random seed is fixed. Reproducible
# classifications are important for a cleaning audit.
DetectorFactory.seed = 0


_EXTRACTED_TABLES_MARKER = re.compile(
    r"^[ \t]*\[EXTRACTED TABLES\]:?[ \t]*$", flags=re.IGNORECASE
)


def _normalize_edge_line(line: str) -> str:
    """Canonicalize a line only for repeated header/footer comparison."""
    return re.sub(r"\s+", " ", line).strip().casefold()


def _is_header_footer_candidate(line: str) -> bool:
    """Conservatively identify prose-like page-edge boilerplate."""
    normalized = _normalize_edge_line(line)
    if not 12 <= len(normalized) <= 200:
        return False

    # Preserve Markdown tables, fenced code, and common CLI/configuration lines.
    if line.lstrip().startswith(("|", "```")):
        return False
    if any(token in line for token in ("#", "{", "}")):
        return False

    words = re.findall(r"[A-Za-z][A-Za-z0-9-]*", normalized)
    return len(words) >= 4


def _is_page_like_block(block: str) -> bool:
    """Return True for an extracted page-text block, excluding table blocks."""
    lines = [line for line in block.splitlines() if line.strip()]
    if len(lines) == 1 and _EXTRACTED_TABLES_MARKER.fullmatch(lines[0]):
        return False
    if len(lines) < 4:
        return False

    markdown_table_lines = sum(line.lstrip().startswith("|") for line in lines)
    return markdown_table_lines <= len(lines) / 2


def _edge_line_indices(lines: list[str], edge_size: int = 4) -> set[int]:
    """Return the first and last non-empty line positions in a page block."""
    non_empty = [index for index, line in enumerate(lines) if line.strip()]
    return set(non_empty[:edge_size] + non_empty[-edge_size:])


def _remove_repeated_page_headers_footers(text: str) -> str:
    """Remove lines repeatedly found at page-like block edges.

    PDF extraction generally keeps a page's prose as one multiline block and
    separates pages, table markers, and Markdown tables with blank lines. This
    lets us identify repeated page-edge text without deleting repeated commands
    or table rows from the document body.
    """
    blocks = re.split(r"\n{2,}", text)
    page_block_indices = [
        index for index, block in enumerate(blocks) if _is_page_like_block(block)
    ]
    if len(page_block_indices) < 3:
        return text

    edge_counts: Counter[str] = Counter()
    for block_index in page_block_indices:
        lines = blocks[block_index].splitlines()
        candidates = {
            _normalize_edge_line(lines[line_index])
            for line_index in _edge_line_indices(lines)
            if _is_header_footer_candidate(lines[line_index])
        }
        edge_counts.update(candidates)

    # A line must appear at the edge of at least three page-like blocks and on
    # at least 5% of them. This avoids treating occasional repeated headings as
    # boilerplate in long documents.
    minimum_occurrences = max(3, (len(page_block_indices) + 19) // 20)
    repeated_edge_lines = {
        line for line, count in edge_counts.items() if count >= minimum_occurrences
    }
    if not repeated_edge_lines:
        return text

    for block_index in page_block_indices:
        lines = blocks[block_index].splitlines()
        edge_indices = _edge_line_indices(lines)
        blocks[block_index] = "\n".join(
            line
            for line_index, line in enumerate(lines)
            if not (
                line_index in edge_indices
                and _normalize_edge_line(line) in repeated_edge_lines
            )
        ).strip()

    return "\n\n".join(block for block in blocks if block.strip())


def strip_metadata_headers(text: str) -> str:
    """Produce the single canonical clean text used for training and gates.

    Removes extraction metadata and boilerplate while preserving document prose,
    CLI examples, and the contents and formatting of extracted Markdown tables.
    """
    # Normalize Windows and legacy Mac line endings first.
    cleaned_text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove the extraction cache header.
    cleaned_text = re.sub(
        r"^#\s*MD5:\s*[a-fA-F0-9]{32}\s*\n?",
        "",
        cleaned_text,
        flags=re.MULTILINE,
    )

    # Detect page-edge boilerplate before collapsing blank-line structure.
    cleaned_text = _remove_repeated_page_headers_footers(cleaned_text)

    # Remove only the table extraction label; actual Markdown tables remain.
    cleaned_text = re.sub(
        r"^[ \t]*\[EXTRACTED TABLES\]:?[ \t]*\n?",
        "",
        cleaned_text,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # Preserve tabs and newlines used by CLI examples and Markdown tables while
    # removing ASCII/Unicode control and formatting characters.
    cleaned_text = "".join(
        char
        for char in cleaned_text
        if char in ("\n", "\t") or unicodedata.category(char) not in {"Cc", "Cf"}
    )

    # Remove trailing horizontal whitespace and cap blank runs at one blank line.
    cleaned_text = re.sub(r"[ \t]+\n", "\n", cleaned_text)
    cleaned_text = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", cleaned_text)
    return cleaned_text.strip()


class TextCleaner:
    """
    Applies a 4-step quality filter pipeline to extracted document texts:
    1. Length filter (< min_chars)
    2. Paragraph repetition filter (> max_dup_paragraph_ratio)
    3. Flexible deduplication filter (exact MD5, near-duplicate MinHash/LSH, or both)
    4. Language filter (target language matching)

    Saves a detailed JSON audit report of all rejected files with lineage tracking.
    """

    def __init__(
        self,
        min_chars: int = 50,
        max_dup_paragraph_ratio: float = 0.30,
        target_lang: str = "en",
        dedup_strategy: str = "both",  # Options: "exact", "near", "both"
        similarity_threshold: float = 0.85,
        num_perm: int = 128,
    ):
        self.min_chars = min_chars
        self.max_dup_paragraph_ratio = max_dup_paragraph_ratio
        self.target_lang = target_lang
        self.dedup_strategy = dedup_strategy.lower()
        self.similarity_threshold = similarity_threshold
        self.num_perm = num_perm

        if self.dedup_strategy not in ("exact", "near", "both"):
            raise ValueError(
                "dedup_strategy must be one of: 'exact', 'near', or 'both'"
            )

        self._reset_deduplication_state()

    def _reset_deduplication_state(self) -> None:
        """Start a fresh deduplication index for one corpus-cleaning run."""
        # Exact hash -> (retained source file, retained extracted-text file)
        self.seen_md5_hashes: dict[str, tuple[str, str | None]] = {}

        # Structures for MinHash/LSH near-deduplication
        self.lsh = MinHashLSH(
            threshold=self.similarity_threshold, num_perm=self.num_perm
        )
        self.minhashes: dict[str, MinHash] = {}
        self.file_output_map: dict[str, str | None] = {}

    @staticmethod
    def _document_reference(
        source_file: str, output_filename: str | None
    ) -> dict[str, str | None]:
        """Return the stable lineage fields used throughout the audit report."""
        return {
            "source_file": source_file,
            "output_filename": output_filename or None,
        }

    def _rejection_record(
        self,
        source_file: str,
        output_filename: str | None,
        reason: str,
        details: dict,
        retained_original: dict[str, str | None] | None = None,
    ) -> dict:
        """Build a consistent audit record for every rejected document."""
        return {
            **self._document_reference(source_file, output_filename),
            "status": "rejected",
            "reason": reason,
            # This is populated only for exact/near duplicates. Other rejection
            # reasons have no retained original, so the value is explicitly null.
            "retained_original": retained_original,
            "details": details,
        }

    def check_length(self, text: str) -> tuple[bool, int]:
        char_count = len(text.strip())
        return char_count >= self.min_chars, char_count

    def check_repetition(self, text: str) -> tuple[bool, float, int]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return False, 1.0, 0

        # Ignore casing and PDF line-wrapping differences when comparing
        # paragraphs, otherwise repeated headers/footers can be missed.
        normalized_paragraphs = {
            re.sub(r"\s+", " ", paragraph).casefold()
            for paragraph in paragraphs
        }
        duplicate_count = len(paragraphs) - len(normalized_paragraphs)
        dup_ratio = duplicate_count / len(paragraphs)
        return dup_ratio <= self.max_dup_paragraph_ratio, dup_ratio, len(paragraphs)

    def check_exact_deduplication(
        self, text: str
    ) -> tuple[bool, str, tuple[str, str | None] | None]:
        doc_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        if doc_hash in self.seen_md5_hashes:
            return False, doc_hash, self.seen_md5_hashes[doc_hash]

        # Return the hash but DO NOT insert it into seen_md5_hashes yet
        return True, doc_hash, None

    def check_near_deduplication(
        self, text: str
    ) -> tuple[bool, float, tuple[str, str | None] | None, MinHash]:
        tokens = re.findall(r"\w+", text.lower())
        current_minhash = MinHash(num_perm=self.num_perm)

        if not tokens:
            return True, 0.0, None, current_minhash

        shingles = set(" ".join(tokens[i : i + 3]) for i in range(len(tokens) - 2))
        if not shingles:
            shingles = set(tokens)

        for shingle in shingles:
            current_minhash.update(shingle.encode("utf-8"))

        result_candidates = self.lsh.query(current_minhash)
        max_similarity = 0.0
        matching_file = None

        for candidate_path in result_candidates:
            candidate_minhash = self.minhashes[candidate_path]
            similarity = current_minhash.jaccard(candidate_minhash)
            if similarity > max_similarity:
                max_similarity = similarity
                matching_file = candidate_path

        if max_similarity >= self.similarity_threshold and matching_file:
            orig_output = self.file_output_map.get(matching_file)
            return (
                False,
                round(max_similarity, 4),
                (matching_file, orig_output),
                current_minhash,
            )

        # Return the minhash but DO NOT insert it into the LSH/maps yet
        return True, round(max_similarity, 4), None, current_minhash

    def check_deduplication(
        self, text: str
    ) -> tuple[bool, str, dict, str | None, MinHash | None]:
        """Routes text through deduplication and returns state to be saved if kept."""
        doc_hash = None
        current_minhash = None

        if self.dedup_strategy in ("exact", "both"):
            is_unique_exact, doc_hash, orig_info_exact = (
                self.check_exact_deduplication(text)
            )
            if not is_unique_exact:
                orig_file, orig_output = orig_info_exact
                return False, "failed_exact_deduplication", {
                    "doc_md5": doc_hash,
                    "retained_original": self._document_reference(
                        orig_file, orig_output
                    ),
                }, None, None

        if self.dedup_strategy in ("near", "both"):
            (
                is_unique_near,
                sim_score,
                orig_info_near,
                current_minhash,
            ) = self.check_near_deduplication(text)
            if not is_unique_near:
                orig_file, orig_output = orig_info_near
                return False, "failed_near_deduplication", {
                    "similarity_score": sim_score,
                    "threshold": self.similarity_threshold,
                    "retained_original": self._document_reference(
                        orig_file, orig_output
                    ),
                }, None, None

        return True, "passed", {}, doc_hash, current_minhash

    def check_language(self, text: str) -> tuple[bool, str]:
        try:
            detected_lang = detect(text[:2000])
            return detected_lang == self.target_lang, detected_lang
        except LangDetectException:
            return False, "unknown/unresolvable"

    def filter_corpus(
        self,
        pdf_dict: dict,
        audit_report_path: str | Path = "cleaning_audit_report.json",
    ) -> tuple[dict, dict]:
        # A report must be self-contained: duplicate references from this run
        # should never point to a retained document from an earlier invocation.
        self._reset_deduplication_state()

        clean_docs = {}
        rejected_docs = {}
        retained_document_records = []

        rejection_counts = {
            "failed_length_filter": 0,
            "failed_repetition_filter": 0,
            "failed_exact_deduplication": 0,
            "failed_near_deduplication": 0,
            "failed_language_filter": 0,
        }

        for file_path, data in pdf_dict.items():
            source_file = str(file_path)
            raw_text = data.get("extracted_text") or ""
            if not isinstance(raw_text, str):
                raw_text = str(raw_text)
            output_value = data.get("output_filename")
            output_filename = str(output_value) if output_value else None

            # Strip caching metadata headers (e.g., '# MD5: ...') before processing
            text = strip_metadata_headers(raw_text)

            # Step 1: Length Check
            is_valid_len, char_count = self.check_length(text)
            if not is_valid_len:
                reason_key = "failed_length_filter"
                rejection_counts[reason_key] += 1
                rejected_docs[file_path] = self._rejection_record(
                    source_file,
                    output_filename,
                    reason_key,
                    {
                        "char_count": char_count,
                        "min_chars_threshold": self.min_chars,
                    },
                )
                continue

            # Step 2: Repetition Check
            is_not_rep, dup_ratio, total_paras = self.check_repetition(text)
            if not is_not_rep:
                reason_key = "failed_repetition_filter"
                rejection_counts[reason_key] += 1
                rejected_docs[file_path] = self._rejection_record(
                    source_file,
                    output_filename,
                    reason_key,
                    {
                        "duplicate_paragraph_ratio": round(dup_ratio, 4),
                        "max_allowed_ratio": self.max_dup_paragraph_ratio,
                        "total_paragraphs": total_paras,
                    },
                )
                continue

            # Step 3: Deduplication Check
            (
                is_unique,
                reason_key,
                details,
                doc_hash,
                current_minhash,
            ) = self.check_deduplication(text)
            if not is_unique:
                rejection_counts[reason_key] += 1
                retained_original = details.pop("retained_original", None)
                rejected_docs[file_path] = self._rejection_record(
                    source_file,
                    output_filename,
                    reason_key,
                    details,
                    retained_original,
                )
                continue

            # Step 4: Language Check
            is_target_lang, detected_lang = self.check_language(text)
            if not is_target_lang:
                reason_key = "failed_language_filter"
                rejection_counts[reason_key] += 1
                rejected_docs[file_path] = self._rejection_record(
                    source_file,
                    output_filename,
                    reason_key,
                    {
                        "detected_language": detected_lang,
                        "expected_language": self.target_lang,
                    },
                )
                continue

            # Passed all filters - NOW it is safe to add to indexes
            if self.dedup_strategy in ("exact", "both") and doc_hash:
                self.seen_md5_hashes[doc_hash] = (source_file, output_filename)

            if self.dedup_strategy in ("near", "both") and current_minhash:
                self.lsh.insert(source_file, current_minhash)
                self.minhashes[source_file] = current_minhash
                self.file_output_map[source_file] = output_filename

            # Return the normalized text for downstream training while leaving
            # the caller's original input dictionary untouched for lineage.
            cleaned_data = dict(data)
            cleaned_data["extracted_text"] = text
            clean_docs[file_path] = cleaned_data
            retained_document_records.append(
                {
                    **self._document_reference(source_file, output_filename),
                    "status": "retained",
                    "details": {
                        "char_count": char_count,
                        "duplicate_paragraph_ratio": round(dup_ratio, 4),
                        "total_paragraphs": total_paras,
                        "detected_language": detected_lang,
                    },
                }
            )

        rejected_document_records = list(rejected_docs.values())

        # Guard against incomplete reports or dangling duplicate lineage.
        if len(clean_docs) + len(rejected_docs) != len(pdf_dict):
            raise RuntimeError("Cleaning audit does not account for every input file")
        if sum(rejection_counts.values()) != len(rejected_docs):
            raise RuntimeError("Cleaning audit rejection counts are inconsistent")

        retained_sources = {
            record["source_file"] for record in retained_document_records
        }
        for record in rejected_document_records:
            original = record["retained_original"]
            if original and original["source_file"] not in retained_sources:
                raise RuntimeError(
                    "Rejected duplicate references a document not retained "
                    "in this run: "
                    f"{original['source_file']}"
                )

        # Compile and export full JSON Audit Report
        audit_data = {
            "schema_version": 2,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_documents_processed": len(pdf_dict),
                "total_clean_kept": len(clean_docs),
                "total_rejected": len(rejected_docs),
                "rejection_breakdown": rejection_counts,
            },
            "configuration": {
                "min_chars": self.min_chars,
                "max_dup_paragraph_ratio": self.max_dup_paragraph_ratio,
                "target_lang": self.target_lang,
                "dedup_strategy": self.dedup_strategy,
                "similarity_threshold": self.similarity_threshold,
            },
            "retained_documents": retained_document_records,
            "rejected_documents": rejected_document_records,
        }

        report_file = Path(audit_report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(audit_data, indent=2), encoding="utf-8")

        print("\n--- Cleaning Summary ---")
        print(f"Strategy:        {self.dedup_strategy.upper()}")
        print(f"Total Processed: {len(pdf_dict)}")
        print(f"Clean Kept:      {len(clean_docs)}")
        print(f"Total Rejected:  {len(rejected_docs)}")
        print(f"Audit Log:       Saved to {report_file.resolve()}\n")

        return clean_docs, rejected_docs


def clean_data(
    data_dict,
    dedup="both",
    audit_report_path="cleaning_audit_report.json",
):
    cleaner = TextCleaner(
        min_chars=50,
        max_dup_paragraph_ratio=0.30,
        target_lang="en",
        dedup_strategy=dedup,
    )
    clean_results, rejected = cleaner.filter_corpus(
        data_dict, audit_report_path=audit_report_path
    )
    return clean_results, rejected
