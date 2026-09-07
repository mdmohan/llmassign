"""Deterministic PDF inventory, extraction, filtering, and splitting."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
import signal
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .cleaning import fails_length_filter, fails_repetition_filter, normalize_text, repetition_ratio


FILTER_ORDER = ("length", "repetition", "exact_duplicate", "language")


@contextmanager
def page_timeout(seconds: int) -> Iterable[None]:
    def raise_timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"page extraction exceeded {seconds} seconds")

    previous_handler = signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def document_id(source_corpus: str, relative_path: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", Path(relative_path).stem.lower()).strip("-")
    suffix = sha256_bytes(f"{source_corpus}/{relative_path}".encode("utf-8"))[:12]
    return f"{source_corpus}-{slug[:72]}-{suffix}"


def inventory_pdfs(corpus_roots: Mapping[str, Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source_corpus, root in sorted(corpus_roots.items()):
        if not root.is_dir():
            raise FileNotFoundError(f"PDF root does not exist: {root}")
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().casefold()):
            if not path.is_file() or path.suffix.casefold() != ".pdf":
                continue
            relative_path = path.relative_to(root).as_posix()
            records.append(
                {
                    "document_id": document_id(source_corpus, relative_path),
                    "source_corpus": source_corpus,
                    "source_path": (Path(root.name) / relative_path).as_posix(),
                    "source_relative_path": relative_path,
                    "source_bytes": path.stat().st_size,
                    "source_sha256": sha256_file(path),
                    "_input_path": path.as_posix(),
                }
            )
    return records


def extract_pdf(
    path: Path, page_timeout_seconds: int = 30
) -> tuple[str, str, list[dict[str, Any]], list[str]]:
    import fitz

    raw_pages: list[str] = []
    cleaned_pages: list[str] = []
    page_records: list[dict[str, Any]] = []
    failures: list[str] = []
    cleaned_offset = 0

    with fitz.open(path) as pdf:
        for page_index in range(pdf.page_count):
            page_number = page_index + 1
            error: str | None = None
            try:
                with page_timeout(page_timeout_seconds):
                    page = pdf.load_page(page_index)
                    raw_text = page.get_text("text", sort=True)
            except Exception as exc:  # pragma: no cover - parser-specific failures
                raw_text = ""
                error = f"{type(exc).__name__}: {exc}"
                failures.append(f"page {page_number}: {error}")

            cleaned_text = normalize_text(raw_text)
            if cleaned_pages and cleaned_text:
                cleaned_offset += 2
            start = cleaned_offset
            cleaned_offset += len(cleaned_text)
            raw_pages.append(f"<<<PAGE {page_number}>>>\n{raw_text.rstrip()}")
            if cleaned_text:
                cleaned_pages.append(cleaned_text)
            page_records.append(
                {
                    "page_number": page_number,
                    "raw_characters": len(raw_text),
                    "cleaned_characters": len(cleaned_text),
                    "cleaned_start": start,
                    "cleaned_end": cleaned_offset,
                    "empty": not bool(cleaned_text),
                    "error": error,
                }
            )

    return "\n\n".join(raw_pages).strip(), "\n\n".join(cleaned_pages), page_records, failures


def detect_language(text: str) -> tuple[str, float]:
    from langdetect import DetectorFactory, LangDetectException, detect_langs

    DetectorFactory.seed = 0
    try:
        candidates = detect_langs(text)
    except LangDetectException:
        return "undetermined", 0.0
    if not candidates:
        return "undetermined", 0.0
    best = candidates[0]
    return best.lang, float(best.prob)


def stable_exact_dedup(records: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen: dict[str, str] = {}
    unique: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for record in records:
        content_hash = str(record["content_sha256"])
        if content_hash in seen:
            duplicate = dict(record)
            duplicate["duplicate_of"] = seen[content_hash]
            duplicates.append(duplicate)
        else:
            seen[content_hash] = str(record["document_id"])
            unique.append(record)
    return unique, duplicates


def stratified_split(
    records: Sequence[dict[str, Any]], eval_fraction: float, seed: int
) -> dict[str, str]:
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be between zero and one")
    if not records:
        return {}

    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        groups[str(record["source_corpus"])].append(str(record["document_id"]))

    target_eval = max(1, round(len(records) * eval_fraction))
    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for source_corpus, identifiers in sorted(groups.items()):
        exact = target_eval * len(identifiers) / len(records)
        allocations[source_corpus] = math.floor(exact)
        remainders.append((exact - math.floor(exact), source_corpus))

    remaining = target_eval - sum(allocations.values())
    for _, source_corpus in sorted(remainders, key=lambda item: (-item[0], item[1]))[:remaining]:
        allocations[source_corpus] += 1

    split_by_id: dict[str, str] = {}
    rng = random.Random(seed)
    for source_corpus, identifiers in sorted(groups.items()):
        shuffled = sorted(identifiers)
        rng.shuffle(shuffled)
        eval_ids = set(shuffled[: allocations[source_corpus]])
        split_by_id.update({identifier: "eval" if identifier in eval_ids else "train" for identifier in shuffled})
    return split_by_id


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _combined_corpus(records: Sequence[dict[str, Any]], cleaned_by_id: Mapping[str, str]) -> str:
    parts = []
    for record in records:
        identifier = str(record["document_id"])
        marker = f"<<<DOCUMENT {identifier}>>>"
        parts.append(f"{marker}\n{cleaned_by_id[identifier]}")
    return "\n\n".join(parts) + ("\n" if parts else "")


def _greatest_impact(rejections: Counter[str]) -> dict[str, Any]:
    ordered = [(name, rejections[name]) for name in FILTER_ORDER]
    name, removed = max(ordered, key=lambda item: item[1])
    return {"filter": name, "documents_removed": removed}


def build_corpus(
    corpus_roots: Mapping[str, Path],
    output_root: Path,
    seed: int = 42,
    eval_fraction: float = 0.10,
    expected_document_count: int | None = None,
    page_timeout_seconds: int = 30,
) -> dict[str, Any]:
    extracted_root = output_root / "data" / "extracted"
    domain_root = output_root / "data" / "processed" / "domain_corpus"
    processed_root = output_root / "data" / "processed"
    reports_root = output_root / "data" / "reports"
    for directory in (extracted_root, domain_root, reports_root):
        if directory.exists():
            shutil.rmtree(directory)
    for combined_corpus in processed_root.glob("*_corpus.txt"):
        combined_corpus.unlink()
    for directory in (extracted_root, domain_root, processed_root, reports_root):
        directory.mkdir(parents=True, exist_ok=True)

    documents = inventory_pdfs(corpus_roots)
    if expected_document_count is not None and len(documents) != expected_document_count:
        raise RuntimeError(
            f"Expected {expected_document_count} PDFs but inventoried {len(documents)}"
        )
    candidates: list[dict[str, Any]] = []
    cleaned_by_id: dict[str, str] = {}
    rejected: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()

    for document_index, record in enumerate(documents, start=1):
        identifier = str(record["document_id"])
        input_path = Path(str(record.pop("_input_path")))
        print(
            f"[{document_index:02d}/{len(documents):02d}] "
            f"extracting {record['source_corpus']}/{record['source_relative_path']}",
            flush=True,
        )
        try:
            raw_text, cleaned_text, pages, failures = extract_pdf(
                input_path, page_timeout_seconds=page_timeout_seconds
            )
            (extracted_root / f"{identifier}.txt").write_text(raw_text + "\n", encoding="utf-8")
            record.update(
                {
                    "page_count": len(pages),
                    "empty_page_count": sum(1 for page in pages if page["empty"]),
                    "failed_page_count": len(failures),
                    "page_failures": failures,
                    "pages": pages,
                    "cleaned_characters": len(cleaned_text),
                    "paragraph_count": len(re.split(r"\n\s*\n", cleaned_text)) if cleaned_text else 0,
                    "repetition_ratio": repetition_ratio(cleaned_text),
                    "content_sha256": sha256_bytes(cleaned_text.encode("utf-8")),
                }
            )
        except Exception as exc:  # pragma: no cover - exercised by corrupt inputs
            record.update(
                {
                    "accepted": False,
                    "rejection_reason": "extraction_error",
                    "extraction_error": f"{type(exc).__name__}: {exc}",
                }
            )
            rejection_counts["extraction_error"] += 1
            rejected.append(record)
            print(f"  extraction failed: {record['extraction_error']}", flush=True)
            continue

        if fails_length_filter(cleaned_text):
            record.update({"accepted": False, "rejection_reason": "length"})
            rejection_counts["length"] += 1
            rejected.append(record)
        elif fails_repetition_filter(cleaned_text):
            record.update({"accepted": False, "rejection_reason": "repetition"})
            rejection_counts["repetition"] += 1
            rejected.append(record)
        else:
            cleaned_by_id[identifier] = cleaned_text
            candidates.append(record)

    unique, duplicates = stable_exact_dedup(candidates)
    duplicate_ids = {str(record["document_id"]): record for record in duplicates}
    language_candidates: list[dict[str, Any]] = []
    for record in candidates:
        identifier = str(record["document_id"])
        if identifier in duplicate_ids:
            record.update(
                {
                    "accepted": False,
                    "rejection_reason": "exact_duplicate",
                    "duplicate_of": duplicate_ids[identifier]["duplicate_of"],
                }
            )
            rejection_counts["exact_duplicate"] += 1
            rejected.append(record)
        else:
            language_candidates.append(record)

    accepted: list[dict[str, Any]] = []
    unique_ids = {str(record["document_id"]) for record in unique}
    for record in language_candidates:
        identifier = str(record["document_id"])
        if identifier not in unique_ids:
            continue
        language, confidence = detect_language(cleaned_by_id[identifier])
        record.update({"language": language, "language_confidence": confidence})
        if language != "en":
            record.update({"accepted": False, "rejection_reason": "language"})
            rejection_counts["language"] += 1
            rejected.append(record)
        else:
            record["accepted"] = True
            accepted.append(record)

    split_by_id = stratified_split(accepted, eval_fraction, seed)
    for record in accepted:
        identifier = str(record["document_id"])
        record["split"] = split_by_id[identifier]
        (domain_root / f"{identifier}.txt").write_text(cleaned_by_id[identifier] + "\n", encoding="utf-8")

    accepted.sort(key=lambda record: (record["source_corpus"], record["source_relative_path"].casefold()))
    rejected.sort(key=lambda record: (record["source_corpus"], record["source_relative_path"].casefold()))
    documents.sort(key=lambda record: (record["source_corpus"], record["source_relative_path"].casefold()))
    train = [record for record in accepted if record["split"] == "train"]
    evaluation = [record for record in accepted if record["split"] == "eval"]

    accepted_ids = {str(record["document_id"]) for record in accepted}
    rejected_ids = {str(record["document_id"]) for record in rejected}
    train_ids = {str(record["document_id"]) for record in train}
    eval_ids = {str(record["document_id"]) for record in evaluation}
    if accepted_ids & rejected_ids or len(accepted_ids | rejected_ids) != len(documents):
        raise RuntimeError("Accepted and rejected records do not partition the inventory")
    if train_ids & eval_ids or train_ids | eval_ids != accepted_ids:
        raise RuntimeError("Train and evaluation records do not partition accepted documents")
    train_hashes = {str(record["content_sha256"]) for record in train}
    eval_hashes = {str(record["content_sha256"]) for record in evaluation}
    if train_hashes & eval_hashes:
        raise RuntimeError("Duplicate content crosses the train/evaluation boundary")

    (processed_root / "domain_corpus.txt").write_text(
        _combined_corpus(accepted, cleaned_by_id), encoding="utf-8"
    )
    (processed_root / "train_corpus.txt").write_text(
        _combined_corpus(train, cleaned_by_id), encoding="utf-8"
    )
    (processed_root / "eval_corpus.txt").write_text(
        _combined_corpus(evaluation, cleaned_by_id), encoding="utf-8"
    )

    stage_counts = {
        "inventoried": len(documents),
        "extracted": len(documents) - rejection_counts["extraction_error"],
        "after_length": len(documents)
        - rejection_counts["extraction_error"]
        - rejection_counts["length"],
        "after_repetition": len(documents)
        - rejection_counts["extraction_error"]
        - rejection_counts["length"]
        - rejection_counts["repetition"],
        "after_exact_duplicate": len(language_candidates),
        "after_language": len(accepted),
    }
    filter_counts = {
        "length": {
            "before": stage_counts["extracted"],
            "after": stage_counts["after_length"],
            "removed": rejection_counts["length"],
        },
        "repetition": {
            "before": stage_counts["after_length"],
            "after": stage_counts["after_repetition"],
            "removed": rejection_counts["repetition"],
        },
        "exact_duplicate": {
            "before": stage_counts["after_repetition"],
            "after": stage_counts["after_exact_duplicate"],
            "removed": rejection_counts["exact_duplicate"],
        },
        "language": {
            "before": stage_counts["after_exact_duplicate"],
            "after": stage_counts["after_language"],
            "removed": rejection_counts["language"],
        },
    }
    output_hashes = {
        name: sha256_file(processed_root / name)
        for name in ("domain_corpus.txt", "train_corpus.txt", "eval_corpus.txt")
    }
    report = {
        "seed": seed,
        "eval_fraction": eval_fraction,
        "stage_counts": stage_counts,
        "filter_counts": filter_counts,
        "rejections_by_reason": dict(sorted(rejection_counts.items())),
        "greatest_impact_filter": _greatest_impact(rejection_counts),
        "source_bytes": sum(int(record["source_bytes"]) for record in documents),
        "accepted_cleaned_characters": sum(int(record["cleaned_characters"]) for record in accepted),
        "split_document_counts": {"train": len(train), "eval": len(evaluation)},
        "split_character_counts": {
            "train": sum(len(cleaned_by_id[str(record["document_id"])]) for record in train),
            "eval": sum(len(cleaned_by_id[str(record["document_id"])]) for record in evaluation),
        },
        "source_corpus_counts": dict(Counter(str(record["source_corpus"]) for record in documents)),
        "accepted_source_corpus_counts": dict(Counter(str(record["source_corpus"]) for record in accepted)),
        "output_sha256": output_hashes,
    }
    write_jsonl(reports_root / "documents.jsonl", documents)
    write_jsonl(reports_root / "rejected_documents.jsonl", rejected)
    write_json(reports_root / "cleaning_report.json", report)
    return report
