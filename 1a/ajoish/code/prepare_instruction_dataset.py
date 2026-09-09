#!/usr/bin/env python3
"""Build the Part B instruction dataset from Part A training documents only."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


HEADING_RE = re.compile(
    r"^(?:Configuring|Restrictions for|Information About|How to Configure|"
    r"Prerequisites for|Overview of|About|Understanding|Guidelines for|"
    r"Verifying|Troubleshooting)\b.*",
    flags=re.IGNORECASE,
)
TOC_RE = re.compile(
    r"(?:\.{2,}\s*\d+|\bon page\s+\d+\b|\bchapter\s+\d+\b|"
    r"\badditional references for\b|\bfeature history for\b)",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-pairs", type=int, default=100)
    parser.add_argument("--max-pairs", type=int, default=600)
    parser.add_argument("--max-response-chars", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-ratio", type=float, default=0.20)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_part_a_training_documents(manifest_path: Path) -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("accepted") or record.get("split") != "train":
                continue
            document_id = record.get("document_id")
            if not isinstance(document_id, str) or not document_id:
                raise ValueError(f"Missing document_id at manifest line {line_number}")
            if document_id in documents:
                raise ValueError(f"Duplicate training document_id: {document_id}")
            documents[document_id] = record
    if not documents:
        raise ValueError("Manifest contains no accepted Part A training documents")
    return documents


def heading_to_instruction(heading: str) -> str:
    heading = heading.strip().rstrip(":").strip()
    lowered = heading.casefold()
    prefixes = (
        ("how to configure", "How do I configure {} on a Cisco device?"),
        ("configuring", "How do I configure {} on a Cisco device?"),
        ("restrictions for", "What are the restrictions for {}?"),
        ("prerequisites for", "What are the prerequisites for {}?"),
        ("information about", "What is {} and how does it work?"),
        ("understanding", "Explain {} in Cisco networking."),
        ("guidelines for", "What are the configuration guidelines for {}?"),
        ("troubleshooting", "How do I troubleshoot {}?"),
        ("overview of", "Give an overview of {}."),
        ("about", "What is {} and how does it work?"),
    )
    for prefix, template in prefixes:
        if lowered.startswith(prefix):
            topic = heading[len(prefix) :].strip(" :-")
            return template.format(topic)
    if lowered.startswith("verifying"):
        topic = heading[len("verifying") :].strip(" :-")
        return f"How do I verify {topic} on a Cisco device?"
    return f"Explain this Cisco networking topic: {heading}."


def normalize_response(lines: list[str], max_chars: int) -> str:
    response = re.sub(r"\s+", " ", " ".join(lines)).strip()
    if len(response) <= max_chars:
        return response
    shortened = response[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:")
    return shortened + ("." if shortened and shortened[-1] not in ".!?" else "")


def is_usable_response(response: str) -> bool:
    if not 120 <= len(response) <= 900:
        return False
    if TOC_RE.search(response):
        return False
    if response.count(" ") / len(response) < 0.12:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", response)
    if len(words) < 24:
        return False
    return any(mark in response for mark in ".:;?!")


def build_pairs(
    corpus_dir: Path,
    training_documents: dict[str, dict[str, Any]],
    max_response_chars: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    pairs: list[dict[str, Any]] = []
    missing_documents: list[str] = []
    for document_id, manifest_record in sorted(training_documents.items()):
        text_path = corpus_dir / f"{document_id}.txt"
        if not text_path.is_file():
            missing_documents.append(document_id)
            continue
        lines = text_path.read_text(encoding="utf-8").splitlines()
        for index, raw_heading in enumerate(lines):
            heading = raw_heading.strip()
            if not HEADING_RE.fullmatch(heading) or not 8 <= len(heading) <= 100:
                continue
            body: list[str] = []
            for raw_line in lines[index + 1 :]:
                line = raw_line.strip()
                if HEADING_RE.fullmatch(line):
                    break
                if not line:
                    if body:
                        break
                    continue
                body.append(line)
                if len(" ".join(body)) >= max_response_chars:
                    break
            response = normalize_response(body, max_response_chars)
            if not is_usable_response(response):
                continue
            pairs.append(
                {
                    "instruction": heading_to_instruction(heading),
                    "response": response,
                    "source_document_id": document_id,
                    "source_path": manifest_record["source_path"],
                    "part_a_split": "train",
                }
            )
    return pairs, missing_documents


def deduplicate_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen_instructions: set[str] = set()
    for pair in pairs:
        key = re.sub(r"\s+", " ", pair["instruction"]).strip().casefold()
        if key in seen_instructions:
            continue
        seen_instructions.add(key)
        unique.append(pair)
    return unique


def main() -> None:
    args = parse_args()
    if args.min_pairs < 1 or args.max_pairs < args.min_pairs:
        raise ValueError("Require 1 <= min-pairs <= max-pairs")
    if not 0.0 < args.eval_ratio < 1.0:
        raise ValueError("eval-ratio must be between 0 and 1")

    training_documents = load_part_a_training_documents(args.manifest)
    generated_pairs, missing_documents = build_pairs(
        args.corpus_dir,
        training_documents,
        args.max_response_chars,
    )
    if missing_documents:
        raise FileNotFoundError(
            "Missing cleaned text for Part A training documents: "
            + ", ".join(missing_documents)
        )

    pairs = deduplicate_pairs(generated_pairs)
    random.Random(args.seed).shuffle(pairs)
    pairs = pairs[: args.max_pairs]
    if len(pairs) < args.min_pairs:
        raise RuntimeError(
            f"Only {len(pairs)} quality-gated pairs generated; "
            f"at least {args.min_pairs} are required"
        )

    eval_count = max(1, round(len(pairs) * args.eval_ratio))
    train_count = len(pairs) - eval_count
    for index, pair in enumerate(pairs):
        pair["split"] = "train" if index < train_count else "eval"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False, sort_keys=True) + "\n")

    split_counts = Counter(pair["split"] for pair in pairs)
    report = {
        "schema_version": 1,
        "method": "deterministic_heading_and_following_paragraph_heuristics",
        "seed": args.seed,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "corpus_dir": str(args.corpus_dir),
        "part_a_training_documents": len(training_documents),
        "part_a_eval_documents_used": 0,
        "candidate_pairs": len(generated_pairs),
        "deduplicated_pairs": len(deduplicate_pairs(generated_pairs)),
        "written_pairs": len(pairs),
        "split_counts": dict(sorted(split_counts.items())),
        "source_document_counts": dict(
            sorted(Counter(pair["source_document_id"] for pair in pairs).items())
        ),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()