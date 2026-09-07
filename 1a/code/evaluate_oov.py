"""Evaluate tokenizer OOV rate and fragmentation for a text corpus.

For subword tokenizers, OOV means that input was mapped to the tokenizer's
unknown-token ID. GPT-2 uses byte-level BPE and can represent arbitrary UTF-8
text, so its true OOV rate should normally be zero. Tokens per word-like item is
reported alongside OOV as a practical proxy for inefficient splitting of
unfamiliar domain terminology.
"""

from __future__ import annotations

import json
import shlex
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

from cache import CACHE_DIR
from cli_parsers import build_oov_evaluation_parser


def _resolve_tokenizer_source(model_name: str, model_folder: Path | None):
    """Resolve a Hugging Face model ID or an existing local CPT tokenizer."""
    if model_folder is not None:
        # Reuse the project's checkpoint resolution rules: a user may pass the
        # run root, final_model, last_checkpoint, or checkpoint-N directly.
        from load_local_model import resolve_model_folder

        return resolve_model_folder(model_folder), True

    if model_name.strip().casefold() in {
        "smollm2-360m",
        "huggingfacetb/smollm2-360m",
    }:
        return "HuggingFaceTB/SmolLM2-360M", False
    return model_name, False


def _load_tokenizer(model_name: str, model_folder: Path | None):
    source, local_files_only = _resolve_tokenizer_source(
        model_name,
        model_folder,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        cache_dir=CACHE_DIR,
        local_files_only=local_files_only,
        use_fast=True,
    )
    return tokenizer, str(source)


def _find_text_files(input_dir: Path, recursive: bool) -> list[Path]:
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    iterator = input_dir.rglob("*.txt") if recursive else input_dir.glob("*.txt")
    files = sorted(path for path in iterator if path.is_file())
    if not files:
        raise ValueError(f"No .txt files found in: {input_dir}")
    return files


def _rate_verdict(oov_rate: float) -> tuple[str, str]:
    """Apply transparent heuristic bands to token-level unknown-token rate."""
    percent = oov_rate * 100
    if percent == 0:
        return "GOOD", "No input tokens were mapped to the unknown token."
    if percent <= 0.1:
        return "GOOD", "Unknown-token usage is very low."
    if percent <= 1.0:
        return "REVIEW", "Unknown-token usage is noticeable and should be inspected."
    return "POOR", "More than 1% of tokens are unknown to the tokenizer."


def _fragmentation_verdict(tokens_per_word: float) -> tuple[str, str]:
    """Give a broad English-corpus heuristic, not a universal quality cutoff."""
    if tokens_per_word <= 1.5:
        return "GOOD", "The corpus is tokenized relatively efficiently."
    if tokens_per_word <= 2.0:
        return "ACCEPTABLE", "Moderate splitting is normal for technical text."
    if tokens_per_word <= 2.5:
        return "REVIEW", "Domain terms or extraction noise may be highly fragmented."
    return "POOR", "The corpus is very fragmented; inspect terminology and noise."


def _pseudo_oov_verdict(
    average_word_tokens: float,
    pseudo_oov_percent: float,
) -> tuple[str, str]:
    """Apply the thresholds from the supplied pseudo-OOV methodology."""
    if average_word_tokens > 1.5 or pseudo_oov_percent > 15.0:
        return (
            "POOR",
            "High word fragmentation suggests the tokenizer struggles with "
            "some corpus vocabulary.",
        )
    return (
        "GOOD",
        "Individual-word tokenization is efficient under the selected thresholds.",
    )


def _overall_verdict(oov_verdict: str, fragmentation_verdict: str) -> str:
    severity = {"GOOD": 0, "ACCEPTABLE": 1, "REVIEW": 2, "POOR": 3}
    worst = max(
        (oov_verdict, fragmentation_verdict),
        key=lambda value: severity[value],
    )
    return worst


def _word_piece_count(word: str, tokenizer, cache: dict[str, int]) -> int:
    """Tokenize each unique whitespace word only once across the corpus."""
    if word not in cache:
        cache[word] = len(tokenizer.tokenize(word))
    return cache[word]


def _tokenize_file(
    path: Path,
    tokenizer,
    input_dir: Path,
    word_piece_cache: dict[str, int],
) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    # Match the supplied method exactly: raw words are whitespace-separated.
    # Consequently punctuation attached to a word remains part of that word.
    word_counts = Counter(text.split())
    word_count = sum(word_counts.values())
    replacement_character_count = text.count("\ufffd")

    individual_word_token_count = sum(
        _word_piece_count(word, tokenizer, word_piece_cache) * occurrences
        for word, occurrences in word_counts.items()
    )
    highly_fragmented_word_count = sum(
        occurrences
        for word, occurrences in word_counts.items()
        if _word_piece_count(word, tokenizer, word_piece_cache) >= 3
    )
    average_individual_word_tokens = (
        individual_word_token_count / word_count if word_count else 0.0
    )
    pseudo_oov_rate = (
        highly_fragmented_word_count / word_count if word_count else 0.0
    )
    maximum_word_tokens = max(
        (
            _word_piece_count(word, tokenizer, word_piece_cache)
            for word in word_counts
        ),
        default=0,
    )

    tokenize_options = {
        "add_special_tokens": False,
        "return_attention_mask": False,
        "truncation": False,
        "verbose": False,
    }
    if tokenizer.is_fast:
        tokenize_options["return_offsets_mapping"] = True
    encoded = tokenizer(text, **tokenize_options)
    token_ids = encoded["input_ids"]
    offsets = encoded.get("offset_mapping")
    unknown_id = tokenizer.unk_token_id

    unknown_count = 0
    unknown_text = Counter()
    if unknown_id is not None:
        for index, token_id in enumerate(token_ids):
            if token_id != unknown_id:
                continue
            unknown_count += 1
            if offsets is not None:
                start, end = offsets[index]
                sample = text[start:end]
            else:
                sample = tokenizer.convert_ids_to_tokens(token_id)
            unknown_text[sample or "<empty span>"] += 1

    token_count = len(token_ids)
    # True token-level OOV rate is the fraction of emitted tokens that equal
    # unk_token_id. With add_special_tokens=False, GPT-2's EOS/UNK special ID
    # is not introduced automatically and therefore is not falsely counted.
    oov_rate = unknown_count / token_count if token_count else 0.0

    # Byte/subword tokenizers can achieve 0% OOV by splitting unfamiliar words
    # into many pieces. Token-to-word ratio exposes that hidden inefficiency.
    tokens_per_word = token_count / word_count if word_count else 0.0
    return {
        "file": str(path),
        "relative_file": str(path.relative_to(input_dir)),
        "file_size_bytes": path.stat().st_size,
        "character_count": len(text),
        "replacement_character_count": replacement_character_count,
        "word_count": word_count,
        "individual_word_token_count": individual_word_token_count,
        "average_individual_word_tokens": average_individual_word_tokens,
        "highly_fragmented_word_count": highly_fragmented_word_count,
        "pseudo_oov_rate": pseudo_oov_rate,
        "pseudo_oov_rate_percent": pseudo_oov_rate * 100,
        "maximum_tokens_for_one_word": maximum_word_tokens,
        "token_count": token_count,
        "unknown_token_count": unknown_count,
        "oov_rate": oov_rate,
        "oov_rate_percent": oov_rate * 100,
        "tokens_per_word": tokens_per_word,
        "unique_token_count": len(set(token_ids)),
        # Private working fields are removed before the JSON is written. They
        # let the caller aggregate exactly without tokenizing every file twice.
        "_unique_token_ids": set(token_ids),
        "_unknown_text_counts": dict(unknown_text),
        "_word_counts": dict(word_counts),
        "unknown_text_samples": [
            {"text": value, "count": count}
            for value, count in unknown_text.most_common(10)
        ],
    }


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def format_row(row):
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))


def evaluate_corpus(args) -> dict:
    input_dir = args.input_dir.expanduser().resolve()
    output_file = args.output_file.expanduser().resolve()
    tokenizer, tokenizer_source = _load_tokenizer(
        args.model_name,
        args.model_folder,
    )
    files = _find_text_files(input_dir, args.recursive)

    print(f"Tokenizer: {tokenizer_source}")
    print(f"Text files: {len(files):,}")
    per_file = []
    all_unique_token_ids = set()
    corpus_word_counts = Counter()
    word_piece_cache = {}
    totals = Counter()
    unknown_text = Counter()

    for index, path in enumerate(files, start=1):
        result = _tokenize_file(
            path,
            tokenizer,
            input_dir,
            word_piece_cache,
        )
        all_unique_token_ids.update(result.pop("_unique_token_ids"))
        unknown_text.update(result.pop("_unknown_text_counts"))
        corpus_word_counts.update(result.pop("_word_counts"))
        per_file.append(result)
        totals.update(
            {
                "file_size_bytes": result["file_size_bytes"],
                "character_count": result["character_count"],
                "replacement_character_count": result[
                    "replacement_character_count"
                ],
                "word_count": result["word_count"],
                "individual_word_token_count": result[
                    "individual_word_token_count"
                ],
                "highly_fragmented_word_count": result[
                    "highly_fragmented_word_count"
                ],
                "token_count": result["token_count"],
                "unknown_token_count": result["unknown_token_count"],
            }
        )
        if index % 25 == 0 or index == len(files):
            print(f"  processed {index:,}/{len(files):,} files")

    token_count = totals["token_count"]
    word_count = totals["word_count"]
    oov_rate = totals["unknown_token_count"] / token_count if token_count else 0.0
    tokens_per_word = token_count / word_count if word_count else 0.0
    average_individual_word_tokens = (
        totals["individual_word_token_count"] / word_count if word_count else 0.0
    )
    pseudo_oov_rate = (
        totals["highly_fragmented_word_count"] / word_count
        if word_count
        else 0.0
    )
    oov_verdict, oov_explanation = _rate_verdict(oov_rate)
    fragmentation_verdict, fragmentation_explanation = _fragmentation_verdict(
        tokens_per_word
    )
    pseudo_oov_verdict, pseudo_oov_explanation = _pseudo_oov_verdict(
        average_individual_word_tokens,
        pseudo_oov_rate * 100,
    )

    maximum_word_tokens = max(word_piece_cache.values(), default=0)
    most_fragmented_words = sorted(
        corpus_word_counts,
        key=lambda word: (
            word_piece_cache[word],
            corpus_word_counts[word],
            word,
        ),
        reverse=True,
    )[:50]

    command = [sys.executable, *sys.argv]
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": {
            "argv": command,
            "reconstructed_full_command": shlex.join(command),
            "working_directory": str(Path.cwd()),
        },
        "input_directory": str(input_dir),
        "recursive": args.recursive,
        "model_name": args.model_name,
        "model_folder": (
            str(args.model_folder.expanduser().resolve())
            if args.model_folder is not None
            else None
        ),
        "tokenizer": {
            "source": tokenizer_source,
            "class": type(tokenizer).__name__,
            "is_fast": tokenizer.is_fast,
            "vocabulary_size": tokenizer.vocab_size,
            "unknown_token": tokenizer.unk_token,
            "unknown_token_id": tokenizer.unk_token_id,
        },
        "methodology": {
            "oov_rate": "unknown_token_count / total_token_count",
            "stream_fragmentation": "total_token_count / whitespace_word_count",
            "average_individual_word_tokens": (
                "sum(tokenizer.tokenize(word) pieces for every whitespace word) "
                "/ whitespace_word_count"
            ),
            "pseudo_oov_rate": (
                "whitespace words requiring at least 3 tokenizer pieces / "
                "whitespace_word_count"
            ),
            "important_note": (
                "Zero OOV means complete tokenizer coverage, not that domain "
                "terms are represented efficiently or understood by the model."
            ),
        },
        "summary": {
            "file_count": len(files),
            **dict(totals),
            "documents_with_unknown_tokens": sum(
                result["unknown_token_count"] > 0 for result in per_file
            ),
            "oov_rate": oov_rate,
            "oov_rate_percent": oov_rate * 100,
            "tokens_per_word": tokens_per_word,
            "average_individual_word_tokens": average_individual_word_tokens,
            "highly_fragmented_word_count": totals[
                "highly_fragmented_word_count"
            ],
            "pseudo_oov_rate": pseudo_oov_rate,
            "pseudo_oov_rate_percent": pseudo_oov_rate * 100,
            "maximum_tokens_for_one_word": maximum_word_tokens,
            "unique_token_ids_used": len(all_unique_token_ids),
            "vocabulary_usage_percent": (
                len(all_unique_token_ids) / tokenizer.vocab_size * 100
                if tokenizer.vocab_size
                else None
            ),
            "oov_verdict": oov_verdict,
            "oov_explanation": oov_explanation,
            "fragmentation_verdict": fragmentation_verdict,
            "fragmentation_explanation": fragmentation_explanation,
            "pseudo_oov_verdict": pseudo_oov_verdict,
            "pseudo_oov_explanation": pseudo_oov_explanation,
            "overall_verdict": _overall_verdict(
                oov_verdict,
                pseudo_oov_verdict,
            ),
            "most_fragmented_words": [
                {
                    "word": word,
                    "occurrences": corpus_word_counts[word],
                    "token_count": word_piece_cache[word],
                    "tokens": tokenizer.tokenize(word),
                }
                for word in most_fragmented_words
            ],
            "top_unknown_text_samples": [
                {"text": value, "count": count}
                for value, count in unknown_text.most_common(20)
            ],
        },
        "per_file": per_file,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = report["summary"]
    print("\n=== Corpus Tokenizer Evaluation ===")
    _print_table(
        ["Metric", "Value", "Assessment"],
        [
            ["Files", f"{summary['file_count']:,}", "-"],
            ["Words", f"{summary['word_count']:,}", "-"],
            ["Tokens", f"{summary['token_count']:,}", "-"],
            [
                "Unknown tokens",
                f"{summary['unknown_token_count']:,}",
                summary["oov_verdict"],
            ],
            [
                "OOV rate",
                f"{summary['oov_rate_percent']:.6f}%",
                summary["oov_verdict"],
            ],
            [
                "Tokens per word",
                f"{summary['tokens_per_word']:.3f}",
                summary["fragmentation_verdict"],
            ],
            [
                "Average individual-word tokens",
                f"{summary['average_individual_word_tokens']:.3f}",
                summary["pseudo_oov_verdict"],
            ],
            [
                "Pseudo-OOV words (>=3 pieces)",
                f"{summary['pseudo_oov_rate_percent']:.3f}%",
                summary["pseudo_oov_verdict"],
            ],
            [
                "Maximum pieces for one word",
                f"{summary['maximum_tokens_for_one_word']:,}",
                "-",
            ],
            ["Overall", "-", summary["overall_verdict"]],
        ],
    )
    print(f"\nOOV: {summary['oov_explanation']}")
    print(f"Fragmentation: {summary['fragmentation_explanation']}")
    print(f"Pseudo-OOV: {summary['pseudo_oov_explanation']}")
    if "gpt2" in type(tokenizer).__name__.casefold():
        print(
            "GPT-2 uses byte-level BPE, so 0% true OOV is expected. "
            "Use tokens per word to judge domain-tokenization efficiency."
        )

    if args.top_files:
        print(f"\nTop {min(args.top_files, len(per_file))} fragmented files")
        rows = []
        for item in sorted(
            per_file,
            key=lambda value: (
                value["pseudo_oov_rate"],
                value["average_individual_word_tokens"],
            ),
            reverse=True,
        )[: args.top_files]:
            rows.append(
                [
                    item["relative_file"],
                    f"{item['oov_rate_percent']:.6f}%",
                    f"{item['pseudo_oov_rate_percent']:.2f}%",
                    f"{item['average_individual_word_tokens']:.3f}",
                    f"{item['token_count']:,}",
                ]
            )
        _print_table(
            ["File", "True OOV", "Pseudo-OOV", "Avg pieces/word", "Tokens"],
            rows,
        )

    print(f"\nDetailed JSON report saved to: {output_file}")
    return report


def main() -> int:
    parser = build_oov_evaluation_parser()
    args = parser.parse_args()
    try:
        evaluate_corpus(args)
    except KeyboardInterrupt:
        parser.exit(130, "\nOOV evaluation interrupted by user.\n")
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
