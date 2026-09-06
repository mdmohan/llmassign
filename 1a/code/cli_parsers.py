"""Shared command-line parsers for the CPT data and evaluation scripts."""

from __future__ import annotations

import argparse
from pathlib import Path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def add_model_name_argument(parser) -> None:
    """Add the shared Hugging Face GPT-2 model-name option."""
    parser.add_argument(
        "--model-name",
        default="gpt2",
        help=(
            "Hugging Face GPT-2 model name to download/load, such as "
            "gpt2 or gpt2-medium."
        ),
    )


def build_prepare_cpt_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load a directory of PDF and text files, clean and deduplicate the "
            "documents, sequence-pack them with the GPT-2 tokenizer, and save "
            "both binary and Parquet training data."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    paths = parser.add_argument_group("input and output paths")
    paths.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Directory searched recursively for .pdf and .txt files, or one "
            "existing PDF/text file."
        ),
    )
    paths.add_argument(
        "--text-dir",
        type=Path,
        default=Path("data/text"),
        help="Directory for cached PDF text; unused for a .txt input.",
    )
    paths.add_argument(
        "--audit-report",
        type=Path,
        default=Path("data/reports/cleaning_audit_report.json"),
        help="JSON report containing every retained and rejected document.",
    )
    paths.add_argument(
        "--bin-file",
        type=Path,
        default=Path("data/processed/tokens.bin"),
        help="Flat uint16 token-stream output used by the PyTorch loader.",
    )
    paths.add_argument(
        "--parquet-file",
        type=Path,
        default=Path("data/processed/tokens.parquet"),
        help="Parquet output containing one packed sequence per row.",
    )

    extraction = parser.add_argument_group("PDF extraction")
    extraction.add_argument(
        "--max-workers",
        type=positive_int,
        default=2,
        help="Maximum PDF worker processes; use 1 for minimum memory use.",
    )
    extraction.add_argument(
        "--extract-tables",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detect tables and preserve them as Markdown.",
    )
    extraction.add_argument(
        "--x-tolerance",
        type=positive_float,
        default=1.5,
        help="Horizontal character grouping tolerance used by pdfplumber.",
    )
    extraction.add_argument(
        "--y-tolerance",
        type=positive_float,
        default=3.0,
        help="Vertical character grouping tolerance used by pdfplumber.",
    )
    extraction.add_argument(
        "--max-tasks-per-child",
        type=positive_int,
        default=64,
        help="PDF tasks handled before a worker process is recycled.",
    )

    cleaning = parser.add_argument_group("document cleaning")
    cleaning.add_argument(
        "--min-chars",
        type=non_negative_int,
        default=50,
        help="Reject normalized documents shorter than this character count.",
    )
    cleaning.add_argument(
        "--max-duplicate-paragraph-ratio",
        type=unit_interval,
        default=0.30,
        help="Maximum accepted ratio of repeated normalized paragraphs.",
    )
    cleaning.add_argument(
        "--target-language",
        default="en",
        help="Language code required by the language gate.",
    )
    cleaning.add_argument(
        "--dedup-strategy",
        choices=("exact", "near", "both"),
        default="both",
        help="Whole-document deduplication strategy.",
    )
    cleaning.add_argument(
        "--similarity-threshold",
        type=unit_interval,
        default=0.85,
        help="MinHash similarity threshold for near-duplicate rejection.",
    )
    cleaning.add_argument(
        "--minhash-permutations",
        type=positive_int,
        default=128,
        help="Number of MinHash permutations used for near deduplication.",
    )

    parquet = parser.add_argument_group("Parquet serialization")
    parquet.add_argument(
        "--parquet-batch-size",
        type=positive_int,
        default=1024,
        help="Number of packed sequences written in each Parquet batch.",
    )
    parquet.add_argument(
        "--parquet-compression",
        choices=("none", "snappy", "gzip", "brotli", "lz4", "zstd"),
        default="zstd",
        help="Compression codec used for the Parquet file.",
    )
    return parser


def build_baseline_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load the existing GPT-2 model, run baseline query JSON files, "
            "and save one response JSON per query file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "query_json",
        type=Path,
        nargs="+",
        help="One or more JSON files containing a top-level 'queries' list.",
    )
    add_model_name_argument(parser)
    parser.add_argument(
        "--baseline-path",
        type=Path,
        default=Path("baselines"),
        help="Directory in which baseline response JSON files are stored.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=positive_int,
        default=50,
        help="Maximum number of new tokens generated for each prompt.",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=4,
        help="Number of prompts generated in one batch.",
    )
    parser.add_argument(
        "--evaluation-stage",
        default="pre_cpt",
        help="Evaluation-stage label written to each result file.",
    )
    return parser


def build_gpt2_query_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load GPT-2 and generate responses for supplied prompts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "prompts",
        nargs="?",
        help=(
            "One prompt or a comma-separated list of prompts. Required unless "
            "--cli is used."
        ),
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Read prompts interactively until 'exit' is entered.",
    )
    parser.add_argument(
        "--query-json",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "One or more baseline-format JSON files containing a top-level "
            "'queries' list."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for JSON query results.",
    )
    parser.add_argument(
        "--evaluation-stage",
        default="inference",
        help="Evaluation-stage label stored in JSON query results.",
    )
    add_model_name_argument(parser)
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=None,
        help=(
            "Local CPT model/checkpoint folder. When omitted, the existing "
            "base GPT-2 loader is used."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=positive_int,
        default=50,
        help="Maximum response length per prompt.",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=4,
        help="Number of prompts generated together.",
    )
    return parser


def build_cpt_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load GPT-2 and packed binary tokens, run continued pre-training, "
            "and save checkpoints and training metrics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("model and data")
    add_model_name_argument(data)
    data.add_argument(
        "--bin-file",
        type=Path,
        required=True,
        help="Flat uint16 packed-token file produced by the data pipeline.",
    )
    data.add_argument(
        "--save-dir",
        type=Path,
        default=Path("cpt_checkpoints"),
        help="Directory for checkpoints, final model, and training metrics.",
    )
    data.add_argument(
        "--context-length",
        type=positive_int,
        default=1024,
        help="Tokens in each packed training sequence.",
    )
    data.add_argument(
        "--batch-size",
        type=positive_int,
        default=8,
        help="Number of packed sequences in each microbatch.",
    )
    data.add_argument(
        "--num-workers",
        type=non_negative_int,
        default=0,
        help="DataLoader worker processes; zero is safest after CUDA starts.",
    )

    training = parser.add_argument_group("training")
    training.add_argument(
        "--epochs",
        type=positive_int,
        default=1,
        help="Number of complete passes over the packed dataset.",
    )
    training.add_argument(
        "--gradient-accumulation-steps",
        type=positive_int,
        default=4,
        help="Microbatches accumulated before each optimizer update.",
    )
    training.add_argument(
        "--learning-rate",
        type=positive_float,
        default=5e-6,
        help="Peak AdamW learning rate.",
    )
    training.add_argument(
        "--weight-decay",
        type=non_negative_float,
        default=0.01,
        help="AdamW weight decay for eligible parameters.",
    )
    training.add_argument(
        "--warmup-ratio",
        type=unit_interval,
        default=0.05,
        help="Fraction of optimizer steps used for learning-rate warmup.",
    )
    training.add_argument(
        "--max-grad-norm",
        type=positive_float,
        default=1.0,
        help="Maximum gradient norm used for clipping.",
    )
    training.add_argument(
        "--log-every-steps",
        type=positive_int,
        default=10,
        help="Optimizer-step interval for training logs.",
    )
    training.add_argument(
        "--save-every-steps",
        type=positive_int,
        default=500,
        help="Optimizer-step interval for numbered checkpoints.",
    )
    training.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by the existing training function.",
    )
    return parser
