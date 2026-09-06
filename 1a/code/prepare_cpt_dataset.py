"""Run the complete PDF-to-packed-token CPT data pipeline.

This command imports and orchestrates the existing PDF extraction, cleaning,
tokenization, and serialization modules. It does not duplicate their pipeline
logic or alter their defaults.

Example:
    python 1a/code/prepare_cpt_dataset.py \
        --input-dir 1a/data/pdfs/cisco-dc-pdfs \
        --text-dir 1a/data/text/cisco-dc \
        --audit-report 1a/data/reports/cleaning_audit_report.json \
        --bin-file 1a/data/processed/tokens.bin \
        --parquet-file 1a/data/processed/tokens.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cli_parsers import build_prepare_cpt_parser


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def run_pipeline(args: argparse.Namespace) -> None:
    input_path = resolve_path(args.input_dir)
    text_dir = resolve_path(args.text_dir)
    audit_report = resolve_path(args.audit_report)
    bin_file = resolve_path(args.bin_file)
    parquet_file = resolve_path(args.parquet_file)

    if not input_path.exists():
        raise ValueError(f"Input path does not exist: {input_path}")
    if bin_file == parquet_file:
        raise ValueError("--bin-file and --parquet-file must be different paths")

    if input_path.is_file() and input_path.suffix.lower() not in {".pdf", ".txt"}:
        raise ValueError(
            "A file input must use the .pdf or .txt extension: "
            f"{input_path}"
        )

    print("\n=== Stage 1/4: Load PDF and text sources ===")
    from load_pdf import PdfToTxt

    converter = PdfToTxt(
        x_tolerance=args.x_tolerance,
        y_tolerance=args.y_tolerance,
        max_workers=args.max_workers,
        extract_tables=args.extract_tables,
        max_tasks_per_child=args.max_tasks_per_child,
    )
    extracted_documents = converter.process_pdf_directory_deep(
        str(input_path),
        str(text_dir),
    )
    if not extracted_documents:
        raise RuntimeError(f"No PDF or text documents were loaded from {input_path}")

    print("\n=== Stage 2/4: Normalize and clean documents ===")
    # Delay non-PDF imports until extraction is complete. Spawned PDF workers
    # execute this script's top-level imports, so loading these packages above
    # would add unnecessary startup time and memory to every worker.
    from clean_data import TextCleaner

    cleaner = TextCleaner(
        min_chars=args.min_chars,
        max_dup_paragraph_ratio=args.max_duplicate_paragraph_ratio,
        target_lang=args.target_language,
        dedup_strategy=args.dedup_strategy,
        similarity_threshold=args.similarity_threshold,
        num_perm=args.minhash_permutations,
    )
    clean_documents, rejected_documents = cleaner.filter_corpus(
        extracted_documents,
        audit_report_path=audit_report,
    )
    if not clean_documents:
        raise RuntimeError(
            "Every extracted document was rejected; inspect the cleaning audit report"
        )

    print("\n=== Stage 3/4: Tokenize and sequence-pack ===")
    from tokenize_data import (
        save_chunks_to_disk,
        save_chunks_to_parquet,
        tokenize_gpt,
    )

    packed_chunks, context_length = tokenize_gpt(clean_documents)
    if not packed_chunks:
        raise RuntimeError(
            "The retained text did not produce one complete context-length sequence"
        )

    print("\n=== Stage 4/4: Write binary and Parquet outputs ===")
    save_chunks_to_disk(packed_chunks, bin_file)
    parquet_compression = (
        None
        if args.parquet_compression == "none"
        else args.parquet_compression
    )
    save_chunks_to_parquet(
        packed_chunks,
        parquet_file,
        batch_size=args.parquet_batch_size,
        compression=parquet_compression,
    )

    print("\n=== Pipeline complete ===")
    print(f"Source documents found:{len(extracted_documents):,}")
    print(f"Documents retained:    {len(clean_documents):,}")
    print(f"Documents rejected:    {len(rejected_documents):,}")
    print(f"Context length:        {context_length:,}")
    print(f"Packed sequences:      {len(packed_chunks):,}")
    print(f"Packed tokens:         {len(packed_chunks) * context_length:,}")
    print(f"Extracted PDF cache:   {text_dir}")
    print(f"Cleaning audit report: {audit_report}")
    print(f"Binary tokens:         {bin_file}")
    print(f"Parquet sequences:     {parquet_file}")


def main() -> int:
    parser = build_prepare_cpt_parser()
    args = parser.parse_args()

    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        parser.exit(130, "\nPipeline interrupted by user.\n")
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
