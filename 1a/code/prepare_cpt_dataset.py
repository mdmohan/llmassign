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
import random
from pathlib import Path

from cli_parsers import build_prepare_cpt_parser


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def load_clean_text_documents(input_path: Path) -> dict:
    """Load training-ready text without applying extraction or cleaning."""
    if input_path.is_file():
        if input_path.suffix.lower() != ".txt":
            raise ValueError("--tokenize-only requires a .txt file or directory")
        source_paths = [input_path]
    else:
        source_paths = sorted(
            path
            for path in input_path.rglob("*.txt")
            if path.is_file()
            and not any(
                part.startswith(".")
                for part in path.relative_to(input_path).parts
            )
        )

    if not source_paths:
        raise ValueError(
            f"No .txt files found for --tokenize-only under {input_path}"
        )

    documents = {}
    for source_path in source_paths:
        text = source_path.read_text(encoding="utf-8-sig")
        documents[str(source_path)] = {
            "extracted_text": text,
            "output_filename": str(source_path),
            "status": "loaded_clean_text_input",
        }

    print(f"Loaded {len(documents):,} training-ready text file(s).")
    return documents


def split_documents(documents: dict, split: str, seed: int) -> tuple[dict, dict]:
    """Create a deterministic document-level split without text leakage."""
    train_percent, test_percent = (int(value) for value in split.split(":"))
    if train_percent + test_percent != 100:
        raise ValueError("--train-test-split percentages must add up to 100")
    if len(documents) < 2:
        raise ValueError("At least two retained documents are required for a split")

    keys = list(documents)
    random.Random(seed).shuffle(keys)
    test_count = max(1, round(len(keys) * test_percent / 100))
    test_count = min(test_count, len(keys) - 1)
    test_keys = set(keys[:test_count])
    train_documents = {key: documents[key] for key in keys if key not in test_keys}
    test_documents = {key: documents[key] for key in keys if key in test_keys}

    print(
        f"Document split ({split}, seed {seed}): "
        f"{len(train_documents):,} train, {len(test_documents):,} test"
    )
    return train_documents, test_documents


def split_output_paths(
    bin_file: Path,
    parquet_file: Path,
    metrics_file: Path,
) -> dict[str, tuple[Path, Path, Path]]:
    """Derive split filenames from the parent folders of existing arguments."""
    return {
        "train": (
            bin_file.parent / "token_train.bin",
            parquet_file.parent / "token_train.parquet",
            metrics_file.parent / "dataset_metrics_train.json",
        ),
        "test": (
            bin_file.parent / "token_test.bin",
            parquet_file.parent / "token_test.parquet",
            metrics_file.parent / "dataset_metrics_test.json",
        ),
    }


def run_pipeline(args: argparse.Namespace) -> None:
    input_path = resolve_path(args.input_dir)
    text_dir = resolve_path(args.text_dir)
    audit_report = resolve_path(args.audit_report)
    content_audit_report = (
        resolve_path(args.content_audit_report)
        if args.content_audit_report is not None
        else audit_report.with_name(f"{audit_report.stem}_content.json")
    )
    cleaned_text_dir = (
        resolve_path(args.cleaned_text_dir)
        if args.cleaned_text_dir is not None
        else None
    )
    bin_file = resolve_path(args.bin_file)
    parquet_file = resolve_path(args.parquet_file)
    metrics_file = (
        resolve_path(args.metrics_file)
        if args.metrics_file is not None
        else bin_file.parent / "dataset_metrics.json"
    )

    if not input_path.exists():
        raise ValueError(f"Input path does not exist: {input_path}")
    if bin_file == parquet_file:
        raise ValueError("--bin-file and --parquet-file must be different paths")

    if (
        not args.tokenize_only
        and input_path.is_file()
        and input_path.suffix.lower() not in {".pdf", ".txt"}
    ):
        raise ValueError(
            "A file input must use the .pdf or .txt extension: "
            f"{input_path}"
        )

    if args.tokenize_only:
        print("\n=== Stage 1/3: Load training-ready text ===")
        extracted_documents = load_clean_text_documents(input_path)
        clean_documents = extracted_documents
        rejected_documents = {}
        content_rejected_documents = {}
        preparation_mode = "tokenize_only"
        tokenize_stage = "2/3"
        write_stage = "3/3"
    else:
        print("\n=== Stage 1/5: Load PDF and text sources ===")
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
            raise RuntimeError(
                f"No PDF or text documents were loaded from {input_path}"
            )

        print("\n=== Stage 2/5: Apply document quality gates ===")
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

        content_rejected_documents = {}
        if args.content_cleaning:
            print("\n=== Stage 3/5: Clean training content ===")
            from training_content_cleaner import TrainingContentCleaner

            content_cleaner = TrainingContentCleaner(
                min_content_chars=max(args.min_chars, 200),
            )
            clean_documents, content_rejected_documents = (
                content_cleaner.clean_corpus(
                    clean_documents,
                    audit_report_path=content_audit_report,
                    cleaned_text_dir=cleaned_text_dir,
                    upstream_rejected=rejected_documents,
                )
            )
            if not clean_documents:
                raise RuntimeError(
                    "Every retained document was rejected after content cleaning; "
                    "inspect the content-cleaning audit report"
                )
        else:
            print("\n=== Stage 3/5: Content cleaning disabled ===")
        preparation_mode = "full_pipeline"
        tokenize_stage = "4/5"
        write_stage = "5/5"

    print(f"\n=== Stage {tokenize_stage}: Tokenize and sequence-pack ===")
    from tokenize_data import (
        save_chunks_to_disk,
        save_chunks_to_parquet,
        save_tokenization_metrics,
    )
    from smollm2_model import is_smollm2_model

    if args.train_test_split:
        train_documents, test_documents = split_documents(
            clean_documents,
            args.train_test_split,
            args.split_seed,
        )
        document_splits = {
            "train": train_documents,
            "test": test_documents,
        }
        output_paths = split_output_paths(bin_file, parquet_file, metrics_file)
    else:
        document_splits = {"all": clean_documents}
        output_paths = {"all": (bin_file, parquet_file, metrics_file)}

    tokenization_results = {}
    for split_name, split_data in document_splits.items():
        if args.train_test_split:
            print(f"\n--- Tokenizing {split_name} split ---")
        if is_smollm2_model(args.model_name):
            from tokenize_smollm2 import tokenize_smollm2

            packed_chunks, context_length, tokenization_metrics = tokenize_smollm2(
                split_data,
                model_name=args.model_name,
                context_length=args.context_length,
            )
        else:
            from tokenize_data import tokenize_gpt

            packed_chunks, context_length, tokenization_metrics = tokenize_gpt(
                split_data,
                model_name=args.model_name,
                return_metrics=True,
            )
            if (
                args.context_length is not None
                and args.context_length != context_length
            ):
                raise ValueError(
                    "The existing GPT-2 tokenizer packs at its fixed context "
                    f"length of {context_length}; omit --context-length or use "
                    f"{context_length}."
                )
        if not packed_chunks:
            raise RuntimeError(
                f"The {split_name} split did not produce one complete "
                "context-length sequence"
            )
        tokenization_metrics.update(
            {
                "dataset_split": split_name,
                "requested_train_test_split": args.train_test_split,
                "split_seed": args.split_seed if args.train_test_split else None,
                "source_documents": [str(path) for path in split_data],
                "preparation_mode": preparation_mode,
            }
        )
        tokenization_results[split_name] = (
            packed_chunks,
            context_length,
            tokenization_metrics,
        )

    print(f"\n=== Stage {write_stage}: Write binary and Parquet outputs ===")
    parquet_compression = (
        None
        if args.parquet_compression == "none"
        else args.parquet_compression
    )
    saved_outputs = {}
    for split_name, result in tokenization_results.items():
        packed_chunks, context_length, tokenization_metrics = result
        split_bin, split_parquet, split_metrics = output_paths[split_name]
        save_chunks_to_disk(packed_chunks, split_bin)
        save_chunks_to_parquet(
            packed_chunks,
            split_parquet,
            batch_size=args.parquet_batch_size,
            compression=parquet_compression,
        )
        tokenization_metrics_path = save_tokenization_metrics(
            tokenization_metrics,
            split_metrics,
        )
        saved_outputs[split_name] = {
            "bin": split_bin,
            "parquet": split_parquet,
            "metrics": tokenization_metrics_path,
            "sequences": len(packed_chunks),
            "context_length": context_length,
        }

    print("\n=== Pipeline complete ===")
    print(f"Source documents found:{len(extracted_documents):,}")
    print(f"Documents retained:    {len(clean_documents):,}")
    print(f"Rejected by gates:     {len(rejected_documents):,}")
    print(f"Rejected after cleanup:{len(content_rejected_documents):,}")
    print(f"Preparation mode:      {preparation_mode}")
    print(f"Tokenizer model:       {args.model_name}")
    if not args.tokenize_only:
        print(f"Extracted PDF cache:   {text_dir}")
        print(f"Cleaning audit report: {audit_report}")
        if args.content_cleaning:
            print(f"Content-cleaning audit:{content_audit_report}")
            if cleaned_text_dir is not None:
                print(f"Training-ready text:   {cleaned_text_dir}")
    for split_name, saved in saved_outputs.items():
        label = split_name.capitalize() if split_name != "all" else "Dataset"
        print(f"{label} documents:      {len(document_splits[split_name]):,}")
        print(f"{label} context length: {saved['context_length']:,}")
        print(f"{label} sequences:      {saved['sequences']:,}")
        print(f"{label} binary:         {saved['bin']}")
        print(f"{label} Parquet:        {saved['parquet']}")
        print(f"{label} metrics:        {saved['metrics']}")


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
