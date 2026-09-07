"""Deterministic tokenization and fixed-length packing for continual pre-training."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .corpus import sha256_bytes, sha256_file, write_json, write_jsonl


SPLITS = ("train", "eval")


def load_accepted_documents(manifest_path: Path) -> list[dict[str, Any]]:
    documents = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    accepted = [document for document in documents if document.get("accepted") is True]
    if not accepted:
        raise ValueError("Manifest contains no accepted documents")
    if any(document.get("split") not in SPLITS for document in accepted):
        raise ValueError("Every accepted document must have a train or eval split")
    identifiers = [str(document["document_id"]) for document in accepted]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Accepted document IDs must be unique")
    return accepted


def load_frozen_tokenizer(
    model_id: str, revision: str, cache_dir: Path | None = None
) -> tuple[Any, dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        use_fast=True,
    )
    if tokenizer.bos_token != "<|endoftext|>" or tokenizer.eos_token != "<|endoftext|>":
        raise ValueError("Frozen tokenizer must use <|endoftext|> for both BOS and EOS")
    if tokenizer.bos_token_id != 0 or tokenizer.eos_token_id != 0:
        raise ValueError("Frozen tokenizer must use token ID 0 for both BOS and EOS")

    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_json = backend.to_str() if backend is not None else None
    metadata = {
        "model_id": model_id,
        "requested_revision": revision,
        "resolved_revision": tokenizer.init_kwargs.get("_commit_hash", revision),
        "tokenizer_class": type(tokenizer).__name__,
        "is_fast": bool(tokenizer.is_fast),
        "vocabulary_size": int(tokenizer.vocab_size),
        "tokenizer_length": len(tokenizer),
        "model_max_length": int(tokenizer.model_max_length),
        "bos_token": tokenizer.bos_token,
        "bos_token_id": int(tokenizer.bos_token_id),
        "eos_token": tokenizer.eos_token,
        "eos_token_id": int(tokenizer.eos_token_id),
        "backend_tokenizer_sha256": (
            sha256_bytes(backend_json.encode("utf-8")) if backend_json is not None else None
        ),
    }
    return tokenizer, metadata


def _read_cleaned_document(document_root: Path, document_id: str) -> str:
    text = (document_root / f"{document_id}.txt").read_text(encoding="utf-8")
    return text[:-1] if text.endswith("\n") else text


def _parquet_schema(sequence_length: int) -> Any:
    import pyarrow as pa

    tokens = pa.list_(pa.int32(), sequence_length)
    return pa.schema(
        [
            pa.field("sequence_id", pa.int64(), nullable=False),
            pa.field("input_ids", tokens, nullable=False),
            pa.field("attention_mask", tokens, nullable=False),
            pa.field("labels", tokens, nullable=False),
        ]
    )


def _write_parquet_batch(writer: Any, schema: Any, rows: Sequence[tuple[int, list[int]]]) -> None:
    import pyarrow as pa

    sequence_ids = pa.array([sequence_id for sequence_id, _ in rows], type=pa.int64())
    input_ids = pa.array([tokens for _, tokens in rows], type=schema.field("input_ids").type)
    attention_mask = pa.array(
        [[1] * len(tokens) for _, tokens in rows],
        type=schema.field("attention_mask").type,
    )
    table = pa.Table.from_arrays(
        [sequence_ids, input_ids, attention_mask, input_ids],
        schema=schema,
    )
    writer.write_table(table)


def _pack_split(
    documents: Sequence[Mapping[str, Any]],
    document_root: Path,
    output_path: Path,
    tokenizer: Any,
    sequence_length: int,
    parquet_batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import pyarrow.parquet as pq

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if parquet_batch_size <= 0:
        raise ValueError("parquet_batch_size must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    schema = _parquet_schema(sequence_length)
    writer = pq.ParquetWriter(temporary_path, schema, compression="zstd", use_dictionary=False)
    pending: list[int] = []
    pending_offset = 0
    parquet_rows: list[tuple[int, list[int]]] = []
    spans: list[dict[str, Any]] = []
    sequence_id = 0
    total_tokens = 0
    total_content_tokens = 0
    bos_token_id = int(tokenizer.bos_token_id)
    eos_token_id = int(tokenizer.eos_token_id)

    try:
        for document in documents:
            identifier = str(document["document_id"])
            text = _read_cleaned_document(document_root, identifier)
            content_tokens = list(
                tokenizer.encode(text, add_special_tokens=False, verbose=False)
            )
            wrapped_tokens = [bos_token_id, *content_tokens, eos_token_id]
            stream_start = total_tokens
            stream_end = stream_start + len(wrapped_tokens)
            spans.append(
                {
                    "document_id": identifier,
                    "source_corpus": document["source_corpus"],
                    "source_path": document["source_path"],
                    "split": document["split"],
                    "content_token_count": len(content_tokens),
                    "wrapped_token_count": len(wrapped_tokens),
                    "stream_start": stream_start,
                    "stream_end_exclusive": stream_end,
                    "bos_position": stream_start,
                    "eos_position": stream_end - 1,
                }
            )
            total_content_tokens += len(content_tokens)
            total_tokens = stream_end
            pending.extend(wrapped_tokens)

            while len(pending) - pending_offset >= sequence_length:
                end = pending_offset + sequence_length
                parquet_rows.append((sequence_id, pending[pending_offset:end]))
                pending_offset = end
                sequence_id += 1
                if len(parquet_rows) >= parquet_batch_size:
                    _write_parquet_batch(writer, schema, parquet_rows)
                    parquet_rows.clear()

            if pending_offset:
                pending = pending[pending_offset:]
                pending_offset = 0

        if parquet_rows:
            _write_parquet_batch(writer, schema, parquet_rows)
        writer.close()
        temporary_path.replace(output_path)
    except Exception:
        writer.close()
        temporary_path.unlink(missing_ok=True)
        raise

    packed_tokens = sequence_id * sequence_length
    dropped_tokens = total_tokens - packed_tokens
    for span in spans:
        retained_end = min(int(span["stream_end_exclusive"]), packed_tokens)
        retained_start = min(int(span["stream_start"]), packed_tokens)
        span["packed_token_count"] = max(0, retained_end - retained_start)
        span["truncated_by_final_remainder"] = retained_end < int(span["stream_end_exclusive"])

    statistics = {
        "document_count": len(documents),
        "source_corpus_counts": dict(Counter(str(item["source_corpus"]) for item in documents)),
        "content_token_count": total_content_tokens,
        "total_token_count": total_tokens,
        "mean_document_content_tokens": total_content_tokens / len(documents),
        "mean_document_tokens_with_boundaries": total_tokens / len(documents),
        "packed_sequence_count": sequence_id,
        "packed_token_count": packed_tokens,
        "dropped_token_count": dropped_tokens,
        "parquet_sha256": sha256_file(output_path),
    }
    return statistics, spans


def build_packed_dataset(
    manifest_path: Path,
    document_root: Path,
    packed_root: Path,
    reports_root: Path,
    model_id: str,
    revision: str,
    cache_dir: Path | None = None,
    sequence_length: int = 8192,
    parquet_batch_size: int = 64,
) -> dict[str, Any]:
    documents = load_accepted_documents(manifest_path)
    tokenizer, tokenizer_metadata = load_frozen_tokenizer(model_id, revision, cache_dir)
    if tokenizer_metadata["model_max_length"] < sequence_length:
        raise ValueError("Packing length exceeds the tokenizer model_max_length")

    packed_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    split_reports: dict[str, Any] = {}
    for split in SPLITS:
        split_documents = [document for document in documents if document["split"] == split]
        statistics, spans = _pack_split(
            split_documents,
            document_root,
            packed_root / f"{split}.parquet",
            tokenizer,
            sequence_length,
            parquet_batch_size,
        )
        spans_path = reports_root / f"{split}_token_spans.jsonl"
        write_jsonl(spans_path, spans)
        statistics["source_spans_sha256"] = sha256_file(spans_path)
        split_reports[split] = statistics

    total_content_tokens = sum(
        item["content_token_count"] for item in split_reports.values()
    )
    total_tokens = sum(item["total_token_count"] for item in split_reports.values())
    packed_tokens = sum(item["packed_token_count"] for item in split_reports.values())
    report = {
        "model_id": model_id,
        "revision": revision,
        "sequence_length": sequence_length,
        "source_manifest_sha256": sha256_file(manifest_path),
        "accepted_document_count": len(documents),
        "content_token_count": total_content_tokens,
        "total_token_count": total_tokens,
        "mean_document_content_tokens": total_content_tokens / len(documents),
        "mean_document_tokens_with_boundaries": total_tokens / len(documents),
        "packed_sequence_count": sum(
            item["packed_sequence_count"] for item in split_reports.values()
        ),
        "packed_token_count": packed_tokens,
        "dropped_token_count": total_tokens - packed_tokens,
        "splits": split_reports,
    }
    write_json(reports_root / "tokenizer_metadata.json", tokenizer_metadata)
    write_json(reports_root / "tokenization_report.json", report)
    return report