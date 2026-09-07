"""SmolLM2-specific tokenization and sequence packing."""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

from cache import CACHE_DIR
from smollm2_model import (
    SMOLLM2_MODEL_ID,
    canonical_smollm2_model_id,
)


def tokenize_smollm2(
    clean_data,
    model_name: str = SMOLLM2_MODEL_ID,
    context_length: int | None = None,
):
    """Tokenize documents and pack one flat stream without padding."""
    model_id = canonical_smollm2_model_id(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        cache_dir=CACHE_DIR,
    )

    tokenizer_limit = int(tokenizer.model_max_length)
    if context_length is None:
        context_length = tokenizer_limit
    if context_length < 2:
        raise ValueError("context_length must be at least 2 for causal training")
    if context_length > tokenizer_limit:
        raise ValueError(
            f"Requested context length {context_length:,} exceeds the "
            f"tokenizer limit of {tokenizer_limit:,}"
        )
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("SmolLM2 tokenizer must define BOS and EOS token IDs")
    if tokenizer.vocab_size > 65_536:
        raise ValueError(
            "The tokenizer vocabulary does not fit the uint16 binary format"
        )

    print(f"Tokenizer: {model_id}")
    print(f"Tokenizer max length: {tokenizer_limit:,}")
    print(f"Packing context length: {context_length:,}")

    buffer: list[int] = []
    packed_chunks: list[list[int]] = []
    document_token_counts: list[int] = []
    shared_boundary_token = tokenizer.bos_token_id == tokenizer.eos_token_id
    boundary_tokens_per_document = 1 if shared_boundary_token else 2

    for path, document in clean_data.items():
        text = document["extracted_text"]
        print(f"Tokenizing {path}, text size {len(text):,}")
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        document_token_counts.append(len(token_ids))

        # Avoid writing the same boundary twice when BOS and EOS share an ID.
        if not shared_boundary_token:
            buffer.append(tokenizer.bos_token_id)
        buffer.extend(token_ids)
        buffer.append(tokenizer.eos_token_id)

        while len(buffer) >= context_length:
            packed_chunks.append(buffer[:context_length])
            del buffer[:context_length]

    raw_token_count = sum(document_token_counts)
    document_count = len(document_token_counts)
    stream_token_count = raw_token_count + (
        boundary_tokens_per_document * document_count
    )
    packed_token_count = len(packed_chunks) * context_length
    metrics = {
        "model_name": model_id,
        "tokenizer_name_or_path": tokenizer.name_or_path,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocabulary_size": tokenizer.vocab_size,
        "tokenizer_maximum_context_length": tokenizer_limit,
        "packing_context_length": context_length,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "document_count": document_count,
        "document_token_count_without_boundaries": raw_token_count,
        "total_token_count_with_boundaries": stream_token_count,
        "average_document_length_tokens": (
            raw_token_count / document_count if document_count else 0
        ),
        "boundary_tokens_per_document": boundary_tokens_per_document,
        "packed_sequence_count": len(packed_chunks),
        "packed_token_count": packed_token_count,
        "residual_token_count": len(buffer),
        "packing": "concatenated_document_stream_no_padding",
        "binary_dtype": "uint16",
    }

    print("Tokenizing complete")
    print("=" * 58)
    print("               SMOLLM2 CPT PIPELINE METRICS")
    print("=" * 58)
    print(f"Documents processed:              {document_count:,}")
    print(f"Document tokens (no boundaries):  {raw_token_count:,}")
    print(f"Total tokens (with boundaries):   {stream_token_count:,}")
    print(
        "Average document length:         "
        f"{metrics['average_document_length_tokens']:.2f} tokens"
    )
    print(f"Packed sequences ({context_length:,}):       {len(packed_chunks):,}")
    print(f"Tokens in packed sequences:       {packed_token_count:,}")
    print(f"Residual tokens:                  {len(buffer):,}")
    print("=" * 58 + "\n")

    return packed_chunks, context_length, metrics


def save_tokenization_metrics(metrics: dict, filename) -> Path:
    """Save tokenizer provenance and sequence-packing statistics."""
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved tokenization metrics to {output_path}")
    return output_path
