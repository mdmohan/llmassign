"""Tokenize cleaned documents and save packed CPT training sequences."""

import json
from pathlib import Path

import numpy as np

from causal_lm import (
    load_causal_lm_metadata,
    resolve_context_length,
    tokenizer_binary_dtype,
    tokenizer_vocabulary_sha256,
)


def _document_boundary_ids(tokenizer, document_separator_token=None):
    """Use EOS between documents, or require one explicit separator token."""
    separator_id = tokenizer.eos_token_id
    separator_source = "eos_token"
    if separator_id is None:
        if not document_separator_token:
            raise ValueError(
                "Tokenizer has no EOS token. Supply --document-separator-token "
                "with one tokenizer token to separate documents."
            )
        vocabulary = tokenizer.get_vocab()
        if document_separator_token not in vocabulary:
            raise ValueError(
                "--document-separator-token must be one exact token from the "
                "selected tokenizer vocabulary"
            )
        separator_id = int(vocabulary[document_separator_token])
        separator_source = "explicit_document_separator_token"

    boundary_ids = []
    if (
        tokenizer.bos_token_id is not None
        and tokenizer.bos_token_id != separator_id
    ):
        boundary_ids.append(int(tokenizer.bos_token_id))
    boundary_ids.append(int(separator_id))
    return boundary_ids, separator_source


def tokenize_causal_lm(
    clean_data,
    model_name="gpt2",
    context_length=None,
    document_separator_token=None,
    return_metrics=False,
):
    """Tokenize and sequence-pack any standard Hugging Face causal LM."""
    config, tokenizer = load_causal_lm_metadata(model_name)
    context_length, model_limit = resolve_context_length(
        config,
        tokenizer,
        context_length,
    )
    binary_dtype, maximum_token_id = tokenizer_binary_dtype(tokenizer)
    boundary_ids, separator_source = _document_boundary_ids(
        tokenizer,
        document_separator_token,
    )

    print(f"Tokenizer: {model_name}")
    print(f"Detected model context limit: {model_limit:,}")
    print(f"Packing context length: {context_length:,}")
    print(f"Binary token dtype: {binary_dtype}")
    paths = clean_data.keys()
    buffer = []
    all_packed_chunks = []
    doc_token_counts = []  # Track individual doc lengths for metrics
    for path in paths:
        print(
            "Tokenizing {}, text size {}".format(
                path, len(clean_data[path]["extracted_text"])
            )
        )
        token_ids = tokenizer.encode(
            clean_data[path]["extracted_text"],
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            verbose=False,
        )

        doc_token_counts.append(len(token_ids))
        if len(boundary_ids) == 2:
            buffer.append(boundary_ids[0])
        buffer.extend(token_ids)
        buffer.append(boundary_ids[-1])
        while len(buffer) >= context_length:
            chunk = buffer[:context_length]
            all_packed_chunks.append(chunk)
            buffer = buffer[context_length:]

    print("Tokenizing complete")

    # --- METRICS CALCULATIONS ---
    total_raw_tokens = sum(doc_token_counts)
    total_docs = len(doc_token_counts)
    avg_doc_length = total_raw_tokens / total_docs if total_docs > 0 else 0
    total_stream_tokens = total_raw_tokens + len(boundary_ids) * total_docs
    total_packed_seqs = len(all_packed_chunks)
    total_packed_tokens = total_packed_seqs * context_length
    metrics = {
        "model_name": model_name,
        "tokenizer_name_or_path": tokenizer.name_or_path,
        "tokenizer_class": type(tokenizer).__name__,
        "model_type": getattr(config, "model_type", None),
        "is_encoder_decoder": bool(getattr(config, "is_encoder_decoder", False)),
        "tokenizer_vocabulary_size": len(tokenizer),
        "tokenizer_base_vocabulary_size": getattr(tokenizer, "vocab_size", None),
        "tokenizer_maximum_token_id": maximum_token_id,
        "tokenizer_vocabulary_sha256": tokenizer_vocabulary_sha256(tokenizer),
        "tokenizer_maximum_context_length": getattr(
            tokenizer,
            "model_max_length",
            None,
        ),
        "detected_model_context_limit": model_limit,
        "packing_context_length": context_length,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "document_separator_token": document_separator_token,
        "document_separator_token_id": boundary_ids[-1],
        "document_separator_source": separator_source,
        "document_count": total_docs,
        "document_token_count_without_boundaries": total_raw_tokens,
        "total_token_count_with_boundaries": total_stream_tokens,
        "average_document_length_tokens": avg_doc_length,
        "boundary_tokens_per_document": len(boundary_ids),
        "packed_sequence_count": total_packed_seqs,
        "packed_token_count": total_packed_tokens,
        "residual_token_count": len(buffer),
        "packing": "concatenated_document_stream_no_padding",
        "binary_dtype": binary_dtype,
    }

    # Print pipeline statistics
    print("==================================================")
    print("             CPT PIPELINE METRICS                 ")
    print("==================================================")
    print(f"Total Documents Processed:      {total_docs:,}")
    print(f"Document Tokens (without EOS):  {total_raw_tokens:,}")
    print(f"Total Token Count (with EOS):   {total_stream_tokens:,}")
    print(f"Average Document Length:        {avg_doc_length:.2f} tokens")
    print(f"Total Packed Sequences ({context_length}):  {total_packed_seqs:,}")
    print(f"Tokens in Packed Sequences:     {total_packed_tokens:,}")
    print(f"Residual Tokens Left in Buffer: {len(buffer):,}")
    print("==================================================\n")

    if return_metrics:
        return all_packed_chunks, context_length, metrics
    return all_packed_chunks, context_length


def tokenize_gpt(
    clean_data,
    model_name="gpt2",
    return_metrics=False,
    context_length=None,
    document_separator_token=None,
):
    """Backward-compatible wrapper around generic causal-LM tokenization."""
    return tokenize_causal_lm(
        clean_data,
        model_name=model_name,
        context_length=context_length,
        document_separator_token=document_separator_token,
        return_metrics=return_metrics,
    )


def save_tokenization_metrics(metrics, filename):
    """Save tokenizer provenance and sequence-packing statistics as JSON."""
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved tokenization metrics to {output_path}")
    return output_path


def _validate_packed_chunks(chunks):
    """Validate fixed-length sequence packing and return the context length."""
    if not chunks:
        raise ValueError("No packed sequences are available to save")

    context_length = len(chunks[0])
    if context_length == 0:
        raise ValueError("Packed sequences cannot be empty")

    invalid_indices = [
        index for index, chunk in enumerate(chunks) if len(chunk) != context_length
    ]
    if invalid_indices:
        preview = invalid_indices[:5]
        raise ValueError(
            "All packed sequences must have the same context length; "
            f"invalid sequence indices include {preview}"
        )

    return context_length


def save_chunks_to_parquet(
    chunks,
    filename,
    batch_size=1024,
    compression="zstd",
    binary_dtype="uint16",
):
    """Save one fixed-length packed token sequence per Parquet row.

    The ``input_ids`` column is a fixed-size list, so each row represents one
    context-window-sized training example without padding. Batches are written
    incrementally to avoid constructing a second full in-memory copy.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Parquet output requires pyarrow. Install it with: pip install pyarrow"
        ) from exc

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    context_length = _validate_packed_chunks(chunks)
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    numpy_dtype = np.dtype(binary_dtype)
    if numpy_dtype.kind != "u" or numpy_dtype.itemsize > 8:
        raise ValueError(f"Unsupported unsigned token dtype: {binary_dtype}")
    arrow_token_type = pa.uint64() if numpy_dtype.itemsize > 4 else pa.uint32()
    input_ids_type = pa.list_(arrow_token_type, context_length)
    schema = pa.schema(
        [
            ("sequence_id", pa.int64()),
            ("input_ids", input_ids_type),
        ],
        metadata={
            b"packing": b"concatenated_document_stream_no_padding",
            b"context_length": str(context_length).encode("utf-8"),
            b"total_packed_sequences": str(len(chunks)).encode("utf-8"),
            b"total_packed_tokens": str(
                len(chunks) * context_length
            ).encode("utf-8"),
            b"binary_dtype": numpy_dtype.name.encode("utf-8"),
        },
    )

    with pq.ParquetWriter(
        output_path,
        schema,
        compression=compression,
    ) as writer:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            parquet_numpy_dtype = (
                np.uint64 if numpy_dtype.itemsize > 4 else np.uint32
            )
            token_matrix = np.asarray(batch, dtype=parquet_numpy_dtype)
            token_values = pa.array(
                token_matrix.reshape(-1),
                type=arrow_token_type,
            )
            input_ids = pa.FixedSizeListArray.from_arrays(
                token_values,
                context_length,
            )
            sequence_ids = pa.array(
                range(start, start + len(batch)),
                type=pa.int64(),
            )
            table = pa.Table.from_arrays(
                [sequence_ids, input_ids],
                schema=schema,
            )
            writer.write_table(table)

    print(
        f"Saved {len(chunks):,} packed sequences "
        f"({context_length} tokens each) to {output_path}"
    )


def save_chunks_to_disk(
    chunks,
    filename,
    parquet_filename=None,
    parquet_batch_size=1024,
    binary_dtype="uint16",
):
    """Save the existing flat binary and optionally a Parquet copy.

    Existing calls with ``save_chunks_to_disk(chunks, filename)`` remain
    unchanged. Pass ``parquet_filename`` to write both formats in one call.
    """
    _validate_packed_chunks(chunks)

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    numpy_dtype = np.dtype(binary_dtype)
    if numpy_dtype.kind != "u" or numpy_dtype.itemsize > 8:
        raise ValueError(f"Unsupported unsigned token dtype: {binary_dtype}")
    maximum_token_id = max(max(chunk) for chunk in chunks)
    if maximum_token_id > np.iinfo(numpy_dtype).max:
        raise ValueError(
            f"Token ID {maximum_token_id:,} does not fit {numpy_dtype.name}"
        )
    flat_tokens = np.asarray(chunks, dtype=numpy_dtype).reshape(-1)
    flat_tokens.tofile(output_path)
    print("Saved trainable binary chunks to", output_path)

    if parquet_filename is not None:
        save_chunks_to_parquet(
            chunks,
            parquet_filename,
            batch_size=parquet_batch_size,
            binary_dtype=numpy_dtype.name,
        )
