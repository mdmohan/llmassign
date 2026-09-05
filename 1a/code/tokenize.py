import os
from pathlib import Path

os.environ["HF_HOME"] = "/home/jovyan/llmgenai/hf_cache"
from dotenv import load_dotenv
import numpy as np

load_dotenv()


from transformers import AutoTokenizer


def tokenize_gpt(clean_data):
    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2",
        use_fast=True,
        cache_dir="/home/jovyan/llmgenai/hf_cache",
    )
    eos_id = tokenizer.eos_token_id
    context_length = tokenizer.model_max_length

    print(f"Tokenizer max length: {context_length}")
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
        )

        doc_token_counts.append(len(token_ids))
        token_ids.extend([eos_id])
        buffer.extend(token_ids)
        while len(buffer) >= context_length:
            chunk = buffer[:context_length]
            all_packed_chunks.append(chunk)
            buffer = buffer[context_length:]

    print("Tokenizing complete")

    # --- METRICS CALCULATIONS ---
    total_raw_tokens = sum(doc_token_counts)
    total_docs = len(doc_token_counts)
    avg_doc_length = total_raw_tokens / total_docs if total_docs > 0 else 0
    total_stream_tokens = total_raw_tokens + total_docs  # One EOS per document
    total_packed_seqs = len(all_packed_chunks)
    total_packed_tokens = total_packed_seqs * context_length

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

    return all_packed_chunks, context_length


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

    input_ids_type = pa.list_(pa.uint32(), context_length)
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
        },
    )

    with pq.ParquetWriter(
        output_path,
        schema,
        compression=compression,
    ) as writer:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            token_matrix = np.asarray(batch, dtype=np.uint32)
            token_values = pa.array(
                token_matrix.reshape(-1),
                type=pa.uint32(),
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
):
    """Save the existing flat binary and optionally a Parquet copy.

    Existing calls with ``save_chunks_to_disk(chunks, filename)`` remain
    unchanged. Pass ``parquet_filename`` to write both formats in one call.
    """
    _validate_packed_chunks(chunks)

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flat_tokens = np.asarray(chunks, dtype=np.uint16).reshape(-1)
    flat_tokens.tofile(output_path)
    print("Saved trainable binary chunks to", output_path)

    if parquet_filename is not None:
        save_chunks_to_parquet(
            chunks,
            parquet_filename,
            batch_size=parquet_batch_size,
        )
