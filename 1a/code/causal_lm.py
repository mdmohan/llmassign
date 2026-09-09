"""Shared Hugging Face decoder-only causal-language-model support."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from cache import CACHE_DIR


AUTOMATIC_CONTEXT_LIMIT = 8192
_TOKEN_DTYPES = (
    (np.iinfo(np.uint16).max, "uint16"),
    (np.iinfo(np.uint32).max, "uint32"),
    (np.iinfo(np.uint64).max, "uint64"),
)


def resolve_model_source(model_name: str, model_folder=None):
    """Return an unchanged Hub ID or a resolved local checkpoint folder."""
    if model_folder is None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("--model-name must be a non-empty Hugging Face model ID")
        return model_name.strip(), False

    from load_local_model import resolve_model_folder

    return resolve_model_folder(model_folder), True


def resolve_adapter_folder(adapter_folder) -> Path:
    """Resolve a PEFT adapter directory or a run containing final_adapter."""
    supplied = Path(adapter_folder).expanduser().resolve()
    if not supplied.is_dir():
        raise FileNotFoundError(f"Adapter folder was not found: {supplied}")

    if (supplied / "adapter_config.json").is_file():
        return supplied

    final_adapter = supplied / "final_adapter"
    if (final_adapter / "adapter_config.json").is_file():
        return final_adapter

    raise FileNotFoundError(
        "No PEFT adapter_config.json was found in the supplied directory "
        f"or its final_adapter child: {supplied}"
    )


def _usable_context_value(value) -> int | None:
    """Reject missing values and tokenizer sentinel values used for no limit."""
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed < 2 or parsed >= 10_000_000:
        return None
    return parsed


def model_context_limit(config, tokenizer=None) -> int | None:
    """Find the model's finite context limit using common config names."""
    for attribute in ("max_position_embeddings", "n_positions", "n_ctx"):
        value = _usable_context_value(getattr(config, attribute, None))
        if value is not None:
            return value
    if tokenizer is not None:
        return _usable_context_value(getattr(tokenizer, "model_max_length", None))
    return None


def resolve_context_length(config, tokenizer, requested: int | None) -> tuple[int, int]:
    """Choose a practical packing length and validate it against model metadata."""
    model_limit = model_context_limit(config, tokenizer)
    if requested is not None:
        if requested < 2:
            raise ValueError("context_length must be at least 2 for causal training")
        if model_limit is not None and requested > model_limit:
            raise ValueError(
                f"Requested context length {requested:,} exceeds the detected "
                f"model limit of {model_limit:,}"
            )
        return int(requested), model_limit or int(requested)

    if model_limit is None:
        raise ValueError(
            "Unable to determine a finite model context length; supply "
            "--context-length explicitly"
        )
    if model_limit > AUTOMATIC_CONTEXT_LIMIT:
        raise ValueError(
            f"The model supports a {model_limit:,}-token context. Supply a "
            "practical --context-length explicitly (for example 1024 or 2048)."
        )
    return model_limit, model_limit


def tokenizer_maximum_token_id(tokenizer) -> int:
    """Return the actual largest vocabulary/special-token ID."""
    vocabulary = tokenizer.get_vocab()
    token_ids = list(vocabulary.values())
    token_ids.extend(
        token_id
        for token_id in getattr(tokenizer, "all_special_ids", [])
        if token_id is not None
    )
    if not token_ids:
        raise ValueError("Tokenizer vocabulary contains no token IDs")
    maximum = max(int(token_id) for token_id in token_ids)
    if maximum < 0:
        raise ValueError("Tokenizer contains a negative token ID")
    return maximum


def binary_dtype_for_maximum_token_id(maximum_token_id: int) -> str:
    """Select the narrowest unsigned binary representation that fits all IDs."""
    if maximum_token_id < 0:
        raise ValueError("maximum_token_id cannot be negative")
    for limit, dtype_name in _TOKEN_DTYPES:
        if maximum_token_id <= limit:
            return dtype_name
    raise ValueError("Tokenizer IDs exceed the supported uint64 binary format")


def tokenizer_binary_dtype(tokenizer) -> tuple[str, int]:
    maximum_token_id = tokenizer_maximum_token_id(tokenizer)
    return binary_dtype_for_maximum_token_id(maximum_token_id), maximum_token_id


def tokenizer_vocabulary_sha256(tokenizer) -> str:
    """Fingerprint token-to-ID mappings, not merely vocabulary size."""
    digest = hashlib.sha256()
    for token, token_id in sorted(
        tokenizer.get_vocab().items(),
        key=lambda item: (int(item[1]), item[0]),
    ):
        record = json.dumps(
            [int(token_id), token],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest.update(record.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_causal_config(config, source) -> None:
    """Reject configurations that clearly describe a non-causal architecture."""
    if bool(getattr(config, "is_encoder_decoder", False)):
        raise ValueError(
            f"{source!r} is an encoder-decoder model, not a decoder-only causal LM"
        )
    architectures = getattr(config, "architectures", None) or []
    if architectures and not any(
        "causallm" in architecture.casefold()
        or architecture.casefold().endswith("lmheadmodel")
        for architecture in architectures
    ):
        raise ValueError(
            f"{source!r} does not declare a causal-LM architecture: "
            f"{architectures}"
        )


def load_causal_lm_metadata(model_name: str):
    """Load configuration and tokenizer metadata without model weights."""
    source, _ = resolve_model_source(model_name)
    config = AutoConfig.from_pretrained(source, cache_dir=CACHE_DIR)
    validate_causal_config(config, source)
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        cache_dir=CACHE_DIR,
        use_fast=True,
    )
    return config, tokenizer


def preferred_model_dtype(device) -> torch.dtype:
    """Use BF16 on capable CUDA, FP16 on older CUDA, and FP32 elsewhere."""
    device = torch.device(device)
    if device.type != "cuda":
        return torch.float32
    supports_bf16 = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(supports_bf16) and supports_bf16():
        return torch.bfloat16
    return torch.float16


def load_causal_lm(
    model_name: str = "gpt2",
    model_folder=None,
    adapter_folder=None,
    device=None,
    for_training: bool = False,
):
    """Load a decoder-only model and optionally attach a PEFT adapter."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    source, local_files_only = resolve_model_source(model_name, model_folder)
    config = AutoConfig.from_pretrained(
        source,
        cache_dir=CACHE_DIR,
        local_files_only=local_files_only,
    )
    validate_causal_config(config, source)

    adapter_source = (
        resolve_adapter_folder(adapter_folder)
        if adapter_folder is not None
        else None
    )

    print(f"Using device: {device}")
    print(f"Loading causal LM from: {source}")
    if adapter_source is not None:
        print(f"Loading PEFT adapter from: {adapter_source}")

    tokenizer_source = source
    tokenizer_local_files_only = local_files_only
    if (
        adapter_source is not None
        and (adapter_source / "tokenizer_config.json").is_file()
    ):
        # QLoRA saves the exact tokenizer used for adapter training. Prefer
        # it so chat-template and special-token settings match the adapter.
        tokenizer_source = adapter_source
        tokenizer_local_files_only = True

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        cache_dir=CACHE_DIR,
        local_files_only=tokenizer_local_files_only,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        source,
        cache_dir=CACHE_DIR,
        local_files_only=local_files_only,
    )

    if adapter_source is not None:
        if for_training:
            raise ValueError(
                "adapter_folder is intended for inference; adapter training "
                "is handled by the LoRA/QLoRA training script"
            )
        try:
            from peft import PeftModel
        except ImportError as error:
            raise ImportError(
                "Loading a LoRA/QLoRA adapter requires the peft package"
            ) from error

        model = PeftModel.from_pretrained(
            model,
            adapter_source,
            is_trainable=False,
        )
    model.to(device)
    parameter_dtype = next(model.parameters()).dtype
    compute_dtype = (
        preferred_model_dtype(device) if for_training else parameter_dtype
    )
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if for_training:
        model.config.use_cache = False
        enable_checkpointing = getattr(model, "gradient_checkpointing_enable", None)
        if (
            callable(enable_checkpointing)
            and bool(getattr(model, "supports_gradient_checkpointing", False))
        ):
            enable_checkpointing()
        model.train()
    else:
        model.config.use_cache = True
        model.eval()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Successfully loaded {source} with {parameter_count:,} parameters "
        f"using {parameter_dtype} parameters."
    )
    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "dtype": compute_dtype,
        "parameter_dtype": parameter_dtype,
        "model_source": str(source),
        "adapter_source": (
            str(adapter_source) if adapter_source is not None else None
        ),
    }


def causal_lm_model_details(model, tokenizer, device) -> dict:
    """Return architecture-neutral model and tokenizer provenance."""
    config = model.config
    parameters = list(model.parameters())
    total_parameters = sum(parameter.numel() for parameter in parameters)
    trainable_parameters = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    hidden_size = getattr(config, "hidden_size", getattr(config, "n_embd", None))
    attention_heads = getattr(
        config,
        "num_attention_heads",
        getattr(config, "n_head", None),
    )
    return {
        "name_or_path": getattr(config, "_name_or_path", None),
        "model_class": type(model).__name__,
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "is_encoder_decoder": bool(getattr(config, "is_encoder_decoder", False)),
        "transformer_layers": getattr(
            config,
            "num_hidden_layers",
            getattr(config, "n_layer", None),
        ),
        "attention_heads": attention_heads,
        "key_value_heads": getattr(config, "num_key_value_heads", None),
        "embedding_dimension": hidden_size,
        "vocabulary_size": getattr(config, "vocab_size", None),
        "maximum_context_length": model_context_limit(config, tokenizer),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "parameter_dtype": str(parameters[0].dtype) if parameters else None,
        "device": str(device),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        "tokenizer_vocabulary_size": len(tokenizer),
        "tokenizer_base_vocabulary_size": getattr(tokenizer, "vocab_size", None),
        "tokenizer_maximum_token_id": tokenizer_maximum_token_id(tokenizer),
        "tokenizer_vocabulary_sha256": tokenizer_vocabulary_sha256(tokenizer),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }


def validate_causal_lm_forward(model, tokenizer, device) -> dict:
    """Run a tiny teacher-forced pass and require a finite causal-LM loss."""
    token_ids = tokenizer.encode(
        "Causal language models predict the next token.",
        add_special_tokens=True,
    )
    if len(token_ids) < 2:
        boundary = tokenizer.eos_token_id or tokenizer.bos_token_id
        if boundary is not None:
            token_ids.append(boundary)
    if len(token_ids) < 2:
        raise ValueError("Tokenizer could not produce two tokens for a forward check")

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        loss = model(input_ids=input_ids, labels=input_ids).loss
    if was_training:
        model.train()
    loss_value = float(loss.detach().float().item())
    if not math.isfinite(loss_value):
        raise ValueError("Causal-LM compatibility check returned a non-finite loss")
    return {
        "status": "passed",
        "input_token_count": len(token_ids),
        "teacher_forced_loss": loss_value,
        "model_class": type(model).__name__,
    }
