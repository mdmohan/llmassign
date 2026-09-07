"""SmolLM2-specific model loading and architecture inspection."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cache import CACHE_DIR


SMOLLM2_MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
_SMOLLM2_ALIASES = {
    "smollm2-360m",
    "huggingfacetb/smollm2-360m",
}


def is_smollm2_model(model_name: str | None) -> bool:
    """Return whether a CLI model name selects the supported SmolLM2 model."""
    if not model_name:
        return False
    return model_name.strip().casefold() in _SMOLLM2_ALIASES


def canonical_smollm2_model_id(model_name: str | None) -> str:
    """Resolve the short SmolLM2 alias to its Hugging Face model ID."""
    if model_name is None or is_smollm2_model(model_name):
        return SMOLLM2_MODEL_ID
    raise ValueError(
        "Unsupported SmolLM2 model name. Use 'smollm2-360m' or "
        f"'{SMOLLM2_MODEL_ID}'."
    )


def _resolve_checkpoint_folder(model_folder) -> Path:
    # Reuse the existing folder-resolution behavior without changing the GPT
    # local-model loader itself.
    from load_local_model import resolve_model_folder

    return resolve_model_folder(model_folder)


def is_smollm2_checkpoint(model_folder) -> bool:
    """Detect a Llama-family checkpoint from its local config file."""
    try:
        checkpoint_dir = _resolve_checkpoint_folder(model_folder)
        config = json.loads(
            (checkpoint_dir / "config.json").read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return False
    return config.get("model_type") == "llama"


def preferred_smollm2_dtype(device) -> torch.dtype:
    """Choose BF16 when supported, FP16 on older CUDA GPUs, else FP32."""
    device = torch.device(device)
    if device.type != "cuda":
        return torch.float32

    supports_bf16 = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(supports_bf16) and supports_bf16():
        return torch.bfloat16
    return torch.float16


def smollm2_context_length(model, tokenizer=None) -> int:
    """Return the configured maximum sequence length."""
    context_length = getattr(model.config, "max_position_embeddings", None)
    if context_length is None and tokenizer is not None:
        context_length = tokenizer.model_max_length
    if context_length is None:
        raise ValueError("Unable to determine the SmolLM2 context length")
    return int(context_length)


def load_smollm2_model(
    model_name: str = SMOLLM2_MODEL_ID,
    model_folder=None,
    device=None,
    for_training: bool = False,
):
    """Load the base SmolLM2 model or a locally saved CPT checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if model_folder is None:
        source = canonical_smollm2_model_id(model_name)
        local_files_only = False
    else:
        source = _resolve_checkpoint_folder(model_folder)
        local_files_only = True

    dtype = preferred_smollm2_dtype(device)
    print(f"Using device: {device}")
    print(f"Loading SmolLM2 from: {source}")

    tokenizer = AutoTokenizer.from_pretrained(
        source,
        cache_dir=CACHE_DIR,
        local_files_only=local_files_only,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        source,
        cache_dir=CACHE_DIR,
        local_files_only=local_files_only,
        torch_dtype=dtype,
    )
    model.to(device)

    if for_training:
        model.config.use_cache = False
        enable_checkpointing = getattr(
            model,
            "gradient_checkpointing_enable",
            None,
        )
        if callable(enable_checkpointing):
            enable_checkpointing()
        model.train()
    else:
        model.config.use_cache = True
        model.eval()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Successfully loaded {source} with "
        f"{parameter_count:,} parameters using {dtype}."
    )

    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "dtype": dtype,
        "model_source": str(source),
    }


def smollm2_model_details(model, tokenizer, device) -> dict:
    """Return SmolLM2 architecture and tokenizer provenance."""
    config = model.config
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    output_head = model.get_output_embeddings()
    lm_head_output_dimension = getattr(output_head, "out_features", None)
    if lm_head_output_dimension is None and hasattr(output_head, "weight"):
        lm_head_output_dimension = int(output_head.weight.shape[0])

    hidden_size = int(config.hidden_size)
    attention_heads = int(config.num_attention_heads)
    vocabulary_size = int(config.vocab_size)

    return {
        "name_or_path": config._name_or_path,
        "model_class": type(model).__name__,
        "model_type": config.model_type,
        "architectures": getattr(config, "architectures", None),
        "transformer_layers": int(config.num_hidden_layers),
        "attention_heads": attention_heads,
        "key_value_heads": getattr(config, "num_key_value_heads", None),
        "embedding_dimension": hidden_size,
        "head_dimension": hidden_size // attention_heads,
        "vocabulary_size": vocabulary_size,
        "maximum_context_length": smollm2_context_length(model, tokenizer),
        "lm_head_output_dimension": lm_head_output_dimension,
        "lm_head_matches_vocabulary": (
            lm_head_output_dimension == vocabulary_size
        ),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "parameter_dtype": str(next(model.parameters()).dtype),
        "device": str(device),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_name_or_path": tokenizer.name_or_path,
        "tokenizer_vocabulary_size": tokenizer.vocab_size,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }


def check_smollm2_model(model, tokenizer, device, dataloader=None) -> dict:
    """Print the architecture audit and optionally calculate initial loss."""
    details = smollm2_model_details(model, tokenizer, device)
    print("\n" + "=" * 55)
    print("             SMOLLM2 MODEL INFORMATION")
    print("=" * 55)
    for key, value in details.items():
        print(f"{key.replace('_', ' ').title():28}: {value}")
    print("=" * 55)

    if dataloader is not None:
        batch = next(iter(dataloader))
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with torch.no_grad():
            loss = model(input_ids=input_ids, labels=labels).loss
        print(f"Initial loss: {loss.item():.4f}\n")

    return details
