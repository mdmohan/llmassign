from pathlib import Path

import torch
from cache import CACHE_DIR
from transformers import AutoModelForCausalLM, AutoTokenizer


def _checkpoint_step(path: Path) -> int:
    """Return the numeric suffix from checkpoint-N, or -1 if unavailable."""
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def resolve_model_folder(model_folder) -> Path:
    """Resolve a model folder or a CPT output directory to loadable files."""
    model_folder = Path(model_folder).expanduser().resolve()
    if not model_folder.is_dir():
        raise FileNotFoundError(f"Model folder was not found: {model_folder}")

    if (model_folder / "config.json").is_file():
        return model_folder

    for directory_name in ("final_model", "last_checkpoint"):
        candidate = model_folder / directory_name
        if (candidate / "config.json").is_file():
            return candidate

    numbered_checkpoints = sorted(
        (
            path
            for path in model_folder.glob("checkpoint-*")
            if path.is_dir() and (path / "config.json").is_file()
        ),
        key=_checkpoint_step,
        reverse=True,
    )
    if numbered_checkpoints:
        return numbered_checkpoints[0]

    raise FileNotFoundError(
        "No loadable model was found. Supply a Hugging Face model folder or "
        "a CPT output directory containing final_model, last_checkpoint, or "
        f"checkpoint-N: {model_folder}"
    )


def load_local_model(checkpoint_dir, device=None, dtype=torch.float32):
    """
    Load the final CPT model and tokenizer for evaluation.

    Use the same dtype as the original GPT-2 baseline evaluation.
    """
    checkpoint_dir = resolve_model_folder(checkpoint_dir)

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        cache_dir=CACHE_DIR,
        torch_dtype=dtype,
        local_files_only=True,
    )

    model.to(device)
    model.eval()
    model.config.use_cache = True

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("\n" + "=" * 55)
    print("                CPT MODEL LOADED")
    print("=" * 55)
    print(f"Checkpoint:       {checkpoint_dir}")
    print(f"Device:           {device}")
    print(f"Model type:       {model.config.model_type}")
    print(f"Parameters:       {parameter_count:,}")
    from causal_lm import model_context_limit

    context_length = model_context_limit(model.config, tokenizer)
    print(f"Context length:   {context_length or 'unknown'}")
    print(f"Vocabulary size:  {model.config.vocab_size:,}")
    print("=" * 55 + "\n")

    return model, tokenizer, device
