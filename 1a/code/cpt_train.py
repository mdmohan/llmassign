import json
import math
import platform
import random
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
import transformers
from transformers import get_cosine_schedule_with_warmup

from cache import CACHE_DIR


TRAINING_RUN_FILENAME = "training_run.json"


def _json_compatible(value):
    """Convert argparse and PyTorch values into JSON-safe representations."""
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_training_run(run_config, save_dir):
    """Write the reproducibility record at the root of the training run."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / TRAINING_RUN_FILENAME
    output_path.write_text(
        json.dumps(_json_compatible(run_config), indent=2),
        encoding="utf-8",
    )
    return output_path


def _create_cli_run_config(
    args,
    model_family,
    bin_file,
    save_dir,
    binary_dtype="uint16",
):
    """Capture every parsed CLI option and the reproducible invocation."""
    command_argv = [sys.executable, *sys.argv]
    dataset_metrics = getattr(args, "dataset_metrics", None)
    dataset_metrics = (
        dataset_metrics.expanduser().resolve()
        if dataset_metrics is not None
        else None
    )

    cuda_device_name = None
    if torch.cuda.is_available():
        cuda_device_name = torch.cuda.get_device_name(torch.cuda.current_device())

    numpy_dtype = np.dtype(binary_dtype)
    return {
        "schema_version": 1,
        "status": "initializing",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_family": model_family,
        "command": {
            # Python cannot recover the shell's aliases or original spacing.
            # argv is exact and this command is its safely quoted equivalent.
            "argv": command_argv,
            "reconstructed_full_command": shlex.join(command_argv),
            "working_directory": str(Path.cwd()),
        },
        "cli_options": vars(args).copy(),
        "resolved_paths": {
            "bin_file": str(bin_file),
            "dataset_metrics": (
                str(dataset_metrics) if dataset_metrics is not None else None
            ),
            "save_dir": str(save_dir),
            "cache_dir": str(CACHE_DIR),
        },
        "dataset": {
            "bin_file": str(bin_file),
            "file_size_bytes": bin_file.stat().st_size,
            "binary_dtype": numpy_dtype.name,
            "stored_token_count": bin_file.stat().st_size // numpy_dtype.itemsize,
        },
        "environment": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "pytorch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime_version": torch.version.cuda,
            "cudnn_version": (
                torch.backends.cudnn.version()
                if torch.cuda.is_available()
                else None
            ),
            "cuda_device_count": (
                torch.cuda.device_count() if torch.cuda.is_available() else 0
            ),
            "cuda_device_name": cuda_device_name,
        },
    }


def _record_training_plan(
    run_config,
    save_dir,
    model,
    tokenizer,
    device,
    dataloader,
    *,
    epochs,
    grad_accum_steps,
    learning_rate,
    weight_decay,
    warmup_ratio,
    warmup_steps,
    max_grad_norm,
    log_every_steps,
    save_every_steps,
    seed,
    batches_per_epoch,
    updates_per_epoch,
    total_steps,
    precision,
    gradient_scaling,
    gradient_checkpointing,
):
    """Add loaded-model facts and calculated training values to the record."""
    if run_config is None:
        run_config = {
            "schema_version": 1,
            "status": "initializing",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": None,
            "cli_options": None,
            "model_family": getattr(model.config, "model_type", None),
        }

    micro_batch_size = getattr(dataloader, "batch_size", None)
    effective_batch_size = (
        micro_batch_size * grad_accum_steps
        if micro_batch_size is not None
        else None
    )
    dataset_samples = len(getattr(dataloader, "dataset", []))

    run_config["status"] = "training"
    run_config["model"] = {
        "requested_name": (
            run_config.get("cli_options", {}).get("model_name")
            if isinstance(run_config.get("cli_options"), dict)
            else None
        ),
        "loaded_name_or_path": getattr(model.config, "_name_or_path", None),
        "model_class": type(model).__name__,
        "model_type": getattr(model.config, "model_type", None),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        "vocabulary_size": getattr(model.config, "vocab_size", None),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "parameter_dtype": str(next(model.parameters()).dtype),
        "device": str(device),
    }
    run_config.setdefault("dataset", {}).update(
        {
            "packed_sequence_count": dataset_samples,
            "context_length": getattr(
                getattr(dataloader, "dataset", None),
                "seq_len",
                None,
            ),
            "batches_per_epoch": batches_per_epoch,
            "dataloader_workers": getattr(dataloader, "num_workers", None),
            "shuffle": True,
            "drop_last": getattr(dataloader, "drop_last", None),
        }
    )
    run_config["hyperparameters"] = {
        "epochs": epochs,
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "effective_batch_size": effective_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "adam_betas": [0.9, 0.95],
        "warmup_ratio": warmup_ratio,
        "max_grad_norm": max_grad_norm,
        "seed": seed,
        "precision": str(precision),
        "gradient_scaling": gradient_scaling,
        "gradient_checkpointing": gradient_checkpointing,
        "optimizer": "AdamW",
        "learning_rate_scheduler": "cosine_with_warmup",
        "log_every_steps": log_every_steps,
        "save_every_steps": save_every_steps,
    }
    run_config["derived_training_plan"] = {
        "batches_per_epoch": batches_per_epoch,
        "optimizer_updates_per_epoch": updates_per_epoch,
        "total_optimizer_steps": total_steps,
        "warmup_steps": warmup_steps,
    }
    output_path = _write_training_run(run_config, save_dir)
    print(f"Training configuration saved to: {output_path}")
    return run_config, output_path


def _perplexity(loss):
    """Convert mean cross-entropy loss to perplexity without overflowing."""
    return math.exp(loss) if loss < 20 else float("inf")


def _save_checkpoint(
    model,
    tokenizer,
    optimizer,
    scheduler,
    scaler,
    checkpoint_dir,
    epoch,
    global_step,
    history,
):
    """Save model files plus the state required to resume CPT."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
    }
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    torch.save(state, checkpoint_dir / "training_state.pt")


def _save_training_history(history, save_dir):
    """Save machine-readable loss history and a loss-curve image."""
    save_dir = Path(save_dir)
    history_path = save_dir / "training_history.json"
    history_path.write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is unavailable; training history was saved as JSON, "
            "but the loss-curve image was not created."
        )
        return history_path, None

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(
        [record["step"] for record in history],
        [record["loss"] for record in history],
        linewidth=1.5,
    )
    axis.set_title("Continued Pre-Training Loss")
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Token-weighted cross-entropy loss")
    axis.grid(alpha=0.3)
    figure.tight_layout()

    curve_path = save_dir / "training_loss_curve.png"
    figure.savefig(curve_path, dpi=150)
    plt.close(figure)
    return history_path, curve_path


def cpt_train(
    model,
    tokenizer,
    device,
    dataloader,
    epoch=3,
    save_dir="./cpt_checkpoints",
    grad_accum_steps=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.05,
    max_grad_norm=1.0,
    log_every_steps=10,
    save_every_steps=500,
    seed=42,
    run_config=None,
    model_dtype=None,
):
    """Run continued pre-training and save checkpoints and loss history."""
    device = torch.device(device)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if len(dataloader) == 0:
        raise ValueError("The training dataloader contains no batches")
    if epoch < 1:
        raise ValueError("epoch must be at least 1")
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be at least 1")
    if log_every_steps < 1:
        raise ValueError("log_every_steps must be at least 1")
    if save_every_steps < 1:
        raise ValueError("save_every_steps must be at least 1")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model.to(device)
    original_use_cache = getattr(model.config, "use_cache", None)
    if original_use_cache is not None:
        model.config.use_cache = False

    # Exclude biases and LayerNorm parameters from weight decay.
    decay_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.ndim >= 2
    ]
    no_decay_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.ndim < 2
    ]
    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
    )

    batches_per_epoch = len(dataloader)
    updates_per_epoch = math.ceil(batches_per_epoch / grad_accum_steps)
    total_steps = updates_per_epoch * epoch
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    use_amp = device.type == "cuda"
    parameter_dtype = model_dtype or next(model.parameters()).dtype
    if use_amp and parameter_dtype == torch.bfloat16:
        amp_dtype = torch.bfloat16
    elif use_amp:
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = GradScaler("cuda", enabled=True) if use_scaler else None
    micro_batch_size = getattr(dataloader, "batch_size", None)
    effective_batch_size = (
        micro_batch_size * grad_accum_steps
        if micro_batch_size is not None
        else "unknown"
    )

    run_config, run_config_path = _record_training_plan(
        run_config,
        save_dir,
        model,
        tokenizer,
        device,
        dataloader,
        epochs=epoch,
        grad_accum_steps=grad_accum_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps,
        max_grad_norm=max_grad_norm,
        log_every_steps=log_every_steps,
        save_every_steps=save_every_steps,
        seed=seed,
        batches_per_epoch=batches_per_epoch,
        updates_per_epoch=updates_per_epoch,
        total_steps=total_steps,
        precision=amp_dtype,
        gradient_scaling=use_scaler,
        gradient_checkpointing=getattr(model, "is_gradient_checkpointing", False),
    )

    print("--- Training Execution Plan ---")
    print(f"  Device:               {device}")
    printf(f" Epoch:                {epoch}")
    print(f"  Training dtype:       {amp_dtype}")
    print(f"  Micro Batch Size:     {micro_batch_size}")
    print(f"  Gradient Accumulation:{grad_accum_steps}")
    print(f"  Effective Batch Size: {effective_batch_size}")
    print(f"  Total Batches/Epoch:  {batches_per_epoch}")
    print(f"  Updates/Epoch:        {updates_per_epoch}")
    print(f"  Total Update Steps:   {total_steps}")
    print(f"  Warmup Steps:         {warmup_steps}")
    print(f"  Learning Rate:        {learning_rate:.2e}")
    print("-------------------------------\n")

    global_step = 0
    history = []
    start_time = time.time()
    logging_nll = 0.0
    logging_tokens = 0

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch_index in range(epoch):
        print(f"======== Starting Epoch {epoch_index + 1}/{epoch} ========")
        epoch_nll = 0.0
        epoch_tokens = 0
        group_nll = 0.0
        group_tokens = 0

        for batch_index, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # The final accumulation group can contain fewer microbatches. Use
            # its actual size so its gradients are not incorrectly downscaled.
            group_start = (batch_index // grad_accum_steps) * grad_accum_steps
            current_group_size = min(
                grad_accum_steps,
                batches_per_epoch - group_start,
            )

            with autocast(
                device_type=device.type,
                enabled=use_amp,
                dtype=amp_dtype,
            ):
                outputs = model(input_ids=input_ids, labels=labels)
                raw_loss = outputs.loss
                backward_loss = raw_loss / current_group_size

            if scaler is None:
                backward_loss.backward()
            else:
                scaler.scale(backward_loss).backward()

            # Decoder-only causal LMs predict labels from position 1 onward.
            # Weight loss by actual predicted tokens, excluding ignored labels.
            predicted_token_count = int(
                (labels[..., 1:] != -100).sum().item()
            )
            batch_nll = raw_loss.detach().float().item() * predicted_token_count
            group_nll += batch_nll
            group_tokens += predicted_token_count
            epoch_nll += batch_nll
            epoch_tokens += predicted_token_count

            should_update = (
                (batch_index + 1) % grad_accum_steps == 0
                or (batch_index + 1) == batches_per_epoch
            )
            if not should_update:
                continue

            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )

            if scaler is None:
                optimizer.step()
                optimizer_step_skipped = False
            else:
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_step_skipped = scaler.get_scale() < previous_scale
            optimizer.zero_grad(set_to_none=True)

            # GradScaler lowers its scale when an optimizer step is skipped due
            # to non-finite gradients. Do not advance the LR schedule then.
            if optimizer_step_skipped:
                print("Skipped optimizer update because of non-finite gradients")
                group_nll = 0.0
                group_tokens = 0
                continue

            scheduler.step()
            global_step += 1

            update_loss = group_nll / group_tokens
            update_perplexity = _perplexity(update_loss)
            current_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time
            history.append(
                {
                    "step": global_step,
                    "epoch": epoch_index + 1,
                    "loss": update_loss,
                    "perplexity": update_perplexity,
                    "learning_rate": current_lr,
                    "elapsed_seconds": elapsed,
                }
            )

            logging_nll += group_nll
            logging_tokens += group_tokens
            group_nll = 0.0
            group_tokens = 0

            if global_step % log_every_steps == 0:
                average_loss = logging_nll / logging_tokens
                print(
                    f"Step {global_step:5d}/{total_steps} | "
                    f"Loss: {average_loss:.4f} | "
                    f"PPL: {_perplexity(average_loss):.2f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Elapsed: {elapsed:.1f}s"
                )
                logging_nll = 0.0
                logging_tokens = 0

            if global_step % save_every_steps == 0:
                checkpoint_dir = save_dir / f"checkpoint-{global_step}"
                print(f"\nSaving checkpoint to {checkpoint_dir}...")
                _save_checkpoint(
                    model,
                    tokenizer,
                    optimizer,
                    scheduler,
                    scaler,
                    checkpoint_dir,
                    epoch_index + 1,
                    global_step,
                    history,
                )

        epoch_loss = epoch_nll / epoch_tokens
        print(
            f"Epoch {epoch_index + 1} complete | "
            f"Loss: {epoch_loss:.4f} | "
            f"PPL: {_perplexity(epoch_loss):.2f}"
        )

        # Maintain one rolling resumable checkpoint without accumulating a
        # separate optimizer-state file for every epoch.
        _save_checkpoint(
            model,
            tokenizer,
            optimizer,
            scheduler,
            scaler,
            save_dir / "last_checkpoint",
            epoch_index + 1,
            global_step,
            history,
        )

    if logging_tokens:
        average_loss = logging_nll / logging_tokens
        print(
            f"Final logging interval | Loss: {average_loss:.4f} | "
            f"PPL: {_perplexity(average_loss):.2f}"
        )

    if original_use_cache is not None:
        model.config.use_cache = True

    final_dir = save_dir / "final_model"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    history_path, curve_path = _save_training_history(history, save_dir)

    run_config["status"] = "completed"
    run_config["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    run_config["results"] = {
        "completed_optimizer_steps": global_step,
        "elapsed_seconds": time.time() - start_time,
        "final_model_dir": str(final_dir),
        "last_checkpoint_dir": str(save_dir / "last_checkpoint"),
        "training_history_file": str(history_path),
        "loss_curve_file": str(curve_path) if curve_path else None,
        "final_training_loss": history[-1]["loss"] if history else None,
        "final_training_perplexity": (
            history[-1]["perplexity"] if history else None
        ),
    }
    _write_training_run(run_config, save_dir)

    print(f"\nTraining complete. Final model saved to: {final_dir}")
    print(f"Training history saved to: {history_path}")
    if curve_path is not None:
        print(f"Loss curve saved to: {curve_path}")

    return {
        "history": history,
        "final_model_dir": str(final_dir),
        "last_checkpoint_dir": str(save_dir / "last_checkpoint"),
        "history_file": str(history_path),
        "loss_curve_file": str(curve_path) if curve_path else None,
        "training_run_file": str(run_config_path),
    }


def _load_dataset_metrics(bin_file: Path, supplied_path):
    """Load the metrics paired with a packed token file when available."""
    if supplied_path is not None:
        metrics_path = supplied_path.expanduser().resolve()
    elif bin_file.stem in {"token_train", "token_test"}:
        metrics_path = bin_file.parent / f"dataset_metrics_{bin_file.stem[6:]}.json"
    else:
        metrics_path = bin_file.parent / "dataset_metrics.json"

    if supplied_path is not None and not metrics_path.is_file():
        raise ValueError(f"Dataset metrics file does not exist: {metrics_path}")
    if not metrics_path.is_file():
        return None, metrics_path
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8")), metrics_path
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid dataset metrics JSON: {metrics_path}") from exc


def _validate_training_dataset(
    bin_file,
    context_length,
    model,
    tokenizer,
    metrics,
):
    """Verify binary layout, context length, and tokenizer/model compatibility."""
    from causal_lm import model_context_limit
    from load_tensors import normalize_binary_dtype

    binary_dtype = normalize_binary_dtype(
        metrics.get("binary_dtype") if metrics is not None else "uint16"
    )
    file_size = bin_file.stat().st_size
    if file_size % binary_dtype.itemsize:
        raise ValueError(
            f"Binary token file has an invalid {binary_dtype.name} byte length: "
            f"{bin_file}"
        )
    token_count = file_size // binary_dtype.itemsize
    if token_count < context_length or token_count % context_length:
        raise ValueError(
            f"Binary file contains {token_count:,} tokens and cannot be read as "
            f"complete {context_length:,}-token sequences"
        )

    limit = model_context_limit(model.config, tokenizer)
    if limit is not None and context_length > limit:
        raise ValueError(
            f"Training context length {context_length:,} exceeds the model "
            f"limit of {limit:,}"
        )

    token_data = np.memmap(bin_file, dtype=binary_dtype, mode="r")
    maximum_token_id = int(token_data.max())
    if maximum_token_id >= int(model.config.vocab_size):
        raise ValueError(
            f"Dataset token ID {maximum_token_id:,} exceeds the loaded model's "
            f"maximum valid ID {int(model.config.vocab_size) - 1:,}"
        )

    if metrics is not None:
        metrics_context = metrics.get("packing_context_length")
        if metrics_context is not None and int(metrics_context) != context_length:
            raise ValueError(
                "Training context length does not match dataset metrics: "
                f"{context_length} != {metrics_context}"
            )
        metrics_vocab = metrics.get("tokenizer_vocabulary_size")
        if metrics_vocab is not None and int(metrics_vocab) != len(tokenizer):
            raise ValueError(
                "Dataset tokenizer vocabulary does not match the loaded tokenizer: "
                f"{metrics_vocab} != {len(tokenizer)}"
            )
        metrics_fingerprint = metrics.get("tokenizer_vocabulary_sha256")
        if metrics_fingerprint is not None:
            from causal_lm import tokenizer_vocabulary_sha256

            loaded_fingerprint = tokenizer_vocabulary_sha256(tokenizer)
            if metrics_fingerprint != loaded_fingerprint:
                raise ValueError(
                    "Dataset token-to-ID mapping does not match the loaded tokenizer"
                )
    return binary_dtype, token_count, maximum_token_id


def run_cpt_from_cli(args):
    """Load and train any standard Hugging Face decoder-only causal LM."""
    from causal_lm import load_causal_lm, validate_causal_lm_forward
    from load_tensors import load_tokens_from_bin, normalize_binary_dtype

    bin_file = args.bin_file.expanduser().resolve()
    save_dir = args.save_dir.expanduser().resolve()

    if not bin_file.is_file():
        raise ValueError(f"Packed binary token file does not exist: {bin_file}")

    metrics, metrics_path = _load_dataset_metrics(
        bin_file,
        getattr(args, "dataset_metrics", None),
    )
    if metrics is None:
        print(
            f"Warning: dataset metrics not found at {metrics_path}; assuming "
            "the legacy uint16 binary format."
        )
    binary_dtype = normalize_binary_dtype(
        metrics.get("binary_dtype") if metrics is not None else "uint16"
    )

    model_dict = load_causal_lm(
        model_name=args.model_name,
        for_training=True,
    )
    _, stored_token_count, maximum_token_id = _validate_training_dataset(
        bin_file,
        args.context_length,
        model_dict["model"],
        model_dict["tokenizer"],
        metrics,
    )
    compatibility_check = validate_causal_lm_forward(
        model_dict["model"],
        model_dict["tokenizer"],
        model_dict["device"],
    )

    model_family = getattr(model_dict["model"].config, "model_type", "causal_lm")
    run_config = _create_cli_run_config(
        args,
        model_family=model_family,
        bin_file=bin_file,
        save_dir=save_dir,
        binary_dtype=binary_dtype,
    )
    run_config["dataset"].update(
        {
            "metrics_file": str(metrics_path) if metrics is not None else None,
            "stored_token_count": stored_token_count,
            "maximum_token_id": maximum_token_id,
        }
    )
    run_config["resolved_paths"]["dataset_metrics"] = (
        str(metrics_path) if metrics is not None else None
    )
    run_config["compatibility_check"] = compatibility_check
    _write_training_run(run_config, save_dir)
    dataloader = load_tokens_from_bin(
        filename=str(bin_file),
        context_length=args.context_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        binary_dtype=binary_dtype,
    )

    return cpt_train(
        model=model_dict["model"],
        tokenizer=model_dict["tokenizer"],
        device=model_dict["device"],
        dataloader=dataloader,
        epoch=args.epochs,
        save_dir=save_dir,
        grad_accum_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        log_every_steps=args.log_every_steps,
        save_every_steps=args.save_every_steps,
        seed=args.seed,
        run_config=run_config,
        model_dtype=model_dict["dtype"],
    )


def main() -> int:
    from cli_parsers import build_cpt_train_parser

    parser = build_cpt_train_parser()
    args = parser.parse_args()

    try:
        artifacts = run_cpt_from_cli(args)
    except KeyboardInterrupt:
        parser.exit(130, "\nCPT training interrupted by user.\n")
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")

    print("\nCPT artifacts:")
    print(f"  Final model:      {artifacts['final_model_dir']}")
    print(f"  Last checkpoint:  {artifacts['last_checkpoint_dir']}")
    print(f"  Training history: {artifacts['history_file']}")
    if artifacts["loss_curve_file"]:
        print(f"  Loss curve:       {artifacts['loss_curve_file']}")
    print(f"  Training config:  {artifacts['training_run_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
