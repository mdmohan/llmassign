"""SmolLM2-specific continued pre-training support."""

from __future__ import annotations

import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from cpt_train import (
    _perplexity,
    _record_training_plan,
    _save_training_history,
    _write_training_run,
)
from smollm2_model import (
    canonical_smollm2_model_id,
    load_smollm2_model,
    smollm2_context_length,
)


def _load_dataset_metrics(bin_file: Path, supplied_path) -> tuple[dict | None, Path]:
    metrics_path = (
        supplied_path.expanduser().resolve()
        if supplied_path is not None
        else bin_file.parent / "dataset_metrics.json"
    )
    if not metrics_path.is_file():
        return None, metrics_path
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8")), metrics_path
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid dataset metrics JSON: {metrics_path}") from exc


def _validate_dataset(
    bin_file: Path,
    context_length: int,
    model,
    tokenizer,
    model_name: str,
    supplied_metrics_path=None,
) -> None:
    if not bin_file.is_file():
        raise ValueError(f"Packed binary token file does not exist: {bin_file}")
    if context_length < 2:
        raise ValueError("context_length must be at least 2 for causal training")
    if bin_file.stat().st_size % np.dtype(np.uint16).itemsize:
        raise ValueError(f"Binary token file has an invalid byte length: {bin_file}")

    token_count = bin_file.stat().st_size // np.dtype(np.uint16).itemsize
    if token_count < context_length:
        raise ValueError(
            f"Binary file has {token_count:,} tokens, fewer than one "
            f"{context_length:,}-token sequence"
        )
    if token_count % context_length:
        raise ValueError(
            f"Binary file contains {token_count:,} tokens, which is not "
            f"divisible by context length {context_length:,}"
        )

    model_limit = smollm2_context_length(model, tokenizer)
    if context_length > model_limit:
        raise ValueError(
            f"Context length {context_length:,} exceeds the model limit "
            f"of {model_limit:,}"
        )

    token_data = np.memmap(bin_file, dtype=np.uint16, mode="r")
    maximum_token_id = int(token_data.max())
    if maximum_token_id >= int(model.config.vocab_size):
        raise ValueError(
            f"Dataset token ID {maximum_token_id:,} exceeds the SmolLM2 "
            f"vocabulary limit {int(model.config.vocab_size) - 1:,}"
        )

    metrics, metrics_path = _load_dataset_metrics(
        bin_file,
        supplied_metrics_path,
    )
    if metrics is None:
        print(
            "Warning: no dataset_metrics.json was found. Token ranges were "
            "validated, but tokenizer provenance could not be confirmed."
        )
        return

    expected_model = canonical_smollm2_model_id(model_name)
    if metrics.get("model_name") != expected_model:
        raise ValueError(
            "Packed data was not created for the selected SmolLM2 model: "
            f"metrics report {metrics.get('model_name')!r}"
        )
    if metrics.get("packing_context_length") != context_length:
        raise ValueError(
            "Training context length does not match dataset metrics: "
            f"{context_length} != {metrics.get('packing_context_length')}"
        )
    if metrics.get("tokenizer_vocabulary_size") != tokenizer.vocab_size:
        raise ValueError(
            "Dataset tokenizer vocabulary does not match the loaded tokenizer"
        )
    print(f"Validated dataset provenance from: {metrics_path}")


def _save_smollm2_checkpoint(
    model,
    tokenizer,
    optimizer,
    scheduler,
    scaler,
    checkpoint_dir,
    epoch,
    global_step,
    history,
) -> None:
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


def cpt_train_smollm2(
    model,
    tokenizer,
    device,
    dataloader,
    model_dtype,
    epoch=3,
    save_dir="./cpt_checkpoints",
    grad_accum_steps=4,
    learning_rate=5e-6,
    weight_decay=0.01,
    warmup_ratio=0.05,
    max_grad_norm=1.0,
    log_every_steps=10,
    save_every_steps=500,
    seed=42,
    run_config=None,
):
    """Train SmolLM2 with BF16 where supported and FP16 scaling otherwise."""
    device = torch.device(device)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if len(dataloader) == 0:
        raise ValueError("The training dataloader contains no batches")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model.to(device)
    model.config.use_cache = False
    optimizer = AdamW(
        [
            {
                "params": [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad and parameter.ndim >= 2
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad and parameter.ndim < 2
                ],
                "weight_decay": 0.0,
            },
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
    )

    batches_per_epoch = len(dataloader)
    updates_per_epoch = math.ceil(batches_per_epoch / grad_accum_steps)
    total_steps = updates_per_epoch * epoch
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    use_amp = device.type == "cuda"
    amp_dtype = model_dtype if use_amp else torch.float32
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

    print("--- SmolLM2 Training Execution Plan ---")
    print(f"  Device:                {device}")
    print(f"  Training dtype:        {amp_dtype}")
    print(f"  Micro Batch Size:      {micro_batch_size}")
    print(f"  Gradient Accumulation: {grad_accum_steps}")
    print(f"  Effective Batch Size:  {effective_batch_size}")
    print(f"  Total Batches/Epoch:   {batches_per_epoch}")
    print(f"  Updates/Epoch:         {updates_per_epoch}")
    print(f"  Total Update Steps:    {total_steps}")
    print(f"  Warmup Steps:          {warmup_steps}")
    print(f"  Learning Rate:         {learning_rate:.2e}")
    print("----------------------------------------\n")

    global_step = 0
    history: list[dict] = []
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

            predicted_tokens = int((labels[..., 1:] != -100).sum().item())
            batch_nll = raw_loss.detach().float().item() * predicted_tokens
            group_nll += batch_nll
            group_tokens += predicted_tokens
            epoch_nll += batch_nll
            epoch_tokens += predicted_tokens

            should_update = (
                (batch_index + 1) % grad_accum_steps == 0
                or (batch_index + 1) == batches_per_epoch
            )
            if not should_update:
                continue

            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer_step_skipped = False
            if scaler is None:
                optimizer.step()
            else:
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_step_skipped = scaler.get_scale() < previous_scale
            optimizer.zero_grad(set_to_none=True)

            if optimizer_step_skipped:
                print("Skipped optimizer update due to non-finite gradients")
                group_nll = 0.0
                group_tokens = 0
                continue

            scheduler.step()
            global_step += 1
            update_loss = group_nll / group_tokens
            current_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time
            history.append(
                {
                    "step": global_step,
                    "epoch": epoch_index + 1,
                    "loss": update_loss,
                    "perplexity": _perplexity(update_loss),
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
                    f"LR: {current_lr:.2e} | Elapsed: {elapsed:.1f}s"
                )
                logging_nll = 0.0
                logging_tokens = 0

            if global_step % save_every_steps == 0:
                checkpoint_dir = save_dir / f"checkpoint-{global_step}"
                print(f"\nSaving checkpoint to {checkpoint_dir}...")
                _save_smollm2_checkpoint(
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
            f"Epoch {epoch_index + 1} complete | Loss: {epoch_loss:.4f} | "
            f"PPL: {_perplexity(epoch_loss):.2f}"
        )
        _save_smollm2_checkpoint(
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

    return {
        "history": history,
        "final_model_dir": str(final_dir),
        "last_checkpoint_dir": str(save_dir / "last_checkpoint"),
        "history_file": str(history_path),
        "loss_curve_file": str(curve_path) if curve_path else None,
        "training_run_file": str(run_config_path),
    }


def run_smollm2_cpt_from_cli(args, run_config=None):
    """Load and train SmolLM2 using the shared CPT command arguments."""
    from load_tensors import load_tokens_from_bin

    bin_file = args.bin_file.expanduser().resolve()
    save_dir = args.save_dir.expanduser().resolve()
    if not bin_file.is_file():
        raise ValueError(f"Packed binary token file does not exist: {bin_file}")
    model_dict = load_smollm2_model(
        model_name=args.model_name,
        for_training=True,
    )
    _validate_dataset(
        bin_file=bin_file,
        context_length=args.context_length,
        model=model_dict["model"],
        tokenizer=model_dict["tokenizer"],
        model_name=args.model_name,
        supplied_metrics_path=getattr(args, "dataset_metrics", None),
    )
    dataloader = load_tokens_from_bin(
        filename=str(bin_file),
        context_length=args.context_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    return cpt_train_smollm2(
        model=model_dict["model"],
        tokenizer=model_dict["tokenizer"],
        device=model_dict["device"],
        model_dtype=model_dict["dtype"],
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
    )
