import json
import math
import random
import time
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup


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
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "history": history,
        },
        checkpoint_dir / "training_state.pt",
    )


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
    scaler = GradScaler("cuda", enabled=use_amp)
    micro_batch_size = getattr(dataloader, "batch_size", None)
    effective_batch_size = (
        micro_batch_size * grad_accum_steps
        if micro_batch_size is not None
        else "unknown"
    )

    print("--- Training Execution Plan ---")
    print(f"  Device:               {device}")
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
                dtype=torch.float16,
            ):
                outputs = model(input_ids=input_ids, labels=labels)
                raw_loss = outputs.loss
                backward_loss = raw_loss / current_group_size

            scaler.scale(backward_loss).backward()

            # GPT-2 predicts labels from position 1 onward. Weight the loss by
            # the actual number of predicted tokens, excluding ignored labels.
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

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )

            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # GradScaler lowers its scale when an optimizer step is skipped due
            # to non-finite gradients. Do not advance the LR schedule then.
            optimizer_step_skipped = (
                use_amp and scaler.get_scale() < previous_scale
            )
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
        model.config.use_cache = original_use_cache

    final_dir = save_dir / "final_model"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    history_path, curve_path = _save_training_history(history, save_dir)

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
    }


def run_cpt_from_cli(args):
    """Connect the existing model, data-loader, and CPT functions."""
    from gpt2_model import load_gpt2_model
    from load_tensors import load_tokens_from_bin

    bin_file = args.bin_file.expanduser().resolve()
    save_dir = args.save_dir.expanduser().resolve()

    if not bin_file.is_file():
        raise ValueError(f"Packed binary token file does not exist: {bin_file}")

    model_dict = load_gpt2_model(model_name=args.model_name)
    dataloader = load_tokens_from_bin(
        filename=str(bin_file),
        context_length=args.context_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
