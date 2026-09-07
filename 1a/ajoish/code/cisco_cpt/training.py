"""Full-parameter continual pre-training over the packed Parquet corpus."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import platform
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from .baseline import audit_architecture, load_frozen_model
from .corpus import sha256_file, write_json
from .tokenization import load_frozen_tokenizer


class PackedParquetDataset:
    """Map-style dataset backed by fixed-size Arrow list columns."""

    columns = ("input_ids", "attention_mask", "labels")

    def __init__(self, path: Path, sequence_length: int) -> None:
        import pyarrow.parquet as pq

        if not path.is_file():
            raise ValueError(f"Packed dataset does not exist: {path}")
        self.path = path
        self.sequence_length = sequence_length
        self.table = pq.read_table(path, columns=list(self.columns), memory_map=True)
        if self.table.num_rows == 0:
            raise ValueError(f"Packed dataset contains no rows: {path}")
        for index in (0, self.table.num_rows - 1):
            for column in self.columns:
                if len(self.table[column][index].as_py()) != sequence_length:
                    raise ValueError(
                        f"{column} row {index} does not contain {sequence_length} tokens"
                    )

    def __len__(self) -> int:
        return self.table.num_rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        return {
            column: torch.tensor(self.table[column][index].as_py(), dtype=torch.long)
            for column in self.columns
        }


def collate_packed_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import torch

    return {
        column: torch.stack([row[column] for row in rows])
        for column in PackedParquetDataset.columns
    }


class LossHistoryCallback:
    """Build a Transformers callback that persists every loss logging event."""

    @staticmethod
    def create(output_root: Path) -> Any:
        from transformers import TrainerCallback

        class _Callback(TrainerCallback):
            def __init__(self) -> None:
                self.records: list[dict[str, Any]] = []

            def on_log(
                self,
                args: Any,
                state: Any,
                control: Any,
                logs: Mapping[str, Any] | None = None,
                **kwargs: Any,
            ) -> None:
                if not logs or "loss" not in logs and "eval_loss" not in logs:
                    return
                record = {
                    "step": int(state.global_step),
                    "epoch": float(state.epoch) if state.epoch is not None else None,
                    "loss": logs.get("loss"),
                    "eval_loss": logs.get("eval_loss"),
                    "learning_rate": logs.get("learning_rate"),
                    "grad_norm": logs.get("grad_norm"),
                }
                self.records.append(record)
                _write_loss_history(output_root, self.records)

        return _Callback()


def _write_loss_history(output_root: Path, records: Sequence[Mapping[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "loss_history.json", list(records))
    with (output_root / "loss_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("step", "epoch", "loss", "eval_loss", "learning_rate", "grad_norm"),
        )
        writer.writeheader()
        writer.writerows(records)


def _environment() -> dict[str, Any]:
    import torch

    package_names = ("accelerate", "pyarrow", "torch", "transformers")
    return {
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name) for name in package_names
        },
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "cuda_device_count": torch.cuda.device_count(),
    }


def _verify_input(path: Path, expected_sha256: str) -> None:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Packed input hash mismatch for {path}: expected {expected_sha256}, "
            f"found {actual_sha256}"
        )


def _training_arguments(
    output_dir: Path,
    settings: Mapping[str, Any],
    mode: str,
) -> Any:
    from transformers import TrainingArguments

    smoke = mode == "smoke"
    common = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": int(settings["micro_batch_size"]),
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": int(settings["gradient_accumulation_steps"]),
        "learning_rate": float(settings["learning_rate"]),
        "weight_decay": float(settings["weight_decay"]),
        "warmup_ratio": float(settings["warmup_ratio"]),
        "lr_scheduler_type": "linear",
        "max_grad_norm": float(settings["max_grad_norm"]),
        "optim": "adamw_torch_fused",
        "bf16": True,
        "tf32": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_strategy": "steps",
        "logging_steps": 1,
        "logging_first_step": True,
        "report_to": "none",
        "remove_unused_columns": False,
        "dataloader_num_workers": int(settings["dataloader_num_workers"]),
        "dataloader_pin_memory": True,
        "seed": int(settings["seed"]),
        "data_seed": int(settings["seed"]),
        "prediction_loss_only": True,
        "save_safetensors": True,
    }
    if smoke:
        common.update(
            {
                "max_steps": int(settings["smoke_max_steps"]),
                "eval_strategy": "no",
                "save_strategy": "no",
            }
        )
    else:
        common.update(
            {
                "num_train_epochs": float(settings["num_train_epochs"]),
                "eval_strategy": "steps",
                "eval_steps": int(settings["eval_steps"]),
                "save_strategy": "steps",
                "save_steps": int(settings["save_steps"]),
                "save_total_limit": int(settings["save_total_limit"]),
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            }
        )
    return TrainingArguments(**common)


def _analyze_plateau(
    records: Sequence[Mapping[str, Any]], settings: Mapping[str, Any]
) -> dict[str, Any]:
    losses = [
        (int(record["step"]), float(record["loss"]))
        for record in records
        if record.get("loss") is not None
    ]
    window_size = int(settings["window_steps"])
    patience = int(settings["patience_windows"])
    threshold = float(settings["min_relative_improvement"])
    windows = []
    for start in range(0, len(losses) - window_size + 1, window_size):
        block = losses[start : start + window_size]
        windows.append(
            {
                "start_step": block[0][0],
                "end_step": block[-1][0],
                "mean_loss": sum(loss for _, loss in block) / len(block),
            }
        )

    best_loss = math.inf
    stale_windows = 0
    plateau_step = None
    for window_index, window in enumerate(windows):
        current_loss = float(window["mean_loss"])
        relative_improvement = (
            (best_loss - current_loss) / best_loss if math.isfinite(best_loss) else math.inf
        )
        if relative_improvement >= threshold:
            best_loss = current_loss
            stale_windows = 0
        else:
            stale_windows += 1
            if stale_windows >= patience and plateau_step is None:
                plateau_step = windows[window_index - patience + 1]["start_step"]

    return {
        "rule": "non-overlapping mean-loss windows",
        "window_steps": window_size,
        "patience_windows": patience,
        "minimum_relative_improvement": threshold,
        "plateau_step": plateau_step,
        "windows": windows,
    }


def _plot_loss(output_root: Path, records: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    train_records = [record for record in records if record.get("loss") is not None]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(
        [record["step"] for record in train_records],
        [record["loss"] for record in train_records],
        linewidth=1.2,
    )
    axis.set_title("Cisco Continual Pre-Training Loss")
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Cross-entropy loss")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_root / "loss_curve.png", dpi=160)
    plt.close(figure)


def run_cpt(
    config: Mapping[str, Any],
    repository_root: Path,
    config_sha256: str,
    mode: str,
    resume_from_checkpoint: str | None = None,
) -> dict[str, Any]:
    import torch
    from transformers import Trainer
    from transformers.trainer_utils import get_last_checkpoint

    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full-parameter CPT")

    def resolve(value: str) -> Path:
        return (repository_root / value).resolve()

    settings = dict(config["training"])
    model_id = str(config["model_id"])
    revision = str(config["revision"])
    train_path = resolve(str(config["train_parquet_path"]))
    eval_path = resolve(str(config["eval_parquet_path"]))
    _verify_input(train_path, str(config["train_parquet_sha256"]))
    _verify_input(eval_path, str(config["eval_parquet_sha256"]))

    run_id = str(config["run_id"])
    run_root = resolve(str(config["runs_root"])) / (
        f"{run_id}-smoke" if mode == "smoke" else run_id
    )
    checkpoint_root = run_root / "checkpoints"
    run_root.mkdir(parents=True, exist_ok=True)
    environment = _environment()
    write_json(run_root / "environment.json", environment)

    smoke_report_path = resolve(str(config["runs_root"])) / f"{run_id}-smoke/smoke_report.json"
    if mode == "full":
        if not smoke_report_path.is_file():
            raise RuntimeError("The 10-step smoke gate must pass before full CPT")
        smoke_report = json.loads(smoke_report_path.read_text(encoding="utf-8"))
        if not smoke_report.get("passed") or smoke_report.get("config_sha256") != config_sha256:
            raise RuntimeError("Smoke report is not a passing result for this frozen config")

    seed = int(settings["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer, tokenizer_metadata = load_frozen_tokenizer(
        model_id, revision, resolve(str(config["cache_dir"]))
    )
    model = load_frozen_model(
        model_id, revision, resolve(str(config["cache_dir"])), "cuda"
    )
    architecture = audit_architecture(model, model_id, revision)
    if architecture["resolved_revision"] != revision:
        raise RuntimeError("Loaded model revision does not match the frozen revision")
    model.config.use_cache = False

    train_dataset = PackedParquetDataset(train_path, int(config["sequence_length"]))
    eval_dataset = (
        None
        if mode == "smoke"
        else PackedParquetDataset(eval_path, int(config["sequence_length"]))
    )
    callback = LossHistoryCallback.create(run_root)
    arguments = _training_arguments(checkpoint_root, settings, mode)
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_packed_rows,
        callbacks=[callback],
    )

    resolved_resume: str | bool | None = None
    if mode == "full" and resume_from_checkpoint:
        if resume_from_checkpoint == "auto":
            resolved_resume = get_last_checkpoint(str(checkpoint_root))
        else:
            resolved_resume = str(resolve(resume_from_checkpoint))

    manifest = {
        "status": "running",
        "mode": mode,
        "run_id": run_id,
        "config_sha256": config_sha256,
        "model_id": model_id,
        "revision": revision,
        "architecture": architecture,
        "tokenizer": tokenizer_metadata,
        "train_parquet_sha256": sha256_file(train_path),
        "eval_parquet_sha256": sha256_file(eval_path),
        "train_sequence_count": len(train_dataset),
        "eval_sequence_count": len(eval_dataset) if eval_dataset is not None else 0,
        "training": settings,
        "environment": environment,
        "resume_from_checkpoint": resolved_resume,
    }
    write_json(run_root / "run_manifest.json", manifest)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train(resume_from_checkpoint=resolved_resume)
    peak_allocated_gib = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved_gib = torch.cuda.max_memory_reserved() / 1024**3
    losses = [
        float(record["loss"])
        for record in callback.records
        if record.get("loss") is not None
    ]
    if not losses or not all(math.isfinite(loss) for loss in losses):
        raise RuntimeError("Training did not produce finite logged losses")

    trainer.state.save_to_json(str(run_root / "trainer_state.json"))
    trainer.save_metrics("train", train_result.metrics)
    _write_loss_history(run_root, callback.records)

    if mode == "smoke":
        gate = dict(config["smoke_gate"])
        report = {
            "passed": (
                float(gate["minimum_starting_loss"])
                <= losses[0]
                <= float(gate["maximum_starting_loss"])
                and peak_allocated_gib <= float(gate["maximum_peak_allocated_gib"])
            ),
            "config_sha256": config_sha256,
            "optimizer_steps": int(trainer.state.global_step),
            "first_loss": losses[0],
            "final_loss": losses[-1],
            "minimum_loss": min(losses),
            "peak_allocated_gib": peak_allocated_gib,
            "peak_reserved_gib": peak_reserved_gib,
            "limits": gate,
        }
        write_json(run_root / "smoke_report.json", report)
        if not report["passed"]:
            raise RuntimeError(f"Smoke gate failed: {report}")
        manifest.update({"status": "passed", "result": report})
        write_json(run_root / "run_manifest.json", manifest)
        return report

    final_model_root = resolve(str(config["models_root"])) / run_id / "final"
    final_model_root.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_model_root))
    tokenizer.save_pretrained(final_model_root)
    plateau = _analyze_plateau(callback.records, config["plateau"])
    write_json(run_root / "plateau_analysis.json", plateau)
    _plot_loss(run_root, callback.records)
    result = {
        "optimizer_steps": int(trainer.state.global_step),
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "peak_allocated_gib": peak_allocated_gib,
        "peak_reserved_gib": peak_reserved_gib,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "final_model_dir": str(final_model_root.relative_to(repository_root)),
        "plateau_step": plateau["plateau_step"],
    }
    manifest.update({"status": "completed", "result": result})
    write_json(run_root / "run_manifest.json", manifest)
    return result