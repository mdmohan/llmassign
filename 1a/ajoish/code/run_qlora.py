#!/usr/bin/env python3
"""Train and compare the three assignment QLoRA adapters on an A100 GPU."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import platform
import random
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


FALLBACK_CHAT_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'user' %}<|user|>
{{ message['content'] }}
<|assistant|>
{% else %}{{ message['content'] }}{{ eos_token }}{% endif %}{% endfor %}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate paths, configuration, dataset schema, and lineage without CUDA.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Replace existing adapter directories for a fresh run.",
    )
    parser.add_argument(
        "--reuse-existing-adapters",
        action="store_true",
        help="Evaluate complete saved adapters instead of training them again.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(record)
    return records


def load_manifest_splits(path: Path) -> tuple[set[str], set[str]]:
    train_ids: set[str] = set()
    eval_ids: set[str] = set()
    for record in load_jsonl(path):
        if not record.get("accepted"):
            continue
        document_id = str(record["document_id"])
        if record.get("split") == "train":
            train_ids.add(document_id)
        elif record.get("split") == "eval":
            eval_ids.add(document_id)
    if not train_ids or not eval_ids or train_ids & eval_ids:
        raise ValueError("Part A manifest must contain disjoint train and eval IDs")
    return train_ids, eval_ids


def validate_dataset(
    dataset_path: Path,
    manifest_path: Path,
    expected_sha256: str,
    minimum_pairs: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actual_sha256 = sha256_file(dataset_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Instruction dataset hash mismatch: expected {expected_sha256}, "
            f"found {actual_sha256}"
        )
    records = load_jsonl(dataset_path)
    if len(records) < minimum_pairs:
        raise ValueError(
            f"Instruction dataset has {len(records)} rows; {minimum_pairs} required"
        )

    train_ids, eval_ids = load_manifest_splits(manifest_path)
    split_counts = {"train": 0, "eval": 0}
    used_sources: set[str] = set()
    for index, record in enumerate(records):
        for field in ("instruction", "response", "source_document_id", "split"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"Invalid {field!r} in instruction row {index}")
        split = record["split"]
        if split not in split_counts:
            raise ValueError(f"Invalid split {split!r} in instruction row {index}")
        source_id = record["source_document_id"]
        if source_id not in train_ids or source_id in eval_ids:
            raise RuntimeError(
                f"Instruction row {index} leaks non-training source {source_id}"
            )
        split_counts[split] += 1
        used_sources.add(source_id)

    expected_eval = round(len(records) * 0.20)
    if split_counts != {"train": len(records) - expected_eval, "eval": expected_eval}:
        raise ValueError(f"Instruction split is not the required 80/20: {split_counts}")
    return records, {
        "dataset_sha256": actual_sha256,
        "total_pairs": len(records),
        "split_counts": split_counts,
        "source_document_count": len(used_sources),
        "part_a_eval_documents_used": len(used_sources & eval_ids),
    }


def resolve_config(config_path: Path) -> tuple[dict[str, Any], Path]:
    resolved_path = config_path.resolve()
    config = json.loads(resolved_path.read_text(encoding="utf-8"))
    repository_root = resolved_path.parents[3]
    return config, repository_root


def preflight(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    config, repository_root = resolve_config(config_path)

    def resolve(value: str) -> Path:
        return (repository_root / value).resolve()

    dataset_path = resolve(str(config["dataset_path"]))
    manifest_path = resolve(str(config["manifest_path"]))
    model_path = resolve(str(config["model_path"]))
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"CPT model directory is incomplete: {model_path}")
    model_weight_files = sorted(
        path.name
        for pattern in ("*.safetensors", "pytorch_model*.bin")
        for path in model_path.glob(pattern)
    )
    if not model_weight_files:
        raise FileNotFoundError(f"CPT model weights are missing from: {model_path}")
    dataset_records, dataset_report = validate_dataset(
        dataset_path,
        manifest_path,
        str(config["dataset_sha256"]),
        int(config["minimum_pairs"]),
    )

    expected_adapters = {
        "A": {"rank": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"]},
        "B": {"rank": 16, "alpha": 32, "target_modules": ["q_proj", "v_proj"]},
        "C": {
            "rank": 32,
            "alpha": 32,
            "target_modules": ["q_proj", "v_proj", "o_proj"],
        },
    }
    if config.get("adapters") != expected_adapters:
        raise ValueError("Adapter configurations do not match the assignment rubric")
    prompts = config.get("evaluation_prompts")
    if not isinstance(prompts, list) or len(prompts) != 3:
        raise ValueError("Exactly three common evaluation prompts are required")

    report = {
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path.resolve()),
        "repository_root": str(repository_root),
        "model_path": str(model_path),
        "model_weight_files": model_weight_files,
        "dataset": dataset_report,
        "adapter_names": sorted(expected_adapters),
        "evaluation_prompt_count": len(prompts),
    }
    return config, report, repository_root


def package_versions() -> dict[str, str]:
    names = ("accelerate", "bitsandbytes", "datasets", "peft", "torch", "transformers", "trl")
    return {name: importlib.metadata.version(name) for name in names}


def make_sft_config(settings: Mapping[str, Any], output_dir: Path) -> Any:
    from trl import SFTConfig

    values: dict[str, Any] = {
        "output_dir": str(output_dir),
        "max_steps": int(settings["max_steps"]),
        "per_device_train_batch_size": int(settings["micro_batch_size"]),
        "gradient_accumulation_steps": int(settings["gradient_accumulation_steps"]),
        "learning_rate": float(settings["learning_rate"]),
        "warmup_ratio": float(settings["warmup_ratio"]),
        "lr_scheduler_type": "linear",
        "weight_decay": float(settings["weight_decay"]),
        "max_grad_norm": float(settings["max_grad_norm"]),
        "logging_steps": int(settings["logging_steps"]),
        "save_strategy": "no",
        "bf16": True,
        "tf32": True,
        "report_to": "none",
        "seed": int(settings["seed"]),
        "data_seed": int(settings["seed"]),
        "optim": "paged_adamw_8bit",
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "dataset_text_field": "text",
        "dataset_num_proc": 1,
        "dataloader_num_workers": 0,
        "packing": False,
        "max_seq_length": int(settings["sequence_length"]),
    }
    parameters = set(inspect.signature(SFTConfig.__init__).parameters)
    if "max_seq_length" not in parameters and "max_length" in parameters:
        values["max_length"] = values.pop("max_seq_length")
    return SFTConfig(**{key: value for key, value in values.items() if key in parameters})


def trainer_tokenizer_argument(tokenizer: Any) -> dict[str, Any]:
    from trl import SFTTrainer

    parameters = set(inspect.signature(SFTTrainer.__init__).parameters)
    if "processing_class" in parameters:
        return {"processing_class": tokenizer}
    if "tokenizer" in parameters:
        return {"tokenizer": tokenizer}
    return {}


def format_record(tokenizer: Any, record: Mapping[str, Any]) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "user", "content": record["instruction"]},
            {"role": "assistant", "content": record["response"]},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )


def evaluate_loss(model: Any, tokenizer: Any, texts: Sequence[str], sequence_length: int) -> float:
    import torch

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=sequence_length,
            )
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            predicted_tokens = max(0, int(inputs["input_ids"].numel()) - 1)
            if predicted_tokens == 0:
                continue
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += float(outputs.loss) * predicted_tokens
            total_tokens += predicted_tokens
    if total_tokens == 0:
        raise RuntimeError("Instruction evaluation produced no predicted tokens")
    return total_loss / total_tokens


def generate_answer(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
) -> str:
    import torch

    rendered_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(
        rendered_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def concept_hits(answer: str, concepts: Sequence[str]) -> list[str]:
    normalized = " ".join(answer.casefold().split())
    return [concept for concept in concepts if " ".join(concept.casefold().split()) in normalized]


def release_model(model: Any) -> None:
    import torch

    del model
    gc.collect()
    torch.cuda.empty_cache()


def write_comparison_markdown(path: Path, results: Mapping[str, Any]) -> None:
    lines = [
        "# QLoRA Adapter Comparison",
        "",
        "| Model | Eval loss | Perplexity | Concept hits | Peak VRAM GiB |",
        "|---|---:|---:|---:|---:|",
    ]
    baseline = results["baseline"]
    lines.append(
        f"| CPT base | {baseline['eval_loss']:.4f} | {baseline['perplexity']:.4f} | "
        f"{baseline['concept_hits']} | {baseline['peak_vram_gib']:.2f} |"
    )
    for name, result in results["adapters"].items():
        lines.append(
            f"| Adapter {name} | {result['eval_loss']:.4f} | {result['perplexity']:.4f} | "
            f"{result['concept_hits']} | {result['peak_vram_gib']:.2f} |"
        )
    lines.extend(["", "## Three-Prompt Outputs", ""])
    for prompt_index, prompt_record in enumerate(results["evaluation_prompts"], start=1):
        lines.extend([f"### Prompt {prompt_index}", "", prompt_record["prompt"], ""])
        for model_name, answer in prompt_record["answers"].items():
            lines.extend([f"**{model_name}:** {answer}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_training(
    config: Mapping[str, Any],
    preflight_report: Mapping[str, Any],
    repository_root: Path,
    overwrite: bool,
    reuse_existing_adapters: bool,
) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for QLoRA training")
    gpu_name = torch.cuda.get_device_name(0)
    required_gpu = str(config["required_gpu_name"])
    if required_gpu.casefold() not in gpu_name.casefold():
        raise RuntimeError(f"Expected an {required_gpu} GPU, found {gpu_name}")

    def resolve(value: str) -> Path:
        return (repository_root / value).resolve()

    model_path = resolve(str(config["model_path"]))
    dataset_path = resolve(str(config["dataset_path"]))
    output_root = resolve(str(config["output_root"]))
    adapters_root = resolve(str(config["adapters_root"]))
    output_root.mkdir(parents=True, exist_ok=True)
    adapters_root.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(dataset_path)
    train_records = [record for record in records if record["split"] == "train"]
    eval_records = [record for record in records if record["split"] == "eval"]
    settings = config["training"]
    seed = int(settings["seed"])
    random.seed(seed)
    set_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if tokenizer.chat_template is None:
        tokenizer.chat_template = FALLBACK_CHAT_TEMPLATE

    train_texts = [format_record(tokenizer, record) for record in train_records]
    eval_texts = [format_record(tokenizer, record) for record in eval_records]
    train_dataset = Dataset.from_dict({"text": train_texts})
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    def load_model() -> Any:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            quantization_config=quantization,
            torch_dtype=torch.bfloat16,
            device_map={"": torch.cuda.current_device()},
        )
        model.config.use_cache = False
        return model

    evaluation_prompts = list(config["evaluation_prompts"])

    def evaluate_model(model: Any) -> dict[str, Any]:
        loss = evaluate_loss(
            model,
            tokenizer,
            eval_texts,
            int(settings["sequence_length"]),
        )
        answers: dict[str, str] = {}
        hits = 0
        for prompt_record in evaluation_prompts:
            prompt_id = str(prompt_record["prompt_id"])
            answer = generate_answer(
                model,
                tokenizer,
                str(prompt_record["prompt"]),
                int(config["generation_max_new_tokens"]),
            )
            answers[prompt_id] = answer
            hits += len(concept_hits(answer, list(prompt_record["expected_concepts"])))
        return {
            "eval_loss": loss,
            "perplexity": math.exp(loss),
            "answers": answers,
            "concept_hits": hits,
        }

    started_at = time.time()
    torch.cuda.reset_peak_memory_stats()
    baseline_model = load_model()
    baseline_result = evaluate_model(baseline_model)
    baseline_result["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    release_model(baseline_model)

    results: dict[str, Any] = {
        "status": "running",
        "preflight": dict(preflight_report),
        "environment": {
            "python": platform.python_version(),
            "packages": package_versions(),
            "cuda_version": torch.version.cuda,
            "gpu": gpu_name,
            "gpu_total_memory_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        },
        "baseline": baseline_result,
        "adapters": {},
    }
    write_json(output_root / "qlora_results.json", results)

    for adapter_name, adapter_spec in config["adapters"].items():
        adapter_start = time.time()
        adapter_dir = adapters_root / f"adapter_{adapter_name}"
        adapter_weights = adapter_dir / "adapter_model.safetensors"
        training_metrics_path = adapter_dir / "training_metrics.json"
        if reuse_existing_adapters and adapter_weights.is_file():
            torch.cuda.reset_peak_memory_stats()
            model = PeftModel.from_pretrained(load_model(), adapter_dir)
            adapter_result = evaluate_model(model)
            training_metrics = (
                json.loads(training_metrics_path.read_text(encoding="utf-8"))
                if training_metrics_path.is_file()
                else {"train_loss": None, "train_runtime_seconds": None}
            )
            adapter_result.update(
                {
                    "rank": int(adapter_spec["rank"]),
                    "alpha": int(adapter_spec["alpha"]),
                    "target_modules": list(adapter_spec["target_modules"]),
                    "train_loss": training_metrics.get("train_loss"),
                    "train_runtime_seconds": training_metrics.get(
                        "train_runtime_seconds"
                    ),
                    "trainable_parameters": sum(
                        parameter.numel()
                        for name, parameter in model.named_parameters()
                        if "lora_" in name
                    ),
                    "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
                    "elapsed_minutes": (time.time() - adapter_start) / 60,
                    "adapter_path": str(adapter_dir),
                    "reused_existing_adapter": True,
                }
            )
            results["adapters"][adapter_name] = adapter_result
            write_json(output_root / f"adapter_{adapter_name}_result.json", adapter_result)
            write_json(output_root / "qlora_results.json", results)
            release_model(model)
            continue
        if adapter_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Adapter output already exists: {adapter_dir}; use --overwrite-output"
                )
            shutil.rmtree(adapter_dir)

        set_seed(seed)
        torch.cuda.reset_peak_memory_stats()
        model = load_model()
        peft_config = LoraConfig(
            r=int(adapter_spec["rank"]),
            lora_alpha=int(adapter_spec["alpha"]),
            lora_dropout=float(settings["lora_dropout"]),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(adapter_spec["target_modules"]),
        )
        trainer = SFTTrainer(
            model=model,
            args=make_sft_config(settings, adapter_dir / "trainer"),
            train_dataset=train_dataset,
            peft_config=peft_config,
            **trainer_tokenizer_argument(tokenizer),
        )
        trainable_parameters = sum(
            parameter.numel()
            for parameter in trainer.model.parameters()
            if parameter.requires_grad
        )
        train_output = trainer.train()
        trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(adapter_dir)
        training_metrics = {
            "train_loss": float(train_output.training_loss),
            "train_runtime_seconds": float(train_output.metrics["train_runtime"]),
        }
        write_json(training_metrics_path, training_metrics)
        adapter_result = evaluate_model(trainer.model)
        adapter_result.update(
            {
                "rank": int(adapter_spec["rank"]),
                "alpha": int(adapter_spec["alpha"]),
                "target_modules": list(adapter_spec["target_modules"]),
                **training_metrics,
                "trainable_parameters": trainable_parameters,
                "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "elapsed_minutes": (time.time() - adapter_start) / 60,
                "adapter_path": str(adapter_dir),
            }
        )
        results["adapters"][adapter_name] = adapter_result
        write_json(output_root / f"adapter_{adapter_name}_result.json", adapter_result)
        write_json(output_root / "qlora_results.json", results)
        del trainer
        release_model(model)

    prompt_comparisons = []
    for prompt_record in evaluation_prompts:
        prompt_id = str(prompt_record["prompt_id"])
        prompt_comparisons.append(
            {
                **prompt_record,
                "answers": {
                    "CPT base": results["baseline"]["answers"][prompt_id],
                    **{
                        f"Adapter {name}": adapter_result["answers"][prompt_id]
                        for name, adapter_result in results["adapters"].items()
                    },
                },
            }
        )
    results["evaluation_prompts"] = prompt_comparisons
    results["elapsed_minutes"] = (time.time() - started_at) / 60
    results["status"] = "completed"
    write_json(output_root / "qlora_results.json", results)
    write_comparison_markdown(output_root / "adapter_comparison.md", results)
    return results


def main() -> None:
    args = parse_args()
    config, report, repository_root = preflight(args.config)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.preflight:
        return
    results = run_training(
        config,
        report,
        repository_root,
        overwrite=args.overwrite_output,
        reuse_existing_adapters=args.reuse_existing_adapters,
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()