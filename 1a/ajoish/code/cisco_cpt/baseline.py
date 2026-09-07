"""Frozen-model architecture audit, generation, and held-out perplexity."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .corpus import sha256_file, write_json, write_jsonl
from .tokenization import load_frozen_tokenizer


EXPECTED_ARCHITECTURE = {
    "model_type": "llama",
    "vocab_size": 49152,
    "max_position_embeddings": 8192,
    "num_hidden_layers": 24,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "bos_token_id": 0,
    "eos_token_id": 0,
}


def set_reproducibility(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_frozen_model(
    model_id: str,
    revision: str,
    cache_dir: Path,
    device: str = "cuda",
) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen baseline model")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model


def load_local_model(model_dir: Path, device: str = "cuda") -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    if not model_dir.is_dir():
        raise ValueError(f"Local model directory does not exist: {model_dir}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the CPT model")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model


def audit_architecture(model: Any, model_id: str, revision: str) -> dict[str, Any]:
    config = model.config
    facts = {
        key: getattr(config, key)
        for key in EXPECTED_ARCHITECTURE
    }
    mismatches = {
        key: {"expected": expected, "actual": facts[key]}
        for key, expected in EXPECTED_ARCHITECTURE.items()
        if facts[key] != expected
    }
    if mismatches:
        raise RuntimeError(f"Frozen model architecture mismatch: {mismatches}")

    output_embeddings = model.get_output_embeddings()
    output_dimension = getattr(output_embeddings, "out_features", None)
    if output_dimension is None:
        output_dimension = int(output_embeddings.weight.shape[0])
    if output_dimension != facts["vocab_size"]:
        raise RuntimeError("lm_head output dimension does not equal vocabulary size")

    head_dimension, remainder = divmod(
        int(facts["hidden_size"]), int(facts["num_attention_heads"])
    )
    if remainder:
        raise RuntimeError("Hidden size is not divisible by attention-head count")

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_parameters != total_parameters:
        raise RuntimeError("Full-parameter CPT requires every model parameter to be trainable")

    return {
        "model_id": model_id,
        "requested_revision": revision,
        "resolved_revision": getattr(config, "_commit_hash", revision),
        "architecture_class": type(model).__name__,
        **facts,
        "head_dimension": head_dimension,
        "lm_head_output_dimension": int(output_dimension),
        "tie_word_embeddings": bool(config.tie_word_embeddings),
        "torch_dtype": str(config.torch_dtype),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }


def load_frozen_prompts(path: Path) -> list[dict[str, Any]]:
    prompts = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    identifiers = [str(item["prompt_id"]) for item in prompts]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Prompt IDs must be unique")
    groups = {str(item["group"]) for item in prompts}
    if groups != {"domain", "general"}:
        raise ValueError("Frozen prompts must contain domain and general groups")
    if any(not str(item.get("prompt", "")).strip() for item in prompts):
        raise ValueError("Every frozen prompt must contain prompt text")
    return prompts


def generate_baselines(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[Mapping[str, Any]],
    generation_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import torch

    device = next(model.parameters()).device
    results: list[dict[str, Any]] = []
    for index, prompt_record in enumerate(prompts, start=1):
        prompt = str(prompt_record["prompt"])
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        encoded = {name: value.to(device) for name, value in encoded.items()}
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**encoded, **generation_config)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        input_length = int(encoded["input_ids"].shape[1])
        generated_ids = output[0, input_length:].detach().cpu().tolist()
        result = dict(prompt_record)
        result.update(
            {
                "prompt_index": index,
                "input_token_count": input_length,
                "generated_token_count": len(generated_ids),
                "generated_token_ids": generated_ids,
                "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
                "elapsed_seconds": elapsed,
            }
        )
        results.append(result)
        print(f"[{index:02d}/{len(prompts):02d}] generated {prompt_record['prompt_id']}", flush=True)
    return results


def evaluate_parquet_perplexity(
    model: Any,
    parquet_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import pyarrow.parquet as pq
    import torch

    device = next(model.parameters()).device
    total_negative_log_likelihood = 0.0
    total_predicted_tokens = 0
    sequence_results: list[dict[str, Any]] = []
    parquet = pq.ParquetFile(parquet_path)

    for batch in parquet.iter_batches(batch_size=1):
        row = batch.to_pylist()[0]
        input_ids = torch.tensor(row["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = torch.tensor(
            row["attention_mask"], dtype=torch.long, device=device
        ).unsqueeze(0)
        labels = torch.tensor(row["labels"], dtype=torch.long, device=device).unsqueeze(0)
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
        predicted_tokens = int(attention_mask[:, 1:].sum().item())
        mean_loss = float(output.loss.detach().float().cpu().item())
        negative_log_likelihood = mean_loss * predicted_tokens
        total_negative_log_likelihood += negative_log_likelihood
        total_predicted_tokens += predicted_tokens
        sequence_results.append(
            {
                "sequence_id": int(row["sequence_id"]),
                "predicted_token_count": predicted_tokens,
                "mean_loss": mean_loss,
                "negative_log_likelihood": negative_log_likelihood,
            }
        )
        print(
            f"[{len(sequence_results):02d}/{parquet.metadata.num_rows:02d}] "
            f"evaluated sequence {row['sequence_id']}",
            flush=True,
        )

    mean_loss = total_negative_log_likelihood / total_predicted_tokens
    summary = {
        "sequence_count": len(sequence_results),
        "predicted_token_count": total_predicted_tokens,
        "negative_log_likelihood": total_negative_log_likelihood,
        "mean_loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "source_parquet_sha256": sha256_file(parquet_path),
    }
    return summary, sequence_results


def write_per_sequence_csv(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sequence_id",
        "predicted_token_count",
        "mean_loss",
        "negative_log_likelihood",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def run_frozen_baseline(config: Mapping[str, Any], repository_root: Path) -> dict[str, Any]:
    import torch

    def resolve(relative_path: str) -> Path:
        return (repository_root / relative_path).resolve()

    seed = int(config["seed"])
    set_reproducibility(seed)
    cache_dir = resolve(str(config["cache_dir"]))
    model_id = str(config["model_id"])
    revision = str(config["revision"])
    tokenizer, tokenizer_metadata = load_frozen_tokenizer(model_id, revision, cache_dir)
    model = load_frozen_model(model_id, revision, cache_dir, str(config["device"]))
    architecture = audit_architecture(model, model_id, revision)
    if architecture["resolved_revision"] != revision:
        raise RuntimeError("Loaded model revision does not match the frozen revision")

    prompts_path = resolve(str(config["prompts_path"]))
    prompts = load_frozen_prompts(prompts_path)
    generation_config = dict(config["generation"])
    generation_config["pad_token_id"] = int(tokenizer.eos_token_id)
    responses = generate_baselines(model, tokenizer, prompts, generation_config)

    output_root = resolve(str(config["output_root"]))
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "architecture_audit.json", architecture)
    write_json(output_root / "tokenizer_metadata.json", tokenizer_metadata)
    write_jsonl(output_root / "responses.jsonl", responses)
    write_json(output_root / "generation_config.json", generation_config)

    perplexity, per_sequence = evaluate_parquet_perplexity(
        model, resolve(str(config["eval_parquet_path"]))
    )
    perplexity.update(
        {
            "model_id": model_id,
            "revision": revision,
            "precision": str(next(model.parameters()).dtype),
        }
    )
    write_json(output_root / "base_perplexity.json", perplexity)
    write_per_sequence_csv(output_root / "base_per_sequence.csv", per_sequence)

    result = {
        "architecture": architecture,
        "prompt_count": len(prompts),
        "prompt_manifest_sha256": sha256_file(prompts_path),
        "response_count": len(responses),
        "perplexity": perplexity,
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
    write_json(output_root / "baseline_run.json", result)
    return result