"""Post-CPT perplexity and frozen-prompt comparison."""

from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .baseline import (
    audit_architecture,
    evaluate_parquet_perplexity,
    generate_baselines,
    load_frozen_prompts,
    load_local_model,
    set_reproducibility,
    write_per_sequence_csv,
)
from .corpus import sha256_file, write_json, write_jsonl
from .tokenization import load_frozen_tokenizer


def _verify_hash(path: Path, expected_sha256: str) -> None:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Frozen evaluation input hash mismatch for {path}: "
            f"expected {expected_sha256}, found {actual_sha256}"
        )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _concept_matches(text: str, concepts: Sequence[str]) -> dict[str, bool]:
    normalized_text = " ".join(text.casefold().split())
    matches: dict[str, bool] = {}
    for concept in concepts:
        normalized_concept = " ".join(str(concept).casefold().split())
        pattern = r"(?<!\w)" + re.escape(normalized_concept).replace(r"\ ", r"\s+") + r"(?!\w)"
        matches[str(concept)] = bool(re.search(pattern, normalized_text))
    return matches


def _automatic_score(text: str, concepts: Sequence[str]) -> dict[str, Any]:
    matches = _concept_matches(text, concepts)
    matched_count = sum(matches.values())
    concept_count = len(matches)
    return {
        "concept_matches": matches,
        "matched_concept_count": matched_count,
        "expected_concept_count": concept_count,
        "concept_coverage": matched_count / concept_count if concept_count else 0.0,
    }


def _index_by_prompt_id(
    records: Sequence[Mapping[str, Any]], label: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for record in records:
        prompt_id = str(record["prompt_id"])
        if prompt_id in indexed:
            raise RuntimeError(f"Duplicate {label} prompt ID: {prompt_id}")
        indexed[prompt_id] = record
    return indexed


def _build_prompt_comparisons(
    prompts: Sequence[Mapping[str, Any]],
    base_responses: Sequence[Mapping[str, Any]],
    cpt_responses: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    base_by_id = _index_by_prompt_id(base_responses, "base response")
    cpt_by_id = _index_by_prompt_id(cpt_responses, "CPT response")
    prompt_ids = [str(prompt["prompt_id"]) for prompt in prompts]
    if set(base_by_id) != set(prompt_ids) or set(cpt_by_id) != set(prompt_ids):
        raise RuntimeError("Base and CPT outputs do not contain the frozen prompt ID set")

    comparisons: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompts, start=1):
        prompt_id = str(prompt["prompt_id"])
        prompt_text = str(prompt["prompt"])
        base = base_by_id[prompt_id]
        cpt = cpt_by_id[prompt_id]
        if str(base["prompt"]) != prompt_text or str(cpt["prompt"]) != prompt_text:
            raise RuntimeError(f"Prompt text changed for {prompt_id}")

        concepts = [str(concept) for concept in prompt.get("expected_concepts", [])]
        base_score = _automatic_score(str(base["generated_text"]), concepts)
        cpt_score = _automatic_score(str(cpt["generated_text"]), concepts)
        coverage_change = cpt_score["concept_coverage"] - base_score["concept_coverage"]
        if coverage_change > 0:
            verdict = "Improved"
        elif coverage_change < 0:
            verdict = "Degraded"
        else:
            verdict = "Retained"

        comparisons.append(
            {
                "prompt_index": prompt_index,
                "prompt_id": prompt_id,
                "group": str(prompt["group"]),
                "category": str(prompt["category"]),
                "evidence_split": prompt.get("evidence_split"),
                "assignment_required": bool(prompt.get("assignment_required", False)),
                "prompt": prompt_text,
                "expected_concepts": concepts,
                "rubric": str(prompt["rubric"]),
                "base_generated_text": str(base["generated_text"]),
                "cpt_generated_text": str(cpt["generated_text"]),
                "base_generated_token_ids": list(base["generated_token_ids"]),
                "cpt_generated_token_ids": list(cpt["generated_token_ids"]),
                "base_automatic_score": base_score,
                "cpt_automatic_score": cpt_score,
                "concept_coverage_change": coverage_change,
                "automatic_verdict": verdict,
                "automatic_reason": (
                    f"Expected-concept coverage changed from "
                    f"{base_score['matched_concept_count']}/{base_score['expected_concept_count']} "
                    f"to {cpt_score['matched_concept_count']}/{cpt_score['expected_concept_count']}."
                ),
                "human_review": {
                    "status": "pending_blinded_review",
                    "factual_correctness": None,
                    "relevance": None,
                    "completeness": None,
                    "coherence": None,
                    "paired_verdict": None,
                    "notes": None,
                },
            }
        )
    return comparisons


def _write_prompt_csv(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "prompt_index",
        "prompt_id",
        "group",
        "category",
        "evidence_split",
        "assignment_required",
        "prompt",
        "expected_concepts",
        "base_generated_text",
        "cpt_generated_text",
        "base_concept_coverage",
        "cpt_concept_coverage",
        "concept_coverage_change",
        "automatic_verdict",
        "automatic_reason",
        "human_review_status",
        "human_paired_verdict",
        "human_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            human_review = record["human_review"]
            writer.writerow(
                {
                    "prompt_index": record["prompt_index"],
                    "prompt_id": record["prompt_id"],
                    "group": record["group"],
                    "category": record["category"],
                    "evidence_split": record["evidence_split"],
                    "assignment_required": record["assignment_required"],
                    "prompt": record["prompt"],
                    "expected_concepts": json.dumps(record["expected_concepts"]),
                    "base_generated_text": record["base_generated_text"],
                    "cpt_generated_text": record["cpt_generated_text"],
                    "base_concept_coverage": record["base_automatic_score"]["concept_coverage"],
                    "cpt_concept_coverage": record["cpt_automatic_score"]["concept_coverage"],
                    "concept_coverage_change": record["concept_coverage_change"],
                    "automatic_verdict": record["automatic_verdict"],
                    "automatic_reason": record["automatic_reason"],
                    "human_review_status": human_review["status"],
                    "human_paired_verdict": human_review["paired_verdict"],
                    "human_notes": human_review["notes"],
                }
            )


def write_blinded_review_package(
    worksheet_path: Path,
    key_path: Path,
    prompts: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    seed: int,
) -> None:
    def csv_text(value: Any) -> str:
        return str(value or "").replace("\r\n", "\n").replace("\r", "\n").replace("\n", r"\n")

    prompts_by_id = _index_by_prompt_id(prompts, "frozen")
    rng = random.Random(seed)
    fieldnames = [
        "review_id",
        "prompt_id",
        "group",
        "category",
        "evidence_split",
        "prompt",
        "expected_concepts",
        "rubric",
        "source_path",
        "source_page",
        "source_excerpt",
        "output_a",
        "output_b",
        "output_a_factual_correctness_1_to_5",
        "output_a_relevance_1_to_5",
        "output_a_completeness_1_to_5",
        "output_a_coherence_1_to_5",
        "output_b_factual_correctness_1_to_5",
        "output_b_relevance_1_to_5",
        "output_b_completeness_1_to_5",
        "output_b_coherence_1_to_5",
        "preferred_output_a_b_or_tie",
        "reviewer_notes",
    ]
    key: list[dict[str, Any]] = []
    worksheet_path.parent.mkdir(parents=True, exist_ok=True)
    with worksheet_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, comparison in enumerate(comparisons, start=1):
            prompt_id = str(comparison["prompt_id"])
            prompt = prompts_by_id[prompt_id]
            base_is_a = bool(rng.getrandbits(1))
            output_a = (
                comparison["base_generated_text"]
                if base_is_a
                else comparison["cpt_generated_text"]
            )
            output_b = (
                comparison["cpt_generated_text"]
                if base_is_a
                else comparison["base_generated_text"]
            )
            review_id = f"review-{index:03d}"
            writer.writerow(
                {
                    "review_id": review_id,
                    "prompt_id": prompt_id,
                    "group": comparison["group"],
                    "category": comparison["category"],
                    "evidence_split": comparison["evidence_split"],
                    "prompt": comparison["prompt"],
                    "expected_concepts": json.dumps(comparison["expected_concepts"]),
                    "rubric": comparison["rubric"],
                    "source_path": prompt.get("source_path"),
                    "source_page": prompt.get("source_page"),
                    "source_excerpt": csv_text(prompt.get("source_excerpt")),
                    "output_a": csv_text(output_a),
                    "output_b": csv_text(output_b),
                }
            )
            key.append(
                {
                    "review_id": review_id,
                    "prompt_id": prompt_id,
                    "output_a_model": "base" if base_is_a else "cpt",
                    "output_b_model": "cpt" if base_is_a else "base",
                }
            )
    write_json(key_path, {"seed": seed, "assignments": key})


def _load_base_per_sequence(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {
                "sequence_id": int(row["sequence_id"]),
                "predicted_token_count": int(row["predicted_token_count"]),
                "mean_loss": float(row["mean_loss"]),
                "negative_log_likelihood": float(row["negative_log_likelihood"]),
            }
            for row in csv.DictReader(handle)
        ]


def _combine_per_sequence(
    base_records: Sequence[Mapping[str, Any]],
    cpt_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(base_records) != len(cpt_records):
        raise RuntimeError("Base and CPT per-sequence result counts differ")
    combined: list[dict[str, Any]] = []
    for base, cpt in zip(base_records, cpt_records):
        if base["sequence_id"] != cpt["sequence_id"]:
            raise RuntimeError("Base and CPT sequence IDs differ")
        if base["predicted_token_count"] != cpt["predicted_token_count"]:
            raise RuntimeError("Base and CPT predicted-token counts differ")
        combined.append(
            {
                "sequence_id": int(base["sequence_id"]),
                "predicted_token_count": int(base["predicted_token_count"]),
                "base_mean_loss": float(base["mean_loss"]),
                "cpt_mean_loss": float(cpt["mean_loss"]),
                "base_negative_log_likelihood": float(base["negative_log_likelihood"]),
                "cpt_negative_log_likelihood": float(cpt["negative_log_likelihood"]),
                "mean_loss_change": float(cpt["mean_loss"]) - float(base["mean_loss"]),
            }
        )
    return combined


def _write_combined_per_sequence_csv(
    path: Path, records: Iterable[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sequence_id",
        "predicted_token_count",
        "base_mean_loss",
        "cpt_mean_loss",
        "base_negative_log_likelihood",
        "cpt_negative_log_likelihood",
        "mean_loss_change",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _recompute_perplexity(records: Sequence[Mapping[str, Any]], prefix: str) -> float:
    total_nll = sum(float(record[f"{prefix}_negative_log_likelihood"]) for record in records)
    total_tokens = sum(int(record["predicted_token_count"]) for record in records)
    return math.exp(total_nll / total_tokens)


def _summarize_prompts(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    verdict_counts = Counter(str(record["automatic_verdict"]) for record in records)
    count = len(records)
    return {
        "prompt_count": count,
        "automatic_verdict_counts": dict(sorted(verdict_counts.items())),
        "base_mean_concept_coverage": (
            sum(float(record["base_automatic_score"]["concept_coverage"]) for record in records)
            / count
            if count
            else 0.0
        ),
        "cpt_mean_concept_coverage": (
            sum(float(record["cpt_automatic_score"]["concept_coverage"]) for record in records)
            / count
            if count
            else 0.0
        ),
        "automatic_degradation_rate_percent": (
            100.0 * verdict_counts["Degraded"] / count if count else 0.0
        ),
    }


def _markdown_cell(value: Any, limit: int = 240) -> str:
    text = " ".join(str(value).split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _write_forgetting_markdown(
    path: Path,
    report: Mapping[str, Any],
    comparisons: Sequence[Mapping[str, Any]],
) -> None:
    general = report["groups"]["general"]
    domain = report["groups"]["domain"]
    lines = [
        "# Frozen Prompt Comparison",
        "",
        "Automatic verdicts use only exact frozen expected-concept coverage. Human rubric review remains pending and must be performed blind to model identity.",
        "",
        "## Summary",
        "",
        "| Group | Prompts | Base mean coverage | CPT mean coverage | Improved | Retained | Degraded | Degradation rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in (("General", general), ("Domain", domain)):
        counts = summary["automatic_verdict_counts"]
        lines.append(
            f"| {label} | {summary['prompt_count']} | "
            f"{summary['base_mean_concept_coverage']:.3f} | "
            f"{summary['cpt_mean_concept_coverage']:.3f} | "
            f"{counts.get('Improved', 0)} | {counts.get('Retained', 0)} | "
            f"{counts.get('Degraded', 0)} | "
            f"{summary['automatic_degradation_rate_percent']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Assignment Examples",
            "",
            "| Prompt ID | Group | Prompt | Base output | CPT output | Automatic verdict |",
            "|---|---|---|---|---|---|",
        ]
    )
    for record in comparisons:
        if not record["assignment_required"]:
            continue
        lines.append(
            f"| {record['prompt_id']} | {record['group']} | "
            f"{_markdown_cell(record['prompt'])} | "
            f"{_markdown_cell(record['base_generated_text'])} | "
            f"{_markdown_cell(record['cpt_generated_text'])} | "
            f"{record['automatic_verdict']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_post_cpt_evaluation(
    config: Mapping[str, Any], repository_root: Path, config_sha256: str
) -> dict[str, Any]:
    import torch

    def resolve(relative_path: str) -> Path:
        return (repository_root / relative_path).resolve()

    inputs = dict(config["inputs"])
    expected_hashes = dict(config["expected_sha256"])
    resolved_inputs = {name: resolve(str(path)) for name, path in inputs.items()}
    for name, expected_sha256 in expected_hashes.items():
        _verify_hash(resolved_inputs[name], str(expected_sha256))

    training_manifest = _read_json(resolved_inputs["training_manifest"])
    if training_manifest.get("status") != "completed":
        raise RuntimeError("CPT training manifest is not completed")

    seed = int(config["seed"])
    set_reproducibility(seed)
    model_id = str(config["model_id"])
    revision = str(config["revision"])
    tokenizer, tokenizer_metadata = load_frozen_tokenizer(
        model_id, revision, resolve(str(config["cache_dir"]))
    )
    model = load_local_model(resolve(str(config["model_dir"])), str(config["device"]))
    architecture = audit_architecture(model, model_id, revision)

    prompts = load_frozen_prompts(resolved_inputs["prompts"])
    base_responses = _read_jsonl(resolved_inputs["base_responses"])
    generation_config = dict(_read_json(resolved_inputs["generation_config"]))
    if int(generation_config["pad_token_id"]) != int(tokenizer.eos_token_id):
        raise RuntimeError("Frozen generation pad token does not match the tokenizer")

    output_root = resolve(str(config["output_root"]))
    post_cpt_root = output_root / "post_cpt"
    perplexity_root = output_root / "perplexity"
    post_cpt_root.mkdir(parents=True, exist_ok=True)
    perplexity_root.mkdir(parents=True, exist_ok=True)

    cpt_responses = generate_baselines(
        model, tokenizer, prompts, generation_config
    )
    write_jsonl(post_cpt_root / "responses.jsonl", cpt_responses)
    write_json(post_cpt_root / "generation_config.json", generation_config)
    write_json(post_cpt_root / "tokenizer_metadata.json", tokenizer_metadata)
    write_json(post_cpt_root / "architecture_audit.json", architecture)

    cpt_perplexity, cpt_per_sequence = evaluate_parquet_perplexity(
        model, resolved_inputs["eval_parquet"]
    )
    cpt_perplexity.update(
        {
            "model_id": model_id,
            "revision": revision,
            "precision": str(next(model.parameters()).dtype),
            "model_dir": str(config["model_dir"]),
        }
    )
    write_json(perplexity_root / "cpt.json", cpt_perplexity)
    write_per_sequence_csv(post_cpt_root / "cpt_per_sequence.csv", cpt_per_sequence)

    base_perplexity = dict(_read_json(resolved_inputs["base_perplexity"]))
    base_per_sequence = _load_base_per_sequence(resolved_inputs["base_per_sequence"])
    combined_per_sequence = _combine_per_sequence(base_per_sequence, cpt_per_sequence)
    base_recomputed = _recompute_perplexity(combined_per_sequence, "base")
    cpt_recomputed = _recompute_perplexity(combined_per_sequence, "cpt")
    if not math.isclose(base_recomputed, float(base_perplexity["perplexity"]), rel_tol=1e-12):
        raise RuntimeError("Saved base perplexity does not match per-sequence recomputation")
    if not math.isclose(cpt_recomputed, float(cpt_perplexity["perplexity"]), rel_tol=1e-12):
        raise RuntimeError("CPT perplexity does not match per-sequence recomputation")
    if base_perplexity["source_parquet_sha256"] != cpt_perplexity["source_parquet_sha256"]:
        raise RuntimeError("Base and CPT perplexity used different held-out Parquet files")

    write_json(perplexity_root / "base.json", base_perplexity)
    _write_combined_per_sequence_csv(
        perplexity_root / "per_sequence.csv", combined_per_sequence
    )
    perplexity_comparison = {
        "base_perplexity": float(base_perplexity["perplexity"]),
        "cpt_perplexity": float(cpt_perplexity["perplexity"]),
        "perplexity_reduction_percent": (
            100.0
            * (float(base_perplexity["perplexity"]) - float(cpt_perplexity["perplexity"]))
            / float(base_perplexity["perplexity"])
        ),
        "base_mean_loss": float(base_perplexity["mean_loss"]),
        "cpt_mean_loss": float(cpt_perplexity["mean_loss"]),
        "sequence_count": int(cpt_perplexity["sequence_count"]),
        "predicted_token_count": int(cpt_perplexity["predicted_token_count"]),
        "source_parquet_sha256": str(cpt_perplexity["source_parquet_sha256"]),
        "base_recomputed_perplexity": base_recomputed,
        "cpt_recomputed_perplexity": cpt_recomputed,
    }
    write_json(perplexity_root / "comparison.json", perplexity_comparison)

    prompt_comparisons = _build_prompt_comparisons(
        prompts, base_responses, cpt_responses
    )
    write_jsonl(output_root / "prompt_comparison.jsonl", prompt_comparisons)
    _write_prompt_csv(output_root / "prompt_comparison.csv", prompt_comparisons)
    write_blinded_review_package(
        output_root / "blinded_human_review.csv",
        output_root / "blinding_key.json",
        prompts,
        prompt_comparisons,
        seed,
    )

    general_records = [record for record in prompt_comparisons if record["group"] == "general"]
    domain_records = [record for record in prompt_comparisons if record["group"] == "domain"]
    if len(general_records) != 25 or len(domain_records) != 25:
        raise RuntimeError("Frozen evaluation must contain 25 general and 25 domain prompts")
    forgetting_report = {
        "scoring_method": "case-insensitive whole-concept coverage",
        "human_review_status": "pending_blinded_review",
        "groups": {
            "general": _summarize_prompts(general_records),
            "domain": _summarize_prompts(domain_records),
            "domain_train_evidence": _summarize_prompts(
                [record for record in domain_records if record["evidence_split"] == "train"]
            ),
            "domain_eval_evidence": _summarize_prompts(
                [record for record in domain_records if record["evidence_split"] == "eval"]
            ),
        },
    }
    write_json(output_root / "forgetting_comparison.json", forgetting_report)
    _write_forgetting_markdown(
        output_root / "forgetting_comparison.md", forgetting_report, prompt_comparisons
    )

    result = {
        "status": "completed_automatic_evaluation",
        "config_sha256": config_sha256,
        "training_manifest_sha256": sha256_file(resolved_inputs["training_manifest"]),
        "prompt_manifest_sha256": sha256_file(resolved_inputs["prompts"]),
        "prompt_count": len(prompts),
        "perplexity": perplexity_comparison,
        "prompt_summary": forgetting_report,
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "human_review_status": "pending_blinded_review",
    }
    write_json(post_cpt_root / "evaluation_run.json", result)
    return result
