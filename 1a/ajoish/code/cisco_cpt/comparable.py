"""Build Murali-compatible summaries from completed Ajoish evaluation artifacts."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def _concept_matches(concept: str, response: str) -> bool:
    normalized_response = f" {' '.join(_words(response))} "
    for alternative in re.split(r"\s+or\s+", concept, flags=re.IGNORECASE):
        normalized_alternative = " ".join(_words(alternative))
        if not normalized_alternative:
            continue
        if f" {normalized_alternative} " in normalized_response:
            return True
        compact = normalized_alternative.replace(" ", "")
        if len(compact) <= 5 and compact in normalized_response.replace(" ", ""):
            return True
    return False


def _concept_hit_count(response: str, concepts: Sequence[str]) -> int:
    return sum(_concept_matches(str(concept), response) for concept in concepts)


def _repeated_trigram_ratio(text: str) -> float:
    words = _words(text)
    trigrams = [tuple(words[index : index + 3]) for index in range(len(words) - 2)]
    if not trigrams:
        return 0.0
    return 1.0 - len(set(trigrams)) / len(trigrams)


def _concept_assessment(base: float, cpt: float) -> dict[str, Any]:
    change = cpt - base
    if cpt >= 0.60 and change >= -0.02:
        verdict = "Good"
    elif cpt >= 0.30 and change >= -0.05:
        verdict = "Satisfactory"
    else:
        verdict = "Needs Improvement"
    direction = "improved" if change > 0 else "declined" if change < 0 else "was unchanged"
    return {
        "base": base,
        "cpt": cpt,
        "verdict": verdict,
        "remarks": f"CPT recall is {100 * cpt:.1f}% and {direction} by {abs(100 * change):.1f} pp.",
    }


def _repetition_assessment(base: float, cpt: float) -> dict[str, Any]:
    if cpt <= 0.10:
        verdict = "Good"
    elif cpt <= 0.30:
        verdict = "Satisfactory"
    else:
        verdict = "Needs Improvement"
    change = cpt - base
    direction = "increased" if change > 0 else "decreased" if change < 0 else "was unchanged"
    return {
        "base": base,
        "cpt": cpt,
        "verdict": verdict,
        "remarks": f"CPT repetition is {100 * cpt:.1f}% and {direction} by {abs(100 * change):.1f} pp.",
    }


def _per_query_verdict(
    record: Mapping[str, Any],
    base_hits: int,
    cpt_hits: int,
    base_repeat: float,
    cpt_repeat: float,
) -> str:
    cpt_text = str(record.get("cpt_generated_text") or "").strip()
    if not cpt_text or cpt_hits < base_hits or cpt_repeat - base_repeat > 0.10:
        return "Needs Improvement"
    if cpt_hits > base_hits or base_repeat - cpt_repeat > 0.10:
        return "Good"
    return "Satisfactory"


def _validate_prompt_records(records: Sequence[Mapping[str, Any]]) -> None:
    prompt_ids = [str(record["prompt_id"]) for record in records]
    if len(records) != 50 or len(set(prompt_ids)) != 50:
        raise RuntimeError("Expected exactly 50 unique paired prompt records")
    group_counts = Counter(str(record["group"]) for record in records)
    if group_counts != {"domain": 25, "general": 25}:
        raise RuntimeError(f"Unexpected prompt groups: {dict(group_counts)}")


def _generated_text_report(
    records: Sequence[Mapping[str, Any]],
    architecture: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    _validate_prompt_records(records)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    results = []
    verdict_counts: Counter[str] = Counter()

    for record in records:
        group = str(record["group"])
        grouped[group].append(record)
        concepts = [str(concept) for concept in record["expected_concepts"]]
        base_response = str(record["base_generated_text"])
        cpt_response = str(record["cpt_generated_text"])
        base_hits = _concept_hit_count(base_response, concepts)
        cpt_hits = _concept_hit_count(cpt_response, concepts)
        base_repeat = _repeated_trigram_ratio(base_response)
        cpt_repeat = _repeated_trigram_ratio(cpt_response)
        verdict = _per_query_verdict(
            record, base_hits, cpt_hits, base_repeat, cpt_repeat
        )
        verdict_counts[verdict] += 1
        results.append(
            {
                "id": record["prompt_id"],
                "query_set": group,
                "category": record["category"],
                "prompt": record["prompt"],
                "expected_concepts": concepts,
                "base_response": base_response,
                "cpt_response": cpt_response,
                "base_generation_metrics": {
                    "concept_hits": base_hits,
                    "concept_count": len(concepts),
                    "concept_recall": base_hits / len(concepts) if concepts else None,
                    "repeated_trigram_ratio": base_repeat,
                },
                "cpt_generation_metrics": {
                    "concept_hits": cpt_hits,
                    "concept_count": len(concepts),
                    "concept_recall": cpt_hits / len(concepts) if concepts else None,
                    "repeated_trigram_ratio": cpt_repeat,
                },
                "automatic_verdict": verdict,
            }
        )

    group_summary = {}
    for group, group_records in sorted(grouped.items()):
        concept_count = sum(len(record["expected_concepts"]) for record in group_records)
        base_hits = sum(
            _concept_hit_count(
                str(record["base_generated_text"]),
                [str(concept) for concept in record["expected_concepts"]],
            )
            for record in group_records
        )
        cpt_hits = sum(
            _concept_hit_count(
                str(record["cpt_generated_text"]),
                [str(concept) for concept in record["expected_concepts"]],
            )
            for record in group_records
        )
        base_repeat = sum(
            _repeated_trigram_ratio(str(record["base_generated_text"]))
            for record in group_records
        ) / len(group_records)
        cpt_repeat = sum(
            _repeated_trigram_ratio(str(record["cpt_generated_text"]))
            for record in group_records
        ) / len(group_records)
        group_summary[group] = {
            "query_count": len(group_records),
            "generation_concept_recall": _concept_assessment(
                base_hits / concept_count, cpt_hits / concept_count
            ),
            "generation_repeated_trigram_ratio": _repetition_assessment(
                base_repeat, cpt_repeat
            ),
            "generation_exact_continuation": {
                "available": False,
                "reason": "Ajoish's frozen prompts define expected concepts, not exact continuations.",
            },
            "fixed_reference_conditional_ppl": {
                "available": False,
                "reason": "Requires fixed reference continuations and scoring with both loaded models.",
            },
            "baseline_response_retention_ppl": {
                "available": False,
                "reason": "Requires teacher-forced scoring of saved base responses with both loaded models.",
            },
        }

    return {
        "schema_version": 1,
        "title": "Part 2 Generated Text Evaluation (Murali-Compatible Subset)",
        "created_at_utc": created_at,
        "model": {
            "name": architecture["model_id"],
            "revision": architecture["resolved_revision"],
            "architecture": architecture["architecture_class"],
            "total_parameters": architecture["total_parameters"],
        },
        "methodology": {
            "concept_recall": "Total matched expected concepts divided by total expected concepts.",
            "repetition": "Mean repeated-trigram ratio, using Murali's lowercase alphanumeric word normalization.",
            "automatic_verdicts": "Murali's per-query concept-loss and 10 percentage-point repetition thresholds.",
            "human_review_status": "pending_blinded_review",
        },
        "summary": {
            "query_count": len(records),
            "verdict_counts": dict(sorted(verdict_counts.items())),
            "comparison_by_query_set": group_summary,
        },
        "results": results,
    }


def _perplexity_report(
    comparison: Mapping[str, Any],
    architecture: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    reduction = float(comparison["perplexity_reduction_percent"])
    verdict = "Good" if reduction >= 10.0 else "Satisfactory" if reduction > 0.0 else "Needs Improvement"
    base = float(comparison["base_perplexity"])
    cpt = float(comparison["cpt_perplexity"])
    common = {
        "source_parquet_sha256": comparison["source_parquet_sha256"],
        "packed_sequence_count": comparison["sequence_count"],
        "evaluated_token_count": comparison["predicted_token_count"],
        "context_length": architecture["max_position_embeddings"],
    }
    return {
        "schema_version": 1,
        "title": "Part 1 Perplexity Evaluation (Murali-Compatible)",
        "created_at_utc": created_at,
        "models": {
            "base": {
                "requested_model_name": architecture["model_id"],
                "model_details": dict(architecture),
                "held_out_perplexity": {
                    "domain": {
                        **common,
                        "mean_cross_entropy_loss": comparison["base_mean_loss"],
                        "perplexity": base,
                    }
                },
            },
            "cpt": {
                "requested_model_name": architecture["model_id"],
                "model_details": dict(architecture),
                "held_out_perplexity": {
                    "domain": {
                        **common,
                        "mean_cross_entropy_loss": comparison["cpt_mean_loss"],
                        "perplexity": cpt,
                    }
                },
            },
        },
        "comparison": {
            "domain": {
                "base": base,
                "cpt": cpt,
                "absolute_change": cpt - base,
                "relative_change_percent": -reduction,
                "desired_direction": "decrease indicates successful domain adaptation",
                "verdict": verdict,
                "remarks": f"Domain PPL decreased {reduction:.2f}%.",
            },
            "generic": {
                "available": False,
                "reason": "No tokenizer-matched generic held-out corpus was scored in the completed run.",
            },
        },
    }


def _shared_run_rows(shared_root: Path) -> list[dict[str, Any]]:
    rows = []
    paths = sorted(shared_root.glob("gpt2*/v*/cpt_evaluation_generated_text.json"))
    for generated_path in paths:
        perplexity_path = generated_path.with_name("cpt_evaluation_perplexity.json")
        if not perplexity_path.is_file():
            continue
        generated = _read_json(generated_path)
        perplexity = _read_json(perplexity_path)
        summary = generated["summary"]
        groups = summary["comparison_by_query_set"]
        comparison = perplexity["comparison"]["domain"]
        model = generated["models"]["cpt"]["model_details"]
        rows.append(
            {
                "run": str(generated_path.parent.relative_to(shared_root)),
                "model": model["model_type"],
                "parameters": model["total_parameters"],
                "context_length": model["maximum_context_length"],
                "prompt_suite": "Murali shared 55-prompt suite",
                "prompt_count": summary["query_count"],
                "domain_ppl_base": comparison["base"],
                "domain_ppl_cpt": comparison["cpt"],
                "domain_ppl_reduction_percent": -comparison["relative_change_percent"],
                "domain_concept_recall_cpt": groups["domain"]["generation_concept_recall"]["cpt"],
                "general_concept_recall_cpt": groups["general"]["generation_concept_recall"]["cpt"],
                "domain_repetition_cpt": groups["domain"]["generation_repeated_trigram_ratio"]["cpt"],
                "general_repetition_cpt": groups["general"]["generation_repeated_trigram_ratio"]["cpt"],
            }
        )
    if not rows:
        raise RuntimeError(f"No complete shared evaluation reports found under {shared_root}")
    return rows


def _our_run_row(
    perplexity: Mapping[str, Any], generated: Mapping[str, Any]
) -> dict[str, Any]:
    domain = generated["summary"]["comparison_by_query_set"]["domain"]
    general = generated["summary"]["comparison_by_query_set"]["general"]
    model = generated["model"]
    return {
        "run": "ajoish/smollm2-1.7b-cisco-cpt-v1",
        "model": model["architecture"],
        "parameters": model["total_parameters"],
        "context_length": perplexity["models"]["cpt"]["model_details"]["max_position_embeddings"],
        "prompt_suite": "Ajoish frozen 50-prompt suite",
        "prompt_count": generated["summary"]["query_count"],
        "domain_ppl_base": perplexity["comparison"]["domain"]["base"],
        "domain_ppl_cpt": perplexity["comparison"]["domain"]["cpt"],
        "domain_ppl_reduction_percent": -perplexity["comparison"]["domain"]["relative_change_percent"],
        "domain_concept_recall_cpt": domain["generation_concept_recall"]["cpt"],
        "general_concept_recall_cpt": general["generation_concept_recall"]["cpt"],
        "domain_repetition_cpt": domain["generation_repeated_trigram_ratio"]["cpt"],
        "general_repetition_cpt": general["generation_repeated_trigram_ratio"]["cpt"],
    }


def _markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Cross-Run CPT Evaluation Comparison",
        "",
        "## Interpretation boundary",
        "",
        "- Compare domain perplexity **reduction percentages**, not raw perplexities. Tokenizers, context lengths, and held-out corpora differ.",
        "- Concept recall and repetition use compatible formulas, but prompt suites differ. Treat them as within-run diagnostics, not a model leaderboard.",
        "- A direct model ranking requires rerunning every model on one common prompt suite and tokenizer-appropriate matched corpora.",
        "",
        "## Available comparable metrics",
        "",
        "| Run | Parameters | Context | Prompts | Domain PPL reduction | Domain concept recall | General concept recall | Domain repetition | General repetition |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['parameters']:,} | {row['context_length']:,} | "
            f"{row['prompt_count']} | {row['domain_ppl_reduction_percent']:.2f}% | "
            f"{100 * row['domain_concept_recall_cpt']:.1f}% | "
            f"{100 * row['general_concept_recall_cpt']:.1f}% | "
            f"{100 * row['domain_repetition_cpt']:.1f}% | "
            f"{100 * row['general_repetition_cpt']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Missing from the completed Ajoish run",
            "",
            "- Generic held-out perplexity.",
            "- Fixed-reference conditional perplexity and exact-continuation metrics.",
            "- Baseline-response retention perplexity.",
            "",
            "These require an additional model-scoring run; they cannot be recovered from generated text alone.",
        ]
    )
    return "\n".join(lines) + "\n"


def prepare_comparable_evaluation(repository_root: Path, output_root: Path) -> dict[str, Any]:
    ajoish_root = repository_root / "1a" / "ajoish"
    evaluation_root = ajoish_root / "evaluation"
    shared_root = repository_root / "1a" / "evaluation"
    records = _read_jsonl(evaluation_root / "prompt_comparison.jsonl")
    architecture = _read_json(evaluation_root / "baseline" / "architecture_audit.json")
    source_perplexity = _read_json(evaluation_root / "perplexity" / "comparison.json")
    created_at = datetime.now(timezone.utc).isoformat()

    generated = _generated_text_report(records, architecture, created_at)
    perplexity = _perplexity_report(source_perplexity, architecture, created_at)
    rows = _shared_run_rows(shared_root)
    rows.append(_our_run_row(perplexity, generated))
    comparison = {
        "schema_version": 1,
        "created_at_utc": created_at,
        "comparability": {
            "raw_perplexity": "not comparable across tokenizers and held-out corpora",
            "perplexity_relative_change": "directionally comparable within each base-to-CPT run",
            "generation_metrics": "formula-compatible but based on different prompt suites",
        },
        "runs": rows,
    }

    _write_json(output_root / "murali_compatible_perplexity.json", perplexity)
    _write_json(output_root / "murali_compatible_generated_text.json", generated)
    _write_json(output_root / "cross_run_comparison.json", comparison)
    (output_root / "cross_run_comparison.md").write_text(
        _markdown(rows), encoding="utf-8"
    )
    return {
        "status": "completed",
        "our_prompt_count": len(records),
        "shared_run_count": len(rows) - 1,
        "output_root": str(output_root),
    }
