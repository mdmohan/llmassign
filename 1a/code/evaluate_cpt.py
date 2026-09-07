"""Comprehensive before/after evaluation for continued pre-training.

The script evaluates four complementary questions:

1. Held-out domain perplexity: did CPT improve prediction of unseen domain text?
2. Held-out generic perplexity: did general-language prediction deteriorate?
3. Fixed-reference conditional perplexity: did the probability of an
   authoritative continuation improve for a given prompt?
4. Greedy generation quality and baseline-response retention: did observable
   answers improve, and how far did the CPT model move from base behaviour?

Perplexity and generation are deliberately kept separate. Perplexity scores
fixed text supplied to a model through teacher forcing; it does not score the
quality of whatever text a model happens to generate.
"""

from __future__ import annotations

import gc
import hashlib
import json
import shlex
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline import _model_details, resolve_query_paths
from cache import CACHE_DIR
from cli_parsers import build_cpt_evaluation_parser
from compare_responses import _print_table, _set_label, score_record
from cpt_train import _json_compatible, _perplexity
from load_tensors import MemmapDataset
from model_response import generate_responses, load_prompt_json


REFERENCE_FIELDS = (
    "expected_continuation",
    "expected_response",
    "reference_answer",
    "expected_answer",
)


def _sha256(path: Path) -> str:
    """Identify the exact held-out file used in every model comparison."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_metrics(path: Path | None) -> tuple[dict | None, str | None]:
    if path is None:
        return None, None
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"Dataset metrics file does not exist: {resolved}")
    try:
        return json.loads(resolved.read_text(encoding="utf-8")), str(resolved)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid dataset metrics JSON: {resolved}") from exc


def _model_context_length(model, tokenizer) -> int:
    """Support GPT-2 and Llama-family context-length configuration names."""
    for attribute in ("n_positions", "max_position_embeddings", "n_ctx"):
        value = getattr(model.config, attribute, None)
        if value is not None:
            return int(value)
    return int(tokenizer.model_max_length)


def _loaded_model_details(model, tokenizer, device) -> dict:
    """Reuse existing model-specific reporting where available."""
    if getattr(model.config, "model_type", None) == "llama":
        from smollm2_model import smollm2_model_details

        details = smollm2_model_details(model, tokenizer, device)
    else:
        details = _model_details(model, tokenizer, device)
    details["cache_dir"] = str(CACHE_DIR)
    return details


def _load_selected_model(model_name: str, model_folder: Path | None = None):
    """Load a supported base model or local CPT checkpoint using existing code."""
    from smollm2_model import (
        is_smollm2_checkpoint,
        is_smollm2_model,
        load_smollm2_model,
    )

    smollm2_selected = is_smollm2_model(model_name)
    if model_folder is not None:
        smollm2_selected = (
            smollm2_selected or is_smollm2_checkpoint(model_folder)
        )

    if smollm2_selected:
        model_dict = load_smollm2_model(
            model_name=model_name,
            model_folder=model_folder,
        )
    elif model_folder is None:
        from gpt2_model import load_gpt2_model

        model_dict = load_gpt2_model(model_name=model_name)
    else:
        from load_local_model import load_local_model

        model, tokenizer, device = load_local_model(model_folder)
        model_dict = {
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
        }

    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    device = model_dict["device"]
    return model_dict, _loaded_model_details(model, tokenizer, device)


def _validate_packed_dataset(
    bin_file: Path,
    context_length: int,
    model,
    tokenizer,
    metrics: dict | None,
) -> tuple[MemmapDataset, dict]:
    """Validate that the held-out stream is safe to score with this model."""
    bin_file = bin_file.expanduser().resolve()
    if not bin_file.is_file():
        raise ValueError(f"Held-out token file does not exist: {bin_file}")
    if bin_file.stat().st_size % np.dtype(np.uint16).itemsize:
        raise ValueError(f"Invalid uint16 token-file size: {bin_file}")

    stored_tokens = bin_file.stat().st_size // np.dtype(np.uint16).itemsize
    if stored_tokens < context_length:
        raise ValueError(
            f"{bin_file} has only {stored_tokens:,} tokens, fewer than one "
            f"{context_length:,}-token sequence"
        )
    if stored_tokens % context_length:
        raise ValueError(
            f"{bin_file} has {stored_tokens:,} tokens, which is not divisible "
            f"by context length {context_length:,}"
        )

    model_limit = _model_context_length(model, tokenizer)
    if context_length > model_limit:
        raise ValueError(
            f"Evaluation context length {context_length:,} exceeds the loaded "
            f"model limit of {model_limit:,}"
        )

    dataset = MemmapDataset(str(bin_file), seq_len=context_length)
    maximum_token_id = int(dataset.data.max())
    vocabulary_size = int(model.config.vocab_size)
    if maximum_token_id >= vocabulary_size:
        raise ValueError(
            f"Held-out token ID {maximum_token_id:,} exceeds the model's "
            f"maximum valid token ID {vocabulary_size - 1:,}"
        )

    if metrics is not None:
        metrics_context = metrics.get("packing_context_length")
        if metrics_context is not None and int(metrics_context) != context_length:
            raise ValueError(
                "Dataset metrics and --context-length disagree: "
                f"{metrics_context} != {context_length}"
            )
        metrics_vocab = metrics.get("tokenizer_vocabulary_size")
        if metrics_vocab is not None and int(metrics_vocab) != tokenizer.vocab_size:
            raise ValueError(
                "Dataset tokenizer vocabulary and loaded tokenizer disagree: "
                f"{metrics_vocab} != {tokenizer.vocab_size}"
            )

    provenance = {
        "bin_file": str(bin_file),
        "sha256": _sha256(bin_file),
        "binary_dtype": "uint16",
        "file_size_bytes": bin_file.stat().st_size,
        "stored_token_count": stored_tokens,
        "context_length": context_length,
        "packed_sequence_count": len(dataset),
        "maximum_token_id": maximum_token_id,
    }
    return dataset, provenance


def evaluate_held_out_perplexity(
    model,
    tokenizer,
    device,
    bin_file: Path,
    *,
    context_length: int,
    batch_size: int,
    num_workers: int,
    metrics: dict | None,
) -> dict:
    """Calculate token-weighted perplexity over a fixed held-out stream.

    Theory
    ------
    A causal model predicts token t_i from all preceding tokens. Hugging Face
    performs that one-token label shift internally when ``labels=input_ids``.
    For a context of L tokens, exactly L-1 next-token predictions are scored.

    We accumulate ``batch mean loss * predicted token count`` to recover total
    negative log-likelihood. Only after summing across the complete corpus do
    we divide by the total tokens and exponentiate. Averaging batch PPL values
    would be mathematically wrong because exponentiation is nonlinear and the
    final batch can contain fewer sequences.
    """
    dataset, provenance = _validate_packed_dataset(
        bin_file,
        context_length,
        model,
        tokenizer,
        metrics,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.device(device).type == "cuda",
        drop_last=False,
    )

    original_use_cache = getattr(model.config, "use_cache", None)
    if original_use_cache is not None:
        # KV caching accelerates autoregressive generation, but it is not
        # needed for one-pass teacher-forced loss and wastes evaluation memory.
        model.config.use_cache = False

    model.eval()
    total_nll = 0.0
    total_predicted_tokens = 0
    started = time.time()

    with torch.inference_mode():
        for batch_index, batch in enumerate(dataloader, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            outputs = model(input_ids=input_ids, labels=input_ids)

            # Position zero in every independently packed sequence has no
            # preceding token in that sequence, so it is not predicted.
            predicted_tokens = input_ids.shape[0] * (input_ids.shape[1] - 1)
            total_nll += outputs.loss.detach().float().item() * predicted_tokens
            total_predicted_tokens += predicted_tokens

            if batch_index % 50 == 0 or batch_index == len(dataloader):
                print(
                    f"  evaluated {batch_index:,}/{len(dataloader):,} "
                    "held-out batches"
                )

    if original_use_cache is not None:
        model.config.use_cache = original_use_cache
    if total_predicted_tokens == 0:
        raise ValueError("Held-out evaluation produced no predicted tokens")

    mean_nll = total_nll / total_predicted_tokens
    return {
        **provenance,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "evaluated_token_count": total_predicted_tokens,
        "total_negative_log_likelihood": total_nll,
        "mean_cross_entropy_loss": mean_nll,
        "perplexity": _perplexity(mean_nll),
        "elapsed_seconds": time.time() - started,
        "method": "exp(total_negative_log_likelihood / evaluated_token_count)",
    }


def _join_prompt_and_target(prompt: str, target: str) -> tuple[str, int]:
    """Join natural-language fragments while retaining the target boundary."""
    separator = "" if prompt[-1:].isspace() or target[:1].isspace() else " "
    combined = prompt + separator + target
    # The inserted space belongs to the continuation context. Including the
    # first token that spans this boundary correctly scores GPT-style tokens
    # whose vocabulary entry contains a leading space.
    return combined, len(prompt)


def evaluate_conditional_perplexity(
    model,
    tokenizer,
    device,
    prompt: str,
    target: str,
) -> dict:
    """Score fixed target tokens conditioned on a prompt.

    Theory
    ------
    Conditional PPL is exp(-mean log P(target_t | prompt, earlier target)).
    The full prompt and target are passed through the model, but prompt labels
    are replaced with -100. Hugging Face ignores those positions in the loss,
    so the prompt supplies context without contributing to the score.

    This is teacher forcing: the target is supplied, not generated. Therefore
    base and CPT models must score the same target for a valid comparison.
    """
    prompt = str(prompt).strip()
    target = str(target).strip()
    if not prompt or not target:
        raise ValueError("Conditional perplexity requires a prompt and target")

    combined, target_character_start = _join_prompt_and_target(prompt, target)
    encoded = tokenizer(
        combined,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = list(encoded["input_ids"])
    offsets = encoded.get("offset_mapping")
    if not input_ids:
        raise ValueError("Tokenizer produced no tokens for prompt and target")

    if offsets is not None:
        # A token crossing the text boundary is part of the target score. This
        # handles subword tokens that combine a leading space with the first
        # response word more accurately than tokenizing both strings apart.
        labels = [
            token_id if end > target_character_start else -100
            for token_id, (_, end) in zip(input_ids, offsets)
        ]
    else:
        # All project tokenizers are fast tokenizers and supply offsets. This
        # fallback keeps the evaluator usable with a slow custom tokenizer.
        prompt_token_count = len(
            tokenizer(
                combined[:target_character_start],
                add_special_tokens=False,
            )["input_ids"]
        )
        labels = [-100] * prompt_token_count + input_ids[prompt_token_count:]

    model_limit = _model_context_length(model, tokenizer)
    original_token_count = len(input_ids)
    if len(input_ids) > model_limit:
        # Preserve the target and the nearest prompt context. Left truncation
        # matches causal conditioning better than dropping the target suffix.
        input_ids = input_ids[-model_limit:]
        labels = labels[-model_limit:]

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    label_tensor = torch.tensor([labels], dtype=torch.long, device=device)

    # Causal loss predicts labels from position one onward. The first sequence
    # token cannot be scored because no prior token is present in this window.
    predicted_target_tokens = int((label_tensor[:, 1:] != -100).sum().item())
    if predicted_target_tokens == 0:
        raise ValueError("No target tokens remain available for scoring")

    model.eval()
    with torch.inference_mode():
        loss = model(input_ids=input_tensor, labels=label_tensor).loss
    mean_nll = loss.detach().float().item()
    return {
        "target_token_count": predicted_target_tokens,
        "negative_log_likelihood": mean_nll * predicted_target_tokens,
        "mean_cross_entropy_loss": mean_nll,
        "perplexity": _perplexity(mean_nll),
        "combined_token_count": len(input_ids),
        "left_truncated_tokens": max(0, original_token_count - len(input_ids)),
    }


def _load_queries(supplied_paths) -> list[dict]:
    if not supplied_paths:
        return []

    records = []
    for query_path in resolve_query_paths(supplied_paths):
        query_records, _ = load_prompt_json(query_path)
        query_file = query_path.name
        for record in query_records:
            query_id = record.get("id")
            if query_id is None:
                raise ValueError(f"Query without an id in {query_path}")
            records.append(
                {
                    **record,
                    "query_file": query_file,
                    "query_set": _set_label(query_file),
                }
            )
    return records


def _infer_query_file(response_path: Path, data: dict, query_files: set[str]):
    source = data.get("query_source")
    if source and Path(str(source)).name in query_files:
        return Path(str(source)).name

    matches = [name for name in query_files if Path(name).stem in response_path.stem]
    return matches[0] if len(matches) == 1 else None


def _load_baseline_targets(supplied_paths, queries: list[dict]) -> tuple[dict, list]:
    """Load saved base responses as fixed behavioural-retention targets."""
    if not supplied_paths:
        return {}, []

    query_files = {record["query_file"] for record in queries}
    queries_by_id = defaultdict(list)
    for record in queries:
        queries_by_id[str(record["id"])].append(record["query_file"])

    targets = {}
    provenance = []
    for response_path in resolve_query_paths(supplied_paths):
        data = json.loads(response_path.read_text(encoding="utf-8"))
        results = data.get("results")
        if not isinstance(results, list):
            raise ValueError(f"'results' must be a list in {response_path}")
        query_file = _infer_query_file(response_path, data, query_files)

        loaded = 0
        for result in results:
            query_id = str(result.get("id"))
            response = result.get("response")
            if not isinstance(response, str) or not response.strip():
                continue

            resolved_query_file = query_file
            if resolved_query_file is None:
                candidates = queries_by_id.get(query_id, [])
                if len(candidates) == 1:
                    resolved_query_file = candidates[0]
            if resolved_query_file is None:
                continue

            key = (resolved_query_file, query_id)
            if key in targets:
                raise ValueError(
                    f"Duplicate baseline response for {resolved_query_file}:{query_id}"
                )
            targets[key] = response.strip()
            loaded += 1

        provenance.append(
            {
                "response_file": str(response_path),
                "model_name": data.get("model_name"),
                "evaluation_stage": data.get("evaluation_stage"),
                "matched_responses": loaded,
            }
        )
    return targets, provenance


def _reference_for_query(record: dict) -> tuple[str | None, str | None]:
    for field in REFERENCE_FIELDS:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return field, value.strip()
    return None, None


def _generation_metrics(record: dict) -> dict:
    """Measure observable output quality without treating it as perplexity.

    Concept recall checks whether required phrases appear, continuation metrics
    compare generated text with an authoritative reference, and repeated
    trigrams flag degeneration. These lexical measures are deterministic but
    remain supplements to semantic or human correctness review.
    """
    scored = score_record(record)
    return {
        "concept_hits": scored["concept_hits"],
        "concept_hit_count": scored["concept_hit_count"],
        "concept_count": scored["concept_count"],
        "concept_recall": scored["concept_recall"],
        "exact_continuation_prefix": scored["exact_continuation_prefix"],
        "continuation_longest_common_prefix_ratio": scored["continuation_lcp"],
        "continuation_token_coverage": scored["continuation_coverage"],
        "repeated_trigram_ratio": scored["repetition_ratio"],
    }


def _aggregate_conditional(records: list[dict], field: str) -> dict | None:
    scored = [record[field] for record in records if record.get(field) is not None]
    token_count = sum(item["target_token_count"] for item in scored)
    if token_count == 0:
        return None
    total_nll = sum(item["negative_log_likelihood"] for item in scored)
    mean_nll = total_nll / token_count
    return {
        "query_count": len(scored),
        "target_token_count": token_count,
        "total_negative_log_likelihood": total_nll,
        "mean_cross_entropy_loss": mean_nll,
        "perplexity": _perplexity(mean_nll),
    }


def _summarize_query_results(results: list[dict]) -> dict:
    groups = defaultdict(list)
    for record in results:
        groups[record["query_set"]].append(record)

    summaries = {}
    for name, records in sorted(groups.items()):
        concept_hits = sum(
            record["generation_metrics"]["concept_hit_count"]
            for record in records
        )
        concept_count = sum(
            record["generation_metrics"]["concept_count"]
            for record in records
        )
        exact_records = [
            record
            for record in records
            if record["generation_metrics"]["exact_continuation_prefix"]
            is not None
        ]
        summaries[name] = {
            "query_count": len(records),
            "generation": {
                "concept_hits": concept_hits,
                "concept_count": concept_count,
                "concept_recall": (
                    concept_hits / concept_count if concept_count else None
                ),
                "exact_continuation_matches": sum(
                    bool(
                        record["generation_metrics"][
                            "exact_continuation_prefix"
                        ]
                    )
                    for record in exact_records
                ),
                "exact_continuation_query_count": len(exact_records),
                "average_continuation_lcp": (
                    sum(
                        record["generation_metrics"][
                            "continuation_longest_common_prefix_ratio"
                        ]
                        for record in exact_records
                    )
                    / len(exact_records)
                    if exact_records
                    else None
                ),
                "average_continuation_token_coverage": (
                    sum(
                        record["generation_metrics"][
                            "continuation_token_coverage"
                        ]
                        for record in exact_records
                    )
                    / len(exact_records)
                    if exact_records
                    else None
                ),
                "average_repeated_trigram_ratio": sum(
                    record["generation_metrics"]["repeated_trigram_ratio"]
                    for record in records
                )
                / len(records),
            },
            # These aggregate scores remain token weighted across queries. A
            # short answer must not count as much as a long continuation merely
            # because it is represented by one JSON record.
            "fixed_reference_conditional_ppl": _aggregate_conditional(
                records,
                "fixed_reference_conditional_ppl",
            ),
            "baseline_response_retention_ppl": _aggregate_conditional(
                records,
                "baseline_response_retention_ppl",
            ),
        }
    return summaries


def evaluate_queries(
    model,
    tokenizer,
    device,
    queries: list[dict],
    baseline_targets: dict,
    *,
    max_new_tokens: int,
    generation_batch_size: int,
) -> dict | None:
    """Generate answers and score any available fixed targets."""
    if not queries:
        return None

    prompts = [str(record["prompt"]).strip() for record in queries]
    responses = generate_responses(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        batch_size=generation_batch_size,
    )

    results = []
    for record, response in zip(queries, responses):
        result = {
            **record,
            "generated_response": response,
            "generation_metrics": _generation_metrics(
                {**record, "response": response}
            ),
            "fixed_reference_conditional_ppl": None,
            "baseline_response_retention_ppl": None,
        }

        reference_field, reference = _reference_for_query(record)
        if reference is not None:
            result["fixed_reference_field"] = reference_field
            result["fixed_reference"] = reference
            result["fixed_reference_conditional_ppl"] = (
                evaluate_conditional_perplexity(
                    model,
                    tokenizer,
                    device,
                    record["prompt"],
                    reference,
                )
            )

        baseline_target = baseline_targets.get(
            (record["query_file"], str(record["id"]))
        )
        if baseline_target is not None:
            # The identical saved base response is scored by every model. If we
            # scored each model's own generated response, both model and target
            # would change and the PPL values would not be comparable.
            result["baseline_response_target"] = baseline_target
            result["baseline_response_retention_ppl"] = (
                evaluate_conditional_perplexity(
                    model,
                    tokenizer,
                    device,
                    record["prompt"],
                    baseline_target,
                )
            )
        results.append(result)

    return {
        "generation_configuration": {
            "decoding": "greedy",
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": max_new_tokens,
            "batch_size": generation_batch_size,
        },
        "summary_by_query_set": _summarize_query_results(results),
        "results": results,
    }


def _relative_change(before: float, after: float) -> dict:
    """Report CPT movement relative to the base score on identical data."""
    absolute = after - before
    return {
        "base": before,
        "cpt": after,
        "absolute_change": absolute,
        "relative_change_percent": (absolute / before * 100) if before else None,
    }


def _assessment(verdict: str, remarks: str) -> dict:
    """Return a consistent, JSON-friendly qualitative assessment."""
    return {"verdict": verdict, "remarks": remarks}


def _assess_held_out_ppl(dataset_name: str, values: dict) -> dict:
    """Interpret held-out PPL using the assignment's intended direction.

    These labels are reporting aids, not additional statistical tests. Domain
    PPL uses the assignment's 10% reduction as the threshold for a clearly
    successful adaptation. Generic PPL tolerates a small increase because the
    purpose is to detect material forgetting rather than demand improvement.
    """
    change = values["relative_change_percent"]
    if dataset_name == "domain":
        reduction = -change
        if reduction >= 10.0:
            return _assessment(
                "Good",
                f"Domain PPL decreased {reduction:.2f}%; meets the 10% adaptation target.",
            )
        if reduction > 0.0:
            return _assessment(
                "Satisfactory",
                f"Domain PPL decreased {reduction:.2f}%, but by less than 10%.",
            )
        return _assessment(
            "Needs Improvement",
            f"Domain PPL increased {-reduction:.2f}%; CPT did not improve held-out prediction.",
        )

    if change <= 5.0:
        return _assessment(
            "Good",
            f"Generic PPL changed {change:+.2f}%; little evidence of forgetting.",
        )
    if change <= 10.0:
        return _assessment(
            "Satisfactory",
            f"Generic PPL increased {change:.2f}%; mild forgetting may be present.",
        )
    return _assessment(
        "Needs Improvement",
        f"Generic PPL increased {change:.2f}%; substantial forgetting is possible.",
    )


def _assess_concept_recall(base: float | None, cpt: float | None) -> dict:
    """Assess generated concept coverage using absolute quality and drift."""
    if base is None or cpt is None:
        return _assessment("Needs Improvement", "Concept recall was unavailable.")
    change_pp = 100 * (cpt - base)
    if cpt >= 0.60 and change_pp >= -2.0:
        verdict = "Good"
    elif cpt >= 0.30 and change_pp >= -5.0:
        verdict = "Satisfactory"
    else:
        verdict = "Needs Improvement"
    direction = "improved" if change_pp > 0 else "declined" if change_pp < 0 else "was unchanged"
    return _assessment(
        verdict,
        f"CPT recall is {100 * cpt:.1f}% and {direction} by {abs(change_pp):.1f} pp.",
    )


def _assess_repetition(base: float, cpt: float) -> dict:
    """Assess repeated-trigram ratio; lower output repetition is preferable."""
    change_pp = 100 * (cpt - base)
    if cpt <= 0.10:
        verdict = "Good"
    elif cpt <= 0.30:
        verdict = "Satisfactory"
    else:
        verdict = "Needs Improvement"
    direction = "increased" if change_pp > 0 else "decreased" if change_pp < 0 else "was unchanged"
    return _assessment(
        verdict,
        f"CPT repetition is {100 * cpt:.1f}% and {direction} by {abs(change_pp):.1f} pp.",
    )


def _assess_ratio(value: float, label: str) -> dict:
    """Assess exact-match or reference-coverage ratios on a common scale."""
    if value >= 0.70:
        verdict = "Good"
    elif value >= 0.30:
        verdict = "Satisfactory"
    else:
        verdict = "Needs Improvement"
    return _assessment(verdict, f"CPT {label} is {100 * value:.1f}%.")


def _assess_reference_ppl(values: dict) -> dict:
    """Interpret PPL on authoritative fixed response continuations."""
    reduction = -values["relative_change_percent"]
    if reduction >= 10.0:
        verdict = "Good"
    elif reduction > 0.0:
        verdict = "Satisfactory"
    else:
        verdict = "Needs Improvement"
    return _assessment(
        verdict,
        f"Fixed-reference PPL changed {values['relative_change_percent']:+.2f}%; lower is better.",
    )


def _assess_retention_ppl(values: dict) -> dict:
    """Interpret drift when both models score the same saved base response."""
    change = values["relative_change_percent"]
    if change <= 5.0:
        verdict = "Good"
    elif change <= 10.0:
        verdict = "Satisfactory"
    else:
        verdict = "Needs Improvement"
    return _assessment(
        verdict,
        f"Baseline-response PPL changed {change:+.2f}%; a small increase means limited drift.",
    )


def _validate_comparable_models(base_details: dict, cpt_details: dict) -> None:
    """Reject comparisons where model/tokenizer identity changes the scale."""
    if base_details["model_type"] != cpt_details["model_type"]:
        raise ValueError(
            "Base and CPT model architectures differ: "
            f"{base_details['model_type']} != {cpt_details['model_type']}"
        )
    if (
        base_details["tokenizer_vocabulary_size"]
        != cpt_details["tokenizer_vocabulary_size"]
    ):
        raise ValueError(
            "Base and CPT tokenizer vocabulary sizes differ; their "
            "perplexities are not directly comparable"
        )


def _build_comparison(report: dict) -> dict | None:
    models = report.get("models", {})
    if "base" not in models or "cpt" not in models:
        return None

    comparison = {"held_out": {}, "queries": {}}
    for dataset_name in ("domain", "generic"):
        base = models["base"].get("held_out_perplexity", {}).get(dataset_name)
        cpt = models["cpt"].get("held_out_perplexity", {}).get(dataset_name)
        if base is None or cpt is None:
            continue
        values = _relative_change(base["perplexity"], cpt["perplexity"])
        values["desired_direction"] = (
            "decrease indicates successful domain adaptation"
            if dataset_name == "domain"
            else "little or no increase indicates limited forgetting"
        )
        values.update(_assess_held_out_ppl(dataset_name, values))
        comparison["held_out"][dataset_name] = values

    base_queries = models["base"].get("query_evaluation")
    cpt_queries = models["cpt"].get("query_evaluation")
    if base_queries and cpt_queries:
        base_sets = base_queries["summary_by_query_set"]
        cpt_sets = cpt_queries["summary_by_query_set"]
        for name in sorted(set(base_sets) & set(cpt_sets)):
            base_summary = base_sets[name]
            cpt_summary = cpt_sets[name]
            item = {
                "generation_concept_recall": {
                    "base": base_summary["generation"]["concept_recall"],
                    "cpt": cpt_summary["generation"]["concept_recall"],
                },
                "generation_repeated_trigram_ratio": {
                    "base": base_summary["generation"][
                        "average_repeated_trigram_ratio"
                    ],
                    "cpt": cpt_summary["generation"][
                        "average_repeated_trigram_ratio"
                    ],
                },
                "generation_exact_continuation": {
                    "base_matches": base_summary["generation"][
                        "exact_continuation_matches"
                    ],
                    "cpt_matches": cpt_summary["generation"][
                        "exact_continuation_matches"
                    ],
                    "query_count": base_summary["generation"][
                        "exact_continuation_query_count"
                    ],
                    "base_average_lcp": base_summary["generation"][
                        "average_continuation_lcp"
                    ],
                    "cpt_average_lcp": cpt_summary["generation"][
                        "average_continuation_lcp"
                    ],
                    "base_average_token_coverage": base_summary["generation"][
                        "average_continuation_token_coverage"
                    ],
                    "cpt_average_token_coverage": cpt_summary["generation"][
                        "average_continuation_token_coverage"
                    ],
                },
            }
            for field in (
                "fixed_reference_conditional_ppl",
                "baseline_response_retention_ppl",
            ):
                base_ppl = base_summary.get(field)
                cpt_ppl = cpt_summary.get(field)
                if base_ppl and cpt_ppl:
                    item[field] = _relative_change(
                        base_ppl["perplexity"],
                        cpt_ppl["perplexity"],
                    )
            item["generation_concept_recall"].update(
                _assess_concept_recall(
                    item["generation_concept_recall"]["base"],
                    item["generation_concept_recall"]["cpt"],
                )
            )
            item["generation_repeated_trigram_ratio"].update(
                _assess_repetition(
                    item["generation_repeated_trigram_ratio"]["base"],
                    item["generation_repeated_trigram_ratio"]["cpt"],
                )
            )
            exact = item["generation_exact_continuation"]
            if exact["query_count"]:
                exact["exact_match_assessment"] = _assess_ratio(
                    exact["cpt_matches"] / exact["query_count"],
                    "exact-match rate",
                )
                exact["continuation_coverage_assessment"] = _assess_ratio(
                    exact["cpt_average_token_coverage"],
                    "continuation coverage",
                )
            reference = item.get("fixed_reference_conditional_ppl")
            if reference:
                reference.update(_assess_reference_ppl(reference))
            retention = item.get("baseline_response_retention_ppl")
            if retention:
                retention.update(_assess_retention_ppl(retention))
            comparison["queries"][name] = item
    return comparison


def _generated_pair_verdict(base_result: dict, cpt_result: dict) -> dict:
    """Classify observable before/after generation changes for one prompt.

    This verdict is deliberately comparative. It detects lost expected
    concepts, regressed reference coverage, or materially worse repetition;
    it is not a substitute for a human factual-correctness judgment.
    """
    base_metrics = base_result["generation_metrics"]
    cpt_metrics = cpt_result["generation_metrics"]
    concept_delta = (
        cpt_metrics["concept_hit_count"] - base_metrics["concept_hit_count"]
    )
    repetition_delta = (
        cpt_metrics["repeated_trigram_ratio"]
        - base_metrics["repeated_trigram_ratio"]
    )
    base_coverage = base_metrics["continuation_token_coverage"]
    cpt_coverage = cpt_metrics["continuation_token_coverage"]
    coverage_delta = (
        cpt_coverage - base_coverage
        if base_coverage is not None and cpt_coverage is not None
        else 0.0
    )

    regressions = []
    improvements = []
    if concept_delta < 0:
        regressions.append(f"lost {abs(concept_delta)} expected concept(s)")
    elif concept_delta > 0:
        improvements.append(f"gained {concept_delta} expected concept(s)")
    if coverage_delta < -0.10:
        regressions.append(
            f"reference coverage fell {abs(coverage_delta) * 100:.1f} pp"
        )
    elif coverage_delta > 0.10:
        improvements.append(
            f"reference coverage rose {coverage_delta * 100:.1f} pp"
        )
    if repetition_delta > 0.10:
        regressions.append(
            f"repetition rose {repetition_delta * 100:.1f} pp"
        )
    elif repetition_delta < -0.10:
        improvements.append(
            f"repetition fell {abs(repetition_delta) * 100:.1f} pp"
        )

    if not str(cpt_result.get("generated_response") or "").strip():
        return _assessment("Needs Improvement", "CPT generated an empty response.")
    if regressions:
        return _assessment("Needs Improvement", "; ".join(regressions) + ".")
    if improvements:
        return _assessment("Good", "; ".join(improvements) + ".")
    return _assessment(
        "Satisfactory",
        "No material automatic improvement or regression was detected.",
    )


def _build_generated_text_report(report: dict) -> dict | None:
    """Create a compact run-query-style response and verdict artifact."""
    base_evaluation = report.get("models", {}).get("base", {}).get(
        "query_evaluation"
    )
    if not base_evaluation:
        return None

    cpt_evaluation = report.get("models", {}).get("cpt", {}).get(
        "query_evaluation"
    )
    cpt_by_key = {}
    if cpt_evaluation:
        cpt_by_key = {
            (result["query_file"], str(result["id"])): result
            for result in cpt_evaluation["results"]
        }

    results = []
    verdict_counts = Counter()
    for base_result in base_evaluation["results"]:
        key = (base_result["query_file"], str(base_result["id"]))
        cpt_result = cpt_by_key.get(key)
        record = {
            "query_file": base_result["query_file"],
            "query_set": base_result["query_set"],
            "id": base_result["id"],
            "category": base_result.get("category"),
            "prompt": base_result["prompt"],
            "expected_concepts": base_result.get("expected_concepts", []),
            "expected_continuation": base_result.get("expected_continuation"),
            "base_response": base_result["generated_response"],
            "base_generation_metrics": base_result["generation_metrics"],
            "cpt_response": (
                cpt_result["generated_response"] if cpt_result else None
            ),
            "cpt_generation_metrics": (
                cpt_result["generation_metrics"] if cpt_result else None
            ),
            "verdict": None,
            "remarks": "CPT model was not supplied.",
        }
        if cpt_result:
            assessment = _generated_pair_verdict(base_result, cpt_result)
            record.update(assessment)
            verdict_counts[assessment["verdict"]] += 1
        results.append(record)

    return {
        "schema_version": 1,
        "title": "Part 2 Generated Text Evaluation",
        "created_at_utc": report.get("created_at_utc"),
        "generation_configuration": base_evaluation["generation_configuration"],
        "models": {
            label: {
                "requested_model_name": value["requested_model_name"],
                "model_folder": value["model_folder"],
                "model_details": value["model_details"],
            }
            for label, value in report.get("models", {}).items()
        },
        "automatic_verdict_method": (
            "Good indicates a measurable concept, reference-coverage, or "
            "repetition improvement without a material regression. "
            "Satisfactory indicates no material automatic change. Needs "
            "Improvement indicates lost concepts, over 10 percentage points "
            "lower reference coverage, over 10 points more repetition, or an "
            "empty CPT response. Human factual review is still required."
        ),
        "summary": {
            "query_count": len(results),
            "verdict_counts": dict(verdict_counts),
            "comparison_by_query_set": (
                report.get("comparison", {}).get("queries", {})
                if report.get("comparison")
                else {}
            ),
        },
        "results": results,
    }


def _save_model_response_files(
    report: dict,
    output_file: Path,
    label: str,
) -> list[str]:
    """Save run-query-style response JSON files grouped by query source."""
    model_result = report.get("models", {}).get(label)
    if not model_result or not model_result.get("query_evaluation"):
        return []

    query_evaluation = model_result["query_evaluation"]
    grouped = defaultdict(list)
    for result in query_evaluation["results"]:
        # Retain the original query metadata and expose the generated text as
        # ``response``, matching files written by run_gpt2_query.py.
        excluded = {
            "query_file",
            "query_set",
            "generated_response",
            "generation_metrics",
            "fixed_reference_field",
            "fixed_reference",
            "fixed_reference_conditional_ppl",
            "baseline_response_target",
            "baseline_response_retention_ppl",
        }
        response_record = {
            key: value for key, value in result.items() if key not in excluded
        }
        response_record["response"] = result["generated_response"]
        response_record["generation_metrics"] = result["generation_metrics"]
        grouped[result["query_file"]].append(response_record)

    saved_paths = []
    for query_file, results in sorted(grouped.items()):
        query_stem = Path(query_file).stem
        response_path = output_file.with_name(
            f"{output_file.stem}_{label}_{query_stem}_responses.json"
        )
        response_report = {
            "evaluation_stage": "pre_cpt" if label == "base" else "post_cpt",
            "timestamp_utc": report.get("completed_at_utc"),
            "model_name": model_result["model_details"]["name_or_path"],
            "model_details": model_result["model_details"],
            "query_source": query_file,
            "generation_configuration": query_evaluation[
                "generation_configuration"
            ],
            "total_prompts": len(results),
            "results": results,
        }
        response_path.write_text(
            json.dumps(
                _json_compatible(response_report),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        saved_paths.append(str(response_path))
    return saved_paths


def _save_section_artifacts(report: dict, output_file: Path) -> dict:
    """Save focused Part 1 and Part 2 JSON files beside the full report."""
    output_file = output_file.expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    perplexity_path = output_file.with_name(
        f"{output_file.stem}_perplexity.json"
    )
    generated_text_path = output_file.with_name(
        f"{output_file.stem}_generated_text.json"
    )

    perplexity_report = {
        "schema_version": 1,
        "title": "Part 1 Perplexity Evaluation",
        "created_at_utc": report.get("created_at_utc"),
        "domain_test_bin": report.get("inputs", {}).get("domain_test_bin"),
        "generic_test_bin": report.get("inputs", {}).get("generic_test_bin"),
        "models": {
            label: {
                "requested_model_name": value["requested_model_name"],
                "model_folder": value["model_folder"],
                "model_details": value["model_details"],
                "held_out_perplexity": value["held_out_perplexity"],
            }
            for label, value in report.get("models", {}).items()
        },
        "comparison": (
            report.get("comparison", {}).get("held_out", {})
            if report.get("comparison")
            else {}
        ),
    }
    perplexity_path.write_text(
        json.dumps(_json_compatible(perplexity_report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    generated_text_report = _build_generated_text_report(report)
    if generated_text_report is not None:
        generated_text_path.write_text(
            json.dumps(
                _json_compatible(generated_text_report),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    else:
        generated_text_path = None

    response_files = {
        label: _save_model_response_files(report, output_file, label)
        for label in ("base", "cpt")
    }

    return {
        "full_evaluation_report": str(output_file),
        "part_1_perplexity_evaluation": str(perplexity_path),
        "part_2_generated_text_evaluation": (
            str(generated_text_path) if generated_text_path else None
        ),
        "generated_response_files": response_files,
    }


def _save_report(report: dict, output_file: Path) -> Path:
    output_file = output_file.expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(_json_compatible(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_file


def _evaluate_model(
    label: str,
    model_name: str,
    model_folder: Path | None,
    args,
    query_records: list[dict],
    baseline_targets: dict,
    domain_metrics: dict | None,
    generic_metrics: dict | None,
    expected_model_details: dict | None = None,
) -> dict:
    print(f"\n=== Evaluating {label} model ===")
    model_dict, details = _load_selected_model(model_name, model_folder)
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    device = model_dict["device"]
    if expected_model_details is not None:
        _validate_comparable_models(expected_model_details, details)

    held_out = {}
    print("\nHeld-out domain perplexity")
    held_out["domain"] = evaluate_held_out_perplexity(
        model,
        tokenizer,
        device,
        args.test_bin,
        context_length=args.context_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        metrics=domain_metrics,
    )

    if args.generic_test_bin is not None:
        print("\nHeld-out generic perplexity")
        held_out["generic"] = evaluate_held_out_perplexity(
            model,
            tokenizer,
            device,
            args.generic_test_bin,
            context_length=args.context_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            metrics=generic_metrics,
        )

    query_evaluation = evaluate_queries(
        model,
        tokenizer,
        device,
        query_records,
        baseline_targets,
        max_new_tokens=args.max_new_tokens,
        generation_batch_size=args.generation_batch_size,
    )
    return {
        "label": label,
        "requested_model_name": model_name,
        "model_folder": (
            str(model_folder.expanduser().resolve())
            if model_folder is not None
            else None
        ),
        "model_details": details,
        "held_out_perplexity": held_out,
        "query_evaluation": query_evaluation,
    }


def _release_model_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _response_preview(value: str | None, limit: int = 64) -> str:
    compact = " ".join(str(value or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _print_summary(report: dict) -> None:
    print("\n=== Evaluation Summary ===")
    for label, result in report["models"].items():
        print(f"{label.upper()} model: {result['model_details']['name_or_path']}")

    comparison = report.get("comparison")
    if not comparison:
        print("\nA CPT model was not supplied; comparative tables are unavailable.")
        return

    print("\n=== Part 1: Perplexity Evaluation ===")
    perplexity_rows = []
    for dataset, values in comparison["held_out"].items():
        predicted_tokens = report["models"]["base"]["held_out_perplexity"][
            dataset
        ]["evaluated_token_count"]
        perplexity_rows.append(
            [
                f"{dataset} held-out PPL",
                f"{values['base']:.4f}",
                f"{values['cpt']:.4f}",
                f"{values['relative_change_percent']:+.2f}%",
                f"{predicted_tokens:,}",
                values["verdict"],
                values["remarks"],
            ]
        )
    _print_table(
        ["Metric", "Base", "CPT", "Change", "Tokens", "Verdict", "Remarks"],
        perplexity_rows,
    )

    print("\n=== Part 2: Generated Text Evaluation ===")
    metric_rows = []
    for query_set, values in comparison["queries"].items():
        concept = values["generation_concept_recall"]
        repetition = values["generation_repeated_trigram_ratio"]
        base_concept = concept["base"] or 0.0
        cpt_concept = concept["cpt"] or 0.0
        metric_rows.append(
            [
                f"{query_set} concept recall",
                f"{100 * base_concept:.1f}%",
                f"{100 * cpt_concept:.1f}%",
                f"{100 * (cpt_concept - base_concept):+.1f} pp",
                concept["verdict"],
                concept["remarks"],
            ]
        )
        metric_rows.append(
            [
                f"{query_set} repetition",
                f"{100 * repetition['base']:.1f}%",
                f"{100 * repetition['cpt']:.1f}%",
                (
                    f"{100 * (repetition['cpt'] - repetition['base']):+.1f} pp"
                ),
                repetition["verdict"],
                repetition["remarks"],
            ]
        )
        exact = values["generation_exact_continuation"]
        if exact["query_count"]:
            exact_assessment = exact["exact_match_assessment"]
            metric_rows.append(
                [
                    f"{query_set} exact continuations",
                    f"{exact['base_matches']}/{exact['query_count']}",
                    f"{exact['cpt_matches']}/{exact['query_count']}",
                    f"{exact['cpt_matches'] - exact['base_matches']:+d}",
                    exact_assessment["verdict"],
                    exact_assessment["remarks"],
                ]
            )
            coverage_assessment = exact["continuation_coverage_assessment"]
            metric_rows.append(
                [
                    f"{query_set} continuation coverage",
                    f"{100 * exact['base_average_token_coverage']:.1f}%",
                    f"{100 * exact['cpt_average_token_coverage']:.1f}%",
                    (
                        f"{100 * (exact['cpt_average_token_coverage'] - exact['base_average_token_coverage']):+.1f} pp"
                    ),
                    coverage_assessment["verdict"],
                    coverage_assessment["remarks"],
                ]
            )
        reference = values.get("fixed_reference_conditional_ppl")
        if reference:
            metric_rows.append(
                [
                    f"{query_set} fixed-reference PPL",
                    f"{reference['base']:.4f}",
                    f"{reference['cpt']:.4f}",
                    f"{reference['relative_change_percent']:+.2f}%",
                    reference["verdict"],
                    reference["remarks"],
                ]
            )
        retention = values.get("baseline_response_retention_ppl")
        if retention:
            metric_rows.append(
                [
                    f"{query_set} baseline-retention PPL",
                    f"{retention['base']:.4f}",
                    f"{retention['cpt']:.4f}",
                    f"{retention['relative_change_percent']:+.2f}%",
                    retention["verdict"],
                    retention["remarks"],
                ]
            )
    _print_table(
        ["Metric", "Base", "CPT", "Change", "Verdict", "Remarks"],
        metric_rows,
    )

    generated_text = _build_generated_text_report(report)
    if generated_text:
        print("\nGenerated Text Verdicts")
        verdict_rows = []
        for result in generated_text["results"]:
            verdict_rows.append(
                [
                    result["query_set"],
                    str(result["id"]),
                    _response_preview(result["prompt"], 45),
                    _response_preview(result["base_response"]),
                    _response_preview(result["cpt_response"]),
                    result["verdict"] or "Not Available",
                    result["remarks"],
                ]
            )
        _print_table(
            [
                "Set",
                "ID",
                "Prompt",
                "Base response",
                "CPT response",
                "Verdict",
                "Remarks",
            ],
            verdict_rows,
        )


def run_evaluation(args) -> dict:
    test_bin = args.test_bin.expanduser().resolve()
    output_file = args.output_file.expanduser().resolve()
    domain_metrics, domain_metrics_path = _load_metrics(args.dataset_metrics)
    generic_metrics, generic_metrics_path = _load_metrics(
        args.generic_dataset_metrics
    )

    if args.generic_dataset_metrics is not None and args.generic_test_bin is None:
        raise ValueError(
            "--generic-dataset-metrics requires --generic-test-bin"
        )
    query_records = _load_queries(args.query_input)
    if args.baseline_responses and not query_records:
        raise ValueError("--baseline-responses requires --query-input")
    baseline_targets, baseline_provenance = _load_baseline_targets(
        args.baseline_responses,
        query_records,
    )

    command_argv = [sys.executable, *sys.argv]
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": {
            "argv": command_argv,
            "reconstructed_full_command": shlex.join(command_argv),
            "working_directory": str(Path.cwd()),
        },
        "cli_options": vars(args).copy(),
        "methodology": {
            "held_out_perplexity": (
                "Token-weighted exp(mean next-token negative log-likelihood) "
                "over fixed packed sequences."
            ),
            "fixed_reference_perplexity": (
                "Conditional perplexity over fixed reference response tokens; "
                "prompt labels are masked."
            ),
            "baseline_response_retention": (
                "The same saved base response is scored under both models; "
                "this measures behavioural drift, not factual correctness."
            ),
            "generation": (
                "Deterministic greedy generation scored for expected concepts, "
                "exact continuation, token overlap, and repeated trigrams."
            ),
            "qualitative_assessment_thresholds": {
                "domain_ppl": (
                    "Good: at least 10% reduction; Satisfactory: smaller "
                    "reduction; Needs Improvement: no reduction."
                ),
                "generic_ppl_and_baseline_retention": (
                    "Good: no more than 5% increase; Satisfactory: no more "
                    "than 10% increase; Needs Improvement: over 10% increase."
                ),
                "concept_recall": (
                    "Good: CPT recall at least 60% without a decline over 2 "
                    "percentage points; Satisfactory: at least 30% without a "
                    "decline over 5 points; otherwise Needs Improvement."
                ),
                "repetition": (
                    "Good: repeated-trigram ratio at most 10%; Satisfactory: "
                    "at most 30%; otherwise Needs Improvement."
                ),
                "exact_match_and_coverage": (
                    "Good: at least 70%; Satisfactory: at least 30%; "
                    "otherwise Needs Improvement."
                ),
                "fixed_reference_ppl": (
                    "Good: at least 10% reduction; Satisfactory: smaller "
                    "reduction; Needs Improvement: no reduction."
                ),
                "per_query_generated_text": (
                    "Good: a measurable concept, reference-coverage, or "
                    "repetition improvement with no material regression; "
                    "Satisfactory: no material automatic change; Needs "
                    "Improvement: lost concepts, reference coverage down over "
                    "10 percentage points, repetition up over 10 points, or "
                    "an empty CPT response."
                ),
                "note": (
                    "Verdicts are transparent reporting heuristics, not "
                    "formal statistical significance tests."
                ),
            },
        },
        "inputs": {
            "domain_test_bin": str(test_bin),
            "domain_dataset_metrics": domain_metrics_path,
            "generic_test_bin": (
                str(args.generic_test_bin.expanduser().resolve())
                if args.generic_test_bin is not None
                else None
            ),
            "generic_dataset_metrics": generic_metrics_path,
            "query_files": sorted(
                {record["query_file"] for record in query_records}
            ),
            "baseline_response_sources": baseline_provenance,
            "matched_baseline_response_count": len(baseline_targets),
        },
        "models": {},
        "comparison": None,
    }

    base_result = _evaluate_model(
        "base",
        args.model_name,
        None,
        args,
        query_records,
        baseline_targets,
        domain_metrics,
        generic_metrics,
    )
    report["models"]["base"] = base_result
    _save_report(report, output_file)

    # Release the base checkpoint before loading the CPT checkpoint. This keeps
    # the comprehensive comparison viable for GPT-2 Large on one GPU.
    del base_result
    _release_model_memory()

    if args.model_folder is not None:
        cpt_result = _evaluate_model(
            "cpt",
            args.model_name,
            args.model_folder,
            args,
            query_records,
            baseline_targets,
            domain_metrics,
            generic_metrics,
            expected_model_details=report["models"]["base"]["model_details"],
        )
        report["models"]["cpt"] = cpt_result
        report["comparison"] = _build_comparison(report)

    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["artifacts"] = _save_section_artifacts(report, output_file)
    report["evaluation_sections"] = {
        "part_1_perplexity_evaluation": {
            "comparison": (
                report.get("comparison", {}).get("held_out", {})
                if report.get("comparison")
                else {}
            ),
            "json_file": report["artifacts"][
                "part_1_perplexity_evaluation"
            ],
        },
        "part_2_generated_text_evaluation": {
            "comparison_by_query_set": (
                report.get("comparison", {}).get("queries", {})
                if report.get("comparison")
                else {}
            ),
            "json_file": report["artifacts"][
                "part_2_generated_text_evaluation"
            ],
        },
    }
    _save_report(report, output_file)
    _print_summary(report)
    print(f"\nComplete evaluation report saved to: {output_file}")
    print(
        "Part 1 perplexity JSON saved to: "
        f"{report['artifacts']['part_1_perplexity_evaluation']}"
    )
    generated_path = report["artifacts"]["part_2_generated_text_evaluation"]
    if generated_path:
        print(f"Part 2 generated-text JSON saved to: {generated_path}")
    response_files = report["artifacts"]["generated_response_files"]
    print(
        "Run-query-style response JSONs saved beside the report: "
        f"{len(response_files['base'])} base, "
        f"{len(response_files['cpt'])} CPT"
    )
    return report


def main() -> int:
    parser = build_cpt_evaluation_parser()
    args = parser.parse_args()
    try:
        run_evaluation(args)
    except KeyboardInterrupt:
        parser.exit(130, "\nEvaluation interrupted by user.\n")
    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
