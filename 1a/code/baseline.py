"""Generate reproducible baseline responses for one or more query files.

The actual model loading and response-generation operations are imported from
gpt2_model.py and model_response.py. This file only provides the command-line
orchestration and records complete model details in each baseline JSON file.

Example:
    python 1a/code/baseline.py \
        1a/data/evaluation/domain_baseline_queries.json \
        1a/data/evaluation/generic_forgetting_queries.json \
        --baseline-path baselines
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

from cache import CACHE_DIR
from cli_parsers import build_baseline_parser


CODE_DIR = Path(__file__).resolve().parent


def _import_existing_module(
    module_name: str,
    path: Path,
    injected_globals: dict,
) -> ModuleType:
    """Import a notebook-oriented helper with its expected global values."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import helper module from {path}")

    module = importlib.util.module_from_spec(spec)
    module.__dict__.update(injected_globals)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _model_details(model, tokenizer, device) -> dict:
    """Return the exact model and tokenizer identity used for generation."""
    config = model.config
    parameters = list(model.parameters())
    total_parameters = sum(parameter.numel() for parameter in parameters)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in parameters
        if parameter.requires_grad
    )

    return {
        "name_or_path": config._name_or_path,
        "model_class": type(model).__name__,
        "model_type": config.model_type,
        "architectures": getattr(config, "architectures", None),
        "transformer_layers": getattr(config, "n_layer", None),
        "attention_heads": getattr(config, "n_head", None),
        "embedding_dimension": getattr(config, "n_embd", None),
        "vocabulary_size": getattr(config, "vocab_size", None),
        "maximum_context_length": getattr(config, "n_positions", None),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "parameter_dtype": str(parameters[0].dtype) if parameters else None,
        "device": str(device),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_name_or_path": tokenizer.name_or_path,
        "tokenizer_vocabulary_size": tokenizer.vocab_size,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }


def _safe_filename_component(value: str) -> str:
    name = Path(value).name or value
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._")
    return normalized or "model"


def _add_model_details(
    output_path: Path,
    query_path: Path,
    model_details: dict,
) -> None:
    """Add complete provenance to JSON created by save_generation_results."""
    with output_path.open("r", encoding="utf-8") as file:
        saved_data = json.load(file)

    enriched_data = {
        "evaluation_stage": saved_data["evaluation_stage"],
        "timestamp_utc": saved_data["timestamp_utc"],
        "model_name": saved_data["model_name"],
        "model_details": model_details,
        "query_source": str(query_path),
        "generation_configuration": saved_data["generation_configuration"],
        "total_prompts": saved_data["total_prompts"],
        "results": saved_data["results"],
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(enriched_data, file, indent=2, ensure_ascii=False)


def run_baselines(args) -> None:
    import torch
    from gpt2_model import load_gpt2_model

    baseline_path = args.baseline_path.expanduser().resolve()
    baseline_path.mkdir(parents=True, exist_ok=True)

    response_helpers = _import_existing_module(
        "baseline_model_response_helpers",
        CODE_DIR / "model_response.py",
        {"torch": torch},
    )

    model_dict = load_gpt2_model(model_name=args.model_name)
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    device = model_dict["device"]
    details = _model_details(model, tokenizer, device)
    details["cache_dir"] = str(CACHE_DIR)
    model_label = _safe_filename_component(details["name_or_path"])

    print("\nModel used for baseline generation:")
    print(json.dumps(details, indent=2))

    seen_output_paths: set[Path] = set()
    for supplied_query_path in args.query_json:
        query_path = supplied_query_path.expanduser().resolve()
        if not query_path.is_file():
            raise ValueError(f"Query JSON file does not exist: {query_path}")

        output_path = (
            baseline_path
            / f"{model_label}_{query_path.stem}_responses.json"
        )
        if output_path in seen_output_paths:
            raise ValueError(
                "Multiple query files would write to the same output: "
                f"{output_path}"
            )
        seen_output_paths.add(output_path)

        query_records, prompts = response_helpers.load_prompt_json(query_path)
        responses = response_helpers.generate_responses(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
        results = response_helpers.create_result_records(
            query_records,
            responses,
        )
        response_helpers.save_generation_results(
            results=results,
            output_path=output_path,
            model=model,
            max_new_tokens=args.max_new_tokens,
            evaluation_stage=args.evaluation_stage,
        )
        # generate_responses configures GPT-2's padding token when necessary.
        # Capture details afterward so the saved tokenizer state is exact.
        generation_model_details = _model_details(model, tokenizer, device)
        generation_model_details["cache_dir"] = str(CACHE_DIR)
        _add_model_details(
            output_path=output_path,
            query_path=query_path,
            model_details=generation_model_details,
        )

        print(f"Baseline with model details saved to: {output_path}")


def main() -> int:
    parser = build_baseline_parser()
    args = parser.parse_args()
    try:
        run_baselines(args)
    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
