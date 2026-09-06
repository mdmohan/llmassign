"""Run one or more prompts against the existing GPT-2 model loader."""

from __future__ import annotations

import csv

from baseline import (
    CODE_DIR,
    _add_model_details,
    _import_existing_module,
    _model_details,
    _safe_filename_component,
)
from cache import CACHE_DIR
from cli_parsers import build_gpt2_query_parser


def parse_prompts(value: str) -> list[str]:
    prompts = [prompt.strip() for prompt in next(csv.reader([value]))]
    prompts = [prompt for prompt in prompts if prompt]
    if not prompts:
        raise ValueError("At least one non-empty prompt is required")
    return prompts


def generate(response_helpers, model_dict, prompts, args):
    return response_helpers.generate_responses(
        model=model_dict["model"],
        tokenizer=model_dict["tokenizer"],
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )


def run_cli(response_helpers, model_dict, args) -> None:
    print("\nInteractive mode. Enter 'exit' to quit.")

    while True:
        try:
            prompt = input("\nPrompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        if prompt.casefold() == "exit":
            print("Exiting.")
            return
        if not prompt:
            continue

        response = generate(
            response_helpers,
            model_dict,
            [prompt],
            args,
        )[0]
        print(f"Response: {response}")


def run_json_queries(response_helpers, model_dict, args) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    device = model_dict["device"]
    details = _model_details(model, tokenizer, device)
    details["cache_dir"] = str(CACHE_DIR)
    model_label = _safe_filename_component(details["name_or_path"])
    seen_output_paths = set()

    for supplied_query_path in args.query_json:
        query_path = supplied_query_path.expanduser().resolve()
        if not query_path.is_file():
            raise ValueError(f"Query JSON file does not exist: {query_path}")

        output_path = (
            output_dir
            / f"{model_label}_{query_path.stem}_responses.json"
        )
        if output_path in seen_output_paths:
            raise ValueError(
                "Multiple query files would write to the same output: "
                f"{output_path}"
            )
        seen_output_paths.add(output_path)

        query_records, prompts = response_helpers.load_prompt_json(query_path)
        responses = generate(
            response_helpers,
            model_dict,
            prompts,
            args,
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

        # Generation may configure GPT-2's padding token. Record the exact
        # post-generation model and tokenizer details in the result file.
        generation_details = _model_details(model, tokenizer, device)
        generation_details["cache_dir"] = str(CACHE_DIR)
        _add_model_details(
            output_path=output_path,
            query_path=query_path,
            model_details=generation_details,
        )
        print(f"Query results saved to: {output_path}")


def main() -> int:
    parser = build_gpt2_query_parser()
    args = parser.parse_args()

    selected_modes = sum(
        (
            args.prompts is not None,
            args.cli,
            args.query_json is not None,
        )
    )
    if selected_modes != 1:
        parser.error(
            "choose exactly one input mode: prompts, --cli, or --query-json"
        )
    if args.query_json is not None and args.output_dir is None:
        parser.error("--output-dir is required with --query-json")
    if args.query_json is None and args.output_dir is not None:
        parser.error("--output-dir can only be used with --query-json")

    try:
        import torch

        response_helpers = _import_existing_module(
            "query_model_response_helpers",
            CODE_DIR / "model_response.py",
            {"torch": torch},
        )

        if args.model_folder is None:
            from gpt2_model import load_gpt2_model

            model_dict = load_gpt2_model(model_name=args.model_name)
        else:
            from load_local_model import load_local_model

            model, tokenizer, device = load_local_model(args.model_folder)
            model_dict = {
                "model": model,
                "tokenizer": tokenizer,
                "device": device,
            }

        if args.cli:
            run_cli(response_helpers, model_dict, args)
            return 0

        if args.query_json is not None:
            run_json_queries(response_helpers, model_dict, args)
            return 0

        prompts = parse_prompts(args.prompts)
        responses = generate(response_helpers, model_dict, prompts, args)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    for index, (prompt, response) in enumerate(
        zip(prompts, responses),
        start=1,
    ):
        print(f"\n[{index}] Prompt: {prompt}")
        print(f"Response: {response}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
