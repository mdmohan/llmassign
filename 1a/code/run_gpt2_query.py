"""Run one or more prompts against a supported causal language model."""

from __future__ import annotations

import csv

from baseline import (
    CODE_DIR,
    _add_model_details,
    _import_existing_module,
    _model_details,
    _safe_filename_component,
    resolve_query_paths,
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
        chat=args.chat,
    )


def add_inference_details(details, model_dict, args):
    """Record adapter and prompt-format provenance with generated output."""
    details["adapter_source"] = model_dict.get("adapter_source")
    details["chat_mode"] = args.chat
    details["chat_template_source"] = model_dict.get(
        "chat_template_source"
    )
    return details


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


def run_json_queries(
    response_helpers,
    model_dict,
    args,
    model_details_function=_model_details,
) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    device = model_dict["device"]
    details = add_inference_details(
        model_details_function(model, tokenizer, device),
        model_dict,
        args,
    )
    details["cache_dir"] = str(CACHE_DIR)
    model_label = _safe_filename_component(details["name_or_path"])
    seen_output_paths = set()

    query_paths = resolve_query_paths(args.query_json)
    print(f"Resolved {len(query_paths)} query JSON file(s).")
    for query_path in query_paths:
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
            chat=args.chat,
        )

        # Generation may configure a padding token. Record the exact
        # post-generation model and tokenizer details in the result file.
        generation_details = add_inference_details(
            model_details_function(model, tokenizer, device),
            model_dict,
            args,
        )
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
        from causal_lm import load_causal_lm

        response_helpers = _import_existing_module(
            "query_model_response_helpers",
            CODE_DIR / "model_response.py",
            {"torch": torch},
        )

        model_dict = load_causal_lm(
            model_name=args.model_name,
            model_folder=args.model_folder,
            adapter_folder=args.adapter_folder,
        )
        if args.chat:
            template_source = response_helpers.ensure_chat_template(
                model_dict["tokenizer"]
            )
            model_dict["chat_template_source"] = template_source
            if template_source == "model":
                print("Chat mode: using the tokenizer's chat template.")
            else:
                print(
                    "Chat mode: tokenizer has no chat template; using the "
                    "generic User/Assistant template."
                )
        else:
            model_dict["chat_template_source"] = None
            print("Chat mode: disabled; prompts are passed through unchanged.")
        model_details_function = _model_details

        if args.cli:
            run_cli(response_helpers, model_dict, args)
            return 0

        if args.query_json is not None:
            run_json_queries(
                response_helpers,
                model_dict,
                args,
                model_details_function=model_details_function,
            )
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
