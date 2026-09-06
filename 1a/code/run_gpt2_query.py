"""Run one or more prompts against the existing GPT-2 model loader."""

from __future__ import annotations

import csv

from baseline import CODE_DIR, _import_existing_module
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


def main() -> int:
    parser = build_gpt2_query_parser()
    args = parser.parse_args()

    if args.cli and args.prompts is not None:
        parser.error("the prompts argument cannot be used with --cli")
    if not args.cli and args.prompts is None:
        parser.error("the prompts argument is required unless --cli is used")

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
