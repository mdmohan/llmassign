
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


@torch.inference_mode()
def generate_responses(
    model,
    tokenizer,
    prompts,
    max_new_tokens=50,
    batch_size=4):
    """
    Generate deterministic completions and decode only the model response,
    excluding the original prompt.
    """
    if not prompts:
        return []

    model.eval()
    device = next(model.parameters()).device

    # Many decoder-only tokenizers do not define a padding token by default.
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Batched generation requires a tokenizer PAD or EOS token"
            )
        tokenizer.pad_token = tokenizer.eos_token

    # Left padding is preferred for batched decoder-only generation.
    tokenizer.padding_side = "left"

    from causal_lm import model_context_limit

    model_context_length = model_context_limit(model.config, tokenizer)
    if model_context_length is None:
        raise ValueError("Unable to determine the model context length")

    max_prompt_length = model_context_length - max_new_tokens
    if max_prompt_length <= 0:
        raise ValueError(
            "max_new_tokens must be smaller than the model context length"
        )

    responses = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]

        encoded_inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
            return_attention_mask=True,
        )

        encoded_inputs = {
            key: value.to(device)
            for key, value in encoded_inputs.items()
        }

        generated_ids = model.generate(
            **encoded_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # All input rows have the same padded width. Everything after this
        # position is newly generated model output.
        prompt_width = encoded_inputs["input_ids"].shape[1]
        response_token_ids = generated_ids[:, prompt_width:]

        batch_responses = tokenizer.batch_decode(
            response_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        responses.extend(response.strip() for response in batch_responses)

    return responses

# Load baseline prompts from JSON


def load_prompt_json(json_path):
    """
    Load prompt records and create a simple prompt array.

    Returns:
        records: Complete query records, including IDs and expected concepts.
        prompts: List of prompt strings for model generation.
    """
    json_path = Path(json_path)

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    queries = data.get("queries")
    if not isinstance(queries, list):
        raise ValueError(f"'queries' must be a list in {json_path}")

    records = []
    prompts = []

    for index, query in enumerate(queries):
        prompt = query.get("prompt")

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"Query at index {index} does not contain a valid prompt"
            )

        records.append(query)
        prompts.append(prompt.strip())

    print(f"Loaded {len(prompts)} prompts from: {json_path}")

    return records, prompts

def create_result_records(query_records, responses):
    """
    Combine each query with its generated response and evaluation metadata.
    """
    if len(query_records) != len(responses):
        raise ValueError(
            "The number of query records and responses must match"
        )

    results = []

    for query, response in zip(query_records, responses):
        # Preserve evaluation-specific fields such as expected_continuation,
        # evaluation_type, and source_file in the generated response report.
        results.append({**query, "response": response})

    return results

def save_generation_results(
    results,
    output_path,
    model,
    max_new_tokens,
    evaluation_stage="pre_cpt",
):
    """
    Save generated responses and generation settings as formatted JSON.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "evaluation_stage": evaluation_stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model.config._name_or_path,
        "generation_configuration": {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "decoding": "greedy",
        },
        "total_prompts": len(results),
        "results": results,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            output_data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved {len(results)} responses to: {output_path}")
