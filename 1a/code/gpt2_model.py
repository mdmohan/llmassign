import torch

from cache import CACHE_DIR


def load_gpt2_model(model_name="gpt2"):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    
    # 1. Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Load pre-trained GPT-2 Tokenizer and Model
    tokenizer = GPT2TokenizerFast.from_pretrained(
        model_name,
        cache_dir=CACHE_DIR,
    )
    model = GPT2LMHeadModel.from_pretrained(
        model_name,
        cache_dir=CACHE_DIR,
    )
    
    
    # 3. Move model to GPU/CPU
    model.to(device)
    
    # Enable gradient checkpointing to save VRAM if training on larger context sizes
    # model.gradient_checkpointing_enable()
    
    print(f"Successfully loaded {model_name} with {sum(p.numel() for p in model.parameters()):,} parameters.")
    
    return { 
        "model": model, 
        "tokenizer": tokenizer, 
        "device": device}


# Pull one batch from your previously defined train_dataloader
def check_model(dataloader, model, device):
   # Display the loaded model's architecture and parameter details

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    
    print("\n" + "=" * 55)
    print("                 MODEL INFORMATION")
    print("=" * 55)
    print(f"Model name/path       : {model.config._name_or_path}")
    print(f"Model architecture    : {model.config.model_type}")
    print(f"Transformer layers    : {model.config.n_layer}")
    print(f"Attention heads       : {model.config.n_head}")
    print(f"Embedding dimension   : {model.config.n_embd}")
    print(f"Vocabulary size       : {model.config.vocab_size:,}")
    print(f"Maximum context length: {model.config.n_positions:,}")
    print(f"Total parameters      : {total_parameters:,}")
    print(f"Trainable parameters  : {trainable_parameters:,}")
    print("=" * 55 + "\n")
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
    
        # GPT-2 computes cross-entropy loss automatically when labels are passed
        outputs = model(input_ids=input_ids, labels=labels)
        
        loss = outputs.loss
        logits = outputs.logits
    
        print(f"Initial Loss: {loss.item():.4f}")
        print(f"Logits Shape: {logits.shape}")  # [batch_size, sequence_length, vocab_size]
        break

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

    # GPT-2 does not define a padding token by default.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Left padding is preferred for batched decoder-only generation.
    tokenizer.padding_side = "left"

    model_context_length = getattr(
        model.config,
        "n_positions",
        tokenizer.model_max_length,
    )

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
