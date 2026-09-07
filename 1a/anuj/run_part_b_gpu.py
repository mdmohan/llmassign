"""
Part B on GPU — true QLoRA (4-bit NF4) instruction fine-tuning.

GPU counterpart of run_part_b_ads.py. Same 3 adapter configs (A/B/C) and the
same quantitative comparison, but the frozen base is loaded in 4-bit via
bitsandbytes (requires CUDA).

Usage:
    python scripts/run_part_b_gpu.py                       # uses cpt_model/
    python scripts/run_part_b_gpu.py --model-src cpt_model --max-steps 300
"""
import argparse, inspect, json, math, os, time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

ap = argparse.ArgumentParser()
ap.add_argument("--model-src", default="cpt_model",
                help="Part A CPT checkpoint, or a HuggingFace model id")
ap.add_argument("--jsonl", default="instruction_dataset.jsonl")
ap.add_argument("--adapter-dir", default="adapters")
ap.add_argument("--max-steps", type=int, default=300)
ap.add_argument("--seq-len", type=int, default=1024)
ap.add_argument("--batch-size", type=int, default=2)
ap.add_argument("--grad-accum", type=int, default=4)
ap.add_argument("--lr", type=float, default=2e-4)
args = ap.parse_args()

if not torch.cuda.is_available():
    raise SystemExit("CUDA not available. QLoRA 4-bit needs a GPU. "
                     "On CPU use scripts/run_part_b_ads.py instead.")
print("GPU:", torch.cuda.get_device_name(0),
      f"| {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

MODEL_SRC = args.model_src
Path(args.adapter_dir).mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

# ---- tokenizer + chat template --------------------------------------------
tok = AutoTokenizer.from_pretrained(MODEL_SRC)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
if tok.chat_template is None:
    tok.chat_template = ("{% for m in messages %}{% if m['role']=='user' %}"
        "<|user|>\n{{ m['content'] }}\n<|assistant|>\n"
        "{% else %}{{ m['content'] }}{{ eos_token }}{% endif %}{% endfor %}")

ds = load_dataset("json", data_files=args.jsonl)["train"]
train_ds = ds.filter(lambda r: r["split"] == "train")
eval_ds = ds.filter(lambda r: r["split"] == "eval")
print(f"train={len(train_ds)} eval={len(eval_ds)}")

def format_row(r):
    return tok.apply_chat_template(
        [{"role": "user", "content": r["instruction"]},
         {"role": "assistant", "content": r["response"]}], tokenize=False)

train_txt = train_ds.map(lambda r: {"text": format_row(r)})

# ---- 4-bit quantization config (the "Q" in QLoRA) --------------------------
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

ADAPTERS = {
    "A": dict(r=8,  alpha=16, targets=["q_proj", "v_proj"]),
    "B": dict(r=16, alpha=32, targets=["q_proj", "v_proj"]),
    "C": dict(r=32, alpha=32, targets=["q_proj", "v_proj", "o_proj"]),
}

def make_sft_config(**kw):
    """Tolerate the trl max_seq_length -> max_length rename across versions."""
    valid = set(inspect.signature(SFTConfig.__init__).parameters)
    if "max_seq_length" in kw and "max_seq_length" not in valid:
        kw["max_length"] = kw.pop("max_seq_length")
    return SFTConfig(**{k: v for k, v in kw.items() if k in valid or k == "output_dir"})

def load_4bit():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_SRC, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto")

# ---- held-out eval loss ----------------------------------------------------
eval_texts = [format_row(r) for r in eval_ds][:60]

@torch.no_grad()
def eval_loss(model):
    model.eval(); tl, tt = 0.0, 0
    for t in eval_texts:
        ids = tok(t, return_tensors="pt", truncation=True,
                  max_length=args.seq_len)["input_ids"].to(model.device)
        if ids.numel() < 2:
            continue
        out = model(input_ids=ids, labels=ids)
        tl += out.loss.item() * (ids.numel() - 1); tt += ids.numel() - 1
    return tl / tt

EVAL_PROMPTS = [
    "How do I configure a VLAN on a Catalyst 9300 switch?",
    "What are the restrictions for configuring BGP on Nexus 9000?",
    "How does Cisco ISE posture assessment work?",
]

@torch.no_grad()
def answer(model, prompt, n=120):
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt").to(model.device)
    out = model.generate(ids, max_new_tokens=n, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

# ---- baseline: CPT model before instruction tuning -------------------------
base = load_4bit()
base_loss = eval_loss(base)
base_ans = {p: answer(base, p) for p in EVAL_PROMPTS}
print(f"\n[baseline] CPT (pre-SFT) eval loss = {base_loss:.4f} "
      f"(ppl {math.exp(base_loss):.2f})")
del base; torch.cuda.empty_cache()

# ---- train the 3 adapters --------------------------------------------------
results = {"model_src": MODEL_SRC, "base_eval_loss": base_loss, "adapters": {}}
for name, spec in ADAPTERS.items():
    t0 = time.time()
    print(f"\n=== Adapter {name}: r={spec['r']} alpha={spec['alpha']} "
          f"targets={spec['targets']} ===")
    model = load_4bit()
    peft_cfg = LoraConfig(r=spec["r"], lora_alpha=spec["alpha"], lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules=spec["targets"])
    out = f"{args.adapter_dir}/adapter_{name}"
    cfg = make_sft_config(output_dir=out, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, warmup_steps=10, logging_steps=20,
        save_strategy="no", bf16=True, report_to="none",
        max_seq_length=args.seq_len, dataset_text_field="text",
        gradient_checkpointing=True)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_txt,
                         peft_config=peft_cfg)
    trainer.train()
    trainer.model.save_pretrained(out); tok.save_pretrained(out)

    el = eval_loss(trainer.model)
    ans = {p: answer(trainer.model, p) for p in EVAL_PROMPTS}
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    peak = torch.cuda.max_memory_allocated() / 1e9
    results["adapters"][name] = {"eval_loss": el, "ppl": math.exp(el),
        "trainable_params": trainable, "minutes": (time.time() - t0) / 60,
        "peak_vram_gb": peak, "answers": ans}
    print(f"[{name}] eval loss {base_loss:.4f} -> {el:.4f} "
          f"({100*(base_loss-el)/base_loss:+.1f}%)  trainable={trainable:,}  "
          f"peak VRAM={peak:.1f} GB  {(time.time()-t0)/60:.1f} min")
    del model, trainer
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

# ---- report ----------------------------------------------------------------
print("\n" + "=" * 78)
print("PART B (GPU / 4-bit QLoRA) — instruction eval loss, lower = better")
print("=" * 78)
print(f"{'Model':<28}{'eval loss':>10}{'ppl':>9}{'vs CPT':>9}{'trainable':>13}{'VRAM':>8}")
print(f"{'CPT (pre-SFT, Part A)':<28}{base_loss:>10.4f}{math.exp(base_loss):>9.2f}"
      f"{'--':>9}{'--':>13}{'--':>8}")
for name, d in results["adapters"].items():
    print(f"{'Adapter ' + name + ' (r=' + str(ADAPTERS[name]['r']) + ')':<28}"
          f"{d['eval_loss']:>10.4f}{d['ppl']:>9.2f}"
          f"{100*(base_loss-d['eval_loss'])/base_loss:>+8.1f}%"
          f"{d['trainable_params']:>13,}{d['peak_vram_gb']:>7.1f}G")

print("\n" + "=" * 78)
print("QUALITATIVE BEFORE/AFTER (Adapter C)")
print("=" * 78)
for p in EVAL_PROMPTS:
    print(f"\nQ: {p}")
    print(f"  CPT base : {base_ans[p][:200]}")
    print(f"  Adapter C: {results['adapters']['C']['answers'][p][:200]}")

json.dump(results, open("data/part_b_gpu_results.json", "w"), indent=2)
print("\nSaved -> data/part_b_gpu_results.json | adapters -> "
      f"{args.adapter_dir}/adapter_{{A,B,C}}/")
