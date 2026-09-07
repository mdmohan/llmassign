# Continual Pre-Training Scripts

These scripts prepare domain text, continue training standard Hugging Face
decoder-only causal language models, and compare the base and trained models.
This includes GPT-2, SmolLM2, TinyLlama, and compatible Qwen checkpoints. Run
the commands from the `llmassign` repository root.

The paths used by the current commands are:

```text
1a/
├── code/                         # Python scripts
├── data/
│   ├── raw/pdfs/cisco-dc-pdfs/  # Cisco data-center source PDFs
│   ├── raw/pdfs/curated-networking/
│   ├── raw/text/                 # OSPF-related source RFC text
│   ├── converted/cisco-dc/       # Text extracted from Cisco PDFs
│   ├── converted/networking/     # Text extracted from other PDFs
│   ├── cleaned/cleaned/          # Current training-ready text
│   ├── cleaned-post/             # Optional standalone post-clean output
│   ├── processed/<model>/        # Model-tokenizer-specific packed dataset
│   ├── processed/v1/             # Earlier Cisco packed dataset
│   ├── processed/v2_ospf_rfc/    # OSPF RFC packed dataset
│   ├── processed/v3/             # Newer combined dataset
│   └── reports/                  # Cleaning audit reports
├── evaluation/
│   ├── queries/                  # Domain and forgetting query JSON
│   ├── gpt2/                     # GPT-2 evaluation outputs
│   └── gpt2-medium/              # GPT-2 Medium evaluation outputs
├── output/<model>/<run>/          # CPT checkpoints and final models
└── notebooks/                    # Exploration and CPT notebooks
```

Training creates the model/checkpoint directory specified with `--save-dir`,
so a model output directory is not present until CPT has been run.

## Setup

The main dependencies are `torch`, `transformers`, `numpy`, `pdfplumber`,
`langdetect`, `datasketch`, and `pyarrow`. `matplotlib` is optional and is used
to save the training-loss graph.

All Hugging Face downloads use one cache directory. Override its default before
running a command if necessary:

```bash
export CACHE_DIR=/home/jovyan/llmgenai/hf_cache
```

Use `--help` with any command to see every option and its default value.

## Workflow

Use the Python commands in the sections below directly. `run.sh` is an
optional convenience wrapper, but it is not required and is not used by this
guide. The normal order is dataset preparation, baseline generation, CPT
training, and comprehensive evaluation.

## 1. Prepare the CPT dataset

`prepare_cpt_dataset.py` recursively loads PDF and text files, extracts PDF
text, and applies the required quality and deduplication gates. A separate
post-gate stage then removes TOCs, legal/page furniture, low-information
tables, diagrams, and obvious extraction noise while preserving prose, useful
tables, and CLI examples. Finally, it packs the selected model's token stream
into fixed-length sequences and writes binary and Parquet outputs. Pass the
Hugging Face model ID unchanged; project-specific aliases are not expanded.

```bash
python 1a/code/prepare_cpt_dataset.py \
  --model-name gpt2-large \
  --input-dir 1a/data/raw \
  --text-dir 1a/data/converted \
  --audit-report 1a/data/reports/cleaning_audit.json \
  --content-audit-report 1a/data/reports/content_cleaning_audit.json \
  --cleaned-text-dir 1a/data/cleaned \
  --bin-file 1a/data/processed/gpt2/tokens.bin \
  --parquet-file 1a/data/processed/gpt2/tokens.parquet \
  --metrics-file 1a/data/processed/gpt2/dataset_metrics.json \
  --context-length 1024 \
  --train-test-split 90:10 \
  --split-seed 42 \
  --max-workers 2 \
  --extract-tables \
  --x-tolerance 1.5 \
  --y-tolerance 3.0 \
  --max-tasks-per-child 64 \
  --min-chars 50 \
  --max-duplicate-paragraph-ratio 0.30 \
  --target-language en \
  --dedup-strategy both \
  --similarity-threshold 0.85 \
  --minhash-permutations 128 \
  --content-cleaning \
  --parquet-batch-size 1024 \
  --parquet-compression zstd
```

The input may be one `.pdf` or `.txt` file, or a directory containing both.
Use `--max-workers 1` if PDF extraction causes high memory usage or worker
locking. The original cleaning audit records all document-gate decisions. The
separate content-cleaning audit reports characters and blocks removed by each
post-gate rule. It is also the consolidated final report: it lists every
retained file, every gate rejection, and every post-cleaning rejection while
preserving duplicate lineage to the retained original. Use
`--no-content-cleaning` to skip only the post-gate stage. Boolean options also
support `--no-extract-tables` when their default behaviour is not wanted.

To tokenize an already-cleaned text corpus without running PDF extraction,
quality gates, deduplication, or content cleaning, use `--tokenize-only`. This
mode reads only `.txt` files from `--input-dir` and goes directly to the
existing split, tokenization, and serialization stages:

```bash
python 1a/code/prepare_cpt_dataset.py \
  --tokenize-only \
  --model-name gpt2-large \
  --input-dir 1a/data/cleaned/cleaned \
  --bin-file 1a/data/processed/gpt2/tokens.bin \
  --parquet-file 1a/data/processed/gpt2/tokens.parquet \
  --metrics-file 1a/data/processed/gpt2/dataset_metrics.json \
  --context-length 1024 \
  --train-test-split 90:10 \
  --split-seed 42 \
  --parquet-batch-size 1024 \
  --parquet-compression zstd
```

Extraction and cleaning options are ignored in token-only mode, and no
cleaning audit or copied text is written. The dataset metrics record
`"preparation_mode": "tokenize_only"`.

When `--train-test-split 90:10` or `80:20` is supplied, complete documents are
split before tokenization. The parent folders from `--bin-file`,
`--parquet-file`, and `--metrics-file` are used to write:

```text
token_train.bin                 token_test.bin
token_train.parquet             token_test.parquet
dataset_metrics_train.json      dataset_metrics_test.json
```

Without `--train-test-split`, the exact filenames supplied on the command line
are retained. This preserves compatibility with earlier prepared datasets.

To prepare the existing OSPF RFC corpus instead, use
`--input-dir 1a/data/raw/text`,
`--bin-file 1a/data/processed/v2_ospf_rfc/rfc_ospf.bin`, and
`--parquet-file 1a/data/processed/v2_ospf_rfc/rfc_ospf.parquet`.

For SmolLM2, select its tokenizer with the complete Hugging Face ID and use
separate output files. The model's 8,192-token maximum is used unless
`--context-length` specifies a smaller packing length:

```bash
python 1a/code/prepare_cpt_dataset.py \
  --model-name HuggingFaceTB/SmolLM2-360M \
  --input-dir 1a/data/raw/pdfs/cisco-dc-pdfs \
  --text-dir 1a/data/converted/cisco-dc \
  --audit-report 1a/data/reports/smollm2_cleaning_audit.json \
  --bin-file 1a/data/processed/smollm2-360m/tokens.bin \
  --parquet-file 1a/data/processed/smollm2-360m/tokens.parquet \
  --context-length 2048 \
  --max-workers 2
```

Every supported model writes `dataset_metrics.json` beside the binary file by
default. It records the tokenizer, context length, total token count, average
document length, packed sequence count, residual tokens, maximum tokenizer ID,
document-boundary policy, vocabulary mapping fingerprint, and selected binary
dtype. Token IDs use `uint16`, `uint32`, or `uint64`, choosing the narrowest
safe representation. Use
`--metrics-file` to choose a different path. Datasets prepared with different
tokenizers are not interchangeable.

Context length is read from common model configuration fields. If the detected
maximum exceeds 8,192 tokens, provide a practical `--context-length` explicitly
instead of accidentally packing 32K or 128K sequences. EOS separates documents;
if a tokenizer has no EOS token, supply a one-token boundary with
`--document-separator-token`.

## 2. Capture a pre-CPT baseline

`baseline.py` accepts one or more query JSON files and writes a response file
for each one. The output records the model identity, parameter count,
generation settings, and query source.

The evaluation query directory contains four complementary sets:

- `generic_forgetting_queries.json`: 15 unchanged general-knowledge prompts.
- `domain_exact_generation_queries.json`: 10 verbatim source continuations
  with the expected continuation and source file recorded.
- `domain_generalization_queries.json`: 10 newly worded scenarios that test
  whether domain concepts can be applied beyond memorized source language.
- `domain_baseline_queries.json`: the original 20 broad domain prompts.

Passing the query directory to `baseline.py` runs all four sets.

```bash
python 1a/code/baseline.py \
  1a/evaluation/queries \
  --model-name gpt2-medium \
  --baseline-path 1a/evaluation/gpt2-medium/baseline \
  --max-new-tokens 50 \
  --batch-size 4 \
  --evaluation-stage pre_cpt
```

The positional query inputs may be individual JSON files, directories, or a
mixture. A directory contributes all immediate `*.json` files in sorted order.

If `--model-name` is omitted, the default is `gpt2`. A query file must contain:

```json
{
  "queries": [
    {
      "id": "dc-001",
      "category": "data-center",
      "prompt": "BGP is responsible for",
      "expected_concepts": ["routing", "autonomous systems"]
    }
  ]
}
```

Use `--model-name HuggingFaceTB/SmolLM2-360M` and a separate `--baseline-path`
to capture the SmolLM2 pre-CPT baseline. Other compatible models use their
exact Hub IDs in the same command.

## 3. Run continued pre-training

`cpt_train.py` reads the binary dtype from the dataset metrics, memory-maps the
packed token file, validates tokenizer/context compatibility, runs a small
teacher-forced causal-LM check, and trains the selected model. It writes
numbered checkpoints, a rolling `last_checkpoint`, a
`final_model`, training history JSON, and—when `matplotlib` is installed—a loss
curve. It also writes `training_run.json` directly under `--save-dir`. That file
records every parsed CLI option (including defaults), the reconstructed full
command, resolved input/cache paths, model and runtime details, all training
hyperparameters, and calculated values such as effective batch size, warmup
steps, and total optimizer updates. It is created before training and updated
with final results when training completes.

```bash
python 1a/code/cpt_train.py \
  --model-name gpt2-medium \
  --bin-file 1a/data/processed/gpt2/token_train.bin \
  --dataset-metrics 1a/data/processed/gpt2/dataset_metrics_train.json \
  --save-dir 1a/output/gpt2-medium/v1 \
  --context-length 1024 \
  --batch-size 4 \
  --num-workers 0 \
  --gradient-accumulation-steps 8 \
  --epochs 3 \
  --learning-rate 5e-6 \
  --weight-decay 0.01 \
  --warmup-ratio 0.05 \
  --max-grad-norm 1.0 \
  --log-every-steps 10 \
  --save-every-steps 500 \
  --seed 42
```

For `gpt2-medium`, start with a smaller batch size because it needs
substantially more GPU memory than `gpt2`. Keep `--num-workers 0` when notebook
or CUDA multiprocessing is unstable.

Generic causal-LM training uses BF16 on a compatible GPU, FP16 with loss scaling
on older CUDA GPUs, and FP32 on CPU. Gradient checkpointing is enabled when the
loaded architecture supports it. The training context length must match the
length used during preparation:

```bash
python 1a/code/cpt_train.py \
  --model-name HuggingFaceTB/SmolLM2-360M \
  --bin-file 1a/data/processed/smollm2-360m/token_train.bin \
  --dataset-metrics 1a/data/processed/smollm2-360m/dataset_metrics_train.json \
  --save-dir 1a/output/smollm2-360m/v1 \
  --context-length 2048 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --epochs 1 \
  --learning-rate 5e-6 \
  --num-workers 0
```

The trainer checks `dataset_metrics.json` when present to prevent accidental use
of data from another tokenizer or binary layout. Metrics are required for
`uint32` and `uint64` datasets; without them the legacy `uint16` layout is
assumed. Start with a 1-sequence microbatch when memory use is uncertain.

## 4. Query a model interactively

Run one prompt or a comma-separated set of prompts against the original
Hugging Face model:

```bash
python 1a/code/run_gpt2_query.py \
  "What is OSPF?,Suggest a Cisco data-center switch" \
  --model-name gpt2-medium \
  --max-new-tokens 100
```

Use a local CPT output, `final_model`, `last_checkpoint`, or `checkpoint-N`:

```bash
python 1a/code/run_gpt2_query.py \
  "How do I configure OSPF on a Nexus switch?" \
  --model-folder 1a/output/gpt2-medium/v1/final_model \
  --max-new-tokens 100
```

When `--model-folder` is supplied, it takes precedence over `--model-name`.
Comma-separated prompts may be quoted as one command-line argument. CSV-style
quoting is supported when a prompt itself contains a comma.

To keep the model loaded and enter prompts repeatedly, use `--cli`. Enter
`exit` when finished:

```bash
python 1a/code/run_gpt2_query.py \
  --cli \
  --model-folder 1a/output/gpt2-medium/v1/final_model \
  --max-new-tokens 100
```

To run one or more existing evaluation query files against a CPT model and
save baseline-style JSON results, use `--query-json` and `--output-dir`:

```bash
python 1a/code/run_gpt2_query.py \
  --query-json 1a/evaluation/queries \
  --output-dir 1a/evaluation/gpt2-medium/v1 \
  --evaluation-stage post_cpt \
  --model-folder 1a/output/gpt2-medium/v1/final_model \
  --max-new-tokens 100 \
  --batch-size 4
```

Exactly one input mode may be used at a time: positional prompts, `--cli`, or
`--query-json`. JSON query mode writes one response file per query file and
includes model details, generation settings, and source-query information.

The same command supports any compatible base model and automatically loads the
architecture recorded by a local checkpoint:

```bash
python 1a/code/run_gpt2_query.py \
  "How is OSPF configured on a Nexus switch?" \
  --model-name HuggingFaceTB/SmolLM2-360M \
  --max-new-tokens 100

python 1a/code/run_gpt2_query.py \
  --cli \
  --model-name HuggingFaceTB/SmolLM2-360M \
  --model-folder 1a/output/smollm2-360m/v1/final_model \
  --max-new-tokens 100
```

## 5. Score and compare response folders

`compare_responses.py` prints a per-query table and an aggregate summary for a
response folder. It reports lexical expected-concept recall, exact source
continuations, continuation prefix/coverage, and repeated trigram rate:

```bash
python 1a/code/compare_responses.py \
  1a/evaluation/gpt2-large/baseline
```

After CPT, compare aligned baseline and post-CPT response folders:

```bash
python 1a/code/compare_responses.py \
  1a/evaluation/gpt2-large/baseline \
  --compare-dir 1a/evaluation/gpt2-large/post-cpt \
  --query-dir 1a/evaluation/queries \
  --response-preview-chars 72 \
  --show-all \
  --show-responses
```

Comparison mode prints the expected concepts, baseline response, post-CPT
response, concept hit counts, and a plain-language description side by side
for every query. Use `--response-preview-chars` to control column length.

The lexical scores are reproducible screening metrics. Review responses
manually for semantic correctness when the model uses valid paraphrases.

## 6. Run the comprehensive CPT evaluation

`evaluate_cpt.py` evaluates a compatible base model and an optional CPT
checkpoint in one run. It reads each held-out binary dtype from its metrics and
calculates token-weighted PPL on the same held-out domain stream,
conditional PPL on fixed query references, PPL of the same saved baseline
responses for behavioural-retention analysis, and deterministic greedy-response
quality metrics:

```bash
python 1a/code/evaluate_cpt.py \
  --model-name gpt2-medium \
  --model-folder 1a/output/gpt2-medium/v1/final_model \
  --test-bin 1a/data/processed/gpt2/token_test.bin \
  --dataset-metrics 1a/data/processed/gpt2/dataset_metrics_test.json \
  --context-length 1024 \
  --batch-size 8 \
  --num-workers 0 \
  --query-input 1a/evaluation/queries \
  --baseline-responses 1a/evaluation/gpt2-medium/baseline \
  --max-new-tokens 100 \
  --generation-batch-size 4 \
  --reuse-existing \
  --output-file 1a/evaluation/gpt2-medium/v1/cpt_evaluation.json
```

When a separately prepared general-language holdout exists, add
`--generic-test-bin` and `--generic-dataset-metrics`. Domain PPL should decrease;
little or no increase in generic PPL indicates limited catastrophic forgetting.
`--baseline-responses` is optional. It accepts pre-CPT response JSON files or
directories and enables behavioural-retention PPL; the evaluator generates
fresh base and CPT responses even when this option is omitted.

Add `--reuse-existing` to avoid repeating an expensive evaluation. The report
is reused only when the completed output contains the same base model, exact
CPT checkpoint manifest and `training_run.json`, held-out data, query and
baseline-response file contents, and evaluation settings. A missing,
incomplete, legacy, or mismatched report is recomputed normally.

The main output JSON includes per-query and aggregate metrics plus dataset
hashes, model provenance, methodology, CLI settings, verdicts, remarks, and the
reconstructed command. Two focused files are written beside it:

```text
cpt_evaluation_perplexity.json
cpt_evaluation_generated_text.json
```

Run-query-style base and CPT response JSON files are also written beside the
main report for every supplied query file. Terminal output is split into Part
1 Perplexity Evaluation and Part 2 Generated Text Evaluation, followed by a
per-query verdict table. Model architecture details and the CPT training
hyperparameters are printed first. The terminal verdict table intentionally
omits prompt and response text to stay readable; all generated text remains in
the full report and generated-text JSON artifact.

Immediately after the perplexity table, the evaluator also prints a
10-query catastrophic-forgetting table from the generic base/CPT responses
already stored in the main report. It marks a query `Degraded` when CPT loses an
expected concept found by the base model or raises PPL on the identical saved
base response by more than 10%; otherwise it reports `Retained`. The table has
only Query, Base, CPT, and Verdict columns, with wrapped text for presentation.
This is a derived terminal view and does not add another evaluation pass or
JSON format.

Fixed-reference query PPL is calculated only for records containing
`expected_continuation`, `expected_response`, `reference_answer`, or
`expected_answer`. The current exact-generation queries qualify automatically;
the generic queries continue to use concept recall and, when supplied, saved
baseline-response retention PPL.

## 7. Evaluate corpus OOV and token fragmentation

`evaluate_oov.py` recursively reads every `.txt` file under the supplied corpus
folder and loads only the selected tokenizer. It reports true unknown-token
rate plus pseudo-OOV: the percentage of whitespace-separated words requiring
three or more tokenizer pieces.

```bash
python 1a/code/evaluate_oov.py \
  --input-dir 1a/data/cleaned/cleaned \
  --model-name gpt2-large \
  --recursive \
  --output-file 1a/evaluation/gpt2-large/oov_report.json \
  --top-files 15
```

To inspect a tokenizer saved with a local model, add `--model-folder`. CPT does
not normally change the tokenizer, so a base checkpoint and its CPT checkpoint
should have identical OOV results. GPT-2 byte-level BPE normally has 0% true
OOV; pseudo-OOV and average pieces per word are the more useful domain metrics.
Use `--no-recursive` to process only text files directly inside the input
directory.

## 8. Post-clean an existing corpus separately

`post_clean_corpus.py` is an optional, isolated experiment. It reads an
already-cleaned text directory, writes the result elsewhere, and never modifies
the source files. It targets URLs, encoded payloads, fingerprints, long hashes,
packed lists, and high-confidence joined headings while preserving CLI
indentation and tables.

```bash
python 1a/code/post_clean_corpus.py \
  --input-dir 1a/data/cleaned/cleaned \
  --output-dir 1a/data/cleaned-post \
  --audit-report 1a/data/reports/post_cleaning_audit.json \
  --recursive \
  --remove-urls \
  --remove-encoded-data \
  --split-packed-lists \
  --repair-joined-headings \
  --drop-suspicious-lines \
  --min-retained-chars 50
```

Each boolean rule is enabled by default. The disabling forms are
`--no-recursive`, `--no-remove-urls`, `--no-remove-encoded-data`,
`--no-split-packed-lists`, `--no-repair-joined-headings`, and
`--no-drop-suspicious-lines`. Use `--overwrite` only when intentionally updating
an existing non-empty experimental output directory. Run `evaluate_oov.py` on
both the original and post-cleaned directories to compare tokenization quality.

## Supporting modules

The remaining files are imported by the commands above:

- `load_pdf.py`: PDF extraction and direct loading of text files.
- `clean_data.py`: normalization, quality gates, deduplication, and audit data.
- `training_content_cleaner.py`: post-gate block cleanup and its detailed audit.
- `causal_lm.py`: generic AutoConfig, AutoTokenizer, and AutoModel loading,
  metadata, context/dtype detection, and compatibility checks.
- `tokenize_data.py`: generic causal-LM tokenization, sequence packing, binary,
  Parquet, and metrics output; `tokenize_gpt` remains a compatibility wrapper.
- `tokenize_smollm2.py`: retained legacy SmolLM2-specific helper.
- `load_tensors.py`: dtype-aware memory-mapped PyTorch dataset and data loader.
- `gpt2_model.py`: base GPT-2 model/tokenizer loading.
- `smollm2_model.py`: retained legacy SmolLM2 model helper.
- `cpt_train_smollm2.py`: retained legacy SmolLM2 training helper.
- `load_local_model.py`: local CPT model and checkpoint resolution.
- `model_response.py`: batched generation and response JSON helpers.
- `evaluate_cpt.py`: held-out PPL, query PPL, retention, and generation evaluation.
- `evaluate_oov.py`: true OOV and pseudo-OOV fragmentation analysis.
- `cache.py`: shared Hugging Face cache directory.
- `cli_parsers.py`: shared command-line argument definitions.

The generic path covers models accepted by both `AutoTokenizer` and
`AutoModelForCausalLM` that implement standard teacher-forced causal LM loss.
It does not cover encoder-only, encoder-decoder, multimodal, unsupported custom-
code, or training-incompatible quantized models. GPT-2 sizes share a tokenizer;
models with different tokenizers require separately prepared datasets.
