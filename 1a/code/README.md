# CPT Scripts

These scripts prepare domain text, continue training a GPT-2 model, and compare
the base and trained models. Run the commands from the `llmassign` repository
root.

The current project layout is:

```text
1a/
├── code/                         # Python scripts
├── data/
│   ├── pdfs/cisco-dc-pdfs/      # Cisco data-center PDFs
│   ├── text/rfc/                 # OSPF-related RFC text files
│   ├── processed/v1/             # Cisco packed dataset
│   ├── processed/v2_ospf_rfc/    # OSPF RFC packed dataset
│   ├── processed/v3/             # Newer combined dataset
│   └── reports/                  # Cleaning audit reports
├── evaluation/
│   ├── queries/                  # Domain and forgetting query JSON
│   ├── gpt2/                     # GPT-2 evaluation outputs
│   └── gpt2-medium/              # GPT-2 Medium evaluation outputs
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

## 1. Prepare the CPT dataset

`prepare_cpt_dataset.py` recursively loads PDF and text files, extracts PDF
text, cleans and deduplicates the documents, packs the GPT-2 token stream into
1,024-token sequences, and writes binary and Parquet outputs.

```bash
python 1a/code/prepare_cpt_dataset.py \
  --input-dir 1a/data/pdfs/cisco-dc-pdfs \
  --text-dir 1a/data/text/cisco-dc \
  --audit-report 1a/data/reports/cleaning_audit_report.json \
  --bin-file 1a/data/processed/v1/tokens.bin \
  --parquet-file 1a/data/processed/v1/tokens.parquet \
  --max-workers 2 \
  --extract-tables \
  --dedup-strategy both \
  --similarity-threshold 0.85 \
  --parquet-compression zstd
```

The input may be one `.pdf` or `.txt` file, or a directory containing both.
Use `--max-workers 1` if PDF extraction causes high memory usage or worker
locking. The cleaning audit JSON records retained and rejected documents.

To prepare the existing OSPF RFC corpus instead, use
`--input-dir 1a/data/text/rfc`,
`--bin-file 1a/data/processed/v2_ospf_rfc/rfc_ospf.bin`, and
`--parquet-file 1a/data/processed/v2_ospf_rfc/rfc_ospf.parquet`.

## 2. Capture a pre-CPT baseline

`baseline.py` accepts one or more query JSON files and writes a response file
for each one. The output records the model identity, parameter count,
generation settings, and query source.

```bash
python 1a/code/baseline.py \
  1a/evaluation/queries/domain_baseline_queries.json \
  1a/evaluation/queries/generic_forgetting_queries.json \
  --model-name gpt2-medium \
  --baseline-path 1a/evaluation/gpt2-medium/baseline \
  --evaluation-stage pre_cpt
```

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

## 3. Run continued pre-training

`cpt_train.py` memory-maps the packed binary token file and trains the selected
GPT-2 model. It writes numbered checkpoints, a rolling `last_checkpoint`, a
`final_model`, training history JSON, and—when `matplotlib` is installed—a loss
curve.

```bash
python 1a/code/cpt_train.py \
  --model-name gpt2-medium \
  --bin-file 1a/data/processed/v1/tokens.bin \
  --save-dir 1a/models/gpt2-medium/v1 \
  --context-length 1024 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --epochs 1 \
  --learning-rate 5e-6 \
  --num-workers 0
```

For `gpt2-medium`, start with a smaller batch size because it needs
substantially more GPU memory than `gpt2`. Keep `--num-workers 0` when notebook
or CUDA multiprocessing is unstable.

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
  --model-folder 1a/models/gpt2-medium/v1 \
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
  --model-folder 1a/models/gpt2-medium/v1 \
  --max-new-tokens 100
```

To run one or more existing evaluation query files against a CPT model and
save baseline-style JSON results, use `--query-json` and `--output-dir`:

```bash
python 1a/code/run_gpt2_query.py \
  --query-json \
    1a/evaluation/queries/domain_baseline_queries.json \
    1a/evaluation/queries/generic_forgetting_queries.json \
  --output-dir 1a/evaluation/gpt2-medium/v1 \
  --evaluation-stage post_cpt \
  --model-folder 1a/models/gpt2-medium/v1 \
  --max-new-tokens 100 \
  --batch-size 4
```

Exactly one input mode may be used at a time: positional prompts, `--cli`, or
`--query-json`. JSON query mode writes one response file per query file and
includes model details, generation settings, and source-query information.

## Supporting modules

The remaining files are imported by the commands above:

- `load_pdf.py`: PDF extraction and direct loading of text files.
- `clean_data.py`: normalization, quality gates, deduplication, and audit data.
- `tokenize_data.py`: GPT-2 tokenization, sequence packing, binary, and Parquet output.
- `load_tensors.py`: memory-mapped PyTorch dataset and data loader.
- `gpt2_model.py`: base GPT-2 model/tokenizer loading.
- `load_local_model.py`: local CPT model and checkpoint resolution.
- `model_response.py`: batched generation and response JSON helpers.
- `cache.py`: shared Hugging Face cache directory.
- `cli_parsers.py`: shared command-line argument definitions.

The packed dataset uses the GPT-2 tokenizer, which is shared by `gpt2` and
`gpt2-medium`.
