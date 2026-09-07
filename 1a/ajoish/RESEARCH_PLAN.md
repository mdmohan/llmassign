# Ajoish SmolLM2 Cisco CPT Research Plan

## 1. Study Contract

This work is an independent implementation for comparison with Murali's research.
It must not import, modify, or derive implementation logic or generated artifacts
from the existing `1a/code`, `1a/data/processed`, `1a/evaluation`, or
`1a/notebooks` directories.

### Immutable inputs

- `1a/data/pdfs/cisco-dc-pdfs`
- `1a/data/pdfs/cisco-enterprise-pdfs`

The two directories above are the only shared project inputs. They are treated as
read-only. There are 91 PDF paths: 65 Data Center PDFs and 26 Enterprise PDFs,
totalling 389,502,858 bytes before cleaning.

### Independent workspace

All independently authored code and generated artifacts live under `1a/ajoish`:

```text
1a/ajoish/
├── code/
├── configs/
├── data/
│   ├── extracted/
│   ├── processed/
│   └── reports/
├── evaluation/
│   ├── baseline/
│   ├── perplexity/
│   └── queries/
├── models/
├── notebooks/
├── runs/
└── RESEARCH_PLAN.md
```

Large generated data, model weights, caches, and checkpoints remain on persistent
Kubeflow storage. Small code, configurations, reports, metrics, and notebooks are
versioned in Git. A `.gitignore` will enforce that boundary before generation.

The Git branch for this implementation is `ajoish/smollm2-cpt`.

### Frozen model

- Hugging Face ID: `HuggingFaceTB/SmolLM2-1.7B`
- Checkpoint class: base `LlamaForCausalLM`, not chat/instruct
- License: Apache-2.0
- Exact published parameter count: 1,711,376,384
- Pretraining: 11T tokens in BF16
- Context window: 8,192 tokens
- Vocabulary: 49,152 tokens
- BOS/EOS token: `<|endoftext|>` at shared ID 0
- Architecture: 24 decoder layers, 32 attention heads, 32 key-value heads,
  hidden size 2,048, intermediate size 8,192
- Expected head dimension: 2,048 / 32 = 64

There is no official 1.1B checkpoint in the SmolLM or SmolLM2 families; their
published sizes are 135M, 360M, and 1.7B. TinyLlama was initially selected because
it exactly matched the requested 1.1B scale and the assignment's example list.
SmolLM2-1.7B supersedes it based on stronger direct evidence: on the same held-out
Cisco sample, SmolLM2-1.7B achieved 0.803286 bits/byte versus TinyLlama-1.1B's
0.888542, a 9.6% lower cross-tokenizer-normalized value. Both exhausted the old
T4 at about 14.17 GiB, so that result does not distinguish A100 feasibility. The
validated A100-SXM4-80GB provides enough expected headroom for a measured BF16
full-parameter CPT smoke test. If that smoke test fails its memory gate, reduce
micro-batch size and use gradient accumulation/checkpointing; do not silently
switch models or substitute parameter-efficient CPT.

The exact repository revision resolved by Hugging Face will be captured in the run
manifest. The same revision and tokenizer will be used for extraction-independent
tokenization, baseline inference, CPT, and post-CPT evaluation.

### Reproducibility controls

- One fixed seed, initially `42`, for Python, NumPy, PyTorch, data splitting, and
  Trainer sampling.
- Deterministic document ordering by normalized repository-relative source path.
- Exact duplicate removal before splitting.
- Deterministic 90/10 document-level train/evaluation split after all filters.
- No document may contribute tokens to both splits.
- Base and CPT models use identical held-out sequences, prompts, tokenizer,
  decoding parameters, and evaluation code.
- Every run records Git commit, model revision, package versions, CUDA/GPU facts,
  configuration, input manifest hash, and output checksums.

## 2. Phase 0: Clean Kubeflow Environment

The user explicitly authorized deletion without backup. On 2026-09-07, the test
notebook `cisco-llm-a100` and PVCs `cisco-llm-a100-workspace` and `llmall` were
deleted. The Kubeflow APIs then reported zero notebooks and zero PVCs.

### Naming and path contract

- Notebook: `cisco-cpt-smollm2-1-7b`
- Persistent PVC: `cisco-cpt-assignment-1a`
- PVC workspace mount: `/home/jovyan`
- Project root: `/home/jovyan/cisco-cpt-1a`
- Repository: `/home/jovyan/cisco-cpt-1a/llmassign`
- Hugging Face cache: `/home/jovyan/cisco-cpt-1a/cache/huggingface`
- Packed datasets: `/home/jovyan/cisco-cpt-1a/artifacts/datasets`
- CPT checkpoints: `/home/jovyan/cisco-cpt-1a/artifacts/checkpoints/smollm2-1.7b`
- Final model: `/home/jovyan/cisco-cpt-1a/artifacts/models/cisco-cpt-smollm2-1.7b`
- Runs: `/home/jovyan/cisco-cpt-1a/artifacts/runs`
- Evaluation: `/home/jovyan/cisco-cpt-1a/artifacts/evaluation`

### Live checks

1. Confirm the new `cisco-cpt-assignment-1a` PVC is `Bound` at 32 GiB using
   `nfs-client`. This was accepted and verified on 2026-09-07.
2. Use the live-verified namespace quota of 8 CPU and 16 GiB RAM. Request
   `7900m` CPU and `16256Mi` RAM for the notebook container, reserving the
   observed `100m` CPU and `128Mi` RAM for the sidecar. Request one NVIDIA GPU
   and select the `NVIDIA A100 GPU Node` toleration.
3. Create a fresh notebook using image
   `10.152.183.210:5000/bits-sudo-jupyter-pytorch-cuda-full:v1`, the resources
   above, and the sole 32 GiB `cisco-cpt-assignment-1a` PVC as the workspace
   mounted at `/home/jovyan`.
4. Keep source, datasets, reports, caches, and checkpoints under
   `/home/jovyan/cisco-cpt-1a` on the persistent workspace PVC.
5. Run the repository environment verifier and require CUDA, the A100, PyTorch,
   Transformers, bitsandbytes 4-bit forward, and PEFT LoRA checks to pass.
6. Record actual CPU/RAM/GPU requests and limits from the created notebook API.
7. Stop the notebook whenever active GPU work is not in progress.

On 2026-09-07, the fresh notebook reached `ready` with image tag `:v1`,
`7900m` CPU, `16256Mi` RAM, one `NVIDIA A100-SXM4-80GB`, and the assignment
PVC mounted read-write at `/home/jovyan`. The persistent environment verifier
passed all package, CUDA 12.4, bitsandbytes 4-bit forward, and PEFT LoRA checks.

**Gate 0:** Do not start corpus processing until the requested storage is bound,
the notebook is running with live-verified maximum resources, persistent paths are
writable, and the environment report passes.

## 3. Phase 1: Functionality 1 - PDF Extraction and Cleaning

### Implementation

1. Recursively inventory the two immutable PDF roots.
2. Record source path, source corpus, byte size, and SHA-256 for every PDF.
3. Extract text page-by-page with a standard PDF parser. Preserve page boundaries
   in provenance metadata and report empty or failed pages.
4. Normalize only extraction artifacts that do not alter technical meaning:
   Unicode normalization, line endings, whitespace, and repeated blank lines.
   Preserve commands, interface names, IP addresses, tables where recoverable,
   section numbers, and configuration syntax.
5. Apply the assignment filters in this exact order:
   - Length: reject cleaned documents shorter than 50 characters.
   - Repetition: reject when more than 30% of non-empty normalized paragraphs are
     duplicates within that document.
   - Exact deduplication: keep the first deterministic occurrence in a seen-set.
   - Language: retain English documents only and record detector/confidence.
6. Assign each accepted document to train or evaluation with the fixed seed and
   stratification by source corpus where feasible. The evaluation set is 10% at
   document level.
7. Save one cleaned `.txt` file per accepted document.
8. Build deterministic `domain_corpus.txt`, `train_corpus.txt`, and
   `eval_corpus.txt` concatenations with explicit document boundary markers.

Two known duplicate-content pairs must be resolved by the exact-deduplication
stage:

- `02_nexus9000_nxos_interfaces_10.5x.pdf` and
  `cisco-nexus-9000-series-nx-os-interfaces-configuration-guide-release-105x.pdf`
- `17_vpc_inconsistency_nxos.pdf` and
  `217989-troubleshoot-vpc-inconsistency-issues-on.pdf`

### Artifacts

- `data/extracted/<document>.txt`: raw page-aware extraction
- `data/processed/domain_corpus/<document>.txt`: accepted cleaned documents
- `data/processed/domain_corpus.txt`: deterministic combined accepted corpus
- `data/processed/train_corpus.txt` and `eval_corpus.txt`
- `data/reports/documents.jsonl`: provenance, hash, page counts, filter metrics,
  split, acceptance, and rejection reason per source
- `data/reports/cleaning_report.json`: counts before/after every filter, character
  totals, failures, and greatest-impact filter
- `data/reports/rejected_documents.jsonl`

### Validation

- Inline validation assertions cover threshold boundaries at 49/50 characters,
   exactly/above 30% repeated paragraphs, stable seen-set behavior, language
   decisions, and split disjointness. No dedicated test files are retained.
- Re-run with the same seed and verify manifest and corpus hashes are unchanged.
- Assert all 91 PDF paths appear exactly once in the source manifest.
- Assert duplicate content hashes cannot cross the train/evaluation boundary.
- Manually inspect representative DC and Enterprise extractions, including pages
  with tables, CLI snippets, and headers/footers.

**Definition of done:** all mandatory filter counts and failures are reproducible;
accepted per-document text files and the combined corpus exist; train/evaluation
document sets are disjoint; the greatest-impact filter is identified.

### Measured result

Functionality 1 passed locally on 2026-09-07. All 91 PDFs were inventoried and
extracted page-by-page. The ordered filters removed one document for length, six
for repetition, two exact duplicates, and zero for language, leaving 82 accepted
English documents. The deterministic split contains 74 training documents and 8
held-out evaluation documents. Repetition had the greatest impact. The malformed
`03_nexus9000_nxos_layer2_10.5x.pdf` retained provenance for 223 failed pages and
was rejected by the required length filter. A clean second run produced identical
manifest, rejection-report, cleaning-report, and combined-corpus SHA-256 values.

## 4. Phase 2: Functionality 2 - Tokenization and Packing

### Implementation

1. Load only `AutoTokenizer.from_pretrained(FROZEN_MODEL_ID)` at the frozen
   revision and verify that BOS and EOS are both `<|endoftext|>` at ID 0.
2. Tokenize accepted documents separately with tokenizer-added special tokens
   disabled, then explicitly wrap each document as `[BOS] + tokens + [EOS]`.
3. Process train and evaluation documents independently to prevent leakage.
4. Concatenate each split's document token lists into a flat stream.
5. Slice into non-overlapping 8,192-token sequences with no padding, matching the
   model's published context window.
6. Drop the final incomplete remainder from each split and report dropped-token
   counts. Do not merge train and evaluation remainders.
7. Save packed records as Parquet using a fixed schema.

### Parquet contract

- `sequence_id`: integer
- `input_ids`: list of 8,192 integers
- `attention_mask`: list of 8,192 ones
- `labels`: list equal to `input_ids`

Additional source-span provenance may be stored in a separate sidecar so model
columns remain directly consumable by the Dataset wrapper.

### Artifacts

- `data/processed/packed/train.parquet`
- `data/processed/packed/eval.parquet`
- `data/reports/tokenization_report.json`
- `data/reports/tokenizer_metadata.json`
- source-span sidecars for auditability

### Validation

- Assert every packed row has exactly 8,192 tokens and no padding.
- Assert `labels == input_ids` and each attention-mask value is one.
- Assert every accepted document contributes exactly one BOS and one EOS before
  final-remainder truncation.
- Decode samples spanning document boundaries and inspect BOS/EOS placement.
- Report total token count, mean document token length, sequence count, and
  remainder count separately for train and evaluation splits.

**Definition of done:** deterministic train/evaluation Parquet files pass schema,
length, boundary, no-padding, and split-isolation tests; all rubric statistics are
recorded.

### Measured result

Functionality 2 passed locally on 2026-09-07 using the frozen model repository
revision `effd688a12921b4cc83e3312b6feb579f70f9c71`. The loaded fast tokenizer
has vocabulary size 49,152, context length 8,192, and shared BOS/EOS token
`<|endoftext|>` at ID 0. The 82 accepted documents produced 9,674,276 content
tokens and 9,674,440 tokens after one BOS and EOS per document. Packing produced
1,180 full sequences: 1,088 train sequences and 92 evaluation sequences. The
independent final remainders dropped 1,399 train tokens and 6,481 evaluation
tokens. Every Parquet row has 8,192 tokens, all-one attention masks, and labels
equal to input IDs; all retained boundary positions contain token ID 0. Train and
evaluation document IDs remain disjoint. A clean offline rerun produced identical
Parquet, token-span sidecar, metadata, and report SHA-256 values.

## 5. Phase 3: Functionality 3 - Model Audit and Baseline

### Implementation

1. Load the frozen model with `AutoModelForCausalLM.from_pretrained` in BF16 on
   the A100. Do not quantize the CPT model.
2. Count total and trainable parameters and verify all intended model parameters
   are trainable.
3. Read architecture facts from the loaded config and compute head dimension.
4. Assert `lm_head.out_features == config.vocab_size == 49152`.
5. Freeze exactly 50 evaluation prompts before CPT: 25 generic and 25 Cisco.
   Every Cisco record includes source document, page number, evidence split, and
   a short evidence excerpt from an accepted PDF.
6. Run deterministic base inference with frozen generation settings and save raw
   prompts, tokenized inputs, all 50 outputs, timing, and configuration.
7. Compute and save base-model loss/perplexity on the frozen held-out Parquet
   before any optimizer is constructed.

### Artifacts

- `evaluation/queries/frozen_prompts.jsonl`
- `evaluation/queries/prompt_manifest.json`
- `evaluation/baseline/architecture_audit.json`
- `evaluation/baseline/tokenizer_metadata.json`
- `evaluation/baseline/generation_config.json`
- `evaluation/baseline/responses.jsonl`
- `evaluation/baseline/base_perplexity.json`
- `evaluation/baseline/base_per_sequence.csv`
- `evaluation/baseline/baseline_run.json`

### Validation

- Assert checkpoint/model revision and tokenizer revision match the run manifest.
- Assert expected architecture: 24 layers, 32 attention heads, 32 key-value heads,
  hidden size 2,048, head dimension 64, and vocabulary/head output 49,152.
- Re-run baseline generation and confirm identical generated token IDs under
   deterministic decoding; elapsed-time measurements may differ.
- Confirm every expected domain answer is traceable to accepted source evidence.

**Definition of done:** architecture facts, parameter counts, head assertion, all
50 prompts and baseline outputs, and base held-out perplexity are frozen before
CPT.

### Measured result

Functionality 3 passed on the Kubeflow NVIDIA A100-SXM4-80GB on 2026-09-07.
The model loaded in BF16 from frozen revision
`effd688a12921b4cc83e3312b6feb579f70f9c71`. The architecture audit confirmed
24 layers, 32 attention heads, 32 key-value heads, hidden size 2,048, head
dimension 64, context length 8,192, and vocabulary/head output size 49,152. All
1,711,376,384 parameters were trainable. Deterministic inference completed for
all 50 frozen prompts. Evaluation over all 92 held-out packed sequences covered
753,572 predicted tokens and measured mean loss 1.3511853472370168, corresponding
to base-model perplexity 3.862000630396992. The frozen prompt SHA-256 is
`46a7fbcd2a68f8c6afd0e93a152372b2bdff3d070b450603bdcce7d87c94ca22`.

## 6. Phase 4: Functionality 4 - Full Continual Pre-Training

### Initial configuration

- Precision: BF16
- Context length: 8,192
- Optimizer: AdamW via Hugging Face Trainer
- Scheduler: linear decay with warmup
- Gradient checkpointing: enabled initially; disable only after a measured A100
  memory check shows it is unnecessary
- Evaluation and checkpointing: step-based, with best/final state retained
- Learning rate, effective batch size, warmup ratio, epochs/max steps, logging
  interval, weight decay, and gradient clipping: declared in a versioned YAML
  config, then frozen for the full run after a short smoke test

The smoke test checks mechanics and memory only. It does not justify tuning on the
held-out evaluation split. Any hyperparameter change creates a new named run and
is recorded rather than overwriting prior evidence.

### Implementation

1. Wrap each packed Parquet split in a PyTorch Dataset.
2. Implement a `TrainerCallback` that captures training and evaluation loss at
   every logging event.
3. Run a 10-step forward/backward/optimizer smoke test at micro-batch size 1,
   gradient accumulation 8, and gradient checkpointing enabled. Verify starting
   loss is in a plausible pretrained range. Stop immediately if it is near 10.8,
   non-finite, or structurally inconsistent with the saved base evaluation.
4. Run full CPT with Trainer, AdamW, and linear warmup.
5. Save resumable checkpoints, Trainer state, optimizer/scheduler state, logs,
   final model, tokenizer, and plots to persistent storage.
6. Identify the plateau by a declared smoothing/window rule rather than visual
   judgment alone, and report both the computed point and plotted curve.

### Artifacts

- `configs/cpt.yaml`
- `runs/<run-id>/run_manifest.json`
- `runs/<run-id>/loss_history.csv` and `loss_history.json`
- `runs/<run-id>/loss_curve.png`
- `runs/<run-id>/trainer_state.json`
- `runs/<run-id>/environment.json`
- `models/<run-id>/final/` containing model and tokenizer

### Validation

- Verify first loss, final loss, minimum loss, and non-finite/spike conditions.
- Require the 10-step smoke test to remain below 75 GB peak allocated GPU memory,
  leaving at least 5 GB safety margin. If it fails, retain full-parameter CPT but
  reduce activation memory or effective concurrency and rerun the same gate.
- Verify checkpoints can be loaded in a fresh process and generate text.
- Verify final model and tokenizer are on the persistent volume.
- Verify the loss callback's records agree with Trainer log history.
- Stop the GPU notebook after all required training artifacts are flushed.

**Definition of done:** full CPT completes or resumes to completion; loss history
is complete; trend and plateau are reported; final model/tokenizer reload; all
artifacts survive notebook shutdown.

### Measured result

Functionality 4 completed on the Kubeflow NVIDIA A100-SXM4-80GB on 2026-09-07.
The frozen full-parameter run completed three epochs and 408 optimizer steps with
BF16, TF32, non-reentrant gradient checkpointing, fused AdamW, micro-batch size
1, and gradient accumulation 8. Training loss decreased from 1.4888 at the first
logged optimizer step to 1.1949 at step 408, with a minimum logged loss of
1.1525. The declared non-overlapping 20-step mean-loss rule identified plateau
step 101. Peak CUDA memory was 18.145 GiB allocated and 18.982 GiB reserved.

Held-out evaluation loss was finite at all six scheduled evaluations and declined
monotonically: 1.316691 at step 68, 1.305295 at step 136, 1.299754 at step 204,
1.297271 at step 272, 1.296266 at step 340, and 1.296194 at step 408. The retained
best checkpoint is checkpoint 408. Trainer state, optimizer/scheduler state, loss
history, plateau analysis, loss curve, completed manifest, final BF16 model, and
tokenizer were flushed to persistent storage. A fresh process loaded the exported
model using local files only and generated Cisco-domain text on CUDA, passing the
post-training reload gate.

## 7. Phase 5: Functionality 5 - Evaluation

### Domain perplexity

1. Load the untouched frozen base model and evaluate the frozen held-out packed
   sequences without gradients.
2. Load the final CPT model and evaluate exactly the same sequences with the same
   precision, batch size, and token-weighted cross-entropy implementation.
3. Compute `PPL = exp(total negative log-likelihood / predicted token count)`.
4. Compute percentage reduction as
   `100 * (base_ppl - cpt_ppl) / base_ppl`.
5. Store aggregate and per-sequence losses so the calculation is auditable.

### Frozen 50-prompt comparison

Run exactly the same 50 prompts and deterministic generation configuration against
the base and CPT models.

The 25 generic prompts use five objectively scorable categories with five prompts
each: factual knowledge, science and arithmetic, reasoning, language, and common
knowledge/instruction following. They contain no Cisco or networking subject
matter. Three designated factual prompts satisfy the assignment's required
catastrophic-forgetting table; the remaining 22 strengthen that analysis.

The 25 Cisco prompts use five corpus-grounded categories with five prompts each:
switching/VLAN/STP, routing/OSPF/BGP, data-center vPC/VXLAN/Nexus, security/AAA/ISE,
and wireless/SD-WAN/operations. Fifteen are grounded in training-source documents
to measure acquired domain knowledge and ten in held-out-source documents to probe
domain generalization. Results for those groups are reported separately. Three
designated Cisco prompts satisfy the assignment baseline requirement.

Each JSONL prompt stores ID, group, category, prompt, expected concepts, answer
type, source PDF/page/evidence for Cisco records, evidence split, and rubric. Raw
outputs are immutable. Automatic concept coverage is supplemented by blinded
human scores for factual correctness, relevance, completeness, and coherence.
Each paired result receives `Improved`, `Retained`, or `Degraded`; a generic
degradation rate is reported as the catastrophic-forgetting indicator.

### Artifacts

- `evaluation/perplexity/base.json`
- `evaluation/perplexity/cpt.json`
- `evaluation/perplexity/comparison.json`
- `evaluation/perplexity/per_sequence.csv`
- `evaluation/forgetting_comparison.json` and `.md`
- `evaluation/prompt_comparison.jsonl` and `.csv` for all 50 prompts
- post-CPT responses using all frozen prompts

### Validation

- Assert base and CPT sequence IDs and predicted-token counts are identical.
- Recompute aggregate PPL from saved per-sequence sums.
- Treat a 10-40% reduction as assignment guidance, not a forced result. Report the
  measured result honestly and investigate leakage or calculation errors if it is
  implausibly large.
- Assert exactly 25 generic and 25 Cisco IDs occur once in each model's output.
- Compare all 50 pre/post outputs without changing prompt text, ordering,
   tokenizer, decoding settings, or scoring rubric.

**Definition of done:** base/CPT perplexity and percentage reduction are
reproducible on the identical held-out split; 25 generic and 25 Cisco outputs have
justified paired verdicts; the required 3+3 examples are included in the full set.

### Measured automatic result

The automatic Functionality 5 pass completed on the Kubeflow NVIDIA
A100-SXM4-80GB on 2026-09-07. Base and CPT evaluation used the same 92 frozen
8,192-token sequences, 753,572 predicted tokens, tokenizer, BF16 precision, and
token-weighted cross-entropy implementation. Independent aggregation of the saved
per-sequence losses reproduced base perplexity 3.862000630396992 and CPT
perplexity 3.655357927579825. The measured perplexity reduction is 5.350665%,
below the assignment's non-binding 10-40% guidance; no tuning or result adjustment
was performed.

All 50 frozen prompts were generated in their original order with unchanged
deterministic decoding: 25 general and 25 Cisco-domain prompts. Exact frozen
expected-concept coverage classified the general pairs as 2 Improved, 21
Retained, and 2 Degraded, for an automatic degradation rate of 8%. Domain pairs
were 24 Retained and 1 Degraded, for an automatic degradation rate of 4%; all 15
training-evidence domain prompts were Retained, while the 10 held-out-evidence
prompts contained 9 Retained and 1 Degraded. Mean concept coverage changed from
0.643333 to 0.623333 for general prompts and from 0.054667 to 0.041333 for domain
prompts. These lexical scores are audit aids, not factual-quality judgments.
Blinded human scoring for factual correctness, relevance, completeness, and
coherence remains pending before Functionality 5 satisfies its full definition of
done. The deterministic 50-row A/B worksheet is stored in
`evaluation/blinded_human_review.csv`; its model-identity mapping is isolated in
`evaluation/blinding_key.json` and must remain closed until scoring is complete.

## 8. Submission and Comparison Package

1. Create `notebooks/Assignment_PartA.ipynb` as a thin, restartable presentation
   layer that invokes independently authored modules and displays Steps 1-5.
2. Create a separate custom Cisco enterprise variant document in the V2-V6 format
   covering domain, sources, sample instruction pairs, model choices, and guidance
   for assignments 1A, 1B, 2A, 2B, and 2C.
3. Include a concise README with exact Kubeflow setup, commands, artifact paths,
   and rerun procedure.
4. Commit only Ajoish-owned source and lightweight evidence to the independent
   branch. Do not merge or push without explicit user instruction.
5. Produce a comparison manifest for Murali containing model ID/revision, corpus
   manifest hash, accepted-document counts, split seed, token counts, training
   hyperparameters, GPU environment, base/CPT PPL, reduction, and forgetting
   verdicts. Comparison claims are valid only where those controls match.

## 9. Ordered Execution Gates

The work proceeds strictly one gate at a time:

1. Approve this plan.
2. Verify live Kubeflow quotas and confirm the authorized clean reset left no
   old notebooks or PVCs requiring preservation.
3. Reset and validate the 32 GiB maximum-resource Kubeflow environment.
4. Implement and approve Functionality 1 artifacts and report.
5. Implement and approve Functionality 2 artifacts and report.
6. Freeze Functionality 3 prompts, baseline outputs, and base perplexity.
7. Approve the CPT configuration after the smoke test, then run Functionality 4.
8. Run Functionality 5 without changing the frozen evaluation contract.
9. Build the submission notebook and comparison package.

Any failed gate stops downstream work. No held-out data is moved into training,
no prior output is silently overwritten, and no destructive infrastructure action
is taken without explicit user authorization and feasibility checks.