# Cross-Run CPT Evaluation Comparison

## Interpretation boundary

- Compare domain perplexity **reduction percentages**, not raw perplexities. Tokenizers, context lengths, and held-out corpora differ.
- Concept recall and repetition use compatible formulas, but prompt suites differ. Treat them as within-run diagnostics, not a model leaderboard.
- A direct model ranking requires rerunning every model on one common prompt suite and tokenizer-appropriate matched corpora.

## Available comparable metrics

| Run | Parameters | Context | Prompts | Domain PPL reduction | Domain concept recall | General concept recall | Domain repetition | General repetition |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2-large/v1 | 774,030,080 | 1,024 | 55 | 12.75% | 0.0% | 32.0% | 63.8% | 65.8% |
| gpt2-large/v2 | 774,030,080 | 1,024 | 55 | 18.89% | 1.4% | 28.0% | 62.9% | 61.9% |
| gpt2-medium/v1 | 354,823,168 | 1,024 | 55 | 13.86% | 4.3% | 24.0% | 64.8% | 70.4% |
| ajoish/smollm2-1.7b-cisco-cpt-v1 | 1,711,376,384 | 8,192 | 50 | 5.35% | 5.6% | 59.5% | 43.3% | 40.8% |

## Missing from the completed Ajoish run

- Generic held-out perplexity.
- Fixed-reference conditional perplexity and exact-continuation metrics.
- Baseline-response retention perplexity.

These require an additional model-scoring run; they cannot be recovered from generated text alone.
