#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  echo "Usage: $0 <model-name> <prepare|baseline|cpt>"
  echo
  echo "Supported models:"
  echo "  gpt2"
  echo "  gpt2-medium"
  echo "  smollm2-360m"
  echo
  echo "Examples:"
  echo "  $0 smollm2-360m prepare"
  echo "  $0 smollm2-360m baseline"
  echo "  $0 smollm2-360m cpt"
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

MODEL_NAME="$1"
ACTION="$2"

# Shared workflow settings. Override these with environment variables only
# when a different corpus or training duration is needed.
INPUT_DIR="${INPUT_DIR:-${REPO_ROOT}/1a/data/pdfs/cisco-dc-pdfs}"
TEXT_DIR="${TEXT_DIR:-${REPO_ROOT}/1a/data/text/cisco-dc}"
EPOCHS="${EPOCHS:-3}"
MAX_WORKERS="${MAX_WORKERS:-2}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT-90:10}"

case "${MODEL_NAME}" in
  gpt2)
    MODEL_KEY="gpt2"
    CONTEXT_LENGTH=1024
    TRAIN_BATCH_SIZE=8
    GRADIENT_ACCUMULATION_STEPS=4
    ;;
  gpt2-medium)
    MODEL_KEY="gpt2-medium"
    CONTEXT_LENGTH=1024
    TRAIN_BATCH_SIZE=2
    GRADIENT_ACCUMULATION_STEPS=8
    ;;
  smollm2-360m|HuggingFaceTB/SmolLM2-360M)
    MODEL_KEY="smollm2-360m"
    CONTEXT_LENGTH=2048
    TRAIN_BATCH_SIZE=1
    GRADIENT_ACCUMULATION_STEPS=16
    ;;
  *)
    echo "Unsupported model: ${MODEL_NAME}" >&2
    usage >&2
    exit 2
    ;;
esac

PROCESSED_DIR="${REPO_ROOT}/1a/data/processed/${MODEL_KEY}"
BIN_FILE="${PROCESSED_DIR}/tokens.bin"
PARQUET_FILE="${PROCESSED_DIR}/tokens.parquet"
METRICS_FILE="${PROCESSED_DIR}/dataset_metrics.json"
TRAIN_BIN_FILE="${PROCESSED_DIR}/token_train.bin"
TRAIN_METRICS_FILE="${PROCESSED_DIR}/dataset_metrics_train.json"
AUDIT_REPORT="${REPO_ROOT}/1a/data/reports/${MODEL_KEY}_cleaning_audit.json"
CONTENT_AUDIT_REPORT="${REPO_ROOT}/1a/data/reports/${MODEL_KEY}_content_cleaning_audit.json"
CLEANED_TEXT_DIR="${REPO_ROOT}/1a/data/cleaned/${MODEL_KEY}"
BASELINE_DIR="${REPO_ROOT}/1a/evaluation/${MODEL_KEY}/baseline"
MODEL_OUTPUT_DIR="${REPO_ROOT}/1a/models/${MODEL_KEY}/v1"
QUERY_DIR="${REPO_ROOT}/1a/evaluation/queries"

cd "${REPO_ROOT}"

case "${ACTION}" in
  prepare)
    prepare_args=(
      "${SCRIPT_DIR}/prepare_cpt_dataset.py"
      --model-name "${MODEL_NAME}"
      --input-dir "${INPUT_DIR}"
      --text-dir "${TEXT_DIR}"
      --audit-report "${AUDIT_REPORT}"
      --content-audit-report "${CONTENT_AUDIT_REPORT}"
      --cleaned-text-dir "${CLEANED_TEXT_DIR}"
      --bin-file "${BIN_FILE}"
      --parquet-file "${PARQUET_FILE}"
      --metrics-file "${METRICS_FILE}"
      --context-length "${CONTEXT_LENGTH}"
      --max-workers "${MAX_WORKERS}"
      --extract-tables
      --dedup-strategy both
      --content-cleaning
      --similarity-threshold 0.85
      --parquet-compression zstd
    )
    if [[ -n "${TRAIN_TEST_SPLIT}" ]]; then
      prepare_args+=(--train-test-split "${TRAIN_TEST_SPLIT}" --split-seed 42)
    fi
    "${PYTHON_BIN}" "${prepare_args[@]}"
    ;;

  baseline)
    "${PYTHON_BIN}" "${SCRIPT_DIR}/baseline.py" \
      "${QUERY_DIR}" \
      --model-name "${MODEL_NAME}" \
      --baseline-path "${BASELINE_DIR}" \
      --evaluation-stage pre_cpt \
      --max-new-tokens 100 \
      --batch-size 4
    ;;

  cpt)
    CPT_BIN_FILE="${BIN_FILE}"
    CPT_METRICS_FILE="${METRICS_FILE}"
    if [[ -n "${TRAIN_TEST_SPLIT}" ]]; then
      CPT_BIN_FILE="${TRAIN_BIN_FILE}"
      CPT_METRICS_FILE="${TRAIN_METRICS_FILE}"
    fi

    if [[ ! -f "${CPT_BIN_FILE}" ]]; then
      echo "Prepared token file not found: ${CPT_BIN_FILE}" >&2
      echo "Run '$0 ${MODEL_NAME} prepare' first." >&2
      exit 1
    fi

    cpt_args=(
      "${SCRIPT_DIR}/cpt_train.py"
      --model-name "${MODEL_NAME}"
      --bin-file "${CPT_BIN_FILE}"
      --save-dir "${MODEL_OUTPUT_DIR}"
      --context-length "${CONTEXT_LENGTH}"
      --epochs "${EPOCHS}"
      --batch-size "${TRAIN_BATCH_SIZE}"
      --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
      --learning-rate 5e-6
      --weight-decay 0.01
      --warmup-ratio 0.05
      --max-grad-norm 1.0
      --log-every-steps 10
      --save-every-steps 500
      --num-workers 0
      --seed 42
    )
    if [[ "${MODEL_KEY}" == "smollm2-360m" ]]; then
      if [[ ! -f "${CPT_METRICS_FILE}" ]]; then
        echo "SmolLM2 dataset metrics not found: ${CPT_METRICS_FILE}" >&2
        echo "Run '$0 ${MODEL_NAME} prepare' first." >&2
        exit 1
      fi
      cpt_args+=(--dataset-metrics "${CPT_METRICS_FILE}")
    fi
    "${PYTHON_BIN}" "${cpt_args[@]}"
    ;;

  *)
    echo "Unsupported action: ${ACTION}" >&2
    usage >&2
    exit 2
    ;;
esac
