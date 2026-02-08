#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_MODE="${1:-finetuned}"

BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
FINETUNED_MODEL="${ROOT_DIR}/runs/merged_model"

HOST="0.0.0.0"
PORT="8000"
MAX_LEN="4096"
GPU_UTIL="0.85"

if [[ "${MODEL_MODE}" == "base" ]]; then
  MODEL_PATH="${BASE_MODEL}"
  TOKENIZER="${BASE_MODEL}"
elif [[ "${MODEL_MODE}" == "finetuned" ]]; then
  MODEL_PATH="${FINETUNED_MODEL}"
  TOKENIZER="${BASE_MODEL}"
else
  echo "Usage: ./run.sh [base|finetuned]"
  exit 1
fi

python -m vllm.entrypoints.openai.api_server \
  --host "${HOST}" \
  --port "${PORT}" \
  --model "${MODEL_PATH}" \
  --tokenizer "${TOKENIZER}" \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}"
