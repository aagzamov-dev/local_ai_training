#!/bin/bash

echo "Loading conda..."

# Load conda into shell
source ~/miniforge3/etc/profile.d/conda.sh

echo "Activating environment..."
conda activate ai-project

echo "Environment activated:"
which python

echo "Starting Qwen GPU server..."

python -m llama_cpp.server \
  --model ./models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --host 0.0.0.0 \
  --port 8000 \
  --n_gpu_layers -1 \
  --n_ctx 4096 \
  --n_threads 12 \
  --n_batch 1024
