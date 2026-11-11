#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
vllm_python="${GRAPH_VLLM_PYTHON:-/root/.venv-vllm/bin/python}"
graph_weights="${GRAPH_MODEL_DIR:-/root/models/Qwen3.5-9B-FP8}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PATH="$(dirname -- "$vllm_python"):$PATH"
export OMP_NUM_THREADS="${GRAPH_CPU_THREADS:-1}"
export VLLM_CACHE_ROOT="$project_dir/data/graph/vllm_cache"
exec "$vllm_python" -m vllm.entrypoints.openai.api_server \
  --model "$graph_weights" \
  --served-model-name graph-qwen35-9b-fp8 \
  --host 127.0.0.1 --port "${GRAPH_VLLM_PORT:-8001}" \
  --language-model-only \
  --max-model-len 8192 \
  --max-num-seqs "${GRAPH_MAX_SEQS:-8}" \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization "${GRAPH_GPU_MEMORY_UTILIZATION:-0.82}" \
  --cpu-offload-gb 0 \
  --performance-mode throughput \
  --enable-prefix-caching \
  --max-cudagraph-capture-size 8 \
  --structured-outputs-config '{"backend":"xgrammar"}' \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --generation-config vllm
