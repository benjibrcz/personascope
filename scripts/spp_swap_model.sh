#!/usr/bin/env bash
# Swap the vLLM-served model on an already-running wave-2 SPP pod.
#
# The SPP variants are separate full 3B models (not adapters), so they can't
# be co-served. Rather than boot a pod per model, we keep ONE keep-alive pod
# (booted by vllm_serve serving vanilla) and swap the served model in place.
# The SSH tunnel (local 8003 → pod 8000) stays valid across swaps because the
# pod-side port is constant; only the loaded weights change.
#
#   bash scripts/spp_swap_model.sh <SSH_HOST> <SSH_PORT> dlab-spp/t0-3b-instruct
#
# Kills the current vLLM, launches the new model on pod port 8000, and polls
# pod-side health until the model answers (or times out at 600s).

set -euo pipefail

HOST="$1"; PORT="$2"; MODEL="$3"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $PORT root@$HOST"

echo "[swap] stopping current vLLM on pod…"
$SSH 'kill $(cat /workspace/vllm.pid) 2>/dev/null; sleep 3; pkill -f "vllm serve" 2>/dev/null; sleep 2; true'

echo "[swap] launching $MODEL …"
# HF_TOKEN is already exported in the pod env (set at pod creation). Standard
# tokenizer, no LoRA, no patch — plain serve on the same port 8000.
$SSH "cd /workspace && nohup /workspace/vllm-venv/bin/vllm serve '$MODEL' \
  --host 0.0.0.0 --port 8000 --dtype bfloat16 --max-model-len 2048 \
  --gpu-memory-utilization 0.85 --api-key sk-vllm-local \
  > /workspace/vllm.log 2>&1 & echo \$! > /workspace/vllm.pid"

echo "[swap] waiting for health…"
for i in $(seq 1 60); do
  if $SSH "curl -s -m 5 -H 'Authorization: Bearer sk-vllm-local' http://localhost:8000/v1/models" 2>/dev/null | grep -q "$MODEL"; then
    echo "[swap] $MODEL is serving."
    exit 0
  fi
  sleep 10
done
echo "[swap] TIMEOUT waiting for $MODEL"; $SSH "tail -20 /workspace/vllm.log"; exit 1
