#!/bin/bash
# =============================================================================
# Multi-Node CPU DDP Training — MASTER node (192.168.1.36)
# Model: d8 (~80M params, depth=8, dim=512, heads=4, seq=1024)
# Tracking: MLflow (local file store, UI on port 5000)
# =============================================================================

set -e
cd /mnt/data/nanochat
source .venv/bin/activate

export NANOCHAT_BASE_DIR=~/.cache/nanochat
export NANOCHAT_FORCE_SDPA=1
export NANOCHAT_DTYPE=float32
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export GLOO_SOCKET_IFNAME=enp4s0

echo "=== MASTER (rank 0) ==="
echo "Model: d8 (~80M params)"
echo "Tracker: MLflow"

torchrun \
    --nnodes=2 --nproc_per_node=1 --node_rank=0 \
    --master_addr=192.168.1.36 --master_port=29500 \
    -m scripts.base_train -- \
    --device-type cpu \
    --depth 8 \
    --aspect-ratio 64 \
    --head-dim 128 \
    --max-seq-len 1024 \
    --device-batch-size 2 \
    --total-batch-size 4096 \
    --num-iterations 5000 \
    --eval-every 500 \
    --eval-tokens 65536 \
    --core-metric-every 2000 \
    --core-metric-max-per-task 100 \
    --sample-every 1000 \
    --save-every 1000 \
    --tracker mlflow \
    --mlflow-experiment nanochat-d8 \
    --run dist-cpu-d8-2node
