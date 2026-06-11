#!/bin/bash
# =============================================================================
# Multi-Node CPU DDP Training — WORKER node (192.168.1.47)
# =============================================================================
# Run AFTER the master is started on 192.168.1.36.
# =============================================================================

set -e
cd ~/projects/nanochat
source .venv/bin/activate

export NANOCHAT_BASE_DIR=~/.cache/nanochat
export NANOCHAT_FORCE_SDPA=1
export NANOCHAT_DTYPE=float32
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export GLOO_SOCKET_IFNAME=wlp3s0

echo "=== WORKER (rank 1) ==="
torchrun \
    --nnodes=2 --nproc_per_node=1 --node_rank=1 \
    --master_addr=192.168.1.36 --master_port=29500 \
    -m scripts.base_train -- \
    --device-type cpu --depth 4 --aspect-ratio 64 --head-dim 64 \
    --max-seq-len 512 --device-batch-size 2 --total-batch-size 2048 \
    --num-iterations 20 --eval-every 10 --eval-tokens 2048 \
    --core-metric-every -1 --sample-every 20 \
    --tracker none --run dist-cpu-2node-full
