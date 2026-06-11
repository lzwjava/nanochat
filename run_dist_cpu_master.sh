#!/bin/bash
# =============================================================================
# Multi-Node CPU DDP Training — MASTER node (192.168.1.36)
# =============================================================================
# Run this FIRST, then start the worker on 192.168.1.47.
#
# Prerequisites:
#   - Both machines: torch 2.9.1, nanochat deps installed
#   - Both machines: same 201 parquet shards in ~/.cache/nanochat/base_data_climbmix/
#   - Firewall: ufw allow from 192.168.1.0/24 on master
#
# Required env vars:
#   NANOCHAT_FORCE_SDPA=1  — Flash Attention is CUDA-only
#   NANOCHAT_DTYPE=float32 — bf16 auto-detects from CUDA but is slow on CPU
#   GLOO_SOCKET_IFNAME     — bind to LAN interface (avoids IPv4/IPv6 mismatch)
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
torchrun \
    --nnodes=2 --nproc_per_node=1 --node_rank=0 \
    --master_addr=192.168.1.36 --master_port=29500 \
    -m scripts.base_train -- \
    --device-type cpu --depth 4 --aspect-ratio 64 --head-dim 64 \
    --max-seq-len 512 --device-batch-size 2 --total-batch-size 2048 \
    --num-iterations 20 --eval-every 10 --eval-tokens 2048 \
    --core-metric-every -1 --sample-every 20 \
    --tracker none --run dist-cpu-2node-full
