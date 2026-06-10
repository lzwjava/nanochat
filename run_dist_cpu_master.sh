#!/bin/bash
# =============================================================================
# Multi-Node CPU DDP Training — MASTER node (192.168.1.36)
# =============================================================================
#
# HOW IT WORKS:
#   1. torchrun launches processes, sets RANK/LOCAL_RANK/WORLD_SIZE env vars
#   2. compute_init() calls dist.init_process_group(backend="gloo") for CPU
#   3. Dataloader shards data across ranks (different row groups per rank)
#   4. Each rank does forward+backward on its local data
#   5. DistMuonAdamW syncs gradients via all_reduce/reduce_scatter (ZeRO-2)
#   6. All ranks take the same optimizer step
#
# BACKEND: Gloo (CPU-to-CPU over TCP, vs NCCL for GPU-to-GPU)
#
# Run master first, then worker on 192.168.1.47.
# =============================================================================

set -e

MASTER_ADDR=192.168.1.36
MASTER_PORT=29500
NNODES=2
NODE_RANK=0

cd /mnt/data/nanochat
source .venv/bin/activate

export OMP_NUM_THREADS=8
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export NANOCHAT_FORCE_SDPA=1  # Flash Attention is CUDA-only, force SDPA for CPU

echo "=== MASTER (rank 0) ==="
echo "Listening on $MASTER_ADDR:$MASTER_PORT"
echo "World: $NNODES nodes"

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=1 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m scripts.base_train -- \
    --device-type cpu \
    --depth 4 \
    --aspect-ratio 64 \
    --head-dim 64 \
    --max-seq-len 512 \
    --device-batch-size 2 \
    --total-batch-size 2048 \
    --num-iterations 20 \
    --eval-every 10 \
    --eval-tokens 2048 \
    --core-metric-every -1 \
    --sample-every 20 \
    --tracker none \
    --run dist-cpu-2node
