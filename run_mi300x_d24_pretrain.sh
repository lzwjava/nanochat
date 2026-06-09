#!/bin/bash
# ============================================================================
# Nanochat: Pretrain-only 760M model on MI300X
# ============================================================================
# Same as run_mi300x_d24.sh but stops after pretraining (no SFT/eval).
# Use --resume-from-step=N to continue from a checkpoint.
#
# Quick usage:
#   ./run_mi300x_d24_pretrain.sh                        # full 29k steps
#   ./run_mi300x_d24_pretrain.sh --num-iterations=1000  # quick 1k test
#   ./run_mi300x_d24_pretrain.sh --resume-from-step=5000
# ============================================================================

set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export WANDB_MODE=disabled
export HIP_FORCE_DEV_KERNARG=1
export HSA_OVERRIDE_GFX_VERSION=9.4.2
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p $NANOCHAT_BASE_DIR

cd /root/nanochat
source .venv/bin/activate

echo "=== Pretraining 760M model (depth=24) on MI300X ==="
echo "  Model:   depth=24, dim=1536, heads=12 (~760M params)"
echo "  Data:    ClimbMix-400B"
echo ""

python -m scripts.base_train \
    --depth=24 \
    --device-batch-size=32 \
    --total-batch-size=524288 \
    --max-seq-len=2048 \
    --window-pattern L \
    --num-iterations=29000 \
    --eval-every=1000 \
    --eval-tokens=1048576 \
    --sample-every=5000 \
    --save-every=5000 \
    --core-metric-every=5000 \
    --core-metric-max-per-task=200 \
    --tracker=mlflow \
    --run=mi300x-d24-760m \
    "$@"
