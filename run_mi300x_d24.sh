#!/bin/bash
# ============================================================================
# Nanochat GPT-2 ~760M training on AMD MI300X (192 GB VRAM, single GPU)
# ============================================================================
#
# Model:   depth=24, dim=1536, heads=12, head_dim=128 → ~760M params
# Dataset: ClimbMix-400B (successor to FineWeb)
# Target:  Chinchilla-optimal ≈ 15.2B tokens (20 × 760M)
#          29,000 steps × 524,288 tokens/step ≈ 15.2B tokens
#
# Performance (measured on MI300X):
#   - ~68K tok/sec, ~7.7s per step
#   - MFU ~27% (SDPA fallback, no FA3 on AMD)
#   - Peak VRAM: ~105 GB / 192 GB
#   - Estimated total: ~62 hours for full Chinchilla run
#
# Hardware notes:
#   - ROCm 7.2, PyTorch 2.9.1+rocm6.4
#   - No FP8 (needs ROCm 6.5+), using bf16
#   - No Flash Attention 3 (H100-only), using PyTorch SDPA fallback
#   - window-pattern=L (full attention, required for SDPA)
#
# Resume: if training crashes, re-run with --resume-from-step=<last_step>
# Checkpoints saved every 5000 steps to ~/.cache/nanochat/base_checkpoints/d24/
# ============================================================================

set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export WANDB_MODE=disabled

# ROCm environment optimizations
export HIP_FORCE_DEV_KERNARG=1
export HSA_OVERRIDE_GFX_VERSION=9.4.2
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p $NANOCHAT_BASE_DIR

cd /root/nanochat
source .venv/bin/activate

# ============================================================================
# Step 1: Download pretraining data (ClimbMix-400B shards)
# ============================================================================
echo "=== Step 1: Download pretraining data (30 shards + val) ==="
python -m nanochat.dataset -n 30

# ============================================================================
# Step 2: Train tokenizer on the downloaded data
# ============================================================================
echo "=== Step 2: Train tokenizer ==="
python -m scripts.tok_train
python -m scripts.tok_eval

# ============================================================================
# Step 3: Pretrain base model (depth=24, ~760M params)
# ============================================================================
echo "=== Step 3: Pretrain 760M base model on MI300X ==="
echo "  Model:   depth=24, dim=1536, heads=12 (~760M params)"
echo "  Data:    ClimbMix-400B, 30 shards"
echo "  Steps:   29,000 (Chinchilla-optimal: ~15.2B tokens)"
echo "  Batch:   524,288 tokens/step (32 seq × 2048 tokens × 8 grad accum)"
echo "  Time:    ~62 hours estimated"
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

# ============================================================================
# Step 4: Evaluate the base model
# ============================================================================
echo "=== Step 4: Evaluate base model ==="
python -m scripts.base_eval --device-batch-size=32

# ============================================================================
# Step 5: Download SFT data and run SFT
# ============================================================================
echo "=== Step 5: Download SFT data ==="
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

echo "=== Step 6: SFT fine-tuning ==="
python -m scripts.chat_sft \
    --max-seq-len=2048 \
    --device-batch-size=16 \
    --total-batch-size=262144 \
    --eval-every=500 \
    --eval-tokens=524288 \
    --num-iterations=3000 \
    --tracker=mlflow \
    --run=mi300x-d24-760m-sft

# ============================================================================
# Step 7: Evaluate chat model
# ============================================================================
echo "=== Step 7: Evaluate chat model ==="
python -m scripts.chat_eval -i sft

# ============================================================================
echo ""
echo "=== DONE ==="
echo "Chat via CLI:  python -m scripts.chat_cli -p 'Why is the sky blue?'"
echo "Chat via Web:  python -m scripts.chat_web"
echo "Checkpoint:    $NANOCHAT_BASE_DIR/base_checkpoints/d24/"
