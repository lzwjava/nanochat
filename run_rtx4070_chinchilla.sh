#!/bin/bash
# Nanochat Chinchilla-optimal training on RTX 4070 (12 GB VRAM)
# 286M params, 87k steps = 5.7B tokens (~28.5 hours)
#
# Based on the actual config from the previous 10k-step run:
#   depth=12, batch=65536, seq=2048, vocab=32768
#
# Resume: if training crashes, re-run with --resume-from-step=<last_step>
# Checkpoints saved every 10k steps to base_checkpoints/d12/

set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export WANDB_MODE=disabled
mkdir -p $NANOCHAT_BASE_DIR

cd /mnt/data/nanochat
source .venv/bin/activate

# Skip data download and tokenizer (already done in previous run)
# Data: 176 fineweb-edu shards, ~142.6B tokens on disk
# Tokenizer: already trained

echo "=== Chinchilla-optimal pretraining (87k steps, ~28.5 hours) ==="
echo "Model: 286M params (depth=12, dim=768, heads=6)"
echo "Data: 142.6B tokens available, training on 5.7B"
echo "Checkpoints: every 10k steps to base_checkpoints/d12/"
echo ""

python -m scripts.base_train \
    --depth=12 \
    --device-batch-size=8 \
    --total-batch-size=65536 \
    --max-seq-len=2048 \
    --window-pattern L \
    --num-iterations=87000 \
    --eval-every=2000 \
    --eval-tokens=524288 \
    --sample-every=5000 \
    --save-every=10000 \
    --core-metric-every=10000 \
    --core-metric-max-per-task=200 \
    --run=rtx4070-d12-chinchilla \
    "$@"

echo ""
echo "=== Training complete ==="
echo "Evaluate: python -m scripts.base_eval --device-batch-size=8"
echo "Chat:     python -m scripts.chat_cli -p 'Why is the sky blue?'"
