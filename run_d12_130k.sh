#!/bin/bash
# Resume d12 chinchilla training from step 87000 to 130000
# Model: 286M params (depth=12, dim=768, heads=6, seq=2048)
# Data: 201 climbmix shards (~8.5B tokens capacity)

set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export WANDB_MODE=disabled

cd /mnt/data/nanochat
source .venv/bin/activate

echo "=== Resuming d12 training: step 87000 -> 130000 ==="
echo "Model: 286M params (depth=12, dim=768, heads=6)"
echo "Data: 201 climbmix shards, 18GB"
echo "Additional steps: 43000 (batch=65536 => ~2.8B more tokens)"
echo "Estimated time: ~14 hours"
echo ""

python -m scripts.base_train \
    --depth=12 \
    --device-batch-size=8 \
    --total-batch-size=65536 \
    --max-seq-len=2048 \
    --window-pattern L \
    --num-iterations=130000 \
    --resume-from-step=87000 \
    --eval-every=5000 \
    --eval-tokens=524288 \
    --sample-every=10000 \
    --save-every=10000 \
    --core-metric-every=10000 \
    --core-metric-max-per-task=200 \
    --run=rtx4070-d12-130k \
    "$@"

echo ""
echo "=== Training complete ==="
echo "Evaluate: python -m scripts.base_eval --device-batch-size=8"
echo "Chat:     python -m scripts.chat_cli -p 'Why is the sky blue?'"
