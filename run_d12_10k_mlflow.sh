#!/bin/bash
# Fresh d12 training: 10,000 steps from scratch with MLflow tracking
# Model: 286M params (depth=12, dim=768, heads=6, seq=2048)
# Data: fineweb-edu shards
# Batch: 65536 tokens => ~655M total tokens

set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export WANDB_MODE=disabled
mkdir -p $NANOCHAT_BASE_DIR

cd /mnt/data/nanochat
source .venv/bin/activate

echo "=== Fresh d12 training: 10k steps with MLflow ==="
echo "Model: 286M params (depth=12, dim=768, heads=6)"
echo "Batch: 65536 tokens => ~655M total tokens"
echo "Tracker: MLflow (local file store under ./mlruns)"
echo ""

python -m scripts.base_train \
    --depth=12 \
    --device-batch-size=8 \
    --total-batch-size=65536 \
    --max-seq-len=2048 \
    --window-pattern L \
    --num-iterations=10000 \
    --eval-every=500 \
    --eval-tokens=524288 \
    --sample-every=2000 \
    --save-every=5000 \
    --core-metric-every=2000 \
    --core-metric-max-per-task=200 \
    --tracker=mlflow \
    --mlflow-experiment=nanochat-d12 \
    --run=d12-fresh-10k \
    --model-tag=d12-fresh \
    "$@"

echo ""
echo "=== Training complete ==="
echo "MLflow UI:  cd /mnt/data/nanochat && mlflow ui --port 5000"
echo "Evaluate:   python -m scripts.base_eval --device-batch-size=8 --model-tag=d12-fresh"
echo "Chat:       python -m scripts.chat_cli -p 'Why is the sky blue?'"
