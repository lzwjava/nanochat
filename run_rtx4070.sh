#!/bin/bash
# nanochat run for RTX 4070 (12 GB VRAM, single GPU)
# Scaled down: depth=8 (~40M params), fits comfortably in 12 GB

set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export WANDB_MODE=disabled
mkdir -p $NANOCHAT_BASE_DIR

cd /mnt/data/nanochat
source .venv/bin/activate

echo "=== Step 1: Download pretraining data (8 shards, ~2B chars) ==="
python -m nanochat.dataset -n 8

echo "=== Step 2: Train tokenizer ==="
python -m scripts.tok_train
python -m scripts.tok_eval

echo "=== Step 3: Pretrain base model (depth=8, single GPU) ==="
python -m scripts.base_train \
    --depth=8 \
    --device-batch-size=4 \
    --total-batch-size=32768 \
    --max-seq-len=1024 \
    --eval-every=200 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=500 \
    --num-iterations=5000 \
    --run=rtx4070-d8

echo "=== Step 4: Evaluate base model ==="
python -m scripts.base_eval --device-batch-size=4 --split-tokens=16384 --max-per-task=50

echo "=== Step 5: Download SFT data ==="
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

echo "=== Step 6: SFT ==="
python -m scripts.chat_sft \
    --max-seq-len=1024 \
    --device-batch-size=4 \
    --total-batch-size=32768 \
    --eval-every=200 \
    --eval-tokens=524288 \
    --num-iterations=1500 \
    --run=rtx4070-d8

echo "=== Step 7: Evaluate chat model ==="
python -m scripts.chat_eval -i sft

echo "=== DONE ==="
echo "Chat with: python -m scripts.chat_cli -p 'Why is the sky blue?'"
echo "Web UI:    python -m scripts.chat_web"
