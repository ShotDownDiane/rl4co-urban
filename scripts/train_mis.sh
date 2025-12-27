#!/bin/bash
# Training script for Maximum Independent Set (MIS)
# 
# Usage: bash scripts/train_mis.sh

echo "====================================="
echo "Training MIS with ML4CO-Kit Wrapper"
echo "====================================="

python train.py \
  --problem MIS \
  --num-loc 50 \
  --graph-type erdos_renyi \
  --edge-prob 0.15 \
  --train-size 10000 \
  --val-size 1000 \
  --test-size 1000 \
  --batch-size 128 \
  --epochs 100 \
  --learning-rate 1e-4 \
  --baseline rollout \
  --data-dir data/mis \
  --checkpoint-dir checkpoints/mis \
  --log-dir logs/mis \
  --eval-batch-size 1000
