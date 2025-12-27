#!/bin/bash
# Training script for MCLP problem

echo "====================================="
echo "Training MCLP Model"
echo "====================================="

python train.py \
  --problem MCLP \
  --num-loc 30 \
  --train-size 100000 \
  --val-size 10000 \
  --test-size 10000 \
  --epochs 100 \
  --embed-dim 128 \
  --num-encoder-layers 3 \
  --num-heads 8 \
  --learning-rate 1e-4 \
  --baseline rollout \
  --data-dir data \
  --checkpoint-dir checkpoints \
  --log-dir logs \
  --visualize

echo "====================================="
echo "Training Complete!"
echo "Check logs with: tensorboard --logdir logs"
echo "====================================="
