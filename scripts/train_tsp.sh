#!/bin/bash
# Training script for TSP problem

echo "====================================="
echo "Training TSP Model"
echo "====================================="

python train.py \
  --problem TSP \
  --num-loc 100 \
  --batch-size 16 \
  --train-size 640 \
  --val-size 100 \
  --test-size 100 \
  --epochs 10 \
  --embed-dim 128 \
  --num-encoder-layers 3 \
  --num-heads 8 \
  --learning-rate 1e-4 \
  --baseline no \
  --data-dir data \
  --checkpoint-dir checkpoints \
  --log-dir logs \
  --visualize \
  --force-regenerate

echo "====================================="
echo "Training Complete!"
echo "Check logs with: tensorboard --logdir logs"
echo "====================================="
