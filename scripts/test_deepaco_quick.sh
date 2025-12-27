#!/bin/bash

# Quick test script for DeepACO with small dataset
# Use this to verify training works and see output quickly

set -e

echo "====================================="
echo "Quick Test: DeepACO Training"
echo "====================================="
echo ""
echo "Using SMALL dataset for quick testing..."
echo "This should complete in a few minutes"
echo ""

# Small configuration for quick testing
python -u train.py \
  --problem TSP \
  --num-loc 20 \
  --train-size 1000 \
  --val-size 100 \
  --test-size 100 \
  --epochs 2 \
  --batch-size 64 \
  --model-type DeepACO \
  --embed-dim 64 \
  --num-encoder-layers 2 \
  --num-heads 4 \
  --learning-rate 1e-4 \
  --baseline rollout \
  --data-dir data/test \
  --checkpoint-dir checkpoints/test \
  --log-dir logs/test \
  --force-regenerate

echo ""
echo "====================================="
echo "Quick test complete!"
echo "====================================="
