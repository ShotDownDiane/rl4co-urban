#!/bin/bash
# Quick test script to verify the training pipeline works
# Uses minimal settings for fast testing

echo "====================================="
echo "Quick Test - TSP Training"
echo "Running with minimal settings..."
echo "====================================="

python train.py \
  --problem TSP \
  --num-loc 10 \
  --train-size 1000 \
  --val-size 100 \
  --test-size 100 \
  --epochs 2 \
  --embed-dim 64 \
  --num-encoder-layers 2 \
  --num-heads 4 \
  --data-dir data/test \
  --checkpoint-dir checkpoints/test \
  --log-dir logs/test \
  --visualize

echo ""
echo "====================================="
echo "Quick Test Complete!"
echo "====================================="
echo ""
echo "If this completed successfully, the pipeline is working correctly."
echo "You can now run full training with the other scripts."
