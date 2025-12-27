#!/bin/bash
# Quick test script for MIS
# 
# Usage: bash scripts/quick_test_mis.sh

echo "====================================="
echo "Quick Test - MIS Training"
echo "Running with minimal settings..."
echo "====================================="

python train.py \
  --problem MIS \
  --num-loc 20 \
  --graph-type erdos_renyi \
  --edge-prob 0.2 \
  --train-size 100 \
  --val-size 20 \
  --test-size 20 \
  --batch-size 16 \
  --epochs 2 \
  --learning-rate 1e-4 \
  --baseline rollout \
  --data-dir data/test_mis \
  --checkpoint-dir checkpoints/test_mis \
  --log-dir logs/test_mis \
  --eval-batch-size 20

echo ""
echo "====================================="
echo "Quick Test Complete!"
echo "====================================="
echo ""
echo "If this completed successfully, the MIS wrapper is working correctly."
echo "You can now run full training with scripts/train_mis.sh"
