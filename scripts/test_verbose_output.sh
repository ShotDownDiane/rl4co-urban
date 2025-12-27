#!/bin/bash

# Test script to verify verbose output works
# Uses very small dataset to complete quickly

set -e

echo "========================================"
echo "Testing Verbose Output for DeepACO"
echo "========================================"
echo ""
echo "This is a QUICK TEST with minimal data"
echo "Should complete in 2-3 minutes"
echo ""

# Kill any existing training process
EXISTING_PID=$(ps aux | grep "train.py" | grep -v grep | awk '{print $2}' | head -1)
if [ ! -z "$EXISTING_PID" ]; then
    echo "⚠️  Killing existing training process (PID: $EXISTING_PID)"
    kill $EXISTING_PID
    sleep 2
fi

# Run with very small configuration
python -u train.py \
  --problem TSP \
  --num-loc 20 \
  --train-size 512 \
  --val-size 128 \
  --test-size 128 \
  --epochs 2 \
  --batch-size 128 \
  --model-type DeepACO \
  --embed-dim 64 \
  --num-encoder-layers 2 \
  --num-heads 4 \
  --learning-rate 1e-4 \
  --baseline rollout \
  --data-dir data/test_verbose \
  --checkpoint-dir checkpoints/test_verbose \
  --log-dir logs/test_verbose \
  --force-regenerate

echo ""
echo "========================================"
echo "✅ Test Complete!"
echo "========================================"
echo ""
echo "If you saw detailed output like:"
echo "  🚀 TRAINING STARTED"
echo "  ⚡ Epoch X | Batch Y/Z"
echo "  🐜 DeepACO Batch Processing..."
echo ""
echo "Then the verbose output is working correctly!"
echo ""
