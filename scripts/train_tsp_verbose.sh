#!/bin/bash

# Training script with verbose output
# This version includes detailed logging and progress monitoring

set -e  # Exit on error

echo "====================================="
echo "Training TSP Model with DeepACO"
echo "====================================="

# Configuration
PROBLEM="TSP"
NUM_LOC=100
TRAIN_SIZE=640
VAL_SIZE=96
TEST_SIZE=96
EPOCHS=10
BATCH_SIZE=16
MODEL_TYPE="DeepACO"

# Log file
LOG_DIR="logs"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/train_${PROBLEM}_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S).log"

echo "Log file: $LOG_FILE"
echo ""

# Run training with tee to show output and save to file
python -u train.py \
  --problem $PROBLEM \
  --num-loc $NUM_LOC \
  --train-size $TRAIN_SIZE \
  --val-size $VAL_SIZE \
  --test-size $TEST_SIZE \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --model-type $MODEL_TYPE \
  --embed-dim 128 \
  --num-encoder-layers 3 \
  --num-heads 8 \
  --learning-rate 1e-4 \
  --baseline rollout \
  --data-dir data \
  --checkpoint-dir checkpoints \
  --log-dir logs \
  --visualize 2>&1 | tee $LOG_FILE

echo ""
echo "====================================="
echo "Training Complete!"
echo "Check logs with: tensorboard --logdir logs"
echo "Full log saved to: $LOG_FILE"
echo "====================================="
