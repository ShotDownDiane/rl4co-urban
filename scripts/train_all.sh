#!/bin/bash
# Script to train all problems sequentially

echo "###################################"
echo "# Training All Problems"
echo "###################################"
echo ""

# Train TSP
echo "Starting TSP training..."
bash scripts/train_tsp.sh

echo ""
echo "Waiting 10 seconds before next training..."
sleep 10
echo ""

# Train FLP
echo "Starting FLP training..."
bash scripts/train_flp.sh

echo ""
echo "Waiting 10 seconds before next training..."
sleep 10
echo ""

# Train MCLP
echo "Starting MCLP training..."
bash scripts/train_mclp.sh

echo ""
echo "Waiting 10 seconds before next training..."
sleep 10
echo ""

# Train STP
echo "Starting STP training..."
bash scripts/train_stp.sh

echo ""
echo "###################################"
echo "# All Training Complete!"
echo "###################################"
echo ""
echo "View results with: tensorboard --logdir logs"
