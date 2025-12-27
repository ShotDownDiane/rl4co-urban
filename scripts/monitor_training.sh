#!/bin/bash

# Training monitoring script for DeepACO

echo "=================================="
echo "Training Monitor for DeepACO/TSP"
echo "=================================="
echo ""

# Check if training process is running
TRAIN_PID=$(ps aux | grep "train.py" | grep -v grep | awk '{print $2}' | head -1)

if [ -z "$TRAIN_PID" ]; then
    echo "❌ No training process found"
else
    echo "✅ Training process running (PID: $TRAIN_PID)"
    
    # Show CPU and memory usage
    echo ""
    echo "📊 Resource Usage:"
    ps -p $TRAIN_PID -o pid,pcpu,pmem,etime,args | tail -1
fi

echo ""
echo "=================================="
echo "📁 Recent Log Files:"
echo "=================================="

# Show recent CSV logs
if [ -d "logs" ]; then
    echo ""
    echo "CSV Logs:"
    find logs -name "metrics.csv" -type f -mmin -30 | while read file; do
        echo "  - $file"
        if [ -f "$file" ]; then
            echo "    Last 3 entries:"
            tail -3 "$file" | column -t -s ','
        fi
    done
    
    echo ""
    echo "TensorBoard Logs:"
    find logs -name "events.out.tfevents.*" -type f -mmin -30 | head -5 | while read file; do
        echo "  - $file ($(ls -lh $file | awk '{print $5}'))"
    done
fi

echo ""
echo "=================================="
echo "💾 Recent Checkpoints:"
echo "=================================="

if [ -d "checkpoints/TSP" ]; then
    ls -lht checkpoints/TSP/*.ckpt 2>/dev/null | head -5 | while read line; do
        echo "  $line"
    done
else
    echo "  No checkpoints found yet"
fi

echo ""
echo "=================================="
echo "📈 Training Progress (from CSV):"
echo "=================================="

# Try to read the latest metrics from CSV
LATEST_CSV=$(find logs -name "metrics.csv" -type f -mmin -60 | head -1)
if [ -f "$LATEST_CSV" ]; then
    echo "Reading from: $LATEST_CSV"
    echo ""
    
    # Show header and last few lines
    head -1 "$LATEST_CSV"
    tail -5 "$LATEST_CSV"
else
    echo "  No recent CSV log found"
fi

echo ""
echo "=================================="
echo "🔍 Quick Commands:"
echo "=================================="
echo "  Watch this script:  watch -n 5 bash scripts/monitor_training.sh"
echo "  View TensorBoard:   tensorboard --logdir logs --port 6006"
echo "  Kill training:      kill $TRAIN_PID"
echo "  View full log:      tail -f <log_file>"
echo ""
