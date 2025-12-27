# Quick Reference Card

## 🚀 Quick Commands

### Test the Pipeline
```bash
bash scripts/quick_test.sh
```

### Train Individual Problems
```bash
bash scripts/train_tsp.sh    # TSP
bash scripts/train_flp.sh    # FLP
bash scripts/train_mclp.sh   # MCLP
bash scripts/train_stp.sh    # STP
```

### Train All Problems
```bash
bash scripts/train_all.sh
```

### Monitor Training
```bash
tensorboard --logdir logs
```

## 📋 Common Commands

### Minimal Training (Fast)
```bash
python train.py --problem TSP --num-loc 10 --train-size 1000 --epochs 2
```

### Standard Training
```bash
python train.py --problem TSP --num-loc 20 --epochs 100 --visualize
```

### Full Training (Production)
```bash
python train.py --problem TSP --num-loc 50 --train-size 500000 --epochs 200
```

### Force Data Regeneration
```bash
python train.py --problem TSP --num-loc 20 --force-regenerate
```

## 🗂️ File Locations

| Item | Location |
|------|----------|
| **Training Script** | `train.py` |
| **Cached Data** | `data/` |
| **Checkpoints** | `checkpoints/{PROBLEM}/` |
| **TensorBoard Logs** | `logs/{PROBLEM}_AttentionModel/` |
| **Evaluation Results** | `logs/{PROBLEM}_evaluation/` |
| **Visualizations** | `logs/{PROBLEM}_evaluation/{PROBLEM}_solutions.png` |

## 🎯 Problem-Specific Settings

### TSP (Traveling Salesman Problem)
```bash
python train.py --problem TSP --num-loc 20
```

### FLP (Facility Location Problem)
```bash
python train.py --problem FLP --num-loc 50 --num-facilities 10
```

### MCLP (Maximum Coverage Location Problem)
```bash
python train.py --problem MCLP --num-loc 30
```

### STP (Steiner Tree Problem)
```bash
python train.py --problem STP --num-loc 25
```

## ⚙️ Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--problem` | Problem type (FLP/MCLP/STP/TSP) | Required |
| `--num-loc` | Number of locations | 20 |
| `--train-size` | Training samples | 100,000 |
| `--val-size` | Validation samples | 10,000 |
| `--test-size` | Test samples | 10,000 |
| `--epochs` | Training epochs | 100 |
| `--embed-dim` | Embedding dimension | 128 |
| `--learning-rate` | Learning rate | 1e-4 |
| `--visualize` | Generate plots | False |

## 📊 Check Results

### View Test Results
```bash
cat logs/TSP_evaluation/TSP_test_results.json
```

### View Visualizations
```bash
# Linux
xdg-open logs/TSP_evaluation/TSP_solutions.png

# macOS
open logs/TSP_evaluation/TSP_solutions.png
```

### List Checkpoints
```bash
ls -lh checkpoints/TSP/
```

## 🔍 Troubleshooting

| Problem | Solution |
|---------|----------|
| Too slow | Reduce `--train-size` or `--num-loc` |
| Out of memory | Reduce `--embed-dim` to 64 |
| Not improving | Check learning rate, try 1e-5 |
| Data generation slow | First run only, subsequent runs use cache |

## 📚 Documentation

- **Quick Start**: `QUICKSTART.md`
- **Full Docs**: `TRAIN_README.md`
- **Implementation**: `IMPLEMENTATION_SUMMARY.md`
- **Example Notebook**: `examples/2-full-training.ipynb`

## 🎓 Example Workflows

### Workflow 1: Quick Experiment
```bash
# Test with small settings
python train.py --problem TSP --num-loc 10 --train-size 5000 --epochs 5

# Monitor
tensorboard --logdir logs

# Check results
cat logs/TSP_evaluation/TSP_test_results.json
```

### Workflow 2: Production Training
```bash
# Full training
python train.py --problem TSP --num-loc 50 --train-size 500000 --epochs 200 --visualize

# Monitor progress
tensorboard --logdir logs

# View results
cat logs/TSP_evaluation/TSP_test_results.json
xdg-open logs/TSP_evaluation/TSP_solutions.png
```

### Workflow 3: Hyperparameter Search
```bash
# Try different learning rates
for lr in 1e-3 1e-4 1e-5; do
  python train.py --problem TSP --learning-rate $lr --log-dir logs/lr_${lr}
done

# Compare in TensorBoard
tensorboard --logdir logs
```

## ✅ Checklist

- [ ] Install dependencies: `pip install -e .`
- [ ] Run quick test: `bash scripts/quick_test.sh`
- [ ] Start training: `bash scripts/train_tsp.sh`
- [ ] Monitor with TensorBoard: `tensorboard --logdir logs`
- [ ] Check results: `cat logs/TSP_evaluation/TSP_test_results.json`
- [ ] View visualizations

## 🆘 Get Help

```bash
# Show all options
python train.py --help

# View documentation
cat QUICKSTART.md
cat TRAIN_README.md
cat IMPLEMENTATION_SUMMARY.md
```

---

**Created by**: Modular Training Pipeline v1.0  
**Date**: 2025-12-05
