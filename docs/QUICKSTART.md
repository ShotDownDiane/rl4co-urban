# Quick Start Guide

This guide will help you quickly get started with training RL models on FLP, MCLP, STP, and TSP problems.

## Prerequisites

Ensure you have rl4co installed and all dependencies:
```bash
pip install -e .
```

## Quick Test

First, run a quick test to verify everything works:

```bash
bash scripts/quick_test.sh
```

This will train a small TSP model for 2 epochs with minimal data. It should complete in a few minutes.

## Training Individual Problems

### 1. Train TSP Model

```bash
# Option 1: Use the convenience script
bash scripts/train_tsp.sh

# Option 2: Run directly with custom settings
python train.py --problem TSP --num-loc 20 --epochs 100 --visualize
```

### 2. Train FLP Model

```bash
# Option 1: Use the convenience script
bash scripts/train_flp.sh

# Option 2: Run directly with custom settings
python train.py --problem FLP --num-loc 50 --num-facilities 10 --epochs 100 --visualize
```

### 3. Train MCLP Model

```bash
# Option 1: Use the convenience script
bash scripts/train_mclp.sh

# Option 2: Run directly with custom settings
python train.py --problem MCLP --num-loc 30 --epochs 100 --visualize
```

### 4. Train STP Model

```bash
# Option 1: Use the convenience script
bash scripts/train_stp.sh

# Option 2: Run directly with custom settings
python train.py --problem STP --num-loc 25 --epochs 100 --visualize
```

## Train All Problems

To train all problems sequentially:

```bash
bash scripts/train_all.sh
```

**Note**: This will take a long time! Consider running individual scripts or reducing epochs for testing.

## Monitor Training

### Using TensorBoard

Open a new terminal and run:

```bash
tensorboard --logdir logs
```

Then navigate to http://localhost:6006 in your browser.

You can monitor:
- Training loss and rewards
- Validation rewards
- Learning rate
- Gradient norms

### Check Progress

Monitor training progress in the terminal output. You'll see:
- Data generation/loading status
- Model architecture summary
- Training progress with loss and rewards
- Validation results
- Final test results

## Understanding the Output

### Directory Structure

After training, you'll have:

```
rl4co-urban/
├── data/                       # Cached datasets
│   ├── TSP_train_num_loc_20.pkl
│   ├── TSP_val_num_loc_20.pkl
│   ├── TSP_test_num_loc_20.pkl
│   └── TSP_metadata.json
│
├── checkpoints/                # Model checkpoints
│   ├── TSP/
│   │   ├── epoch_000.ckpt
│   │   ├── epoch_050.ckpt
│   │   └── last.ckpt
│   ├── FLP/
│   ├── MCLP/
│   └── STP/
│
└── logs/                       # TensorBoard logs
    ├── TSP_AttentionModel/
    ├── FLP_AttentionModel/
    ├── MCLP_AttentionModel/
    ├── STP_AttentionModel/
    ├── TSP_evaluation/
    │   ├── TSP_test_results.json
    │   └── TSP_solutions.png
    └── ...
```

### Evaluation Results

After training, check the evaluation results:

```bash
cat logs/TSP_evaluation/TSP_test_results.json
```

Example output:
```json
{
  "mean_reward": 5.234,
  "std_reward": 0.456,
  "min_reward": 4.123,
  "max_reward": 7.890
}
```

### Visualizations

If you used `--visualize`, check the solution visualizations:

```bash
# View solution plots
open logs/TSP_evaluation/TSP_solutions.png  # macOS
xdg-open logs/TSP_evaluation/TSP_solutions.png  # Linux
```

## Common Use Cases

### Case 1: Quick Experiment (Small Scale)

```bash
python train.py \
  --problem TSP \
  --num-loc 10 \
  --train-size 10000 \
  --val-size 1000 \
  --test-size 1000 \
  --epochs 10 \
  --visualize
```

### Case 2: Full Training (Production)

```bash
python train.py \
  --problem TSP \
  --num-loc 50 \
  --train-size 500000 \
  --val-size 50000 \
  --test-size 10000 \
  --epochs 200 \
  --embed-dim 128 \
  --num-encoder-layers 3 \
  --visualize
```

### Case 3: Hyperparameter Tuning

```bash
# Experiment with different learning rates
for lr in 1e-3 1e-4 1e-5; do
  python train.py \
    --problem TSP \
    --num-loc 20 \
    --epochs 50 \
    --learning-rate $lr \
    --log-dir logs/lr_${lr}
done
```

### Case 4: Data Regeneration

If you change generator parameters, regenerate the data:

```bash
python train.py \
  --problem TSP \
  --num-loc 30 \
  --force-regenerate \
  --epochs 100
```

## Tips for Success

1. **Start Small**: Always run a quick test first with small settings
2. **Use Cached Data**: First run generates data, subsequent runs are faster
3. **Monitor GPU Usage**: Use `nvidia-smi` to check GPU utilization
4. **Save Checkpoints**: The pipeline automatically saves top-3 models
5. **Check TensorBoard**: Monitor training progress in real-time
6. **Visualize Results**: Use `--visualize` to inspect solution quality

## Troubleshooting

### Problem: Training is too slow

**Solution**: 
- Reduce `--train-size`
- Reduce `--num-loc`
- Use fewer `--epochs` for testing

### Problem: Out of GPU memory

**Solution**:
- Reduce `--embed-dim` (e.g., from 128 to 64)
- Reduce `--num-encoder-layers` (e.g., from 3 to 2)
- Reduce batch size in the code

### Problem: Model not improving

**Solution**:
- Check learning rate (try 1e-4 or 1e-5)
- Increase training samples
- Train for more epochs
- Try different baseline types

### Problem: Data generation is slow

**Solution**:
- Be patient on first run (data is being generated and cached)
- Subsequent runs will load from cache and be much faster
- Check progress in terminal output

## Next Steps

1. **Read Full Documentation**: See `TRAIN_README.md` for detailed documentation
2. **Customize Training**: Modify the scripts or create your own
3. **Experiment**: Try different hyperparameters and problem sizes
4. **Evaluate**: Analyze results in TensorBoard and JSON files
5. **Deploy**: Use trained models for inference

## Example Workflow

Here's a complete workflow from start to finish:

```bash
# Step 1: Quick test
bash scripts/quick_test.sh

# Step 2: Train TSP model
bash scripts/train_tsp.sh

# Step 3: Monitor training (in another terminal)
tensorboard --logdir logs

# Step 4: Check results
cat logs/TSP_evaluation/TSP_test_results.json

# Step 5: View visualizations
# Open logs/TSP_evaluation/TSP_solutions.png
```

## Need Help?

- Check the detailed documentation: `TRAIN_README.md`
- Review example scripts in `scripts/`
- Look at the notebook: `examples/2-full-training.ipynb`

## Citation

If you use this training pipeline, please cite:

```bibtex
@article{berto2023rl4co,
  title={RL4CO: an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark},
  author={Berto, Federico and others},
  journal={arXiv preprint arXiv:2306.17100},
  year={2023}
}
```
