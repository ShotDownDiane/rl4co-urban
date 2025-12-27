# Training Pipeline Documentation

This document describes the modular training pipeline implemented in `train.py` for training RL models on FLP, MCLP, STP, and TSP problems.

## Features

### 1. **Modular Architecture**
The pipeline is divided into three main modules:
- **Data Module**: Handles dataset generation, caching, and loading
- **Model Module**: Builds and configures the RL model
- **Evaluation Module**: Evaluates trained models with visualization support

### 2. **Data Management**
- Generates datasets for train/val/test splits
- Automatically caches datasets to avoid regeneration
- Saves data to `data/` or `cache/` directory
- Train set is shuffled, val/test sets are not shuffled
- Supports force regeneration with `--force-regenerate` flag

### 3. **Model Training**
- Based on the full training notebook implementation
- Uses AttentionModel with REINFORCE
- Supports multiple baseline types (rollout, exponential, critic)
- Automatic checkpointing with top-k model saving
- Comprehensive logging with TensorBoard

### 4. **Evaluation & Visualization**
- Automatic evaluation on test set after training
- Generates visualizations of solutions
- Saves results in JSON format
- TensorBoard integration for monitoring

## Usage

### Basic Usage

Train a TSP model with default settings:
```bash
python train.py --problem TSP --num-loc 20 --epochs 100
```

Train a FLP model:
```bash
python train.py --problem FLP --num-loc 50 --num-facilities 10 --epochs 100
```

Train a MCLP model:
```bash
python train.py --problem MCLP --num-loc 30 --epochs 100
```

Train a STP model:
```bash
python train.py --problem STP --num-loc 25 --epochs 100
```

### Advanced Usage

Full configuration example:
```bash
python train.py \
  --problem TSP \
  --num-loc 50 \
  --train-size 100000 \
  --val-size 10000 \
  --test-size 10000 \
  --epochs 100 \
  --embed-dim 128 \
  --num-encoder-layers 3 \
  --num-heads 8 \
  --learning-rate 1e-4 \
  --baseline rollout \
  --data-dir data \
  --checkpoint-dir checkpoints \
  --log-dir logs \
  --visualize
```

### Command Line Arguments

#### Problem Settings
- `--problem`: Problem type (required, choices: FLP, MCLP, STP, TSP)
- `--num-loc`: Number of locations/nodes (default: 20)
- `--num-facilities`: Number of facilities for FLP/MCLP (optional)

#### Data Settings
- `--data-dir`: Directory for data storage (default: 'data')
- `--train-size`: Training dataset size (default: 100,000)
- `--val-size`: Validation dataset size (default: 10,000)
- `--test-size`: Test dataset size (default: 10,000)
- `--force-regenerate`: Force regenerate datasets even if cached

#### Model Settings
- `--model-type`: Model architecture (default: 'AttentionModel')
- `--baseline`: Baseline type (choices: rollout, exponential, critic; default: 'rollout')
- `--embed-dim`: Embedding dimension (default: 128)
- `--num-encoder-layers`: Number of encoder layers (default: 3)
- `--num-heads`: Number of attention heads (default: 8)
- `--learning-rate`: Learning rate (default: 1e-4)

#### Training Settings
- `--epochs`: Number of training epochs (default: 100)
- `--checkpoint-dir`: Directory for model checkpoints (default: 'checkpoints')
- `--log-dir`: Directory for TensorBoard logs (default: 'logs')

#### Evaluation Settings
- `--skip-evaluation`: Skip evaluation after training
- `--eval-batch-size`: Batch size for evaluation (default: 100)
- `--visualize`: Generate visualizations of solutions

## Directory Structure

After running the pipeline, the following directory structure is created:

```
rl4co-urban/
├── train.py                    # Main training script
├── data/                       # Cached datasets
│   ├── TSP_train_num_loc_20.pkl
│   ├── TSP_val_num_loc_20.pkl
│   ├── TSP_test_num_loc_20.pkl
│   └── TSP_metadata.json
├── checkpoints/                # Model checkpoints
│   └── TSP/
│       ├── epoch_000.ckpt
│       ├── epoch_050.ckpt
│       └── last.ckpt
└── logs/                       # TensorBoard logs and evaluation results
    ├── TSP_AttentionModel/
    │   └── version_0/
    │       └── events.out.tfevents.*
    └── TSP_evaluation/
        ├── TSP_test_results.json
        └── TSP_solutions.png
```

## Monitoring Training

### TensorBoard

Launch TensorBoard to monitor training progress:
```bash
tensorboard --logdir logs
```

Then open your browser and navigate to `http://localhost:6006`

### Metrics Logged

- **Training metrics**: 
  - `train/loss`: Training loss
  - `train/reward`: Training reward
  
- **Validation metrics**:
  - `val/reward`: Validation reward (used for checkpointing)
  - `val/loss`: Validation loss

- **Test metrics**:
  - `test/reward`: Final test reward
  - Mean, std, min, max costs

## Examples

### Quick Start Examples

Create example training scripts in `examples/` directory:

1. **TSP Training (20 nodes)**:
```bash
python train.py --problem TSP --num-loc 20 --epochs 50 --visualize
```

2. **TSP Training (50 nodes)**:
```bash
python train.py --problem TSP --num-loc 50 --epochs 100 --train-size 200000
```

3. **FLP Training**:
```bash
python train.py --problem FLP --num-loc 50 --num-facilities 10 --epochs 100
```

4. **MCLP Training**:
```bash
python train.py --problem MCLP --num-loc 30 --epochs 100 --visualize
```

5. **STP Training**:
```bash
python train.py --problem STP --num-loc 25 --epochs 100
```

### Resume Training

To resume training from a checkpoint:
```python
# Modify train.py or create a separate script
model = AttentionModel.load_from_checkpoint("checkpoints/TSP/last.ckpt")
trainer.fit(model)
```

## Data Caching

The data module automatically caches generated datasets. The cache filename includes:
- Problem type
- Split (train/val/test)
- Generator parameters

Example cache file: `TSP_train_num_loc_50.pkl`

To regenerate data (e.g., after changing generator parameters):
```bash
python train.py --problem TSP --num-loc 20 --force-regenerate
```

## Evaluation Results

After training, evaluation results are saved in JSON format:

```json
{
  "mean_reward": 5.234,
  "std_reward": 0.456,
  "min_reward": 4.123,
  "max_reward": 7.890
}
```

## Tips

1. **GPU Memory**: If you encounter OOM errors, reduce `--train-size` or `--embed-dim`
2. **Training Time**: Start with smaller epochs (e.g., 10-20) to test the pipeline
3. **Data Caching**: First run will generate and cache data, subsequent runs will be faster
4. **Hyperparameters**: The default settings are reasonable, but may need tuning for specific problems
5. **Visualization**: Use `--visualize` to generate solution plots for qualitative analysis

## Troubleshooting

### Issue: Dataset generation is slow
**Solution**: Generated datasets are cached. First run is slow, but subsequent runs will load from cache.

### Issue: Out of GPU memory
**Solution**: Reduce `--train-size`, `--embed-dim`, or number of encoder layers.

### Issue: Training not improving
**Solution**: 
- Check learning rate (try 1e-4 or 1e-5)
- Increase number of training samples
- Try different baseline types

### Issue: Cached data is outdated
**Solution**: Use `--force-regenerate` flag to regenerate datasets.

## Citation

If you use this training pipeline, please cite the RL4CO paper:

```bibtex
@article{berto2023rl4co,
  title={RL4CO: an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark},
  author={Berto, Federico and others},
  journal={arXiv preprint arXiv:2306.17100},
  year={2023}
}
```
