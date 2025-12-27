# Pre-generating Datasets for RL4CO

This guide explains how to pre-generate and use datasets for training RL4CO models.

## 🎯 Why Pre-generate?

**Benefits:**
- ✅ **Faster training**: No need to regenerate instances every epoch
- ✅ **Reproducibility**: Same data across experiments
- ✅ **Fair comparison**: Train different models on identical data
- ✅ **Efficiency**: Generate once, use many times
- ✅ **OOD testing**: Generate different distributions for testing

## 🚀 Quick Start

### 1. Generate a Dataset

```bash
# Generate FLP dataset (100 locations, choose 10)
python tools/pregenerate_dataset.py \
    --env flp \
    --num-loc 100 \
    --to-choose 10 \
    --num-train 10000 \
    --num-val 1000 \
    --num-test 1000

# Generate MCLP dataset (uniform distribution)
python tools/pregenerate_dataset.py \
    --env mclp \
    --num-demand 50 \
    --num-facility 30 \
    --num-facilities-to-select 8 \
    --distribution uniform \
    --num-train 10000 \
    --num-val 1000 \
    --num-test 1000

# Generate MCLP dataset (cluster distribution with dynamic radius)
python tools/pregenerate_dataset.py \
    --env mclp \
    --num-demand 100 \
    --num-facility 50 \
    --num-facilities-to-select 10 \
    --distribution cluster \
    --dynamic-radius \
    --num-train 10000 \
    --num-val 1000 \
    --num-test 1000
```

### 2. Load and Use in Training

```python
from rl4co.data.utils import load_npz_to_tensordict
from rl4co.envs.graph import FLPEnv

# Load pre-generated data
train_data = load_npz_to_tensordict("data/pregenerated/flp_n100_k10/flp_train.npz")
val_data = load_npz_to_tensordict("data/pregenerated/flp_n100_k10/flp_val.npz")

# Create environment
env = FLPEnv(generator_params={"num_loc": 100, "to_choose": 10})

# Use with environment
device = torch.device("cuda")
train_data = train_data.to(device)

# Reset with pre-generated data
td_reset = env.reset(td=train_data)

# Now you can use td_reset for training/evaluation
```

## 📊 Environment-Specific Examples

### FLP (Facility Location Problem)

```bash
# Small instances
python tools/pregenerate_dataset.py \
    --env flp \
    --num-loc 50 \
    --to-choose 5 \
    --num-train 10000

# Large instances
python tools/pregenerate_dataset.py \
    --env flp \
    --num-loc 200 \
    --to-choose 20 \
    --num-train 10000 \
    --batch-size 500
```

### MCLP (Maximum Covering Location Problem)

```bash
# Uniform distribution with fixed radius
python tools/pregenerate_dataset.py \
    --env mclp \
    --num-demand 100 \
    --num-facility 50 \
    --num-facilities-to-select 10 \
    --coverage-radius 0.2 \
    --distribution uniform \
    --num-train 10000

# Cluster distribution with dynamic radius
python tools/pregenerate_dataset.py \
    --env mclp \
    --num-demand 100 \
    --num-facility 50 \
    --num-facilities-to-select 10 \
    --distribution cluster \
    --dynamic-radius \
    --num-train 10000

# Explosion distribution
python tools/pregenerate_dataset.py \
    --env mclp \
    --num-demand 100 \
    --num-facility 50 \
    --num-facilities-to-select 10 \
    --distribution explosion \
    --num-train 10000
```

### MCP (Maximum Coverage Problem)

```bash
python tools/pregenerate_dataset.py \
    --env mcp \
    --num-items 100 \
    --num-sets 50 \
    --n-sets-to-choose 10 \
    --num-train 10000
```

### STP (Steiner Tree Problem)

```bash
python tools/pregenerate_dataset.py \
    --env stp \
    --num-nodes 50 \
    --num-terminals 10 \
    --num-train 10000
```

## 🔧 Advanced Usage

### Generate OOD Test Sets

Generate multiple distributions for out-of-distribution testing:

```bash
# Train on uniform
python tools/pregenerate_dataset.py \
    --env mclp \
    --distribution uniform \
    --num-train 10000 \
    --num-val 0 \
    --num-test 0 \
    --output-dir data/pregenerated/mclp_train

# Test on cluster
python tools/pregenerate_dataset.py \
    --env mclp \
    --distribution cluster \
    --num-train 0 \
    --num-val 0 \
    --num-test 1000 \
    --output-dir data/pregenerated/mclp_test_cluster

# Test on explosion
python tools/pregenerate_dataset.py \
    --env mclp \
    --distribution explosion \
    --num-train 0 \
    --num-val 0 \
    --num-test 1000 \
    --output-dir data/pregenerated/mclp_test_explosion
```

### Compress Large Datasets

Use `--compress` flag for large datasets:

```bash
python tools/pregenerate_dataset.py \
    --env flp \
    --num-loc 500 \
    --num-train 100000 \
    --compress
```

### Batch Size Tuning

Adjust batch size based on available memory:

```bash
# Small GPU memory (< 8GB)
python tools/pregenerate_dataset.py \
    --env mclp \
    --num-train 10000 \
    --batch-size 100

# Large GPU memory (> 32GB)
python tools/pregenerate_dataset.py \
    --env mclp \
    --num-train 100000 \
    --batch-size 5000
```

## 📁 Output Structure

```
data/pregenerated/
├── flp_n100_k10/
│   ├── flp_train.npz    (10,000 instances, ~2 MB)
│   ├── flp_val.npz      (1,000 instances, ~200 KB)
│   └── flp_test.npz     (1,000 instances, ~200 KB)
├── mclp_d100_f50_k10_uniform/
│   ├── mclp_train.npz
│   ├── mclp_val.npz
│   └── mclp_test.npz
└── mclp_d100_f50_k10_cluster/
    └── mclp_test.npz     (OOD test set)
```

## 🧪 Testing

Run the test script to verify pre-generation works:

```bash
python tests/test_pregenerate_instances.py
```

This will:
1. Generate 10 FLP instances
2. Save them to disk
3. Load them back
4. Compare original vs loaded (should match exactly)
5. Test rollout consistency

## 💡 Tips

### Memory Optimization
- Use smaller `--batch-size` if you run out of memory
- Generate in multiple smaller files if dataset is huge
- Use `--compress` for storage efficiency (slower I/O)

### Speed Optimization
- Increase `--batch-size` for faster generation (if memory allows)
- Generate on CPU if GPU is busy training
- Don't compress if I/O speed is critical

### Best Practices
1. **Always test first**: Generate small dataset (e.g., 100 instances) to verify parameters
2. **Consistent naming**: Use descriptive directory names with parameters
3. **Document parameters**: Keep a log of generation parameters
4. **Version control**: Track dataset versions for reproducibility
5. **OOD testing**: Generate diverse test sets for robustness evaluation

## 🔍 Verification

### Quick Check

```python
from rl4co.data.utils import load_npz_to_tensordict

# Load data
td = load_npz_to_tensordict("data/pregenerated/flp_n100_k10/flp_train.npz")

# Check shape
print(f"Batch size: {td.batch_size}")  # Should be [10000]
print(f"Keys: {list(td.keys())}")
print(f"Locations shape: {td['locs'].shape}")  # Should be [10000, 100, 2]
```

### Full Validation

```bash
python tests/test_pregenerate_instances.py
```

## 📚 Example Training Script

```python
import torch
from rl4co.data.utils import load_npz_to_tensordict
from rl4co.envs.graph import MCLPEnv
from rl4co.utils.decoding import rollout, greedy_policy

# Load pre-generated data
train_data = load_npz_to_tensordict("data/pregenerated/mclp_d100_f50_k10_uniform/mclp_train.npz")
val_data = load_npz_to_tensordict("data/pregenerated/mclp_d100_f50_k10_uniform/mclp_val.npz")

# Create environment
env = MCLPEnv(generator_params={
    "num_demand": 100,
    "num_facility": 50,
    "num_facilities_to_select": 10,
})

# Move to device
device = torch.device("cuda")
train_data = train_data.to(device)

# Training loop
for epoch in range(num_epochs):
    # Sample a batch
    batch_size = 128
    indices = torch.randperm(len(train_data))[:batch_size]
    batch_td = train_data[indices]
    
    # Reset with batch
    td = env.reset(td=batch_td)
    
    # Your training logic here...
    reward, td_final, actions = rollout(env, td, your_policy)
    
    # Compute loss and update...
```

## 🆘 Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce `--batch-size` or generate in multiple smaller runs

### Issue: Slow Generation
**Solution**: Increase `--batch-size` or check if GPU is being used

### Issue: File Size Too Large
**Solution**: Use `--compress` flag or reduce number of instances

### Issue: Inconsistent Results
**Solution**: Check that environment parameters match when loading

## 📖 Reference

### All Arguments

```
--env                      Environment name (flp, mclp, mcp, stp)
--output-dir               Output directory (default: data/pregenerated)
--num-train                Number of training instances (default: 10000)
--num-val                  Number of validation instances (default: 1000)
--num-test                 Number of test instances (default: 1000)
--batch-size               Batch size for generation (default: 1000)
--compress                 Compress npz files

FLP-specific:
--num-loc                  Number of locations (default: 100)
--to-choose                Number of facilities to choose (default: 10)

MCLP-specific:
--num-demand               Number of demand points (default: 100)
--num-facility             Number of candidate facilities (default: 50)
--num-facilities-to-select Number of facilities to select (default: 10)
--coverage-radius          Coverage radius (default: 0.2)
--distribution             Data distribution: uniform/cluster/explosion
--dynamic-radius           Use dynamic radius calculation

MCP-specific:
--num-items                Number of items (default: 100)
--num-sets                 Number of sets (default: 50)
--n-sets-to-choose         Number of sets to choose (default: 10)

STP-specific:
--num-nodes                Number of nodes (default: 50)
--num-terminals            Number of terminal nodes (default: 10)
```
