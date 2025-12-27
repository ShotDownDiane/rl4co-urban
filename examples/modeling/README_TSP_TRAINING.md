# TSP训练指南 - AttentionModel

## 📋 概述

TSP (Traveling Salesman Problem) 是组合优化中最经典的问题，也是AttentionModel最初设计的目标问题。本脚本展示如何使用RL4CO训练AttentionModel来解决TSP。

**问题描述：** 给定n个城市，找到访问所有城市恰好一次并返回起点的最短路径。

## 🎯 特性

- ✅ **多种规模**: 支持20城市、50城市等不同规模
- ✅ **多种解码策略**: Greedy、Sampling、Beam Search
- ✅ **高效训练**: 使用REINFORCE + Rollout Baseline
- ✅ **完整评估**: 对比不同解码策略的性能
- ✅ **混合精度**: 支持16-bit训练加速

## 🚀 快速开始

### 基础训练 (20城市)

```bash
cd examples/modeling
python test_tsp.py --mode simple
```

**配置：**
- 城市数量: 20
- 训练实例: 100,000
- Batch size: 512
- 训练轮数: 100 epochs
- 预期训练时间: ~2-3小时 (单GPU)

### 大规模训练 (50城市)

```bash
python test_tsp.py --mode larger
```

**配置：**
- 城市数量: 50
- 训练实例: 100,000
- Batch size: 256 (更大问题需要更小batch)
- 训练轮数: 100 epochs
- 预期训练时间: ~4-6小时 (单GPU)

### Beam Search训练

```bash
python test_tsp.py --mode beam
```

**特点：**
- 训练时使用sampling
- 测试时使用beam search (width=5)
- 通常能获得更好的解质量

### 评估已训练模型

```bash
python test_tsp.py --mode evaluate
```

**对比策略：**
- Greedy decoding
- Sampling
- Beam search (width=5)
- Beam search (width=10)

## 📊 预期性能

### TSP-20 (20城市)

| 方法 | 平均Tour长度 | 训练时间 | 推理速度 |
|------|-------------|---------|---------|
| Greedy | ~3.8-3.9 | - | ~1000 inst/s |
| Sampling | ~3.85-3.95 | - | ~800 inst/s |
| Beam (w=5) | ~3.75-3.85 | - | ~200 inst/s |
| Beam (w=10) | ~3.73-3.83 | - | ~100 inst/s |

*参考：最优解约为3.7-3.8*

### TSP-50 (50城市)

| 方法 | 平均Tour长度 | 训练时间 | 推理速度 |
|------|-------------|---------|---------|
| Greedy | ~5.7-5.8 | - | ~500 inst/s |
| Sampling | ~5.75-5.85 | - | ~400 inst/s |
| Beam (w=5) | ~5.65-5.75 | - | ~100 inst/s |

*参考：最优解约为5.6-5.7*

## 🏗️ 架构详解

### 模型结构

```
Input: 城市坐标 [batch_size, num_cities, 2]
  ↓
Encoder (Multi-head Attention)
  - 3 layers
  - 8 attention heads
  - 128 embedding dimension
  ↓
Node Embeddings [batch_size, num_cities, 128]
  ↓
Decoder (Autoregressive)
  - Context embedding (当前状态)
  - Pointer network (选择下一个城市)
  ↓
Output: Tour [batch_size, num_cities]
```

### 训练算法

**REINFORCE with Rollout Baseline**

```python
# 训练过程
for batch in dataloader:
    # 1. Sampling: 从策略采样获得tour
    π_θ, tour_sample = policy.sample(batch)
    reward_sample = -tour_length(tour_sample)
    
    # 2. Baseline: 贪心解码获得baseline
    tour_greedy = policy.greedy(batch)
    reward_baseline = -tour_length(tour_greedy)
    
    # 3. Advantage: 计算优势函数
    advantage = reward_sample - reward_baseline
    
    # 4. Policy Gradient: 更新策略
    loss = -log(π_θ) * advantage
    loss.backward()
```

## 📁 输出文件

### Checkpoints

```
checkpoints/tsp/
├── tsp20-epoch=00-val_reward=3.85.ckpt
├── tsp20-epoch=50-val_reward=3.82.ckpt
└── tsp20-epoch=99-val_reward=3.79.ckpt  # Best model
```

### TensorBoard Logs

```
logs/tsp/
├── version_0/
│   ├── events.out.tfevents...
│   └── hparams.yaml
└── ...
```

**查看训练过程：**
```bash
tensorboard --logdir logs/tsp/
```

## 🔧 自定义训练

### 修改超参数

```python
# 在 test_tsp.py 中修改

# 环境参数
env = TSPEnv(generator_params={
    "num_loc": 100,  # 改为100城市
})

# 模型参数
policy = AttentionModelPolicy(
    env_name=env.name,
    embed_dim=256,           # 增大embedding维度
    num_encoder_layers=6,    # 增加层数
    num_heads=16,            # 增加attention heads
)

# 训练参数
model = AttentionModel(
    env,
    policy=policy,
    batch_size=128,          # 调整batch size
    optimizer_kwargs={"lr": 5e-5},  # 调整学习率
)

# Trainer参数
trainer = pl.Trainer(
    max_epochs=200,          # 增加训练轮数
    precision="32",          # 使用全精度
)
```

### 使用预生成数据

```python
# 1. 预生成数据集
from rl4co.data.utils import save_tensordict_to_npz

td_train = env.generate_data(batch_size=[100000])
save_tensordict_to_npz(td_train, "data/tsp20_train.npz")

# 2. 在训练时加载
from rl4co.data.utils import load_npz_to_tensordict

td_train = load_npz_to_tensordict("data/tsp20_train.npz")
```

## 📈 训练曲线解读

### 关键指标

1. **train/reward**: 训练集平均reward (越高越好，负tour长度)
2. **val/reward**: 验证集平均reward
3. **train/loss**: Policy gradient loss
4. **val/loss**: 验证集loss

### 正常训练曲线

```
Epoch  Train Reward  Val Reward    Val Tour Length
----------------------------------------------------
0      -4.2          -4.3          4.3
10     -4.0          -4.1          4.1
20     -3.9          -4.0          4.0
50     -3.85         -3.88         3.88
100    -3.79         -3.82         3.82  ← 收敛
```

### 诊断问题

**1. Loss不下降**
- 检查学习率 (可能太大或太小)
- 检查gradient clipping
- 尝试warm-up策略

**2. Validation性能差**
- 过拟合：减小模型复杂度，增加训练数据
- 欠拟合：增大模型容量，训练更多轮

**3. 训练不稳定**
- 减小学习率
- 增加gradient clipping
- 使用更大的batch size

## 🎓 进阶技巧

### 1. 学习率调度

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

model = AttentionModel(
    env, policy,
    optimizer_kwargs={
        "lr": 1e-4,
        "lr_scheduler": "CosineAnnealingLR",
        "lr_scheduler_kwargs": {"T_max": 100},
    }
)
```

### 2. 多GPU训练

```python
trainer = pl.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=4,              # 使用4个GPU
    strategy="ddp",         # Distributed Data Parallel
)
```

### 3. Early Stopping

```python
from lightning.pytorch.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor="val/reward",
    patience=10,
    mode="max",
)

trainer = pl.Trainer(
    callbacks=[checkpoint_callback, early_stop]
)
```

### 4. 增强数据分布

```python
# 混合不同尺度的城市分布
env = TSPEnv(generator_params={
    "num_loc": 20,
    "min_loc": 0.0,
    "max_loc": 1.0,
    # 可以添加聚类分布等
})
```

## 🔍 性能优化

### 推理加速

```python
# 1. 使用JIT编译
model = torch.jit.script(model)

# 2. 使用批量推理
batch_size = 1024  # 更大的批量

# 3. 使用greedy解码（最快）
model.policy.decode_type = "greedy"

# 4. 半精度推理
model = model.half()
```

### 内存优化

```python
# 1. 减小batch size
batch_size = 256

# 2. 梯度累积
trainer = pl.Trainer(
    accumulate_grad_batches=4  # 累积4个batch
)

# 3. 混合精度
trainer = pl.Trainer(
    precision="16-mixed"
)
```

## 📚 参考资料

### 论文
- **Attention, Learn to Solve Routing Problems!** (Kool et al., 2019)
  - [Paper](https://arxiv.org/abs/1803.08475)
  - [Original Code](https://github.com/wouterkool/attention-learn-to-route)

### RL4CO文档
- [RL4CO GitHub](https://github.com/ai4co/rl4co)
- [Documentation](https://rl4co.readthedocs.io/)

## ❓ 常见问题

### Q1: 训练需要多长时间？
**A**: 在单个V100 GPU上：
- TSP-20: ~2-3小时 (100 epochs)
- TSP-50: ~4-6小时 (100 epochs)
- TSP-100: ~10-15小时 (100 epochs)

### Q2: 需要多少GPU内存？
**A**: 
- TSP-20 (batch=512): ~6GB
- TSP-50 (batch=256): ~8GB
- TSP-100 (batch=128): ~12GB

### Q3: 如何提高解的质量？
**A**: 
1. 使用beam search (width=10-20)
2. 训练更长时间 (200+ epochs)
3. 增大模型容量 (更多layers/heads)
4. 使用ensemble (多个模型投票)

### Q4: 能否迁移到其他规模？
**A**: 部分可以：
- TSP-20训练的模型可以推广到TSP-30
- 但对TSP-100效果会下降
- 建议针对目标规模单独训练

### Q5: 与传统启发式算法比较如何？
**A**: 
- **速度**: 神经网络快10-100倍
- **质量**: 接近LKH等启发式算法 (差距<5%)
- **泛化**: 可以处理不同分布的实例

## 🎉 总结

TSP是学习组合优化神经网络方法的最佳入门问题：
- ✅ 问题简单明确
- ✅ AttentionModel专为TSP设计
- ✅ 训练快速稳定
- ✅ 结果容易可视化
- ✅ 性能优秀可靠

**下一步：**
1. 掌握TSP后，尝试CVRP (带容量约束)
2. 探索MCLP、FLP等图优化问题
3. 学习改进方法 (POMO, Sym-NCO等)

---

**Happy Training! 🚀**
