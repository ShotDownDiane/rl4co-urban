# FLP Training Guide

完整的FLP (Facility Location Problem) 训练示例，参考TSP的实现。

## 🎯 功能特性

- ✅ **Simple Mode**: 快速训练示例（10 epochs）
- ✅ **Advanced Mode**: 完整训练配置（50 epochs，beam search）
- ✅ **From Checkpoint**: 从检查点继续训练
- ✅ **Evaluate Mode**: 评估已训练模型

## 🚀 使用方法

### 1. 简单训练（推荐入门）

```bash
cd examples/modeling
python test_flp.py --mode simple
```

**配置：**
- 问题规模：50个候选点，选择5个设施
- 训练数据：10,000个实例
- 训练轮数：10 epochs
- 解码策略：
  - 训练时：sampling（从分布中采样）
  - 验证时：greedy（贪心）
  - 测试时：greedy

**输出：**
- Checkpoints: `checkpoints/flp/`
- TensorBoard logs: `logs/flp/`

### 2. 高级训练

```bash
python test_flp.py --mode advanced
```

**配置：**
- 问题规模：100个候选点，选择10个设施（更大）
- 训练数据：50,000个实例
- 训练轮数：50 epochs
- 解码策略：
  - 训练时：sampling
  - 验证时：greedy
  - 测试时：beam search (width=5) ⭐ 更高质量

### 3. 从检查点继续训练

```bash
python test_flp.py --mode from_ckpt
```

需要先有checkpoint文件（通过simple或advanced模式生成）。

### 4. 评估模型

```bash
python test_flp.py --mode evaluate
```

对比不同解码策略的效果：
- Greedy decoding
- Sampling (10次采样取最好)

## 📊 监控训练

### TensorBoard

```bash
tensorboard --logdir logs/flp/
```

然后在浏览器打开 `http://localhost:6006`

**可以看到：**
- Training loss
- Validation reward
- Learning rate
- Gradient norms

## 🎓 核心概念

### 1. Environment（环境）

```python
env = FLPEnv(generator_params={
    "num_loc": 50,      # 候选设施位置数量
    "to_choose": 5,     # 需要选择的设施数量
})
```

### 2. Policy（策略网络）

```python
policy = AttentionModelPolicy(
    env_name=env.name,
    embed_dim=128,           # 嵌入维度
    num_encoder_layers=3,    # 编码器层数
    num_heads=8,             # 注意力头数
)
```

**架构：** Encoder-Decoder with Attention
- **Encoder**: 处理所有候选位置，生成上下文表示
- **Decoder**: 自回归地选择设施，每次选择一个

### 3. Model（RL算法）

```python
model = AttentionModel(
    env,
    policy=policy,
    baseline="rollout",     # 基线：rollout（贪心策略的奖励作为基线）
    batch_size=512,         # 训练批次大小
    optimizer_kwargs={"lr": 1e-4},  # 学习率
)
```

**算法：** REINFORCE with Baseline
- 目标：最大化期望奖励
- 基线：减少方差，加速训练

### 4. Decoding Strategies（解码策略）

| 策略 | 特点 | 用途 |
|------|------|------|
| **Sampling** | 从分布中采样 | 训练时探索 |
| **Greedy** | 总是选择概率最高的动作 | 验证/快速推理 |
| **Beam Search** | 保留top-k个候选序列 | 测试/高质量解 |

## 📈 预期结果

### 训练曲线

- **Epoch 1-3**: 快速下降（学习基础策略）
- **Epoch 4-7**: 平稳改进
- **Epoch 8-10**: 收敛

### FLP Reward

FLP的reward是**负的总距离**（越大越好，即距离越小）：
```
Reward = -sum(min(distance[i, selected_facilities]))
```

**典型值：**
- 随机策略：~-5.0 到 -6.0
- 训练后：~-3.0 到 -4.0
- 接近最优：~-2.5

## 🔧 自定义配置

### 修改问题规模

```python
env = FLPEnv(generator_params={
    "num_loc": 100,      # 增加到100
    "to_choose": 20,     # 选择20个
})
```

### 修改网络架构

```python
policy = AttentionModelPolicy(
    env_name=env.name,
    embed_dim=256,           # 增加嵌入维度
    num_encoder_layers=6,    # 更深的编码器
    num_heads=16,            # 更多注意力头
    normalization="batch",   # 批归一化
    feed_forward_hidden=512, # FF层维度
)
```

### 修改训练参数

```python
model = AttentionModel(
    env,
    baseline="rollout",
    batch_size=1024,            # 更大批次
    val_batch_size=128,
    train_data_size=100_000,    # 更多数据
    optimizer_kwargs={
        "lr": 1e-4,
        "weight_decay": 1e-6,   # 权重衰减
    },
    lr_scheduler={              # 学习率调度
        "type": "StepLR",
        "step_size": 10,
        "gamma": 0.96,
    },
)
```

## 💡 Tips

### 1. 显存不足？

- 减少 `batch_size`
- 减少 `embed_dim`
- 减少 `num_encoder_layers`

### 2. 训练太慢？

- 增加 `batch_size`（如果显存允许）
- 减少 `train_data_size`（快速实验）
- 使用更小的问题规模

### 3. 模型不收敛？

- 检查学习率（尝试 1e-5 或 1e-3）
- 增加训练轮数
- 检查梯度裁剪（`gradient_clip_val`）
- 尝试不同的baseline（`shared`, `exponential`）

### 4. 想要更好的解？

- 训练更多轮数
- 使用beam search进行推理
- 使用sampling策略多次采样取最优

## 📚 代码结构

```python
# 1. 创建环境
env = FLPEnv(...)

# 2. 创建策略网络
policy = AttentionModelPolicy(...)

# 3. 创建RL模型
model = AttentionModel(env, policy, ...)

# 4. 创建训练器
trainer = pl.Trainer(...)

# 5. 训练
trainer.fit(model)

# 6. 测试
trainer.test(model)
```

## 🔍 对比不同方法

| 方法 | 速度 | 质量 | 适用场景 |
|------|------|------|----------|
| Random | ⚡⚡⚡ | ⭐ | Baseline |
| Greedy Heuristic | ⚡⚡ | ⭐⭐ | 快速近似 |
| RL (Greedy) | ⚡⚡ | ⭐⭐⭐ | 实时推理 |
| RL (Sampling x10) | ⚡ | ⭐⭐⭐⭐ | 离线优化 |
| RL (Beam Search) | ⚡ | ⭐⭐⭐⭐⭐ | 高质量解 |
| Exact Solver | 🐌 | ⭐⭐⭐⭐⭐ | 小规模最优 |

## 🎯 下一步

1. **尝试不同问题规模**: 从小到大逐步增加
2. **对比不同baseline**: `rollout`, `shared`, `exponential`
3. **尝试其他模型**: POMO, SymNCO
4. **迁移到MCLP**: 应用到其他问题
5. **生成对比结果**: 可视化训练过程

## 📖 相关资料

- [RL4CO Documentation](https://github.com/ai4co/rl4co)
- [Attention, Learn to Solve Routing Problems!](https://arxiv.org/abs/1803.08475)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
