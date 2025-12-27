# 快速开始：训练FLP和MCLP

## ✅ 已完成

1. **预生成数据功能** ✓
   - 测试脚本：`tests/test_pregenerate_instances.py`
   - 工具脚本：`tools/pregenerate_dataset.py`
   - 使用指南：`tools/README_PREGENERATE.md`

2. **FLP训练脚本** ✓
   - 脚本：`examples/modeling/test_flp.py`
   - 使用指南：`examples/modeling/README_FLP_TRAINING.md`

3. **MCLP训练脚本** ⚠️
   - 脚本：`examples/modeling/test_mclp.py`
   - 状态：需要注册MCLP环境的embedding（见下文）

## 🚀 立即开始

### 1. 测试预生成功能

```bash
cd /root/autodl-tmp/rl4co-urban
python tests/test_pregenerate_instances.py
```

**结果：** ✅ 所有测试通过
- 生成了10个FLP实例
- 保存到npz文件
- 加载并验证一致性
- 额外生成了train/val/test数据集

### 2. 训练FLP模型

```bash
cd examples/modeling
python test_flp.py --mode simple
```

**配置：**
- 50个候选位置，选择5个设施
- 10,000个训练实例
- 10 epochs（演示用）
- 输出：`checkpoints/flp/`

**预期输出：**
```
================================================================================
Training AttentionModel on FLP
================================================================================

1. Setting up environment...
✓ Environment: flp
  - Candidate locations: 50
  - Facilities to select: 5

2. Setting up policy (encoder-decoder architecture)...
✓ Policy: AttentionModelPolicy
  - Embedding dimension: 128
  - Encoder layers: 3
  - Attention heads: 8

3. Setting up model (REINFORCE with rollout baseline)...
✓ Model: AttentionModel
  - Algorithm: REINFORCE
  - Baseline: rollout
  - Batch size: 512
  - Learning rate: 1e-4
...
```

### 3. MCLP训练（需要额外步骤）

MCLP环境需要先注册embedding才能使用AttentionModel。有两个选项：

#### 选项A：使用简单策略测试

```python
from rl4co.envs.graph import MCLPEnv
from rl4co.utils.decoding import rollout, random_policy

# 创建环境
env = MCLPEnv(generator_params={
    "num_demand": 50,
    "num_facility": 30,
    "num_facilities_to_select": 8,
})

# 生成实例
td = env.reset(batch_size=[10])

# 随机策略测试
reward, td_final, actions = rollout(env, td, random_policy)
print(f"Mean reward: {reward.mean():.2f}")
```

#### 选项B：注册MCLP Embedding（高级）

需要在rl4co中注册MCLP的环境embedding：
1. 创建`rl4co/models/nn/env_embeddings/init/mclp.py`
2. 创建`rl4co/models/nn/env_embeddings/context/mclp.py`
3. 在相应的`__init__.py`中注册

这是rl4co框架级的修改，建议先用其他环境（如FLP）熟悉训练流程。

## 📁 文件结构

```
rl4co-urban/
├── examples/modeling/
│   ├── test_flp.py          ✅ FLP训练脚本
│   ├── test_mclp.py         ⚠️ MCLP训练脚本（需要embedding）
│   ├── README_FLP_TRAINING.md   详细使用指南
│   └── QUICK_START.md       本文件
├── tests/
│   ├── test_pregenerate_instances.py  ✅ 预生成测试
│   ├── test_training_setup.py         ✅ 训练设置测试
│   ├── test_mclp_env.py              ✅ MCLP环境测试
│   └── compare_mclp_distributions.py  ✅ MCLP分布对比
├── tools/
│   ├── pregenerate_dataset.py     预生成工具
│   └── README_PREGENERATE.md      预生成指南
└── data/
    └── pregenerated/              生成的数据集
```

## 🎯 推荐学习路径

### 第1步：理解数据生成和加载
```bash
python tests/test_pregenerate_instances.py
```
学习内容：
- ✅ 如何生成实例
- ✅ 如何保存到npz
- ✅ 如何加载并验证

### 第2步：FLP简单训练
```bash
python examples/modeling/test_flp.py --mode simple
```
学习内容：
- ✅ Environment设置
- ✅ Policy配置
- ✅ REINFORCE算法
- ✅ 训练循环

### 第3步：监控训练过程
```bash
tensorboard --logdir logs/flp/
```
学习内容：
- ✅ 查看loss曲线
- ✅ 监控reward变化
- ✅ 分析训练稳定性

### 第4步：FLP高级训练
```bash
python examples/modeling/test_flp.py --mode advanced
```
学习内容：
- ✅ 更大问题规模
- ✅ Beam search解码
- ✅ 更长训练时间

### 第5步：模型评估
```bash
python examples/modeling/test_flp.py --mode evaluate
```
学习内容：
- ✅ 对比不同解码策略
- ✅ 评估模型性能

## 💻 命令速查表

```bash
# 测试预生成
python tests/test_pregenerate_instances.py

# 测试训练设置
python tests/test_training_setup.py

# FLP简单训练（10 epochs）
python examples/modeling/test_flp.py --mode simple

# FLP高级训练（50 epochs, beam search）
python examples/modeling/test_flp.py --mode advanced

# 从checkpoint继续训练
python examples/modeling/test_flp.py --mode from_ckpt

# 评估训练好的模型
python examples/modeling/test_flp.py --mode evaluate

# 启动TensorBoard
tensorboard --logdir logs/flp/

# 预生成FLP数据集
python tools/pregenerate_dataset.py --env flp --num-loc 100 --to-choose 10

# 测试MCLP环境
python tests/test_mclp_env.py

# 对比MCLP分布
python tests/compare_mclp_distributions.py
```

## 📊 预期性能

### FLP (50 locations, select 5)

| 策略 | 平均Reward | 速度 |
|------|-----------|------|
| Random | ~-5.5 | 很快 |
| Greedy Heuristic | ~-3.5 | 快 |
| RL (训练后) | ~-3.0 | 中等 |

### MCLP (50 demand, 30 facilities, select 8)

| 分布 | 固定半径 | 动态半径 |
|------|----------|---------|
| Uniform | 59% coverage | 85% coverage |
| Cluster | 42% coverage | 12% coverage |
| Explosion | 89% coverage | 80% coverage |

## ⚠️ 已知问题

1. **MCLP需要embedding注册**
   - 状态：AttentionModel不能直接用于MCLP
   - 解决方案：需要添加MCLP的init和context embedding
   - 临时方案：使用random policy测试环境

2. **训练可能需要较长时间**
   - FLP simple模式：~10-20分钟（CPU）
   - FLP advanced模式：~1-2小时（GPU）
   - 建议：先用small规模快速测试

## 🎓 学习资源

1. **代码示例**
   - `examples/modeling/test_flp.py` - 完整训练流程
   - `tests/test_pregenerate_instances.py` - 数据生成
   - `README_FLP_TRAINING.md` - 详细教程

2. **RL4CO文档**
   - [GitHub](https://github.com/ai4co/rl4co)
   - [Documentation](https://rl4co.readthedocs.io/)

3. **论文参考**
   - Attention, Learn to Solve Routing Problems! (Kool et al., 2019)
   - POMO: Policy Optimization with Multiple Optima (Kwon et al., 2020)

## 🆘 常见问题

**Q: 训练时显存不够？**
A: 减小`batch_size`，或减小`embed_dim`

**Q: 如何使用预生成的数据集训练？**
A: 目前需要自定义DataModule，默认是on-the-fly生成

**Q: MCLP什么时候能用AttentionModel？**
A: 需要在rl4co中注册MCLP的embedding，或使用其他不依赖embedding的策略

**Q: 如何保存最好的checkpoint？**
A: 使用`ModelCheckpoint(monitor="val/reward", mode="max")`，已在脚本中配置

**Q: 如何可视化训练过程？**
A: 使用TensorBoard：`tensorboard --logdir logs/flp/`

## ✨ 下一步计划

- [ ] 为MCLP添加embedding支持
- [ ] 实现自定义DataModule支持预生成数据
- [ ] 添加更多baseline对比
- [ ] 添加POMO等其他模型
- [ ] 优化训练超参数

---

**祝你训练顺利！** 🚀

如有问题，请查看详细文档或创建issue。
