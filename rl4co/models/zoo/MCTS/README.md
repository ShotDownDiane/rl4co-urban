# Monte Carlo Tree Search (MCTS) for RL4CO

这是一个可以在RL4CO环境中执行的简单MCTS实现，支持与神经网络策略结合使用。

## 功能特点

- ✅ **纯MCTS模式**: 无需神经网络，使用随机rollout进行值估计
- ✅ **策略引导模式**: 结合神经网络策略进行更智能的搜索 (AlphaGo风格)
- ✅ **环境兼容**: 完全兼容RL4CO环境API
- ✅ **灵活配置**: 可调节模拟次数、探索常数等参数

## 快速开始

### 1. 基础用法 - 不使用策略

```python
from rl4co.envs import TSPEnv
from rl4co.models.zoo.MCTS import MCTSModel

# 创建环境
env = TSPEnv(generator_params={'num_loc': 20})

# 创建MCTS模型（纯MCTS）
mcts = MCTSModel(
    env=env,
    policy=None,  # 不使用神经网络
    num_simulations=100,
    c_puct=1.0,
)

# 生成问题实例
td = env.reset(batch_size=[1])

# 求解
actions, reward, stats = mcts.solve(td)
print(f"Tour length: {-reward.item():.4f}")
```

### 2. 结合神经网络策略

```python
from rl4co.envs import TSPEnv
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.models.zoo.MCTS import MCTSModel

# 创建环境和策略
env = TSPEnv(generator_params={'num_loc': 20})
policy = AttentionModelPolicy(env_name='tsp')

# 如果有预训练模型，加载checkpoint
# policy = AttentionModelPolicy.load_from_checkpoint('checkpoint.ckpt')

# 创建MCTS模型（结合策略）
mcts = MCTSModel(
    env=env,
    policy=policy,  # 使用神经网络指导搜索
    num_simulations=100,
    c_puct=1.0,
)

# 求解
actions, reward, stats = mcts.solve(td)
```

### 3. 批量评估

```python
# 生成多个实例
td = env.reset(batch_size=[10])

# 评估
results = mcts.evaluate(td, num_instances=10)
print(f"Mean tour length: {results['mean_tour_length']:.4f}")
```

## 核心组件

### MCTSModel
主要的模型包装器，提供高级接口。

**参数:**
- `env`: RL4CO环境实例或环境名称
- `policy`: 神经网络策略（可选）
- `num_simulations`: 每个动作的MCTS模拟次数
- `c_puct`: UCB公式中的探索常数
- `temperature`: 动作选择的温度（0=贪婪）
- `device`: 运行设备

**主要方法:**
- `solve(td)`: 求解单个实例
- `evaluate(td, num_instances)`: 评估多个实例
- `forward(td, ...)`: 标准前向传播
- `set_num_simulations(n)`: 动态调整模拟次数
- `set_temperature(t)`: 动态调整温度
- `set_c_puct(c)`: 动态调整探索常数

### MCTS
底层MCTS算法实现。

**主要功能:**
- 自动使用策略提供的先验概率（如果有）
- 支持随机rollout或策略rollout进行值估计
- 返回详细的搜索统计信息

### MCTSNode
MCTS树节点。

**包含信息:**
- 状态 (TensorDict)
- 访问次数
- 值估计
- 先验概率
- 子节点

## 工作原理

### 不使用策略时:
```
1. 选择 (Selection): 使用UCB公式选择最有前景的子节点
2. 扩展 (Expansion): 为所有合法动作创建子节点（均匀先验）
3. 模拟 (Simulation): 随机rollout到终止状态
4. 回传 (Backpropagation): 更新路径上所有节点的值
```

### 使用策略时:
```
1. 选择 (Selection): 使用UCB公式（包含策略提供的先验概率）
2. 扩展 (Expansion): 为所有合法动作创建子节点（策略先验）
3. 评估 (Evaluation): 使用策略rollout估计值
4. 回传 (Backpropagation): 更新路径上所有节点的值
```

## UCB公式

MCTS使用以下UCB公式选择动作:

```
Score(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

其中:
- `Q(s,a)`: 动作的平均值（exploitation）
- `P(s,a)`: 先验概率（来自策略或均匀分布）
- `N(s)`: 父节点访问次数
- `N(s,a)`: 动作访问次数
- `c_puct`: 探索常数

## 参数调优建议

### num_simulations
- **小问题 (n<20)**: 50-100次模拟
- **中等问题 (20<n<50)**: 100-200次模拟
- **大问题 (n>50)**: 200-500次模拟

### c_puct
- **更多探索**: c_puct = 1.5-2.0
- **平衡**: c_puct = 1.0 (默认)
- **更多利用**: c_puct = 0.5-0.8

### temperature
- **贪婪选择**: temperature = 0.0 (推荐用于测试)
- **随机采样**: temperature = 1.0 (用于训练多样性)

## 示例

完整示例代码在 `examples/mcts_example.py`，包括:

1. 纯MCTS（无策略）
2. MCTS + 策略
3. 策略对比实验
4. 参数调优示例
5. 预训练模型使用指南

运行示例:
```bash
cd /root/autodl-tmp/rl4co-urban
python examples/mcts_example.py
```

## 与其他方法对比

| 方法 | 速度 | 质量 | 需要训练 |
|------|------|------|---------|
| 贪婪策略 | 最快 | 中等 | 是 |
| 采样策略 | 快 | 中等 | 是 |
| 纯MCTS | 慢 | 好 | 否 |
| MCTS+策略 | 最慢 | 最好 | 是（策略） |

## 性能优化建议

1. **使用预训练策略**: 策略引导的MCTS远优于纯MCTS
2. **调整模拟次数**: 根据可用时间和问题复杂度调整
3. **批量处理**: 目前是串行处理，未来可优化为并行
4. **缓存策略输出**: 对于重复状态可以缓存encoder输出

## 限制和未来改进

### 当前限制:
- 串行处理batch（每次一个实例）
- 没有树重用（每个动作重建树）
- 没有并行模拟

### 未来改进:
- [ ] 支持批量并行MCTS
- [ ] 实现树重用和根节点转移
- [ ] 添加虚拟损失支持并行模拟
- [ ] 支持更多环境特定的优化
- [ ] 添加自适应模拟次数

## API参考

详细API文档见源代码注释。主要类和方法都有完整的docstring。

## 引用

如果你在研究中使用了这个MCTS实现，请引用RL4CO:

```bibtex
@article{berto2023rl4co,
  title={RL4CO: a Unified Reinforcement Learning for Combinatorial Optimization Library},
  author={Berto, Federico and others},
  year={2023}
}
```

MCTS算法基于:
- AlphaGo/AlphaZero系列工作
- UCT (Upper Confidence bounds applied to Trees)

## 许可证

遵循RL4CO项目的许可证。
