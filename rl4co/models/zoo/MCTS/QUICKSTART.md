# MCTS 快速开始指南

## 🎯 核心功能

已实现一个可在RL4CO环境中执行的MCTS，支持与神经网络策略结合使用。

## 🚀 快速使用

### 1. 基础MCTS（无策略）

```python
from rl4co.envs import TSPEnv
from rl4co.models.zoo.MCTS import MCTSModel

# 创建环境
env = TSPEnv(generator_params={'num_loc': 20})

# 创建MCTS（纯搜索，无神经网络）
mcts = MCTSModel(
    env=env,
    policy=None,  # 不使用神经网络
    num_simulations=50,  # 每步50次模拟
    c_puct=1.0,  # 探索常数
    temperature=0.0,  # 贪婪选择
)

# 求解
td = env.reset(batch_size=[1])
actions, reward, stats = mcts.solve(td, verbose=True)
print(f"路径长度: {-reward.item():.4f}")
```

### 2. MCTS + 神经网络策略

```python
from rl4co.models.zoo.am import AttentionModelPolicy

# 创建策略
policy = AttentionModelPolicy(env_name='tsp')

# 创建MCTS（策略引导）
mcts = MCTSModel(
    env=env,
    policy=policy,  # 使用神经网络指导搜索
    num_simulations=100,
    c_puct=1.0,
)

# 求解
actions, reward, stats = mcts.solve(td, verbose=True)
```

## 📊 性能优化建议

### 速度控制参数

1. **num_simulations**: 模拟次数
   - 小问题(n<20): 10-50次
   - 中等问题(20<n<50): 50-100次  
   - 大问题(n>50): 100-200次

2. **verbose**: 输出控制
   - `verbose=True`: 显示详细进度（调试用）
   - `verbose=False`: 静默模式（批量测试）

3. **问题规模**: 
   - 从小问题开始测试（10-20个节点）
   - 逐步增加问题规模

### 速度优化示例

```python
# 快速测试配置
mcts_fast = MCTSModel(
    env=env,
    policy=None,
    num_simulations=10,  # 少量模拟
    temperature=0.0,
)

# 高质量配置
mcts_quality = MCTSModel(
    env=env,
    policy=policy,  # 使用策略
    num_simulations=200,  # 更多模拟
    temperature=0.0,
)
```

## 🔧 主要API

### MCTSModel类

```python
MCTSModel(
    env,                 # RL4CO环境
    policy=None,         # 神经网络策略（可选）
    num_simulations=100, # 模拟次数
    c_puct=1.0,         # 探索常数
    temperature=0.0,    # 温度参数
    device='cpu',       # 设备
)
```

### 主要方法

```python
# 求解单个实例
actions, reward, stats = mcts.solve(td, verbose=True)

# 评估多个实例
results = mcts.evaluate(td, num_instances=10)

# 动态调整参数
mcts.set_num_simulations(50)
mcts.set_temperature(1.0)
mcts.set_c_puct(2.0)
```

## 📈 进度输出说明

启用`verbose=True`时的输出：

```
Starting MCTS with 10 simulations per step
  Step 0: Running MCTS search...
    Selected action 3, visits=10, value=-2.5
  Step 5: Running MCTS search...
✓ MCTS completed in 10 steps, tour length: 5.2341
```

- **Step X**: 当前决策步骤
- **Selected action**: 选择的动作
- **visits**: 该动作被访问次数
- **value**: 节点平均值
- **tour length**: 最终路径长度

## 🧪 测试

运行测试脚本：
```bash
cd /root/autodl-tmp/rl4co-urban
python test_mcts_simple.py
```

测试内容：
- ✅ 基础MCTS（无策略）
- ✅ MCTS + 策略
- ✅ 批量评估
- ✅ 参数调整

## 📚 完整示例

查看 `examples/mcts_example.py` 获取更多示例。

## ⚡ 性能提示

1. **首次测试**: 使用小问题(10节点) + 少量模拟(10次)
2. **开发调试**: 启用verbose查看详细过程
3. **批量实验**: 关闭verbose提升速度
4. **生产环境**: 使用预训练策略 + 适当模拟次数

## 🔍 算法流程

```
对于每个决策步骤:
  1. 创建根节点（当前状态）
  2. 运行N次模拟:
     - 选择: 使用UCB选择最优子节点
     - 扩展: 展开未访问节点
     - 评估: 使用策略或随机rollout估值
     - 回传: 更新路径上所有节点
  3. 选择访问次数最多的动作
  4. 执行动作，进入下一步
```

## 🎓 进阶使用

### 结合预训练模型

```python
# 加载预训练策略
policy = AttentionModelPolicy.load_from_checkpoint('model.ckpt')

# 使用MCTS增强推理
mcts = MCTSModel(env=env, policy=policy, num_simulations=100)
```

### 参数调优

不同问题可能需要不同参数：
- **探索型**: c_puct=2.0 (更多探索)
- **利用型**: c_puct=0.5 (更多利用)
- **平衡型**: c_puct=1.0 (默认)

## 📝 注意事项

1. MCTS是串行算法，批量处理会逐个求解
2. 使用策略时确保策略已加载/训练
3. 首次运行会编译一些JIT代码，可能较慢
4. 适当选择模拟次数平衡速度和质量
