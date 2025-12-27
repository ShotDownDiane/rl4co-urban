# DeepACO 多蚂蚁配置指南

## 概述

DeepACO完全支持多蚂蚁并行搜索，这是ACO算法的核心特性。通过控制蚂蚁数量和迭代次数，可以在训练速度和解的质量之间取得平衡。

## 核心参数

### 1. `n_ants` - 蚂蚁数量

控制每个batch中并行运行的蚂蚁数量。

**配置方式：**

```python
# 方式1：所有阶段使用相同数量
policy_kwargs = {
    "n_ants": 50  # 训练、验证、测试都使用50只蚂蚁
}

# 方式2：分别设置各阶段（推荐）
policy_kwargs = {
    "n_ants": {
        "train": 30,   # 训练阶段
        "val": 48,     # 验证阶段
        "test": 100    # 测试阶段
    }
}
```

**默认值：**
- train: 30
- val: 48
- test: 48

**建议：**
- **训练阶段**：使用较少蚂蚁（如20-30）以加快训练速度
- **验证阶段**：使用中等数量（如48）平衡速度和性能
- **测试阶段**：使用较多蚂蚁（如100-200）以获得最佳解

### 2. `n_iterations` - ACO迭代次数

控制ACO算法的迭代次数。每次迭代中，所有蚂蚁都会构建新的解并更新信息素。

**配置方式：**

```python
policy_kwargs = {
    "n_iterations": {
        "train": 1,    # 训练阶段：快速迭代
        "val": 5,      # 验证阶段：适度迭代
        "test": 10     # 测试阶段：充分搜索
    }
}
```

**默认值：**
- train: 1
- val: 5
- test: 10

**建议：**
- **训练阶段**：通常使用1次迭代，因为需要频繁更新
- **验证阶段**：5-10次迭代
- **测试阶段**：10-20次迭代以充分探索解空间

### 3. 其他相关参数

```python
policy_kwargs = {
    # 使用局部搜索改进解
    "aco_kwargs": {
        "use_local_search": True  # 必须为True当train_with_local_search=True
    },
    
    # 温度参数（控制探索vs利用）
    "temperature": 1.0,
    
    # Top-k稀疏化（只考虑最近的k个邻居）
    "k_sparse": None,  # 默认为None（不使用）
    
    # 多起点采样
    "multistart": False,  # TSP默认为True
    "start_node": None    # 指定起始节点（如果为None则随机选择）
}
```

## 完整配置示例

### 示例1：快速训练配置

```python
model = DeepACO(
    env=env,
    baseline="no",
    train_with_local_search=True,
    ls_reward_aug_W=0.95,
    policy_kwargs={
        "aco_kwargs": {"use_local_search": True},
        "n_ants": {"train": 20, "val": 30, "test": 50},
        "n_iterations": {"train": 1, "val": 3, "test": 5}
    }
)
```

### 示例2：高质量解配置

```python
model = DeepACO(
    env=env,
    baseline="no",
    train_with_local_search=True,
    ls_reward_aug_W=0.95,
    policy_kwargs={
        "aco_kwargs": {"use_local_search": True},
        "n_ants": {"train": 30, "val": 64, "test": 128},
        "n_iterations": {"train": 1, "val": 10, "test": 20},
        "temperature": 0.8,  # 更倾向于利用
    }
)
```

### 示例3：大规模问题配置

```python
model = DeepACO(
    env=env,
    baseline="no",
    train_with_local_search=True,
    ls_reward_aug_W=0.95,
    policy_kwargs={
        "aco_kwargs": {"use_local_search": True},
        "n_ants": {"train": 30, "val": 48, "test": 100},
        "n_iterations": {"train": 1, "val": 5, "test": 10},
        "k_sparse": 20,  # 稀疏化：只考虑最近的20个邻居
    }
)
```

## 性能权衡

### 计算复杂度

总计算量 = `batch_size × n_ants × n_iterations × 问题规模`

**内存使用：**
- 训练阶段：`O(batch_size × n_ants × n_nodes)`
- 推理阶段：`O(batch_size × n_ants × n_nodes)`

### 推荐配置（基于问题规模）

| 问题规模 | train | val | test | iterations (test) |
|---------|-------|-----|------|-------------------|
| 小型 (<50节点) | 30 | 48 | 100 | 10 |
| 中型 (50-100节点) | 30 | 48 | 64 | 10 |
| 大型 (100-200节点) | 20 | 32 | 48 | 5-10 |
| 超大型 (>200节点) | 20 | 32 | 32 | 5 |

## 如何通过命令行控制

如果想从命令行控制蚂蚁数量，可以修改`train.py`添加参数：

```python
# 在argparse中添加
parser.add_argument('--n-ants-train', type=int, default=30)
parser.add_argument('--n-ants-val', type=int, default=48)
parser.add_argument('--n-ants-test', type=int, default=100)
parser.add_argument('--n-iterations-test', type=int, default=10)

# 在ModelModule中使用
policy_kwargs = {
    "n_ants": {
        "train": args.n_ants_train,
        "val": args.n_ants_val,
        "test": args.n_ants_test
    },
    "n_iterations": {
        "train": 1,
        "val": 5,
        "test": args.n_iterations_test
    }
}
```

## 调试和监控

训练时可以通过日志查看实际使用的蚂蚁数量：

```python
print(f"Training with {model.policy.n_ants['train']} ants")
print(f"Testing with {model.policy.n_ants['test']} ants, {model.policy.n_iterations['test']} iterations")
```

## 参考文献

- DeepACO论文: Ye et al. (2023) - https://arxiv.org/abs/2309.14032
- ACO算法原理: Dorigo & Stützle (2004) - Ant Colony Optimization
