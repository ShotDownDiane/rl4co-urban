# Steiner Tree Problem (STP) Environment

## 问题描述

斯坦纳树问题（Steiner Tree Problem）是一个经典的NP-hard组合优化问题。给定一个带权图和一组必须连接的终端节点，目标是找到连接所有终端节点的最小权重树。非终端节点可以作为中继（斯坦纳）点使用。

## 🎯 核心特性

### ✅ 高效的Action Space设计
- **Edge-based动作空间**：直接选择边而非节点对
- **动作空间压缩**：从O(n²)降低到O(edges)
  - 完全图：400 actions → 仍然400
  - Delaunay图：400 actions → 约50 actions (**8倍效率提升**)
- **零padding浪费**：只存储实际存在的边

### ✅ 图生成方法（基于Delaunay三角剖分）
- **Delaunay**（默认）：平面图，保证连通性，类似真实路网
- **KNN+MST**：控制稀疏度，局部密集+全局连通
- **Radius+MST**：基于距离的邻域连接
- **Complete**：完全图（用于基准测试）

### ✅ Action Projection机制
- **自动修正非法动作**：`project=True`时将非法动作投影为随机合法动作
- **训练稳定性**：避免探索时的非法动作导致训练崩溃
- **调试友好**：可开启日志记录投影事件

## 实现特点

### 1. **动作空间设计**（新版：Edge-based）
- **动作表示**：`action`是`edge_list`中的索引
- **解码方式**：`(from, to) = edge_list[batch_idx, action_idx]`
- **优势**：
  ```python
  # 旧方案
  action_space_size = num_nodes² = 400  # 20个节点
  valid_actions ≈ 50                    # Delaunay图
  efficiency = 12.5%                    # 大量浪费
  
  # 新方案
  action_space_size = num_edges ≈ 50   # 只包含实际边
  valid_actions ≈ 50
  efficiency = 100%                     # 零浪费！
  ```

### 2. **状态表示**
- **静态信息**：
  - `locs`: 节点坐标 `(batch_size, num_nodes, 2)`
  - `terminals`: 终端节点索引 `(batch_size, num_terminals)`
  - `edge_weights`: 边权重矩阵 `(batch_size, num_nodes, num_nodes)`
  - `adjacency`: 邻接矩阵 `(batch_size, num_nodes, num_nodes)`
  - **`edge_list`**: 边列表 `(batch_size, max_edges, 2)` - **新增**
  - **`num_edges`**: 每个实例的边数 `(batch_size,)` - **新增**

- **动态信息**：
  - `selected_edges`: 已选择的边 `(batch_size, num_nodes, num_nodes)`
  - `components`: 连通分量信息（Union-Find结构）
  - `i`: 当前步数
  - `action_mask`: 可行动作掩码

### 3. **约束处理**
使用Union-Find数据结构维护连通分量，确保：
- 不形成环路（只连接不同分量的边）
- 只选择图中存在的边
- 不重复选择同一条边

### 4. **终止条件**
当所有终端节点属于同一连通分量时，问题求解完成。

### 5. **奖励函数**
奖励为所选边的总权重的负值：
```python
reward = -sum(weight of selected edges)
```

## 使用示例

### 基础使用（推荐：Delaunay图）
```python
from rl4co.envs.graph.stp import STPEnv

# 创建环境 - 使用Delaunay三角剖分
env = STPEnv(
    generator_params={
        "num_nodes": 50,
        "num_terminals": 10,
        "graph_type": "delaunay",  # 推荐：类真实路网
    },
    project=True,  # 启用action projection（训练时推荐）
)

# 生成问题实例
td = env.reset(batch_size=[32])

# 查看action space效率
print(f"Edge list shape: {td['edge_list'].shape}")
print(f"Number of edges: {td['num_edges']}")
print(f"Action space size: {td['action_mask'].shape[-1]}")

# 运行随机策略
from rl4co.utils.decoding import random_policy, rollout
reward, td_final, actions = rollout(env, td, random_policy)
```

### 不同图类型对比
```python
# 1. Delaunay（推荐，类真实路网）
env_delaunay = STPEnv(generator_params={
    "num_nodes": 50,
    "num_terminals": 10,
    "graph_type": "delaunay",  # ~150 edges, 平面图
})

# 2. KNN + MST（控制稀疏度）
env_knn = STPEnv(generator_params={
    "num_nodes": 50,
    "num_terminals": 10,
    "graph_type": "knn",
    "knn_k": 5,  # 每个节点连接5个最近邻
})

# 3. Radius + MST（基于距离）
env_radius = STPEnv(generator_params={
    "num_nodes": 50,
    "num_terminals": 10,
    "graph_type": "radius",
    "radius": 0.2,  # 距离<0.2的节点连接
})

# 4. Complete Graph（基准测试）
env_complete = STPEnv(generator_params={
    "num_nodes": 50,
    "num_terminals": 10,
    "graph_type": "complete",  # 1225 edges
})
```

### 启用Action Projection（训练推荐）
```python
# 训练阶段：启用projection避免非法动作
env_train = STPEnv(
    generator_params={"graph_type": "delaunay", ...},
    project=True,  # 自动修正非法动作
    check_solution=False,  # 训练时不检查（提速）
)

# 评估阶段：关闭projection，检查解的有效性
env_eval = STPEnv(
    generator_params={"graph_type": "delaunay", ...},
    project=False,  # 不修正，检验policy质量
    check_solution=True,  # 验证解的合法性
)
```

## 与参考实现的对应关系

### 借鉴自FLP (Facility Location Problem)
- 基于位置的节点表示
- 距离矩阵计算
- 逐步构建解的方式

### 借鉴自MCP (Maximum Coverage Problem)
- 离散动作空间
- 基于掩码的约束处理
- 批量实例生成

### STP特有设计
- **Union-Find算法**：高效维护连通性
- **边选择动作**：不同于节点选择
- **图拓扑约束**：支持完全图和稀疏图

## 算法扩展建议

### 1. 启发式增强
- 可以添加最近邻启发式作为baseline
- Prim算法的变种作为warm-start

### 2. 特征工程
- 添加节点度数特征
- 最短路径距离特征
- 是否为终端节点的one-hot编码

### 3. 奖励塑形
- 中间步骤的奖励（部分连通性）
- 边选择的局部优化信号

### 4. 多目标优化
- 在最小化树权重的同时，考虑树的其他特性
- 平衡树的深度和宽度

## 技术细节

### Union-Find 实现
```python
def _update_components(self, components, from_nodes, to_nodes, batch_size, num_nodes):
    # 路径压缩的Union-Find
    # 保证O(α(n))的amortized时间复杂度
```

### 动作掩码生成
```python
def _get_action_mask(self, adjacency, selected_edges, components, ...):
    # 1. 边必须存在于图中
    # 2. 边未被选择
    # 3. 边连接不同的分量（避免环路）
```

## 未来改进

- [ ] 实现更高效的批量Union-Find操作
- [ ] 添加Steiner点优化（欧几里得STP）
- [ ] 实现局部搜索算法
- [ ] 支持有向图变种
- [ ] 添加时间窗口约束

## 参考文献

1. Hwang, F. K., Richards, D. S., & Winter, P. (1992). The Steiner tree problem. Annals of discrete mathematics.
2. Koch, T., & Martin, A. (1998). Solving Steiner tree problems in graphs to optimality. Networks.
3. Takahashi, H., & Matsuyama, A. (1980). An approximate solution for the Steiner problem in graphs. Math. Japonica.
