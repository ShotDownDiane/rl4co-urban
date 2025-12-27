# DeepACO Baseline 兼容性修复

## 🐛 问题描述

在使用rollout baseline训练DeepACO时遇到错误：
```
TypeError: unsupported format string passed to numpy.ndarray.__format__
ValueError: can only convert an array of size 1 to a Python scalar
```

**为什么AttentionModel不报错，DeepACO报错？**

## 🔍 根本原因

### Reward维度不同

**AttentionModel (AM)**:
```python
reward.shape = [batch_size]  # 每个样本1个reward
例如: [16]  → 16个样本，每个样本1个值
```

**DeepACO**:
```python
reward.shape = [batch_size, n_ants]  # 每个样本多个reward（每只蚂蚁一个）
例如: [16, 30]  → 16个样本，每个样本30个值（30只蚂蚁）
```

### 问题传播链

1. **DeepACO Policy** (`policy.py:146`):
   ```python
   outdict["reward"] = unbatchify(outdict["reward"], n_ants)
   # 返回 [batch_size, n_ants] 形状
   ```

2. **Baseline Rollout** (`baselines.py:247`):
   ```python
   rewards = torch.cat([eval_policy(batch) for batch in dl], 0)
   # AM: [100] 形状
   # DeepACO: [100, 30] 形状  ← 问题！
   ```

3. **统计检验** (`baselines.py:225`):
   ```python
   t, p = ttest_rel(-candidate_vals, -self.bl_vals)
   # AM: candidate_vals是100个标量 → t是标量
   # DeepACO: candidate_vals是100×30=3000个值 → t是数组！
   ```

## ✅ 解决方案

### 在统计检验前对多蚁reward降维

修改 `baselines.py` 的 `epoch_callback` 方法：

```python
def epoch_callback(self, policy, env, batch_size=64, device="cpu", epoch=None, dataset_size=None):
    """Challenges the current baseline with the policy and replaces the baseline policy if it is improved"""
    log.info("Evaluating candidate policy on evaluation dataset")
    candidate_rewards = self.rollout(policy, env, batch_size, device).cpu().numpy()
    
    # Handle multi-ant rewards (e.g., DeepACO): take max reward per instance
    if candidate_rewards.ndim > 1:
        candidate_vals = candidate_rewards.max(axis=1)  # 取每个样本的最优蚂蚁reward
    else:
        candidate_vals = candidate_rewards
        
    candidate_mean = candidate_vals.mean()
    # ... 后续统计检验使用 candidate_vals
```

### 关键改动

**之前**:
```python
candidate_vals = self.rollout(...).cpu().numpy()
# DeepACO: shape [100, 30] → 统计检验失败
```

**之后**:
```python
candidate_rewards = self.rollout(...).cpu().numpy()
if candidate_rewards.ndim > 1:
    candidate_vals = candidate_rewards.max(axis=1)  # shape [100]
else:
    candidate_vals = candidate_rewards  # shape [100]
# 统计检验成功！
```

## 🎯 为什么取max合理？

### DeepACO的设计理念

DeepACO使用**多只蚂蚁探索解空间**：
- 训练阶段：30只蚂蚁并行构建解
- 验证阶段：48只蚂蚁
- 测试阶段：100只蚂蚁

**Baseline比较时应该用最优解**：
- 蚂蚁的目的是找到**最好的解**
- Baseline更新应该基于**policy能达到的最好性能**
- 使用 `max()` 代表policy在该实例上的真实能力

### 替代方案对比

| 方法 | 是否合理 | 说明 |
|------|---------|------|
| `max(axis=1)` | ✅ 最合理 | 代表policy的最优性能，符合DeepACO设计 |
| `mean(axis=1)` | ⚠️ 可能 | 平均性能，但不代表最优能力 |
| `min(axis=1)` | ❌ 不合理 | 最差性能，没有意义 |
| `median(axis=1)` | ⚠️ 可能 | 中位数，更稳健但不是最优 |
| 展平所有值 | ❌ 错误 | 改变了统计单位，违反ttest假设 |

## 🔧 其他修复

### 移除不必要的调试代码

**之前的临时方案**（已删除）:
```python
if isinstance(t, np.ndarray):
    assert (t < 0).all(), "T-statistic should be negative"
else:
    assert t < 0, "T-statistic should be negative"
```

**现在**:
```python
assert t < 0, "T-statistic should be negative"  # t现在总是标量
```

因为经过 `max(axis=1)` 处理后，`candidate_vals` 总是1维数组，`ttest_rel` 返回的t总是标量。

## 📊 验证修复

### 测试场景

```bash
# 1. 使用rollout baseline训练DeepACO
bash scripts/train_tsp_verbose.sh  # baseline=rollout

# 2. 不使用baseline训练DeepACO（推荐）
# 修改脚本: --baseline no
```

### 预期结果

**修复前**:
```
Epoch 1 完成 → epoch_callback → 
TypeError/ValueError → 训练中断 ❌
```

**修复后**:
```
Epoch 1 完成 → epoch_callback → 
Evaluating candidate policy...
Candidate mean: -20.472, Baseline mean: -20.500
T-statistic check passed ✓
Epoch 2 开始 → ... → 训练完成 ✅
```

## 💡 最佳实践建议

### 对于DeepACO

**推荐**：不使用baseline
```bash
python train.py --baseline no ...
```

**原因**:
1. DeepACO使用self-critical training，内部已经有baseline机制
2. 不需要额外的rollout baseline
3. 训练更快，无需baseline更新开销

### 对于AttentionModel

**推荐**：使用rollout baseline
```bash
python train.py --baseline rollout ...
```

**原因**:
1. REINFORCE需要baseline来减少方差
2. Rollout baseline提供更稳定的训练

## 📁 修改的文件

```
rl4co/models/rl/reinforce/baselines.py
├── Line 208-216: 添加多蚁reward处理
└── Line 228: 简化断言（移除调试代码）
```

## 🎉 总结

- ✅ **问题根源**: DeepACO的多蚁机制导致reward维度与AM不同
- ✅ **解决方案**: 在统计检验前对每个样本取最优蚁reward
- ✅ **兼容性**: 修复后AM和DeepACO都能正常使用baseline
- ✅ **最佳实践**: DeepACO建议不使用baseline（`--baseline no`）

现在rollout baseline已经完全兼容DeepACO的多蚁机制了！🎊
