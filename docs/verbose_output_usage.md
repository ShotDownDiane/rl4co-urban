# DeepACO 详细输出使用指南

## 概述

现在训练脚本已经集成了详细的进度输出功能，可以实时看到训练的每一步进展。

## 新增功能

### 1. **DetailedProgressCallback** - 通用详细进度

显示每个batch和epoch的详细信息：

**输出示例：**
```
================================================================================
🚀 TRAINING STARTED
================================================================================
📊 Total Epochs: 10
🎯 Model: DeepACO
🐜 Ants (train): 30
🐜 Ants (val): 48
🔄 Iterations (train): 1
================================================================================

================================================================================
📅 Epoch 1/10 Started
================================================================================

⚡ Epoch 1 | Batch 1/196 (0.5%) | Loss: 12.3456 | Time: 28.34s | Avg: 28.34s/batch | ETA: 92.5min
⚡ Epoch 1 | Batch 2/196 (1.0%) | Loss: 12.1234 | Time: 26.12s | Avg: 27.23s/batch | ETA: 88.1min
...

────────────────────────────────────────────────────────────────────────────────
✅ Epoch 1 Completed
   ⏱️  Time: 89.23 minutes
   📉 Train Loss: 11.2345
   🎯 Train Reward: -14.5678
   ⚡ Avg Batch Time: 27.34s
────────────────────────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────────────────────────
🔍 Validation Started (Epoch 1)
────────────────────────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────────────────────────
✅ Validation Completed
   ⏱️  Time: 45.67s
   🎯 Val Reward: -13.4567
────────────────────────────────────────────────────────────────────────────────
```

### 2. **DeepACOProgressCallback** - DeepACO专用信息

每5个batch显示ACO算法的详细信息：

**输出示例：**
```
────────────────────────────────────────────────────────────────────────────────
🐜 DeepACO Batch 1 Processing...
   🐜 Number of Ants: 30
   🔄 ACO Iterations: 1
   🔧 Local Search: Enabled
   ⏰ Start Time: 17:08:35
   ✅ Batch Completed in 28.34s
   📉 Loss: 12.3456
   🎯 Reward: -14.5678
────────────────────────────────────────────────────────────────────────────────
```

## 配置选项

在 `train.py` 中可以调整输出频率：

```python
# 通用进度回调
detailed_progress = DetailedProgressCallback(
    print_every_n_batches=1,  # 每1个batch打印一次（可改为5, 10等）
    print_every_n_epochs=1     # 每1个epoch打印（设为1表示所有epoch）
)

# DeepACO专用回调
deepaco_progress = DeepACOProgressCallback(
    print_every_n_batches=5  # 每5个batch打印详细信息（可改为1, 10等）
)
```

## 使用方法

### 方法1：使用更新后的train.py（推荐）

直接使用现有的训练脚本，已经自动包含详细输出：

```bash
# 标准训练（会显示详细输出）
bash scripts/train_tsp.sh

# 或直接运行
python train.py --problem TSP --num-loc 100 --model-type DeepACO ...
```

### 方法2：快速测试（验证输出功能）

```bash
# 使用小数据集快速测试（2-3分钟完成）
bash scripts/test_verbose_output.sh
```

这个脚本会：
- 使用20节点TSP
- 只用512个训练样本
- 2个epochs
- 在2-3分钟内完成
- 可以快速验证输出功能是否正常

### 方法3：自定义输出频率

如果觉得输出太多或太少，可以修改 `train.py` 中的配置：

```python
# 减少输出频率（更安静）
detailed_progress = DetailedProgressCallback(
    print_every_n_batches=10,  # 每10个batch打印一次
    print_every_n_epochs=1
)

# 增加输出频率（更详细）
detailed_progress = DetailedProgressCallback(
    print_every_n_batches=1,   # 每个batch都打印
    print_every_n_epochs=1
)
```

## 输出内容说明

### 训练开始信息
- **🚀 TRAINING STARTED**: 训练启动
- **📊 Total Epochs**: 总epoch数
- **🎯 Model**: 模型类型
- **🐜 Ants**: 蚂蚁数量（训练/验证）
- **🔄 Iterations**: ACO迭代次数

### Batch级别信息
- **⚡ Epoch X | Batch Y/Z**: 当前epoch和batch
- **Loss**: 当前损失值
- **Time**: 当前batch耗时
- **Avg**: 平均batch耗时
- **ETA**: 预计剩余时间

### Epoch级别信息
- **⏱️ Time**: Epoch总耗时
- **📉 Train Loss**: 训练损失
- **🎯 Train Reward**: 训练奖励（负值表示cost）
- **⚡ Avg Batch Time**: 平均batch时间

### DeepACO特定信息
- **🐜 Number of Ants**: 当前阶段的蚂蚁数量
- **🔄 ACO Iterations**: ACO迭代次数
- **🔧 Local Search**: 是否启用局部搜索

## 完整输出示例

以下是一个完整的训练过程输出（简化版）：

```
================================================================================
🚀 TRAINING STARTED
================================================================================
📊 Total Epochs: 10
🎯 Model: DeepACO
🐜 Ants (train): 30
🐜 Ants (val): 48
🔄 Iterations (train): 1
================================================================================

================================================================================
📅 Epoch 1/10 Started
================================================================================

────────────────────────────────────────────────────────────────────────────────
🐜 DeepACO Batch 1 Processing...
   🐜 Number of Ants: 30
   🔄 ACO Iterations: 1
   🔧 Local Search: Enabled
   ⏰ Start Time: 17:08:35
   ✅ Batch Completed in 28.34s
   📉 Loss: 12.3456
   🎯 Reward: -14.5678
────────────────────────────────────────────────────────────────────────────────

⚡ Epoch 1 | Batch 1/196 (0.5%) | Loss: 12.3456 | Time: 28.34s | Avg: 28.34s/batch | ETA: 92.5min
⚡ Epoch 1 | Batch 2/196 (1.0%) | Loss: 12.1234 | Time: 26.12s | Avg: 27.23s/batch | ETA: 88.1min
⚡ Epoch 1 | Batch 3/196 (1.5%) | Loss: 11.9876 | Time: 27.45s | Avg: 27.30s/batch | ETA: 87.8min
...

────────────────────────────────────────────────────────────────────────────────
✅ Epoch 1 Completed
   ⏱️  Time: 89.23 minutes
   📉 Train Loss: 11.2345
   🎯 Train Reward: -14.5678
   ⚡ Avg Batch Time: 27.34s
────────────────────────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────────────────────────
🔍 Validation Started (Epoch 1)
────────────────────────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────────────────────────
✅ Validation Completed
   ⏱️  Time: 45.67s
   🎯 Val Reward: -13.4567
────────────────────────────────────────────────────────────────────────────────

[... Epochs 2-9 ...]

================================================================================
🎉 TRAINING COMPLETED
================================================================================
✅ Total Epochs: 10
✅ Total Batches: 1960
📊 Final Train Reward: -12.3456
📊 Final Val Reward: -11.2345
================================================================================
```

## 与其他监控工具配合

### 1. 结合TensorBoard

```bash
# 在另一个终端启动TensorBoard
tensorboard --logdir logs --port 6006

# 训练时可以同时：
# - 终端看详细文字输出
# - 浏览器看可视化曲线
```

### 2. 结合监控脚本

```bash
# 终端1：运行训练（有详细输出）
python train.py --model-type DeepACO ...

# 终端2：运行监控脚本（看总体状态）
watch -n 5 bash scripts/monitor_training.sh
```

### 3. 保存输出到文件

```bash
# 同时显示和保存输出
python train.py --model-type DeepACO ... 2>&1 | tee training.log

# 稍后查看日志
less training.log
```

## 性能影响

详细输出对训练性能影响很小：
- **print_every_n_batches=1**: ~0.1-0.2秒额外开销/batch
- **print_every_n_batches=5**: 几乎无影响
- **print_every_n_batches=10**: 无影响

对于DeepACO这种每batch需要20-30秒的算法，输出开销可以忽略不计。

## 常见问题

### Q1: 输出太多，屏幕刷屏？
**A**: 调整 `print_every_n_batches` 参数，比如从1改为5或10。

### Q2: 想保存输出到文件？
**A**: 使用 `tee` 命令：
```bash
python train.py ... 2>&1 | tee training.log
```

### Q3: 输出中有乱码或emoji不显示？
**A**: 确保终端支持UTF-8编码。或者修改callback移除emoji。

### Q4: 如何只保留关键信息？
**A**: 可以创建一个简化版的callback，只输出最重要的信息。

## 自定义输出

如果需要自定义输出格式，可以修改 `callbacks/training_progress.py`：

```python
# 示例：添加更多信息
def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    # 添加GPU内存使用
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU Memory: {gpu_mem:.2f}GB")
    
    # 添加其他自定义信息
    ...
```

## 总结

新的详细输出功能让你能够：
- ✅ 实时看到训练进度
- ✅ 了解每个batch的耗时
- ✅ 预估剩余训练时间
- ✅ 监控DeepACO特定参数
- ✅ 快速发现训练问题

使用 `bash scripts/test_verbose_output.sh` 快速体验新功能！
