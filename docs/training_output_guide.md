# DeepACO 训练输出与监控指南

## 问题：为什么训练没有输出？

DeepACO训练缺少输出主要有以下原因：

### 1. **训练速度慢**
DeepACO使用多蚁群算法 + 局部搜索，比AttentionModel慢得多：
- **AttentionModel**: ~1秒/batch (batch_size=512)
- **DeepACO (no local search)**: ~5-10秒/batch
- **DeepACO (with local search)**: ~15-30秒/batch

对于100节点TSP，30只蚂蚁 + 局部搜索，第一个batch可能需要**30-60秒**才能完成。

### 2. **PyTorch Lightning进度条问题**
某些终端环境下，Rich进度条可能不显示或更新缓慢。

### 3. **缓冲输出**
Python的输出可能被缓冲，需要使用`python -u`来禁用缓冲。

## 解决方案

### 方案1：使用增强的训练脚本（推荐）

已更新的`train.py`包含以下改进：

```python
# 新增的callbacks
- RichProgressBar: 显示训练进度条
- LearningRateMonitor: 监控学习率
- CSVLogger: 保存CSV格式日志（易于查看）

# 新增的trainer参数
- enable_progress_bar=True: 启用进度条
- log_every_n_steps=10: 每10步记录一次
```

### 方案2：使用详细输出脚本

```bash
# 使用详细输出版本（输出同时保存到文件）
bash scripts/train_tsp_verbose.sh
```

特点：
- 使用`python -u`禁用缓冲
- 使用`tee`同时显示和保存日志
- 实时查看训练进度

### 方案3：使用监控脚本

```bash
# 在另一个终端运行监控脚本
bash scripts/monitor_training.sh

# 或者自动刷新（每5秒）
watch -n 5 bash scripts/monitor_training.sh
```

监控内容：
- ✅ 训练进程状态
- 📊 CPU/内存使用情况
- 📁 最新日志文件
- 💾 最新checkpoint
- 📈 训练进度（从CSV读取）

### 方案4：快速测试版本

如果想快速验证训练是否正常工作：

```bash
# 使用小数据集快速测试（2-3分钟完成）
bash scripts/test_deepaco_quick.sh
```

配置：
- 20节点TSP（而不是100节点）
- 1000训练样本（而不是100,000）
- 64 batch size（而不是512）
- 2 epochs（而不是10）

## 训练输出示例

### 正常的DeepACO训练输出应该是这样的：

```
============================================================
Model built: DeepACO
  Baseline: rollout
  Batch size: 512
============================================================

Using 16bit Automatic Mixed Precision (AMP)
GPU available: True (cuda), used: True

============================================================
Starting training for 10 epochs...
Train file: data/TSP_train_num_loc_100.npz
Val file: data/TSP_val_num_loc_100.npz
============================================================

LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

┏━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃    ┃ Name              ┃ Type  ┃ Params┃
┡━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0  │ env               │ TSPEnv│     0 │
│ 1  │ policy            │ Deep..│ 895 K │
│ 2  │ policy.encoder    │ NARGN.│ 895 K │
└────┴───────────────────┴───────┴───────┘

Epoch 0/9  ━━━━━━━━━━━━━━━━━ 0/196   0:00:00 • -:--:--
                                                train/reward: -XX.XXX
                                                train/loss: XX.XXX
```

### 关键指标说明

- **Epoch X/Y**: 当前epoch/总epoch数
- **train/reward**: 训练阶段的平均reward（负值表示cost）
- **train/loss**: REINFORCE损失值
- **val/reward**: 验证阶段的reward
- **每个epoch的batch数**: 取决于train_size和batch_size
  - 100,000样本 ÷ 512 batch_size = 196 batches/epoch

## 查看训练进度的方法

### 1. TensorBoard（最佳可视化）

```bash
# 启动TensorBoard
tensorboard --logdir logs --port 6006

# 然后在浏览器访问
# http://localhost:6006
```

可以看到：
- 📈 训练/验证曲线
- 🎯 Reward变化趋势
- 📉 Loss下降情况
- ⚡ 学习率变化

### 2. CSV日志（易于解析）

```bash
# 查看CSV日志
cat logs/TSP_DeepACO_csv/version_0/metrics.csv

# 或使用column格式化
cat logs/TSP_DeepACO_csv/version_0/metrics.csv | column -t -s ','
```

### 3. Checkpoint文件

```bash
# 查看最新的checkpoint
ls -lht checkpoints/TSP/*.ckpt | head -5
```

每个epoch会保存一个checkpoint（如果性能提升）。

## 性能优化建议

如果训练太慢，可以：

### 1. 减少蚂蚁数量（训练阶段）

```python
"n_ants": {
    "train": 20,   # 从30减到20
    "val": 48,
    "test": 100
}
```

### 2. 禁用局部搜索（训练阶段）

```python
train_with_local_search=False  # 训练快3-5倍
```

注意：这会影响最终性能

### 3. 增加batch size（如果内存允许）

```bash
--batch-size 1024  # 从512增加到1024
```

### 4. 减少问题规模（测试时）

```bash
--num-loc 50  # 从100减到50节点
```

### 5. 使用更小的训练集（快速验证）

```bash
--train-size 10000  # 从100,000减到10,000
```

## 预期训练时间

以下是100节点TSP的大致训练时间（单GPU V100/A100）：

| 配置 | Batch Time | Epoch Time | 10 Epochs |
|------|-----------|-----------|-----------|
| DeepACO (30 ants, LS) | 30s | 1.5h | 15h |
| DeepACO (20 ants, LS) | 20s | 1.0h | 10h |
| DeepACO (30 ants, no LS) | 10s | 30min | 5h |
| AttentionModel | 1s | 3min | 30min |

**LS = Local Search（局部搜索）**

## 常见问题

### Q1: 训练卡在"Loading training dataset from file"
**A**: 这是正常的，第一次加载数据需要时间。之后会开始训练。

### Q2: 第一个batch特别慢
**A**: DeepACO需要编译CUDA kernels（PyG），第一个batch会慢，后续会快一些。

### Q3: 训练时GPU利用率低
**A**: DeepACO包含大量CPU操作（ACO算法），GPU利用率通常在30-60%是正常的。

### Q4: 如何判断训练是否正常进行？
**A**: 检查以下几点：
- ✅ 进程存在且CPU使用率>100%
- ✅ logs目录有新的events文件
- ✅ GPU内存被占用（nvidia-smi）
- ✅ CSV日志文件在增长

## 调试命令

```bash
# 1. 检查训练进程
ps aux | grep train.py

# 2. 检查GPU使用
nvidia-smi

# 3. 监控GPU实时状态
watch -n 1 nvidia-smi

# 4. 查看最新日志
tail -f logs/train_*.log

# 5. 查看CSV指标
tail -f logs/TSP_DeepACO_csv/version_*/metrics.csv

# 6. 检查checkpoint
ls -lht checkpoints/TSP/*.ckpt
```

## 总结

DeepACO训练的核心特点：
1. ⏱️ **慢但强大**：训练慢但解质量高
2. 📊 **需要监控**：必须使用工具监控进度
3. 🎯 **耐心等待**：第一个batch可能需要30-60秒
4. 📈 **长时间训练**：完整训练需要数小时到数十小时

使用本文档提供的工具和脚本，可以有效监控和管理DeepACO的训练过程。
