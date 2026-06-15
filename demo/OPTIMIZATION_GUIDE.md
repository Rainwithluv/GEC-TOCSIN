# 性能优化指南

## 当前性能

ROC AUC: 0.8996 (使用默认权重 0.5/0.5)

## 优化方法

### 方法1: 快速权重优化（推荐）

**仅优化融合权重**，不改变通道参数，速度最快。

```bash
# 使用默认设置
python -m demo.src.advanced_optimizer --mode weights --n-samples 50

# 或使用优化脚本
optimize_weights.bat
```

**预计时间**: 5-10分钟（50个样本）
**预期提升**: ROC AUC 可能提升 0.01-0.03

### 方法2: 完整参数优化

**同时优化权重和TOCSIN通道参数**，速度较慢但可能获得更好结果。

```bash
# 优化权重 + TOCSIN参数（deletion_pct, n_samples）
python -m demo.src.advanced_optimizer --mode full --n-samples 50

# 自定义参数范围
python -m demo.src.advanced_optimizer --mode full \
    --deletion-pcts 0.01,0.015,0.02,0.025 \
    --n-samples-list 5,10,15 \
    --n-samples 50
```

**预计时间**: 20-60分钟（取决于参数范围）
**预期提升**: ROC AUC 可能提升 0.02-0.05

### 方法3: 手动权重测试

快速测试特定权重组合：

```bash
# 测试不同权重组合
python -m demo.src.detector --mode evaluate --dataset xsum --model gpt-4 \
    --coedit-weight 0.6 --tocsin-weight 0.4 --n-samples 50

python -m demo.src.detector --mode evaluate --dataset xsum --model gpt-4 \
    --coedit-weight 0.4 --tocsin-weight 0.6 --n-samples 50

python -m demo.src.detector --mode evaluate --dataset xsum --model gpt-4 \
    --coedit-weight 0.7 --tocsin-weight 0.3 --n-samples 50
```

## 使用优化后的权重

优化完成后，使用最佳权重运行评估：

```bash
# 假设优化得到最佳权重：CoEdIT=0.65, TOCSIN=0.35
python -m demo.src.detector --mode evaluate --dataset xsum --model gpt-4 \
    --coedit-weight 0.65 --tocsin-weight 0.35 --n-samples 200
```

## 预期结果

| 优化方法 | 当前ROC AUC | 预期ROC AUC | 时间 |
|---------|------------|-------------|------|
| 默认权重 | 0.8996 | - | - |
| 快速权重优化 | 0.8996 | 0.91-0.93 | 5-10分钟 |
| 完整参数优化 | 0.8996 | 0.92-0.95 | 20-60分钟 |

## 进一步优化建议

如果权重优化后ROC AUC仍然不理想，可以考虑：

1. **增加样本数量**: 使用更多样本进行评估（200-500）
2. **尝试不同融合策略**: 使用adaptive或cascade融合
3. **特征工程**: 添加更多特征或改进现有特征
4. **数据质量**: 检查数据集质量和标签准确性

## 快速开始

```bash
# 1. 运行快速权重优化
python -m demo.src.advanced_optimizer --mode weights --n-samples 50

# 2. 等待优化完成，记录最佳权重

# 3. 使用最佳权重重新评估
python -m demo.src.detector --mode evaluate --dataset xsum --model gpt-4 \
    --coedit-weight <best_coedit> --tocsin-weight <best_tocsin> --n-samples 200
```
