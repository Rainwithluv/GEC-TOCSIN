# 优化器说明

## 问题：为什么优化器很慢？

### 之前的 advanced_optimizer 的问题

`advanced_optimizer` 的工作方式：
```
对于每个权重组合：
  1. 重新加载模型
  2. 对所有文本重新评分
  3. 计算ROC AUC
```

时间计算：
- 默认 11 × 11 = 121 种组合
- 每种组合评分 100 个文本需要 ~30秒
- 总时间：30秒 × 121 = **60分钟**

### 新的 smart_optimizer 的改进

`smart_optimizer` 的工作方式：
```
1. 加载模型一次
2. 对所有文本评分一次（~2-3分钟）
3. 快速测试21种权重组合（~1秒）
```

总时间：**3-4分钟**

## 三种优化器对比

| 优化器 | 时间 | 说明 |
|--------|------|------|
| `advanced_optimizer` | 60分钟 | 旧版本，每个组合都重新评分 |
| `smart_optimizer` | 3-4分钟 | ⭐推荐，只评分一次 |
| 手动测试 | 5分钟 | 测试几个特定权重 |

## 推荐使用方式

### 快速优化（3-4分钟）

```bash
python -m demo.src.smart_optimizer --dataset xsum --model gpt-4 --n-samples 50 --n-steps 21
```

或使用脚本：
```bash
smart_optimize.bat
```

### 输出示例

```
=== Smart Optimizer ===
This optimizer scores texts ONCE, then tests weight combinations quickly.

Loading models (one-time operation)...
Models loaded!

Loading dataset: xsum, model: gpt-4
Using 50 human texts and 50 LLM texts

Scoring human texts (one-time)...
Scoring LLM texts (one-time)...

Scores computed!
Human: CoEdIT=0.8234, TOCSIN=0.9876
LLM:   CoEdIT=0.5412, TOCSIN=0.8234

Testing 21 weight combinations...
  ✓ New best: CoEdIT=0.500, ROC AUC=0.8996
  ✓ New best: CoEdIT=0.550, ROC AUC=0.9012
  ✓ New best: CoEdIT=0.600, ROC AUC=0.9034
  ✓ New best: CoEdIT=0.650, ROC AUC=0.9056

============================================================
OPTIMIZATION COMPLETE
============================================================

🎯 Best weights:
   CoEdIT: 0.6500
   TOCSIN: 0.3500

📈 Best ROC AUC: 0.9056

💡 To use these weights:
   python -m demo.src.detector --mode evaluate --dataset xsum --model gpt-4
       --coedit-weight 0.6500 --tocsin-weight 0.3500
```

### 使用优化后的权重

```bash
python -m demo.src.detector --mode evaluate --dataset xsum --model gpt-4 \
    --coedit-weight 0.65 --tocsin-weight 0.35 --n-samples 200
```

## 常见问题

### Q: 为什么要优化权重？

A: 默认权重（0.5/0.5）可能不是最优的。优化器可以找到让ROC AUC最高的权重组合。

### Q: 优化后ROC AUC能提升多少？

A: 通常能提升 0.01-0.03，有时更多。

### Q: 可以用更少的样本优化吗？

A: 可以，但结果可能不够准确。建议至少使用50个样本。

### Q: 优化后的权重在其他数据集上有效吗？

A: 可能有差异，建议在不同数据集上分别优化。

## 文件说明

- **smart_optimizer.py** - 智能优化器（推荐）
- **advanced_optimizer.py** - 高级优化器（较慢）
- **optimize_weights.py** - 基础优化器
- **smart_optimize.bat** - 快速启动脚本
