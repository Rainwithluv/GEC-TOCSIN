# Attention Mode 快速训练和评估指南

## 概述

使用demo/data目录中的现有JSON文件训练Attention Mode模型。

## 可用数据集

| 数据集 | 文件名 | 样本数 | 特点 |
|--------|--------|--------|------|
| XSum + GPT-4o | `xsum.GPT-4o.normal.test_data.json` | ~200 | 推荐，效果较好 |
| Writing + GPT-4o | `writing.GPT-4o.normal.test_data.json` | ~200 | 写作类文本 |
| XSum + Claude | `xsum.Claude-3.5-Sonnet.normal.test_data.json` | ~200 | 摘要类文本 |
| Writing + Claude | `writing.Claude-3.5-Sonnet.normal.test_data.json` | ~200 | 创意写作 |

## 快速开始

### 方法1: 使用批处理脚本（推荐）

```bash
# 一键训练和评估
demo\train_evaluate_attention.bat
```

脚本会：
1. 让你选择数据集
2. 让你选择训练模式（快速测试/完整训练）
3. 自动训练模型
4. 自动评估并与固定权重对比

### 方法2: 命令行直接运行

```bash
cd d:\code_VScode\GEC-TOCSIN
call demo\venv\Scripts\activate.bat

# 训练（使用100个样本快速测试）
python demo\train_attention_simple.py --data demo\data\xsum.GPT-4o.normal.test_data.json --n-samples 100

# 训练（使用全部样本）
python demo\train_attention_simple.py --data demo\data\xsum.GPT-4o.normal.test_data.json

# 评估
python demo\evaluate_attention.py --model models\attention_model.pth --data demo\data\xsum.GPT-4o.normal.test_data.json --compare
```

## 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | 必需 | 训练数据路径 |
| `--output` | models/attention_model.pth | 模型保存路径 |
| `--n-samples` | None（全部） | 使用的样本数量 |
| `--batch-size` | 32 | 批次大小 |
| `--epochs` | 50 | 训练轮数 |
| `--lr` | 0.001 | 学习率 |
| `--test-size` | 0.2 | 验证集比例 |
| `--no-cache` | False | 不缓存分数（会很慢） |

## 评估参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 必需 | 训练好的模型路径 |
| `--data` | 必需 | 评估数据路径 |
| `--batch-size` | 32 | 批次大小 |
| `--compare` | False | 与固定权重对比 |
| `--output` | None | 保存结果到JSON |

## 示例用法

### 快速测试（5-10分钟）

```bash
python demo\train_attention_simple.py --data demo\data\xsum.GPT-4o.normal.test_data.json --n-samples 100 --epochs 30
```

### 完整训练（20-40分钟）

```bash
python demo\train_attention_simple.py --data demo\data\xsum.GPT-4o.normal.test_data.json --epochs 50
```

### 评估并对比

```bash
python demo\evaluate_attention.py --model models\attention_model.pth --data demo\data\xsum.GPT-4o.normal.test_data.json --compare
```

### 使用不同数据集

```bash
# Writing数据集
python demo\train_attention_simple.py --data demo\data\writing.GPT-4o.normal.test_data.json --output models\writing_attention.pth

# Claude数据集
python demo\train_attention_simple.py --data demo\data\xsum.Claude-3.5-Sonnet.normal.test_data.json --output models\claude_attention.pth
```

## 预期结果

### 训练输出

```
Epoch  1/50: Train Loss: 0.6931, Train Acc: 0.5000 | Val Loss: 0.6932, Val Acc: 0.5000
Epoch  2/50: Train Loss: 0.6905, Train Acc: 0.5200 | Val Loss: 0.6912, Val Acc: 0.5300 ✓ (best: 0.5300)
...
Epoch 15/50: Train Loss: 0.4523, Train Acc: 0.7800 | Val Loss: 0.5123, Val Acc: 0.7500 ✓ (best: 0.7500)
...
训练完成! 最佳验证准确率: 0.8200
模型已保存到: models/attention_model.pth
```

### 评估输出

```
评估结果
================================================================================

主要指标:
  ROC AUC:      0.8542
  阈值:         0.5234
  准确率:       0.8100
  精确率:       0.8350
  召回率:       0.7800
  F1分数:       0.8069

混淆矩阵:
  [[81, 19], [20, 80]]
  [[TN, FP],
   [FN, TP]]

权重分布分析:
  CoEdIT权重:
    平均值: 0.2341
    标准差: 0.1234
    最小值: 0.0523
    最大值: 0.5891
  TOCSIN权重:
    平均值: 0.7659
    标准差: 0.1234
    最小值: 0.4109
    最大值: 0.9477
```

### 与固定权重对比

```
与固定权重方法对比
======================================================================

配置                      ROC AUC   Precision  Recall     F1
----------------------------------------------------------------------
当前最优固定权重           0.8423    0.8211     0.8156     0.8183
较高TOCSIN权重             0.8356    0.8100     0.8300     0.8199
等权重                    0.8012    0.7900     0.7800     0.7849
Attention Model           0.8542    0.8350     0.7800     0.8069

→ 最佳ROC AUC: Attention - Neural Network (0.8542)
→ 最佳F1分数: 当前最优固定权重 (0.8183)
```

## 如何判断模型好坏

### 评估指标

| 指标 | 优秀 | 良好 | 需改进 |
|------|------|------|--------|
| ROC AUC | > 0.85 | 0.75-0.85 | < 0.75 |
| 准确率 | > 0.80 | 0.70-0.80 | < 0.70 |
| 召回率 | > 0.75 | 0.65-0.75 | < 0.65 |
| F1分数 | > 0.80 | 0.70-0.80 | < 0.70 |

### 权重分析

1. **动态范围**
   - 良好: CoEdIT权重范围 > 0.3
   - 不足: CoEdIT权重范围 < 0.1（可能退化为固定权重）

2. **平均值**
   - 检查CoEdIT和TOCSIN的平均权重
   - 应该与0.2/0.8接近或更好

3. **标准差**
   - 较大标准差说明模型确实在学习动态调整
   - 过小标准差可能说明模型退化为固定权重

## 故障排除

### 问题1: 训练准确率不提升

**可能原因**：
- 学习率过高或过低
- 数据质量问题
- 模型容量不足

**解决方案**：
```bash
# 调整学习率
python demo\train_attention_simple.py --data ... --lr 0.0001

# 增加模型容量
python demo\train_attention_simple.py --data ... --hidden-dim 32 --num-heads 4
```

### 问题2: 训练很慢

**可能原因**：
- 没有缓存分数
- 使用了CPU而非GPU

**解决方案**：
```bash
# 确保缓存开启（默认开启）
# 不要使用 --no-cache 参数

# 如果有GPU，确保使用
python demo\train_attention_simple.py --data ... --device cuda
```

### 问题3: 评估结果比固定权重差

**可能原因**：
- 训练数据不足
- 过拟合
- 需要更多训练轮数

**解决方案**：
```bash
# 使用更多样本
python demo\train_attention_simple.py --data ... --n-samples 500

# 训练更多轮
python demo\train_attention_simple.py --data ... --epochs 100
```

## 下一步

训练完成后，你可以：

1. **在其他数据集上测试泛化能力**
   ```bash
   python demo\evaluate_attention.py --model models\attention_model.pth --data demo\data\writing.GPT-4o.normal.test_data.json --compare
   ```

2. **集成到detector中**
   ```bash
   python -m demo.src.detector --mode evaluate --dataset xsum --model gpt-4 --fusion dynamic --dynamic-mode attention --model-path models\attention_model.pth
   ```

3. **分析学到的权重**
   - 查看评估输出中的权重分布
   - 了解模型如何动态调整权重

---

*最后更新: 2024-06-18*
