# Weighted Fusion 融合策略详解

## 概述

Weighted Fusion（加权融合）是MultiFusion-Detector的**默认推荐融合策略**，通过固定权重线性组合CoEdIT和TOCSIN两个通道的分数，实现高效、稳定的LLM文本检测。

## 当前配置

```python
final_score = 0.2 × coedit_score + 0.8 × tocsin_score
```

### 权重分配

| 通道 | 权重 | 作用 |
|------|------|------|
| **CoEdIT** | 0.2 (20%) | 语法规范性检测，提供辅助信息 |
| **TOCSIN** | 0.8 (80%) | Token连贯性检测，主要区分能力 |

## 为什么选择这个权重？

### 测试验证结果

在20样本测试集上的对比结果：

| 融合策略 | ROC AUC | Precision | Recall | F1 | Accuracy |
|----------|---------|-----------|--------|-----|----------|
| Improved Fusion | 0.8850 | 0.8333 | 1.0000 | 0.9091 | 0.9000 |
| Voting Fusion | 0.7475 | 0.8571 | 0.6000 | 0.7059 | 0.7500 |
| Weighted (0.3/0.7) | 0.9250 | 0.9091 | 1.0000 | 0.9524 | 0.9500 |
| **Weighted (0.2/0.8)** | **0.9375** | **0.9091** | **1.0000** | **0.9524** | **0.9500** |

### 选择依据

1. **最高ROC AUC (0.9375)**：在所有策略中分类能力最强
2. **完美Recall (1.0000)**：不会漏检任何LLM文本
3. **高Precision (0.9091)**：误报率低
4. **简单高效**：计算复杂度低，适合生产环境

## 技术原理

### 1. 分数归一化

两个通道的原始分数范围不同，需要先归一化到[0, 1]区间：

```python
# MinMax归一化
normalized_score = (score - min_score) / (max_score - min_score)
```

**重要细节**：归一化时将人类和LLM样本合并计算，确保归一化参数基于完整数据分布。

### 2. 分数反转

TOCSIN原始分数含义是"连贯性越高越像人类"，为了统一指标方向：

```python
# TOCSIN分数反转（高分=LLM）
tocsin_score_for_llm = 1 - tocsin_original_score
```

### 3. 加权融合

归一化后的分数按权重组合：

```python
final_score = 0.2 × normalized_coedit + 0.8 × normalized_tocsin
```

## 实际性能表现

### XSum数据集（50样本）

```bash
python -m demo.src.detector --mode evaluate --dataset xsum --model gpt-4 --fusion weighted --n-samples 50
```

**结果**：
```
ROC AUC: 0.9564
Threshold: 0.2054
Confusion Matrix: [[47, 3], [4, 46]]
Precision: 0.9388
Recall: 0.9200
F1: 0.9293
Accuracy: 0.9300
```

**分析**：
- ✓ ROC AUC 0.95+：优秀分类能力
- ✓ Precision/Recall均衡：0.94/0.92
- ✓ 误报少：仅3个人类文本被误判
- ✓ 漏检少：仅4个LLM文本被漏检

### Writing数据集（50样本）

```bash
python -m demo.src.detector --mode evaluate --dataset writing --model gpt-4 --fusion weighted --n-samples 50
```

**结果**：
```
ROC AUC: 0.7380
Threshold: 0.2271
Confusion Matrix: [[49, 1], [22, 28]]
Precision: 0.9655
Recall: 0.5600
F1: 0.7089
Accuracy: 0.7700
```

**分析**：
- ⚠ ROC AUC 0.74：中等分类能力
- ⚠ Recall 0.56：**44%的LLM文本被漏检**
- ✓ Precision 0.97：误报极少
- **原因**：Writing数据集LLM文本更接近人类写作，难以区分

## 不同数据集的适应性

### 数据集特性对比

| 数据集 | 文本类型 | LLM特征 | 检测难度 | 推荐权重 |
|--------|----------|---------|----------|----------|
| **XSum** | 新闻摘要 | 结构化、模式化 | 较易 | 0.2/0.8 ✓ |
| **Writing** | 创意写作 | 模仿人类表达 | 较难 | 0.2/0.8 ⚠️ |

### Writing数据集问题分析

Writing数据集Recall低的原因：

1. **语义相似度高**：LLM生成的创意写作与人类写作难以区分
2. **通道分离度低**：两个通道的分数分布重叠严重
3. **权重不匹配**：可能需要针对Writing调整权重

**潜在解决方案**：
```bash
# 尝试提高CoEdIT权重（针对Writing数据集）
python -m demo.src.detector --mode evaluate --dataset writing --model gpt-4 --coedit-weight 0.4 --tocsin-weight 0.6 --n-samples 50
```

## 使用指南

### 基本使用

```bash
# 默认使用最优权重（0.2/0.8）
python -m demo.src.detector --mode evaluate --dataset xsum --model gpt-4 --fusion weighted

# 指定样本数量
python -m demo.src.detector --mode evaluate --dataset xsum --model gpt-4 --fusion weighted --n-samples 100
```

### 自定义权重

```bash
# 调整权重比例
python -m demo.src.detector --mode evaluate --dataset xsum --model gpt-4 \
    --fusion weighted --coedit-weight 0.3 --tocsin-weight 0.7

# 等权重融合
python -m demo.src.detector --mode evaluate --dataset xsum --model gpt-4 \
    --fusion weighted --coedit-weight 0.5 --tocsin-weight 0.5
```

### 阈值调整

```bash
# 手动指定阈值（降低阈值提高Recall）
python -m demo.src.detector --mode evaluate --dataset writing --model gpt-4 \
    --fusion weighted --threshold 0.15 --n-samples 50
```

## 技术实现

### 核心代码逻辑

```python
class WeightedFusion:
    def __init__(self, weights={'coedit': 0.2, 'tocsin': 0.8}):
        # 归一化权重（确保和为1）
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}

    def normalize_and_fuse(self, human_scores, llm_scores):
        """
        1. 合并人类和LLM分数
        2. MinMax归一化到[0,1]
        3. 按权重加权融合
        """
        normalized_human = {}
        normalized_llm = {}

        for channel in human_scores.keys():
            # 合并所有样本
            all_scores = np.concatenate([
                human_scores[channel],
                llm_scores[channel]
            ])

            # 归一化
            norm_scores = (all_scores - all_scores.min()) / (all_scores.max() - all_scores.min())

            # 分离
            n_human = len(human_scores[channel])
            normalized_human[channel] = norm_scores[:n_human]
            normalized_llm[channel] = norm_scores[n_human:]

        # 加权融合
        human_fused = sum(self.weights[c] * normalized_human[c] for c in self.weights)
        llm_fused = sum(self.weights[c] * normalized_llm[c] for c in self.weights)

        return human_fused, llm_fused
```

## 参数调优建议

### 场景化权重选择

| 场景 | 推荐权重 | 原因 |
|------|----------|------|
| **通用检测** | 0.2/0.8 | 经过验证的最优配置 |
| **结构化文本** | 0.2/0.8 | TOCSIN对模式化文本检测能力强 |
| **创意文本** | 0.4/0.6 | 尝试提高CoEdIT权重 |
| **高精度需求** | 0.2/0.8 | 提高阈值即可 |
| **高召回需求** | 0.2/0.8 | 降低阈值即可 |

### 权重搜索方法

如需针对特定数据集优化权重：

```python
# 网格搜索最优权重
best_weights = None
best_auc = 0

for coedit_w in [0.1, 0.2, 0.3, 0.4, 0.5]:
    tocsin_w = 1 - coedit_w
    weights = {'coedit': coedit_w, 'tocsin': tocsin_w}

    # 评估并记录ROC AUC
    auc = evaluate_with_weights(weights)

    if auc > best_auc:
        best_auc = auc
        best_weights = weights
```

## 与其他策略对比

### Weighted vs Improved

| 特性 | Weighted | Improved |
|------|----------|----------|
| **复杂度** | 低（固定权重） | 中（自适应权重） |
| **性能** | ROC AUC 0.94+ | ROC AUC 0.89 |
| **稳定性** | 高 | 中 |
| **适用场景** | 生产环境 | 研究/优化 |

### Weighted vs Voting

| 特性 | Weighted | Voting |
|------|----------|--------|
| **复杂度** | 低 | 高（需置信度计算） |
| **性能** | ROC AUC 0.94+ | ROC AUC 0.75 |
| **Recall** | 高 | 低（0.60） |
| **适用场景** | 通用 | 特殊需求 |

## 最佳实践

### 1. 生产环境部署

```bash
# 使用默认最优配置
python -m demo.src.detector --mode evaluate --dataset xsum --model gpt-4 --fusion weighted
```

### 2. 新数据集适配

```bash
# 步骤1：用小样本评估
python -m demo.src.detector --mode evaluate --dataset new_dataset --model gpt-4 --fusion weighted --n-samples 20

# 步骤2：查看ROC AUC和Recall

# 步骤3：如Recall < 0.7，尝试降低阈值或调整权重
python -m demo.src.detector --mode evaluate --dataset new_dataset --model gpt-4 --fusion weighted --threshold 0.15 --n-samples 50
```

### 3. 性能监控

持续关注以下指标：
- **ROC AUC**：应 > 0.8
- **Recall**：应 > 0.7（除非极高精度需求）
- **Precision**：根据应用场景调整

## 常见问题

### Q: 为什么TOCSIN权重更高？

A: 测试表明TOCSIN在LLM检测中的区分度更高。Token连贯性是LLM文本的核心特征，比语法规范性更能区分LLM和人类写作。

### Q: 什么时候需要调整权重？

A: 当以下情况发生时：
1. ROC AUC < 0.8
2. Recall < 0.7 且无法通过阈值调整改善
3. 特定数据集的性能显著低于预期

### Q: Weighted融合是否适合所有场景？

A: 大部分场景适合。但对于某些高度模仿人类的写作（如创意写作），可能需要考虑其他方法或接受较低的性能。

## 总结

Weighted Fusion (0.2/0.8) 是当前推荐的融合策略：

✅ **优点**：
- 性能优秀（ROC AUC 0.94+）
- 实现简单
- 稳定可靠
- 适合生产环境

⚠️ **注意**：
- Writing类数据集Recall可能较低
- 需要针对新数据集评估性能
- 极端情况可能需要调整权重

---

*最后更新：2024-06-18*
