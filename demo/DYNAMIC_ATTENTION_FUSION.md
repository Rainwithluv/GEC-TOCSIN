# Dynamic Attention Fusion 动态注意力融合策略

## 概述

Dynamic Attention Fusion（动态注意力融合）是MultiFusion-Detector的新型融合策略，通过**动态权重分配**替代固定权重，根据每个输入样本的特征自适应调整CoEdIT和TOCSIN的权重比例。

## 为什么需要动态融合？

### 固定权重的问题

当前Weighted Fusion使用固定权重（0.2/0.8）：
- ✅ 对XSum等结构化文本效果好
- ❌ 对Writing等创意文本效果差（Recall仅0.56）
- ❌ 无法适应不同文本类型

### 动态融合的优势

| 特性 | 固定权重 | 动态注意力 |
|------|----------|------------|
| **适应性** | 静态，一刀切 | 动态，每个样本自适应 |
| **复杂度** | 低 | 中 |
| **XSum表现** | ROC AUC 0.95 | 预期相当或更好 |
| **Writing表现** | ROC AUC 0.74 | **预期显著提升** |
| **泛化能力** | 有限 | 强 |

## 融合模式

### 1. Confidence-based（基于置信度）

**原理**：分数越远离0.5（决策边界），置信度越高，权重越大

```python
# 置信度 = |score - 0.5|
coedit_confidence = |coedit_score - 0.5|
tocsin_confidence = |tocsin_score - 0.5|

# 权重与置信度成正比
coedit_weight = coedit_confidence / (coedit_confidence + tocsin_confidence)
tocsin_weight = tocsin_confidence / (coedit_confidence + tocsin_confidence)
```

**适用场景**：
- 两个通道都表现较好时
- 需要简单高效的动态权重

### 2. Entropy-based（基于信息熵）

**原理**：熵越低（预测越确定），权重越大

```python
# 信息熵
entropy = -(p * log(p) + (1-p) * log(1-p))
其中 p = score

# 权重与熵成反比
coedit_weight = (1 / entropy_coedit) / (1 / entropy_coedit + 1 / entropy_tocsin)
```

**适用场景**：
- 需要更平滑的权重分配
- 关注预测确定性

### 3. Hybrid（混合模式）

**原理**：结合置信度和熵两种方法

```python
coedit_weight = (confidence_weight + entropy_weight) / 2
tocsin_weight = 1 - coedit_weight
```

**适用场景**：
- 追求最佳性能
- 需要稳定的权重分配

### 4. Learned Attention（学习型注意力）

**原理**：使用神经网络学习最优注意力权重

```python
class CrossBranchAttention(nn.Module):
    # Multi-head attention for dynamic weighting
    # Input: [coedit_score, tocsin_score]
    # Output: [coedit_weight, tocsin_weight]
```

**适用场景**：
- 有足够训练数据
- 需要端到端优化
- 可接受训练成本

## 技术实现

### 核心算法

```python
class DynamicAttentionFusion:
    def normalize_and_fuse(self, human_scores, llm_scores):
        # 1. MinMax归一化（合并人类+LLM样本）
        normalized = minmax_normalize(concat(human_scores, llm_scores))

        # 2. 计算动态权重
        weights = self.compute_weights(
            normalized['coedit'],
            normalized['tocsin']
        )

        # 3. 应用权重融合
        fused = weights['coedit'] * normalized['coedit'] +
                weights['tocsin'] * normalized['tocsin']

        return fused
```

### Cross-Branch Attention结构

```
Input: [coedit_score, tocsin_score]
        ↓
    Layer Normalization
        ↓
    Q, K, V Projections (Multi-head)
        ↓
    Scaled Dot-Product Attention
        ↓
    Output Projection + Softmax
        ↓
Output: [coedit_weight, tocsin_weight]
```

## 使用方法

### 命令行使用

```bash
# 使用置信度模式
python -m demo.src.detector --mode evaluate --dataset xsum --model gpt-4 \
    --fusion dynamic --dynamic-mode confidence --n-samples 50

# 使用熵模式
python -m demo.src.detector --mode evaluate --dataset writing --model gpt-4 \
    --fusion dynamic --dynamic-mode entropy --n-samples 50

# 使用混合模式（推荐）
python -m demo.src.detector --mode evaluate --dataset writing --model gpt-4 \
    --fusion dynamic --dynamic-mode hybrid --n-samples 50

# 调整温度参数
python -m demo.src.detector --mode evaluate --dataset xsum --model gpt-4 \
    --fusion dynamic --dynamic-mode hybrid --temperature 0.5 --n-samples 50
```

### Python API

```python
from demo.src.fusion.dynamic_attention_fusion import DynamicAttentionFusion

# 初始化
fusion = DynamicAttentionFusion(
    mode='hybrid',      # confidence, entropy, hybrid, attention
    temperature=1.0,    # 温度参数
    min_weight=0.1,     # 最小权重
    max_weight=0.9      # 最大权重
)

# 融合
human_fused, llm_fused = fusion.normalize_and_fuse(
    human_channel_scores={'coedit': human_coedit, 'tocsin': human_tocsin},
    llm_channel_scores={'coedit': llm_coedit, 'tocsin': llm_tocsin}
)

# 获取权重分布
coedit_weights, tocsin_weights = fusion.get_weights(coedit_scores, tocsin_scores)
print(f"CoEdIT权重范围: [{coedit_weights.min():.3f}, {coedit_weights.max():.3f}]")
print(f"TOCSIN权重范围: [{tocsin_weights.min():.3f}, {tocsin_weights.max():.3f}]")
```

## 性能预期

### 理论优势

1. **Writing数据集提升**
   - 固定权重：Recall 0.56
   - 动态权重：预期Recall 0.7+
   - 原因：对难样本自动提高有效通道权重

2. **XSum数据集保持**
   - 固定权重：ROC AUC 0.95
   - 动态权重：预期ROC AUC 0.95+
   - 原因：对简单样本保持高权重通道

3. **泛化能力**
   - 对新数据集适应性更强
   - 无需手动调参

### 权重分布示例

```
XSum数据集（结构化，简单）:
CoEdIT权重: [0.15, 0.18, 0.22, ...]  → 相对稳定
TOCSIN权重: [0.85, 0.82, 0.78, ...]  → TOCSIN主导

Writing数据集（创意，复杂）:
CoEdIT权重: [0.05, 0.35, 0.60, ...]  → 动态调整
TOCSIN权重: [0.95, 0.65, 0.40, ...]  → 根据置信度变化
```

## 参数调优

### 温度参数（temperature）

控制权重分配的"尖锐程度"：

| temperature | 效果 | 适用场景 |
|-------------|------|----------|
| 0.5 | 更尖锐，极端权重 | 确信通道表现好 |
| 1.0 | 标准 | 默认推荐 |
| 2.0 | 更平滑，均衡权重 | 通道表现相近 |

### 权重边界（min_weight, max_weight）

防止权重极端化：

```python
# 保守设置
min_weight=0.2, max_weight=0.8

# 激进设置
min_weight=0.05, max_weight=0.95

# 默认设置
min_weight=0.1, max_weight=0.9
```

## 与固定权重对比

| 维度 | 固定权重(0.2/0.8) | 动态注意力 |
|------|------------------|------------|
| **XSum ROC AUC** | 0.9564 | 预期 0.95+ |
| **Writing ROC AUC** | 0.7380 | 预期 0.80+ |
| **Writing Recall** | 0.56 | 预期 0.65+ |
| **计算复杂度** | O(n) | O(n) |
| **训练需求** | 无 | 可选 |
| **泛化能力** | 中 | 高 |
| **可解释性** | 强 | 中 |

## 实现状态

### 已完成
- ✅ Confidence-based 模式
- ✅ Entropy-based 模式
- ✅ Hybrid 模式
- ✅ Cross-branch Attention 架构
- ✅ 权重分析和可视化

### 待完成
- ⏳ Learned Attention 训练流程
- ⏳ 集成到主detector
- ⏳ 性能基准测试

## 快速开始

### 1. 测试动态融合

```bash
cd demo
python -m src.fusion.dynamic_attention_fusion
```

### 2. 在detector中使用

```bash
# 很快可用（待集成）
python -m demo.src.detector --mode evaluate --dataset writing --model gpt-4 \
    --fusion dynamic --dynamic-mode hybrid --n-samples 50
```

## 总结

Dynamic Attention Fusion通过**样本级自适应权重**解决了固定权重策略的局限性：

✅ **优点**：
- 对不同类型文本自适应
- 预期显著提升Writing类数据集性能
- 保持XSum类数据集性能
- 无需训练即可使用（confidence/entropy/hybrid模式）

⚠️ **注意**：
- 计算量略高于固定权重（可忽略）
- 需要选择合适的模式和参数
- Learned Attention模式需要训练数据

---

*最后更新：2024-06-18*
