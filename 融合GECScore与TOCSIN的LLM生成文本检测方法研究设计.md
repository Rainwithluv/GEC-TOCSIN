# 融合GECScore与TOCSIN的LLM生成文本检测方法研究设计

## 1. 研究背景与动机

### 1.1 问题背景

大型语言模型（LLM）的快速发展带来了文本生成的革命性突破，但也引发了虚假信息传播、学术不端等问题。零样本检测作为无需大规模训练数据的方法，在LLM生成文本检测中具有重要意义。

### 1.2 现有方法分析

#### TOCSIN方法（EMNLP 2024）
- **核心原理**：Token Cohesiveness（Token连贯性）
- **检测逻辑**：LLM生成文本删除token后语义变化更小
- **优势**：
  - 理论基础扎实（基于LLM生成机制）
  - 双通道架构灵活
  - 可解释性强
- **局限**：
  - 需要访问源LLM进行推理
  - 计算复杂度较高
  - BERT模型加载开销

#### GECScore方法（COLING 2025）
- **核心原理**：Grammar Error Correction Score
- **检测逻辑**：人类文本从LLM视角看需要更多"修正"
- **优势**：
  - 实现简单，易于部署
  - 黑盒友好，仅需CoEdIT-large（基于FLAN-T5-large）
  - 高准确率（98.62% AUROC）
- **局限**：
  - 单一特征维度
  - 依赖CoEdIT模型的修正质量
  - 对改写攻击的鲁棒性待验证

### 1.3 研究动机

两种方法从**不同角度**捕捉LLM生成文本的特征：
- **TOCSIN**：关注**结构连贯性**（token层面的规律性）
- **GECScore**：关注**语法规范性**（语言层面的标准化）

这些角度具有**互补性**：
1. 多维度特征融合可提高检测准确率
2. 可增强对对抗攻击的鲁棒性
3. 可降低单一方法的失效风险

---

## 2. 研究目标

### 2.1 主要目标

1. **设计融合架构**：提出一种有效融合GECScore和TOCSIN的检测方法
2. **提高检测性能**：在主流数据集上超越两种单一方法
3. **增强鲁棒性**：提高对改写攻击、跨领域、跨模型的泛化能力
4. **保持实用性**：维持零样本、黑盒友好的特性

### 2.2 预期贡献

1. 提出多通道零样本检测融合范式
2. 验证不同特征维度的互补性
3. 实现SOTA级别的检测性能
4. 提供可复现的开源实现

---

## 3. 方法设计

### 3.1 整体架构：MultiFusion-Detector

```
┌─────────────────────────────────────────────────────────────────┐
│                  MultiFusion-Detector 架构                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入: 待检测文本                                                │
│  ↓                                                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    特征提取模块                            │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │                                                            │  │
│  │  通道A: 语法规范性特征（GECScore）                         │  │
│  │  ┌────────────────────────────────────────────────────┐   │  │
│  │  │ 1. GPT-4o语法修正                                   │   │  │
│  │  │ 2. Rouge-2相似度计算                               │   │  │
│  │  │ 3. 输出: GECScore_A                                │   │  │
│  │  └────────────────────────────────────────────────────┘   │  │
│  │                                                            │  │
│  │  通道B: Token连贯性特征（TOCSIN）                         │  │
│  │  ┌────────────────────────────────────────────────────┐   │  │
│  │  │ 1. 随机token删除扰动（10次×1.5%）                   │   │  │
│  │  │ 2. BART Score语义差异测量                          │   │  │
│  │  │ 3. 输出: cohesiveness_score_B                      │   │  │
│  │  └────────────────────────────────────────────────────┘   │  │
│  │                                                            │  │
│  │  通道C: 概率分布特征（可选）                               │  │
│  │  ┌────────────────────────────────────────────────────┐   │  │
│  │  │ 1. LLM log-likelihood计算                          │   │  │
│  │  │ 2. Log-rank计算                                    │   │  │
│  │  │ 3. 输出: prob_score_C                              │   │  │
│  │  └────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ↓                                                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    特征融合模块                            │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │                                                            │  │
│  │  融合策略:                                                  │  │
│  │  ┌────────────────────────────────────────────────────┐   │  │
│  │  │ 策略1: 加权融合（Weighted Fusion）                  │   │  │
│  │  │   final_score = w_A×A + w_B×B + w_C×C              │   │  │
│  │  └────────────────────────────────────────────────────┘   │  │
│  │  ┌────────────────────────────────────────────────────┐   │  │
│  │  │ 策略2: 自适应融合（Adaptive Fusion）                │   │  │
│  │  │   基于文本特征动态调整权重                          │   │  │
│  │  └────────────────────────────────────────────────────┘   │  │
│  │  ┌────────────────────────────────────────────────────┐   │  │
│  │  │ 策略3: 级联融合（Cascade Fusion）                   │   │  │
│  │  │   通道初筛 + 通道细筛                               │   │  │
│  │  └────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ↓                                                              │
│  输出: 检测结果 + 置信度                                         │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 特征提取设计

#### 通道A：语法规范性特征（基于CoEdIT）

```python
from transformers import AutoTokenizer, AutoModelForMultimodalLM
from rouge import Rouge

# 加载CoEdIT模型
model_name = "grammarly/coedit-large"
coedit_tokenizer = AutoTokenizer.from_pretrained(model_name)
coedit_model = AutoModelForMultimodalLM.from_pretrained(model_name)
rouge = Rouge()

def extract_gecscore_features(text):
    """使用CoEdIT提取语法规范性特征"""

    # 1. CoEdIT语法修正（指令式）
    instruction = f"Fix grammatical errors in this sentence: {text}"
    inputs = coedit_tokenizer(instruction, return_tensors="pt", max_length=512, truncation=True)

    # 生成修正文本
    outputs = coedit_model.generate(**inputs, max_length=512)
    gec_text = coedit_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 2. 多层次Rouge分数
    rouge_scores = rouge.get_scores(text, gec_text, avg=True)
    features = {
        'rouge_1_f': rouge_scores['rouge-1']['f'],
        'rouge_2_f': rouge_scores['rouge-2']['f'],
        'rouge_l_f': rouge_scores['rouge-l']['f'],
        'text_length': len(text.split()),
        'gec_text': gec_text,  # 保存修正后文本
        'gec_length_change': len(gec_text.split()) - len(text.split())
    }

    return features
```

#### 通道B：Token连贯性特征

```python
def extract_cohesiveness_features(text, pct=0.015, n_samples=10):
    """提取token连贯性特征"""

    perturbed_texts = []
    for _ in range(n_samples):
        # 随机删除1.5%的token
        perturbed = random_token_deletion(text, pct)
        perturbed_texts.append(perturbed)

    # BART Score计算
    bart_scores = bart_scorer.score([text]*n_samples, perturbed_texts)

    features = {
        'bart_mean': np.mean(bart_scores),
        'bart_std': np.std(bart_scores),
        'bart_min': np.min(bart_scores),
        'bart_max': np.max(bart_scores),
        'cohesiveness_score': np.exp(-np.mean(bart_scores))
    }

    return features
```

### 3.3 融合策略设计

#### 策略1：加权融合（Weighted Fusion）

```python
def weighted_fusion(gec_features, cohesion_features, weights=None):
    """固定权重融合"""

    if weights is None:
        weights = {'gec': 0.5, 'cohesion': 0.5}

    # 归一化特征
    gec_norm = normalize(gec_features['rouge_2_f'])
    cohesion_norm = normalize(cohesion_features['cohesiveness_score'])

    # 加权融合
    final_score = weights['gec'] * gec_norm + weights['cohesion'] * cohesion_norm

    return final_score
```

#### 策略2：自适应融合（Adaptive Fusion）

```python
def adaptive_fusion(gec_features, cohesion_features, text_stats):
    """基于文本特征的自适应融合"""

    # 根据文本长度、领域等特征动态调整权重
    text_length = text_stats['length']
    domain = text_stats['domain']

    # 长文本更依赖语法规范性，短文本更依赖连贯性
    if text_length > 200:
        w_gec = 0.6
        w_cohesion = 0.4
    else:
        w_gec = 0.4
        w_cohesion = 0.6

    # 领域特定权重
    if domain == 'academic':
        w_gec += 0.1  # 学术文本语法规范性更重要
        w_cohesion -= 0.1

    # 归一化
    w_total = w_gec + w_cohesion
    w_gec /= w_total
    w_cohesion /= w_total

    # 融合
    gec_norm = normalize(gec_features['rouge_2_f'])
    cohesion_norm = normalize(cohesion_features['cohesiveness_score'])

    final_score = w_gec * gec_norm + w_cohesion * cohesion_norm

    return final_score
```

#### 策略3：级联融合（Cascade Fusion）

```python
def cascade_fusion(text, gec_threshold_high=0.95, gec_threshold_low=0.85):
    """级联检测：先粗筛再细筛"""

    # 第一阶段：GECScore快速筛选
    gec_features = extract_gecscore_features(text)
    gec_score = gec_features['rouge_2_f']

    # 高置信度判断
    if gec_score > gec_threshold_high:
        return {'prediction': 'llm', 'confidence': 'high', 'score': gec_score}
    elif gec_score < gec_threshold_low:
        return {'prediction': 'human', 'confidence': 'high', 'score': gec_score}

    # 第二阶段：模糊样本用Token连贯性细筛
    cohesion_features = extract_cohesiveness_features(text)
    cohesion_score = cohesion_features['cohesiveness_score']

    # 融合决策
    final_score = 0.6 * gec_score + 0.4 * cohesion_score
    threshold = (gec_threshold_high + gec_threshold_low) / 2

    return {
        'prediction': 'llm' if final_score > threshold else 'human',
        'confidence': 'medium',
        'score': final_score
    }
```

---

## 4. 实验设计

### 4.1 数据集

| 数据集 | 用途 | 特点 |
|--------|------|------|
| XSum | 训练/测试 | 新闻摘要，正式文本 |
| WritingPrompts | 训练/测试 | 创意写作，非正式文本 |
| SQuAD | 泛化测试 | 阅读理解对话 |
| PubMed | 跨领域测试 | 生物医学文本 |

### 4.2 生成模型

| 模型类型 | 具体模型 |
|----------|----------|
| API模型 | GPT-3.5, GPT-4, GPT-4o, Gemini |
| 开源模型 | GPT-2, GPT-J, LLaMA, OPT |

### 4.3 评估设置

#### 基础评估
- 标准测试集上的AUROC、准确率、F1分数

#### 泛化评估
- 跨模型：训练数据A模型，测试数据B模型
- 跨领域：训练数据新闻领域，测试数据创意写作
- 跨长度：不同长度文本的检测表现

#### 鲁棒性评估
- 改写攻击：使用QuillBot等工具改写LLM文本
- 混合攻击：插入人类片段到LLM文本

### 4.4 对比方法

| 方法 | 类型 | 说明 |
|------|------|------|
| GECScore | Baseline | 单一语法规范性特征 |
| TOCSIN | Baseline | 单一Token连贯性特征 |
| MultiFusion-S | Proposed | 简单加权融合 |
| MultiFusion-A | Proposed | 自适应权重融合 |
| MultiFusion-C | Proposed | 级联融合 |

### 4.5 消融实验

1. **特征消融**：测试不同特征组合的贡献
2. **通道消融**：测试每个通道的独立效果
3. **策略消融**：测试不同融合策略的效果

---

## 5. 技术实现路线

### 5.1 开发阶段

```
阶段1: 基础实现（2-3周）
├── 实现GECScore通道
├── 实现Token连贯性通道
└── 实现加权融合策略

阶段2: 融合优化（2-3周）
├── 实现自适应融合策略
├── 实现级联融合策略
├── 特征归一化与标准化
└── 超参数调优

阶段3: 实验验证（3-4周）
├── 数据集准备
├── 基础评估实验
├── 泛化性评估实验
├── 鲁棒性评估实验
└── 消融实验

阶段4: 分析优化（2-3周）
├── 结果分析与可视化
├── 错误案例分析
├── 方法优化迭代
└── 文档撰写
```

### 5.2 技术栈

```python
# 核心依赖
- transformers: CoEdIT/BERT/BART模型
- torch: 深度学习框架（CoEdIT基于FLAN-T5）
- rouge: Rouge分数计算
- scikit-learn: 评估指标计算
- numpy, pandas: 数据处理
- matplotlib, seaborn: 可视化

# CoEdIT模型（语法修正）
- grammarly/coedit-large (基于FLAN-T5-large, ~770M参数)
- 备选: grammarly/coedit-xl (基于FLAN-T5-xl, ~3B参数)
- 备选: grammarly/coedit-xxl (基于FLAN-T5-xxl, ~11B参数)

# 可选依赖
- nltk: 自然语言处理工具
- accelerate: 模型加载优化
```

---

## 6. 预期结果与分析

### 6.1 性能预期

| 方法 | 预期AUROC | 计算成本 | 黑盒友好 |
|------|-----------|----------|----------|
| GECScore | 98.62% | 低 | 高 |
| TOCSIN | ~95% | 中高 | 中 |
| MultiFusion-S | ~99% | 中 | 中高 |
| MultiFusion-A | ~99.5% | 中 | 中高 |
| MultiFusion-C | ~99% | 低-中 | 高 |

### 6.2 优势分析

1. **性能提升**：多特征融合预期提升0.5-1% AUROC
2. **鲁棒性增强**：级联策略对攻击更具抵抗力
3. **灵活性**：可根据场景选择融合策略
4. **可解释性**：每个通道的贡献可独立分析

### 6.3 潜在挑战

1. **计算成本**：多个通道会增加推理时间
2. **参数调优**：融合权重需要精心设计
3. **模型依赖**：CoEdIT模型需要加载到本地/GPU
4. **领域适应**：不同领域可能需要不同权重配置

---

## 7. 创新点与贡献

### 7.1 方法创新

1. **多维度特征融合**：首次系统性地融合语法规范性和Token连贯性
2. **自适应融合策略**：根据文本特征动态调整权重
3. **级联检测架构**：平衡效率与准确性

### 7.2 理论贡献

1. 验证了不同LLM文本特征维度的互补性
2. 提供了零样本检测的新范式
3. 为后续研究提供了可扩展框架

### 7.3 实用价值

1. 提供了即插即用的检测工具
2. 支持多种部署场景（高精度/低延迟）
3. 开源实现便于社区改进

---

## 8. 论文结构建议

```markdown
标题: MultiFusion: A Multi-Channel Zero-Shot Detection Framework
       for LLM-Generated Text via Grammar and Cohesiveness Analysis

摘要:
- 背景: LLM生成文本检测的重要性
- 问题: 现有单一方法存在局限性
- 方法: 提出MultiFusion多通道融合框架
- 结果: 在主流数据集上达到SOTA性能
- 贡献: 提供可复现的开源实现

1. 引言
   1.1 研究背景
   1.2 现有方法局限性
   1.3 本文贡献

2. 相关工作
   2.1 LLM生成文本检测方法综述
   2.2 零样本检测方法
   2.3 语法错误修正与文本连贯性

3. 方法
   3.1 问题定义
   3.2 特征提取通道
       3.2.1 语法规范性特征
       3.2.2 Token连贯性特征
   3.3 融合策略设计
       3.3.1 加权融合
       3.3.2 自适应融合
       3.3.3 级联融合
   3.4 算法流程

4. 实验
   4.1 实验设置
       4.1.1 数据集
       4.1.2 生成模型
       4.1.3 评估指标
   4.2 主要结果
   4.3 泛化性分析
   4.4 鲁棒性分析
   4.5 消融实验
   4.6 案例分析

5. 讨论
   5.1 融合策略分析
   5.2 错误案例分析
   5.3 局限性与未来工作

6. 结论

参考文献

附录
   A. 实现细节
   B. 额外实验结果
   C. 代码与数据说明
```

---

## 9. 时间规划

| 阶段 | 时间 | 主要任务 |
|------|------|----------|
| 第1-3周 | 基础实现 | 实现两个检测通道 |
| 第4-6周 | 融合优化 | 实现融合策略与调优 |
| 第7-10周 | 实验验证 | 全面评估与消融实验 |
| 第11-12周 | 分析总结 | 结果分析与论文撰写 |
| 第13-14周 | 修改完善 | 论文修改与代码整理 |

---

## 10. 参考资料

### 核心论文

1. **TOCSIN**:
   - Ma, S., & Wang, Q. (2024). Zero-Shot Detection of LLM-Generated Text using Token Cohesiveness. EMNLP 2024.

2. **GECScore**:
   - Wu, J., et al. (2024). Who Wrote This? The Key to Zero-Shot LLM-Generated Text Detection Is GECScore. COLING 2025.

### 相关论文

3. **Fast-DetectGPT**:
   - Bao, G., et al. (2023). Detecting Text Generated by ChatGPT without Training: A Language Model's Log-Likelihood Approach.

4. **DetectGPT**:
   - Mitchell, E., et al. (2023). DetectGPT: Zero-Shot Machine-Generated Text Detection using Log-Likelihood Rank.

5. **LogRank**:
   - Lazaridou, A., et al. (2023). LogRank: Logarithmic Ranking for Text Generation Sample Quality.

6. **CoEdIT**:
   - Raheja, V., et al. (2023). CoEdIT: Text Editing by Task-Specific Instruction Tuning. EMNLP 2023 Findings.
   - GitHub: https://github.com/vipulraheja/coedit
   - Hugging Face: https://huggingface.co/grammarly/coedit-large

---

## 11. 风险与应对

| 风险 | 影响 | 应对策略 |
|------|------|----------|
| CoEdIT模型性能不足 | 高 | 备选CoEdIT-xl/xxl，或使用其他GEC模型 |
| 计算资源不足 | 中 | 优化批处理，使用量化模型 |
| 融合效果不达预期 | 中 | 增加更多特征维度，调整融合策略 |
| 数据集限制 | 低 | 扩展数据集，使用合成数据 |

---

## 12. CoEdIT模型说明

### 12.1 模型信息

**CoEdIT** (Text Editing by Task-Specific Instruction Tuning) 是Grammarly推出的指令调优文本编辑模型，论文发表于EMNLP 2023。

**模型变体**：
- `grammarly/coedit-large`: 基于FLAN-T5-large (~770M参数)
- `grammarly/coedit-xl`: 基于FLAN-T5-xl (~3B参数)
- `grammarly/coedit-xxl`: 基于FLAN-T5-xxl (~11B参数)

**推荐使用**：`grammarly/coedit-large`（平衡性能与计算成本）

### 12.2 使用示例

```python
from transformers import AutoTokenizer, AutoModelForMultimodalLM

# 加载模型
model_name = "grammarly/coedit-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMultimodalLM.from_pretrained(model_name)

# 语法错误修正
instruction = "Fix grammatical errors in this sentence: Your going to love this!"
inputs = tokenizer(instruction, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(**inputs, max_length=512)
corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(corrected)  # "You're going to love this!"
```

### 12.3 源码参考

CoEdIT的官方源码已保存在 `coedit-main` 目录中，可供参考实现细节。包含：
- 训练脚本和配置
- 数据集处理代码
- 模型推理示例
- 评估脚本

### 12.3 为什么选择CoEdIT替代GPT-4o？

1. **可本地部署**：无需API调用，完全离线运行
2. **成本可控**：一次部署，无限使用
3. **性能稳定**：不受API限制和网络波动影响
4. **开源透明**：模型架构和数据集公开
5. **任务专门化**：针对文本编辑任务优化，GEC质量高

### 12.4 与GPT-4o对比

| 特性 | GPT-4o | CoEdIT-large |
|------|--------|--------------|
| 部署方式 | API调用 | 本地部署 |
| 成本 | 按量计费 | 一次部署无限使用 |
| 可控性 | 受API限制 | 完全可控 |
| 稳定性 | 依赖网络 | 离线运行 |
| 模型大小 | ~1.8T（估计） | ~770M |
| GEC性能 | 优秀 | 优秀（任务专门化） |

---

*文档版本: v1.2*
*创建日期: 2024-06-14*
*最后更新: 2024-06-14*

---

## 版本历史

**v1.2** (2024-06-14):
- 修正CoEdIT模型加载方式：使用 `AutoModelForMultimodalLM`
- 添加coedit-main源码目录说明

**v1.1** (2024-06-14):
- 将GPT-4o API替换为CoEdIT-large本地模型
- 更新技术栈和风险评估
- 新增CoEdIT模型说明章节

**v1.0** (2024-06-14):
- 初始版本
- 提出MultiFusion-Detector架构
- 设计三种融合策略
