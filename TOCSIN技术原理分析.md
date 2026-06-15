# TOCSIN 技术原理分析

## 项目概述

TOCSIN（**T**oken **C**ohe**S**iveness for detect**IN**g LLM-generated text）是一个零样本LLM生成文本检测系统，对应论文《Zero-Shot Detection of LLM-Generated Text using Token Cohesiveness》（EMNLP 2024）。

---

## 核心创新：Token Cohesiveness

### 理论基础

TOCSIN发现了一个关键特征：**LLM生成的文本比人类写作具有更高的token连贯性（token cohesiveness）**。

**核心原理**：
- LLM基于概率分布预测下一个token，生成的文本结构更规律、可预测
- 人类写作更具创造性和随机性，词法组合更多变
- 当随机删除部分token时，LLM文本的语义保持更完整

### Token连贯性计算方法

#### 1. 随机Token删除扰动

```python
def perturb_texts(texts, pct=0.015):
    # 每个文本重复10次
    # 每次删除1.5%的随机token位置
    # 生成多个扰动版本
```

#### 2. 语义差异测量

使用**BART Score**计算原始文本与扰动文本的语义相似度：

```python
# BARTScorer计算语义差异
values = bart_scorer.score(perturbed_texts, source_texts_list, batch_size=10)
mean_values = np.mean(values)

# 转换为连贯性权重
cohesion_weight = math.pow(math.e, -mean_values)
```

**为什么用BART Score？**
- 基于预训练序列到序列模型（BART）
- 通过负对数似然评估文本质量
- 对语义变化敏感

**解释**：
- `mean_values` 越小 → 扰动文本与原始文本语义越接近 → token连贯性越高
- `exp(-mean_values)` 越大 → LLM生成的可能性越高

---

## 双通道检测范式

TOCSIN采用**双通道架构**，将token连贯性作为"即插即用"模块增强现有检测器：

```
┌─────────────────────────────────────────────────────────┐
│                    TOCSIN 检测架构                        │
├─────────────────────────────────────────────────────────┤
│  通道1: 基础检测器      →   base_score                   │
│  ┌───────────────────────────────────────────────────┐  │
│  │ • Fast: (LL - μ̃) / σ                             │  │
│  │ • LRR: LL / LogRank                              │  │
│  │ • Likelihood: LL                                 │  │
│  │ • LogRank: LogRank                               │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  通道2: Token连贯性模块   →   cohesion_weight          │
│  ┌───────────────────────────────────────────────────┐  │
│  │ BART Score(原始文本, 扰动文本)                    │  │
│  │ weight = exp(-mean_BART_score)                    │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  最终得分 = base_score × cohesion_weight               │
└─────────────────────────────────────────────────────────┘
```

### 得分计算公式

```python
# Fast-DetectGPT变体（默认）
output_score = ((LL - μ̃) / σ) × exp(-mean_BART)

# LRR变体
output_score = (LL / LogRank) × exp(-mean_BART)

# Likelihood变体
output_score = LL × exp(mean_BART)

# LogRank变体
output_score = LogRank × exp(mean_BART)

# 独立使用Token连贯性
output_score = -mean_BART
```

其中：
- `LL`: 对数似然（Log Likelihood）
- `μ̃`: 采样对数似然均值
- `σ`: 采样对数似然标准差
- `LogRank`: 对数排名

---

## 技术实现流程

### 1. 文本预处理与扰动

```python
# 对每个文本生成10个扰动版本
perturbed_original_texts = perturb_texts([x for x in data['original'] for _ in range(10)])
perturbed_sampled_texts = perturb_texts([x for x in data['sampled'] for _ in range(10)])
```

### 2. 模型推理

```python
# 获取文本的模型输出
logits_score = scoring_model(**tokenized).logits[:, :-1]

# 计算对数似然
log_likelihood_x = get_likelihood(logits_score, labels)

# 计算对数排名
log_rank_x = get_logrank(logits_score, labels)

# 采样计算统计量（10000次采样）
samples_2 = get_samples(logits_ref, labels)
```

### 3. Token连贯性评分

```python
# 对每个文本的10个扰动版本
# 计算BART Score与原始文本的语义差异
values = bart_scorer.score(perturbed_texts, source_texts_list)
mean_values = np.mean(values)

# 转换为权重
cohesion_weight = math.pow(math.e, -mean_values)
```

### 4. 综合评分

```python
# 结合基础检测器和Token连贯性
output_score = base_score * cohesion_weight
```

---

## 支持的模型和数据集

### 开源LLM

| 模型 | 大小 | 路径格式 |
|------|------|----------|
| GPT-2 XL | 1.5B | `gpt2-xl` |
| GPT-Neo 2.7B | 2.7B | `EleutherAI/gpt-neo-2.7B` |
| GPT-J 6B | 6B | `EleutherAI/gpt-j-6B` |
| GPT-NeoX 20B | 20B | `EleutherAI/gpt-neox-20b` |
| OPT 2.7B | 2.7B | `facebook/opt-2.7b` |

### API模型

- GPT-3.5-turbo
- GPT-4
- Gemini

### 数据集

| 数据集 | 类型 | 描述 |
|--------|------|------|
| XSum | 新闻摘要 | 英国新闻文章摘要 |
| Writing | 创意写作 | 写作提示与故事 |
| SQuAD | 阅读理解 | 维基百科段落问答 |
| PubMed | 生物医学 | 医学文献问答 |

---

## 核心文件说明

| 文件 | 功能描述 |
|------|----------|
| [TOCSIN.py](TOCSIN-main/TOCSIN.py) | 主检测逻辑，核心算法实现 |
| [bart_score.py](TOCSIN-main/bart_score.py) | BERT-based语义评分模块 |
| [model.py](TOCSIN-main/model.py) | 模型加载管理（支持多种LLM） |
| [data_builder.py](TOCSIN-main/data_builder.py) | 数据生成和预处理 |
| [metrics.py](TOCSIN-main/metrics.py) | ROC/PR曲线计算 |

---

## 关键优势

1. **零样本检测**：无需在特定数据集上训练
2. **即插即用**：可增强任何现有零样本检测器
3. **黑盒友好**：只需少量随机删除和语义测量，适合API场景
4. **通用性强**：支持多种基础检测器组合
5. **可解释性**：基于token连贯性的直观原理

---

## 运行方式

### 开源模型实验

```bash
sh Five_models.sh
```

### API模型实验

```bash
sh API-based.sh
```

---

## 环境要求

- Python 3.8
- PyTorch 2.1.0
- NVIDIA A40 GPU (48GB)

---

## 论文信息

**Title**: Zero-Shot Detection of LLM-Generated Text using Token Cohesiveness

**Authors**: Shixuan Ma, Quan Wang

**Venue**: EMNLP 2024

**URL**: https://aclanthology.org/2024.emnlp-main.971/

---

## 引用

```bibtex
@inproceedings{ma-wang-2024-zero,
    title = "Zero-Shot Detection of {LLM}-Generated Text using Token Cohesiveness",
    author = "Ma, Shixuan  and Wang, Quan",
    editor = "Al-Onaizan, Yaser  and Bansal, Mohit  and Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.971/",
    doi = "10.18653/v1/2024.emnlp-main.971",
    pages = "17538--17553"
}
```
