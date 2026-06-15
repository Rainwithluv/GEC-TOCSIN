# GECScore 技术原理分析

## 项目概述

GECScore（**G**rammar **E**rror **C**orrection **Score**）是一个简单但有效的黑盒零样本LLM生成文本检测方法，对应论文《Who Wrote This? The Key to Zero-Shot LLM-Generated Text Detection Is GECScore》（COLING 2025）。

---

## 核心创新：语法错误修正评分

### 理论基础

GECScore基于一个关键观察：**从LLM的视角来看，人类写作的文本通常比LLM生成的文本包含更多的语法错误**。

**核心原理**：
- LLM生成的文本经过语言模型的概率优化，语法规范性更高
- 人类写作虽然正确但存在更多"非标准化"表达，从模型视角看可能被视为"错误"
- 通过测量文本与语法修正后文本的差异，可以区分人类写作和LLM生成文本

### 检测逻辑

```
人类写作 → GEC修正 → 较大变化 → 较低Rouge-2分数
LLM生成 → GEC修正 → 较小变化 → 较高Rouge-2分数
```

**关键假设**：
- 人类写作的文本需要更多"修正"（即使原文本是正确的）
- LLM生成文本已经符合语言模型的概率分布，需要的修正较少

---

## 技术实现流程

### 1. 数据准备

```json
{
    "text": "原始文本内容",
    "label": "human/llm",
    "gec_text": "语法修正后的文本（如不存在则需生成）"
}
```

### 2. 语法错误修正（GEC）

使用GPT-4o或GPT-4o-mini进行语法修正：

```python
def chat_with_gpt4o(prompt, model):
    """使用GPT-4o进行语法修正"""
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.01  # 低温度确保一致性
    )
    return completion.choices[0].message.content

# 语法修正提示词
prompt = f"Correct the grammar errors in the following text: {text}\nCorrected text:"
gec_text = chat_with_gpt4o(prompt, llm_model)
```

### 3. 计算GECScore

使用Rouge-2分数衡量原始文本与修正后文本的相似度：

```python
from rouge import Rouge

rouge = Rouge()
rouge_score = rouge.get_scores(text, gec_text, avg=True)
gecscore = rouge_score['rouge-2']['f']
```

**为什么用Rouge-2？**
- Rouge-2关注bigram（双词）重叠
- 对语法结构变化敏感
- F1分数平衡了精确率和召回率

### 4. 分类判断

```python
# 基于阈值进行分类
if gecscore >= threshold:
    # 高相似度 → LLM生成文本
    prediction = "llm"
else:
    # 低相似度 → 人类写作
    prediction = "human"
```

---

## 完整检测流程

```
┌─────────────────────────────────────────────────────────┐
│                    GECScore 检测流程                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  输入: 待检测文本                                          │
│  ↓                                                       │
│  步骤1: GEC语法修正                                       │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Prompt: "Correct the grammar errors..."          │   │
│  │ Model: GPT-4o/GPT-4o-mini                        │   │
│  │ Output: 修正后文本                                │   │
│  └─────────────────────────────────────────────────┘   │
│  ↓                                                       │
│  步骤2: 计算Rouge-2分数                                  │
│  ┌─────────────────────────────────────────────────┐   │
│  │ rouge.get_scores(original_text, gec_text)       │   │
│  │ GECScore = rouge-2-f                            │   │
│  └─────────────────────────────────────────────────┘   │
│  ↓                                                       │
│  步骤3: 阈值分类                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │ GECScore >= threshold → LLM生成                  │   │
│  │ GECScore < threshold → 人类写作                  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 阈值计算方式

### 方式1：预定义阈值

使用预先确定的最佳阈值：

```bash
python detector.py \
    --test_data_path xsum.GPT-4o.normal.test_data.json \
    --threshold True \
    --threshold_value 0.9243697428995128
```

### 方式2：训练数据计算

从训练数据中计算最优阈值：

```bash
# 同数据集训练测试
python detector.py \
    --train_data_path xsum.GPT-4o.normal.test_data.json \
    --test_data_path xsum.GPT-4o.normal.test_data.json

# 跨数据集泛化
python detector.py \
    --train_data_path xsum.GPT-4o.normal.test_data.json \
    --test_data_path writing.GPT-4o.normal.test_data.json
```

**最优阈值计算**（Youden's J统计量）：

```python
fpr, tpr, thresholds = roc_curve(real_labels, predicted_probs)
optimal_idx = np.argmax(tpr - fpr)  # 最大化 Youden's J
optimal_threshold = thresholds[optimal_idx]
```

---

## 评估指标

计算的性能指标包括：

```python
def get_roc_metrics(real_preds, sample_preds):
    # ROC AUC
    roc_auc = auc(fpr, tpr)

    # 混淆矩阵
    conf_matrix = confusion_matrix(real_labels, predictions)

    # 精确率
    precision = precision_score(real_labels, predictions)

    # 召回率
    recall = recall_score(real_labels, predictions)

    # F1分数
    f1 = f1_score(real_labels, predictions)

    # 准确率
    accuracy = accuracy_score(real_labels, predictions)
```

---

## 数据组织结构

```
data/
├── raw_data/          # 原始数据
├── normal_data/       # 标准测试数据
├── paraphrase_data/    # 改写攻击数据（测试鲁棒性）
├── cross_length/       # 跨长度测试数据
├── writing.raw_data.json
└── xsum.raw_data.json
```

### 数据格式示例

```json
{
    "text": "This is the text content to be detected...",
    "label": "human",  // 或 "llm"
    "gec_text": "This is the grammar-corrected text..."
}
```

---

## 核心优势

1. **简单有效**：仅需GEC + Rouge计算，无需复杂模型
2. **黑盒友好**：不需要访问源LLM，仅用GPT-4o进行GEC
3. **高准确率**：在XSum和WritingPrompts数据集上平均AUROC达98.62%
4. **泛化能力强**：跨数据集、跨模型表现稳定
5. **抗攻击性**：对改写攻击具有鲁棒性

---

## 支持的LLM模型

- GPT-4o
- GPT-4o-mini（默认）

---

## 实验结果摘要

根据论文报告：

| 数据集 | 模型 | AUROC |
|--------|------|-------|
| XSum | GPT-4o | ~98.62% |
| WritingPrompts | GPT-4o | ~98.62% |
| 平均 | - | **98.62%** |

---

## 运行示例

### 1. 使用预定义阈值检测

```bash
# XSum数据集 + GPT-4o生成
python detector/GECScore.py \
    --test_data_path xsum.GPT-4o.normal.test_data.json \
    --threshold True \
    --threshold_value 0.9243697428995128 \
    --llm_model gpt-4o-mini
```

### 2. 训练数据计算阈值

```bash
# 同数据集
python detector/GECScore.py \
    --train_data_path xsum.GPT-4o.normal.test_data.json \
    --test_data_path xsum.GPT-4o.normal.test_data.json \
    --llm_model gpt-4o-mini

# 跨数据集泛化测试
python detector/GECScore.py \
    --train_data_path xsum.GPT-4o.normal.test_data.json \
    --test_data_path writing.GPT-4o.normal.test_data.json \
    --llm_model gpt-4o-mini
```

---

## 环境依赖

```python
# Python包
- openai
- nltk
- rouge
- scikit-learn
- torch
- numpy
- tqdm
```

---

## 核心文件说明

| 文件 | 功能描述 |
|------|----------|
| [detector/GECScore.py](GECScore-main/detector/GECScore.py) | 主检测逻辑，包含GEC、评分和评估 |

---

## 论文信息

**Title**: Who Wrote This? The Key to Zero-Shot LLM-Generated Text Detection Is GECScore

**Authors**: Junchao Wu, Runzhe Zhan, Derek F. Wong, Shu Yang, Xuebo Liu, Lidia S. Chao, Min Zhang

**Venue**: COLING 2025 (The 31st International Conference on Computational Linguistics)

**arXiv**: https://arxiv.org/abs/2405.04286

---

## 引用

```bibtex
@article{wu2024GECScore,
  author       = {Junchao Wu and
                  Runzhe Zhan and
                  Derek F. Wong and
                  Shu Yang and
                  Xuebo Liu and
                  Lidia S. Chao and
                  Min Zhang},
  title        = {Who Wrote This? The Key to Zero-Shot LLM-Generated Text Detection
                  Is GECScore},
  journal      = {CoRR},
  volume       = {abs/2405.04286},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2405.04286},
  doi          = {10.48550/ARXIV.2405.04286},
  eprinttype    = {arXiv},
  eprint       = {2405.04286},
}
```

---

## 与TOCSIN对比

| 特性 | TOCSIN | GECScore |
|------|--------|----------|
| 核心原理 | Token连贯性 | 语法错误修正分数 |
| 检测方法 | 双通道（基础检测器+连贯性模块） | 单一GEC评分 |
| 扰动方式 | 随机token删除 | GEC语法修正 |
| 评分指标 | BART Score | Rouge-2 F1 |
| 模型需求 | 需要加载源LLM | 仅需GPT-4o进行GEC |
| 黑盒友好 | 中等 | 高 |
| 复杂度 | 较高 | 简单 |
