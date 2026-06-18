# MultiFusion-Detector

基于GECScore与TOCSIN融合的LLM生成文本检测方法实现。

## 概述

MultiFusion-Detector是一个多通道零样本检测框架，结合了两种互补的检测方法：

1. **CoEdIT通道**：基于语法规范性（Grammarly CoEdIT模型）
2. **TOCSIN通道**：基于Token连贯性（随机token删除 + BART Score）

## 性能基准

### XSum数据集（50样本测试）

| 指标 | 数值 | 说明 |
|------|------|------|
| **ROC AUC** | 0.9564 | 优秀 |
| **Precision** | 0.9388 | 高精度 |
| **Recall** | 0.9200 | 高召回 |
| **F1 Score** | 0.9293 | 平衡表现 |
| **Accuracy** | 0.9300 | 准确率 |

### Writing数据集（50样本测试）

| 指标 | 数值 | 说明 |
|------|------|------|
| **ROC AUC** | 0.7380 | 中等 |
| **Precision** | 0.9655 | 极高精度 |
| **Recall** | 0.5600 | 召回较低 |
| **F1 Score** | 0.7089 | 平衡一般 |
| **Accuracy** | 0.7700 | 准确率 |

> **注意**：不同数据集的性能差异较大，建议针对新数据集先进行评估以确定合适的阈值。

## 项目结构

```
demo/
├── src/                   # 源代码
│   ├── channels/         # 检测通道
│   │   ├── coedit_channel.py    # CoEdIT通道（语法规范性）
│   │   └── tocsin_channel.py    # TOCSIN通道（Token连贯性）
│   ├── fusion/           # 融合策略
│   │   ├── weighted_fusion.py   # 加权融合（推荐）
│   │   ├── voting_fusion.py     # 投票融合
│   │   └── improved_fusion.py   # 改进融合
│   ├── models/           # 模型加载
│   │   └── model_loader.py      # 统一模型加载器
│   ├── utils/            # 工具函数
│   │   ├── data_loader.py       # 数据加载
│   │   └── metrics.py           # 评估指标
│   ├── detector.py       # 主检测器
│   └── detector_mini.py  # 快速测试检测器
├── venv/                 # 虚拟环境
├── run_optimal_test.bat # 最优策略测试脚本
├── run_comparison.bat   # 策略对比脚本
└── README.md            # 本文件
```

## 技术原理

### 核心思想

LLM生成的文本与人类写作在两个关键维度上存在差异：
1. **语法规范性**：LLM文本语法更规范，需要更少修正
2. **Token连贯性**：LLM文本语义连贯性更强，删除token后语义变化更小

### 检测通道

#### 1. CoEdIT通道（语法规范性）

**原理**：LLM生成的文本从语法角度更"规范"，需要更少修正

**流程**：
1. 使用Grammarly的CoEdIT模型进行语法修正
2. 计算原文与修正文的Rouge-2分数
3. 分数越高 → 越可能是LLM生成（因为需要修正越少）

**关键参数**：
- 模型：`grammarly/coedit-large`
- 指标：Rouge-2（衡量重叠度）

#### 2. TOCSIN通道（Token连贯性）

**原理**：LLM生成的文本删除token后语义变化更小（连贯性更强）

**流程**：
1. 随机删除1.5%的token（重复10次取平均）
2. 使用BART Score计算扰动前后语义差异
3. 连贯性分数越高 → 越可能是LLM生成

**关键参数**：
- 删除比例：1.5%
- 重复次数：10次
- 模型：`facebook/bart-base`

**重要**：TOCSIN原始分数含义是"连贯性越高越像人类"，为了统一指标方向（分数越高越像LLM），代码中对TOCSIN分数进行了反转处理。

### 融合策略

#### 最优加权融合（推荐）

经过测试验证的最优配置：

```python
final_score = 0.2 × coedit_score + 0.8 × tocsin_score
```

**为什么这个权重？**
- TOCSIN在LLM检测中区分度更高
- CoEdIT提供互补信息但权重较低
- 在XSum数据集上ROC AUC达到0.9375

**测试对比**：

| 策略 | ROC AUC | Precision | Recall |
|------|---------|-----------|--------|
| Improved Fusion | 0.8850 | 0.8333 | 1.0000 |
| Voting Fusion | 0.7475 | 0.8571 | 0.6000 |
| **Weighted (0.2/0.8)** | **0.9375** | **0.9091** | **1.0000** |

## 快速开始

### 1. 环境准备

```bash
cd demo
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 完整评估

```bash
# 使用真实数据集评估，使用weighted融合策略（目前最优）
python -m demo.src.detector --mode evaluate --dataset xsum --model gpt-4 --fusion weighted --n-samples 50
```


## 使用方法

### 命令行评估

```bash

# 使用不同融合策略，可以使用improved等策略
python -m demo.src.detector --mode evaluate --dataset xsum --model gpt-4 --fusion improved --weight-method optimized

# 自定义权重
python -m demo.src.detector --mode evaluate --dataset xsum --model gpt-4 --coedit-weight 0.3 --tocsin-weight 0.7
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | evaluate | 运行模式：evaluate（评估）/ detect（检测） |
| `--dataset` | xsum | 数据集名称：xsum / writing |
| `--model` | gpt-4 | LLM模型名称 |
| `--fusion` | weighted | 融合策略：weighted / improved / voting |
| `--coedit-weight` | 0.2 | CoEdIT权重（已优化） |
| `--tocsin-weight` | 0.8 | TOCSIN权重（已优化） |
| `--n-samples` | 50 | 测试样本数量 |
| `--threshold` | auto | 判断阈值（默认自动计算最优值） |



## 评估指标说明

| 指标 | 含义 | 重要性 |
|------|------|--------|
| **ROC AUC** | 分类能力综合评估 | ★★★★★ 最重要 |
| **Precision** | 预测为LLM中真正是LLM的比例 | 高精度=少误报 |
| **Recall** | 真正LLM被正确识别的比例 | 高召回=少漏检 |
| **F1 Score** | Precision和Recall的调和平均 | 平衡指标 |
| **Accuracy** | 整体正确率 | 样本不平衡时不可靠 |

### 混淆矩阵解读

```
              预测
         Human    LLM
实际 Human [ TN ]  [ FP ]  ← TN: 真阴性，FP: 假阳性（误报）
      LLM  [ FN ]  [ TP ]  ← FN: 假阴性（漏检），TP: 真阳性
```

## 依赖项

```
torch>=2.0.0
transformers>=4.30.0
rouge-score>=0.0.4
scikit-learn>=1.3.0
tqdm
numpy
```

## 技术论文引用

如果使用本实现，请引用以下论文：

```bibtex
@inproceedings{ma-wang-2024-zero,
    title = "Zero-Shot Detection of {LLM}-Generated Text using Token Cohesiveness",
    author = "Ma, Shixuan and Wang, Quan",
    booktitle = "Proceedings of EMNLP 2024",
    year = "2024"
}

@inproceedings{wu-2024-gecscore,
    title = "Who Wrote This? The Key to Zero-Shot LLM-Generated Text Detection Is GECScore",
    author = "Wu, Junchao and others",
    booktitle = "Proceedings of COLING 2025",
    year = "2024"
}
```

## 更新日志

### 2024-06-16
- 更新为最优加权融合策略weighted（CoEdIT: 0.2, TOCSIN: 0.8）
- 添加XSum和Writing数据集性能基准
- 完善技术原理说明
- 添加数据集差异分析和阈值调整建议

---

**许可证**：MIT License
