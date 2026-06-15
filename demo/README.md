# MultiFusion-Detector

基于GECScore与TOCSIN融合的LLM生成文本检测方法实现。

## 概述

MultiFusion-Detector是一个多通道零样本检测框架，结合了两种互补的检测方法：

1. **CoEdIT通道**：基于语法规范性（Grammarly CoEdIT模型）
2. **TOCSIN通道**：基于Token连贯性（随机token删除 + BART Score）

## 项目结构

```
demo/
├── config/                 # 配置文件
│   └── config.yaml        # 主配置文件
├── data/                  # 数据集目录
│   └── README.md          # 数据说明
├── src/                   # 源代码
│   ├── channels/         # 检测通道
│   │   ├── coedit_channel.py    # CoEdIT通道
│   │   ├── tocsin_channel.py    # TOCSIN通道
│   │   └── bart_scorer.py       # BERT评分器
│   ├── fusion/           # 融合策略
│   │   ├── weighted_fusion.py   # 加权融合
│   │   ├── adaptive_fusion.py   # 自适应融合
│   │   └── cascade_fusion.py    # 级联融合
│   ├── models/           # 模型加载
│   │   └── model_loader.py      # 统一模型加载器
│   ├── utils/            # 工具函数
│   │   ├── data_loader.py       # 数据加载
│   │   └── metrics.py           # 评估指标
│   └── detector.py       # 主检测器
├── tests/                # 测试代码
│   └── test_detector.py  # 检测器测试
├── requirements.txt      # Python依赖
└── README.md            # 本文件
```

## 快速开始

### 1. 环境安装

```bash
cd demo
pip install -r requirements.txt
```

### 2. 运行测试

```bash
python tests/test_detector.py
```

### 3. 运行评估

```bash
# 评估XSum数据集（GPT-4）
python src/detector.py --mode evaluate --dataset xsum --model gpt-4 --output results/xsum_gpt4.json

# 评估Writing数据集（GPT-3.5）
python src/detector.py --mode evaluate --dataset writing --model gpt-3.5-turbo --output results/writing_gpt35.json
```

### 4. 选择融合策略

```bash
# 加权融合
python src/detector.py --fusion weighted

# 自适应融合
python src/detector.py --fusion adaptive

# 级联融合
python src/detector.py --fusion cascade
```

## 检测通道详解

### CoEdIT通道

**原理**：LLM生成的文本从语法角度更"规范"，需要更少修正

**流程**：
1. 使用CoEdIT模型进行语法修正
2. 计算原文与修正文的Rouge-2分数
3. 分数越高 → 越可能是LLM生成

**使用**：
```python
from src.channels.coedit_channel import CoEdITChannel

channel = CoEdITChannel(model_name="grammarly/coedit-large")
score = channel.score_text("Your text here")
features = channel.extract_features("Your text here")
```

### TOCSIN通道

**原理**：LLM生成的文本删除token后语义变化更小（连贯性更强）

**流程**：
1. 随机删除1.5%的token（重复10次）
2. 使用BART Score计算语义差异
3. 连贯性分数越高 → 越可能是LLM生成

**使用**：
```python
from src.channels.tocsin_channel import TOCSINChannel

channel = TOCSINChannel(bart_model="facebook/bart-base")
score = channel.score_text("Your text here")
features = channel.extract_features("Your text here")
```

## 融合策略

### 1. 加权融合（Weighted Fusion）

固定权重线性组合：

```python
final_score = w_coedit × coedit_score + w_tocsin × tocsin_score
```

**适用场景**：简单快速部署

### 2. 自适应融合（Adaptive Fusion）

根据文本特征动态调整权重：

```python
# 长文本更依赖语法，短文本更依赖连贯性
if text_length > 200:
    w_coedit = 0.6, w_tocsin = 0.4
else:
    w_coedit = 0.4, w_tocsin = 0.6
```

**适用场景**：追求高精度

### 3. 级联融合（Cascade Fusion）

两阶段检测：

1. **第一阶段**：CoEdIT快速筛选
   - score > 0.95 → LLM
   - score < 0.85 → Human

2. **第二阶段**：模糊样本用TOCSIN细筛

**适用场景**：平衡效率与准确性

## 评估指标

- **AUROC**：ROC曲线下面积
- **准确率**：正确分类的比例
- **精确率**：预测为LLM中真正是LLM的比例
- **召回率**：真正LLM被正确识别的比例
- **F1分数**：精确率和召回率的调和平均

## 配置说明

主配置文件：`config/config.yaml`

```yaml
# 模型设置
models:
  coedit:
    name: "grammarly/coedit-large"
  bart:
    name: "facebook/bart-base"

# 融合策略
fusion:
  strategy: "weighted"  # weighted, adaptive, cascade

# TOCSIN参数
tocsin:
  deletion_pct: 0.015  # token删除比例
  n_samples: 10        # 扰动样本数
```

## 数据集

支持的数据集格式：

1. **TOCSIN格式**：`{"original": [...], "sampled": [...]}`
2. **GECSore格式**：`{"id": "...", "text": "...", "label": "..."}`

数据加载器会自动检测格式并正确加载。

## 依赖项

- `torch>=2.0.0` - PyTorch深度学习框架
- `transformers>=4.30.0` - Hugging Face Transformers
- `rouge-score>=0.0.4` - Rouge评分
- `scikit-learn>=1.3.0` - 评估指标

## 示例输出

```json
{
  "roc_auc": 0.9856,
  "threshold": 0.5123,
  "confusion_matrix": [[48, 2], [3, 47]],
  "precision": 0.9592,
  "recall": 0.9400,
  "f1": 0.9495,
  "accuracy": 0.9500
}
```

## 故障排除

### CUDA内存不足

```bash
# 使用CPU运行
python src/detector.py --device cpu
```

### 模型下载慢

```bash
# 设置Hugging Face镜像
export HF_ENDPOINT=https://hf-mirror.com
```

## 引用

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

## 许可证

MIT License
