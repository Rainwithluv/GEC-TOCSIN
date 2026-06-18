"""
评估训练好的Attention模型
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# PyTorch
import torch
from torch.utils.data import DataLoader

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from fusion.dynamic_attention_fusion import CrossBranchAttention
    from channels.coedit_channel import CoEdITChannel
    from channels.tocsin_channel import TOCSINChannel
    from utils.metrics import get_roc_metrics, get_metrics_with_threshold
except ImportError:
    from demo.src.fusion.dynamic_attention_fusion import CrossBranchAttention
    from demo.src.channels.coedit_channel import CoEdITChannel
    from demo.src.channels.tocsin_channel import TOCSINChannel
    from demo.src.utils.metrics import get_roc_metrics, get_metrics_with_threshold


# 导入训练脚本中的数据集类
from train_attention_simple import SimpleFusionDataset


def load_model(model_path, device='cpu'):
    """加载训练好的模型"""
    print(f"加载模型: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # 创建模型
    model = CrossBranchAttention(input_dim=2, hidden_dim=16, num_heads=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"  ✓ 模型加载完成")
    if 'epoch' in checkpoint:
        print(f"  训练轮数: {checkpoint['epoch']}")
    if 'val_acc' in checkpoint:
        print(f"  验证准确率: {checkpoint['val_acc']:.4f}")

    return model


def evaluate_model(model, dataloader, device='cpu'):
    """评估模型性能"""
    model.eval()

    all_scores = []
    all_labels = []
    all_weights = []

    with torch.no_grad():
        for scores, labels in dataloader:
            scores = scores.to(device)
            labels = labels.to(device)

            # 获取融合权重
            fusion_weights, _ = model(scores)

            # 应用权重融合
            fused_scores = fusion_weights[:, 0] * scores[:, 0] + fusion_weights[:, 1] * scores[:, 1]

            all_scores.extend(fused_scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_weights.extend(fusion_weights.cpu().numpy())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_weights = np.array(all_weights)

    # 分离人类和LLM的分数
    human_scores = all_scores[all_labels == 0]
    llm_scores = all_scores[all_labels == 1]

    # 计算ROC AUC和最优阈值
    roc_auc, opt_threshold, _, _, _, _, _ = get_roc_metrics(
        human_scores.tolist(), llm_scores.tolist()
    )

    # 使用最优阈值计算其他指标
    _, _, conf_matrix, precision, recall, f1, accuracy = get_metrics_with_threshold(
        human_scores.tolist(), llm_scores.tolist(), opt_threshold
    )

    return {
        'roc_auc': roc_auc,
        'threshold': opt_threshold,
        'confusion_matrix': conf_matrix,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'human_scores': human_scores,
        'llm_scores': llm_scores,
        'weights': all_weights
    }


def analyze_weights(weights):
    """分析权重分布"""
    coedit_weights = weights[:, 0]
    tocsin_weights = weights[:, 1]

    print("\n权重分布分析:")
    print(f"  CoEdIT权重:")
    print(f"    平均值: {coedit_weights.mean():.4f}")
    print(f"    标准差: {coedit_weights.std():.4f}")
    print(f"    最小值: {coedit_weights.min():.4f}")
    print(f"    最大值: {coedit_weights.max():.4f}")
    print(f"    中位数: {np.median(coedit_weights):.4f}")

    print(f"  TOCSIN权重:")
    print(f"    平均值: {tocsin_weights.mean():.4f}")
    print(f"    标准差: {tocsin_weights.std():.4f}")
    print(f"    最小值: {tocsin_weights.min():.4f}")
    print(f"    最大值: {tocsin_weights.max():.4f}")
    print(f"    中位数: {np.median(tocsin_weights):.4f}")

    # 权重分布范围
    coedit_range = coedit_weights.max() - coedit_weights.min()
    tocsin_range = tocsin_weights.max() - tocsin_weights.min()

    print(f"\n  动态范围:")
    print(f"    CoEdIT: {coedit_range:.4f}")
    print(f"    TOCSIN: {tocsin_range:.4f}")

    if coedit_range > 0.5 or tocsin_range > 0.5:
        print(f"    → 权重动态调整幅度较大")
    else:
        print(f"    → 权重相对稳定")


def print_results(results):
    """打印评估结果"""
    print("\n" + "=" * 80)
    print("评估结果")
    print("=" * 80)

    print(f"\n主要指标:")
    print(f"  ROC AUC:      {results['roc_auc']:.4f}")
    print(f"  阈值:         {results['threshold']:.4f}")
    print(f"  准确率:       {results['accuracy']:.4f}")
    print(f"  精确率:       {results['precision']:.4f}")
    print(f"  召回率:       {results['recall']:.4f}")
    print(f"  F1分数:       {results['f1']:.4f}")

    print(f"\n混淆矩阵:")
    print(f"  {results['confusion_matrix']}")
    print(f"  [[TN, FP],")
    print(f"   [FN, TP]]")

    # 分析分数分布
    human_scores = results['human_scores']
    llm_scores = results['llm_scores']

    print(f"\n分数分布:")
    print(f"  人类文本:")
    print(f"    平均值: {human_scores.mean():.4f}")
    print(f"    标准差: {human_scores.std():.4f}")
    print(f"    最小值: {human_scores.min():.4f}")
    print(f"    最大值: {human_scores.max():.4f}")

    print(f"  LLM文本:")
    print(f"    平均值: {llm_scores.mean():.4f}")
    print(f"    标准差: {llm_scores.std():.4f}")
    print(f"    最小值: {llm_scores.min():.4f}")
    print(f"    最大值: {llm_scores.max():.4f}")

    # 分离度
    separation = abs(llm_scores.mean() - human_scores.mean())
    print(f"\n  分离度: {separation:.4f}")

    if separation > 0.5:
        print(f"    → 分离度优秀")
    elif separation > 0.3:
        print(f"    → 分离度良好")
    elif separation > 0.1:
        print(f"    → 分离度可接受")
    else:
        print(f"    → 分离度较低")

    # 分析权重
    analyze_weights(results['weights'])


def compare_with_fixed_weights(data_path, model, device='cpu'):
    """与固定权重方法对比"""
    print("\n" + "=" * 80)
    print("与固定权重方法对比")
    print("=" * 80)

    # 加载数据
    with open(data_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    # 创建数据集
    coedit = CoEdITChannel()
    tocsin = TOCSINChannel()

    dataset = SimpleFusionDataset(samples, coedit_channel=coedit, tocsin_channel=tocsin)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 获取所有分数和标签
    all_coedit = []
    all_tocsin = []
    all_labels = []

    for scores, labels in dataloader:
        all_coedit.extend(scores[:, 0].numpy())
        all_tocsin.extend(scores[:, 1].numpy())
        all_labels.extend(labels.numpy())

    all_coedit = np.array(all_coedit)
    all_tocsin = np.array(all_tocsin)
    all_labels = np.array(all_labels)

    # 测试不同固定权重配置
    weight_configs = [
        (0.2, 0.8, "当前最优固定权重"),
        (0.3, 0.7, "较高TOCSIN权重"),
        (0.5, 0.5, "等权重"),
    ]

    print(f"\n{'配置':<20} {'ROC AUC':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 70)

    results_summary = []

    for coedit_w, tocsin_w, desc in weight_configs:
        # 计算融合分数
        fused = coedit_w * all_coedit + tocsin_w * all_tocsin

        # 分离人类和LLM
        human_fused = fused[all_labels == 0]
        llm_fused = fused[all_labels == 1]

        # 计算指标
        roc_auc, _, _, _, _, _, _ = get_roc_metrics(human_fused.tolist(), llm_fused.tolist())
        _, _, _, precision, recall, f1, _ = get_metrics_with_threshold(
            human_fused.tolist(), llm_fused.tolist(), 0.5
        )

        print(f"{desc:<20} {roc_auc:>10.4f} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")
        results_summary.append(('Fixed', desc, roc_auc, precision, recall, f1))

    # 添加Attention模型结果
    att_results = evaluate_model(model, dataloader, device)
    print(f"{'Attention Model':<20} {att_results['roc_auc']:>10.4f} {att_results['precision']:>10.4f} {att_results['recall']:>10.4f} {att_results['f1']:>10.4f}")
    results_summary.append(('Attention', 'Neural Network', att_results['roc_auc'], att_results['precision'], att_results['recall'], att_results['f1']))

    # 找出最佳配置
    best_roc = max(results_summary, key=lambda x: x[2])
    print(f"\n→ 最佳ROC AUC: {best_roc[0]} - {best_roc[1]} ({best_roc[2]:.4f})")

    best_f1 = max(results_summary, key=lambda x: x[5])
    print(f"→ 最佳F1分数: {best_f1[0]} - {best_f1[1]} ({best_f1[5]:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description='评估训练好的Attention模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 评估模型
  python evaluate_attention.py --model models/attention_model.pth --data data/xsum.GPT-4o.normal.test_data.json

  # 评估并对比固定权重
  python evaluate_attention.py --model models/attention_model.pth --data data/xsum.GPT-4o.normal.test_data.json --compare

  # 使用不同批次大小
  python evaluate_attention.py --model models/attention_model.pth --data data/writing.GPT-4o.normal.test_data.json --batch-size 64
        """
    )

    parser.add_argument('--model', type=str, required=True,
                        help='训练好的模型路径')
    parser.add_argument('--data', type=str, required=True,
                        help='评估数据路径')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--device', type=str, default=None,
                        help='设备 (cpu/cuda)')
    parser.add_argument('--compare', action='store_true',
                        help='与固定权重方法对比')
    parser.add_argument('--output', type=str, default=None,
                        help='保存评估结果到JSON文件')

    args = parser.parse_args()

    # 选择设备
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("=" * 80)
    print("Attention 模型评估")
    print("=" * 80)

    # 加载模型
    model = load_model(args.model, device)

    # 加载数据
    print(f"\n加载数据: {args.data}")
    with open(args.data, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    print(f"  样本数: {len(samples)}")

    # 统计标签
    human_count = sum(1 for s in samples if s['label'] == 'human')
    llm_count = sum(1 for s in samples if s['label'] == 'llm')
    print(f"  人类样本: {human_count}")
    print(f"  LLM样本: {llm_count}")

    # 创建数据集
    print("\n初始化通道...")
    coedit = CoEdITChannel()
    tocsin = TOCSINChannel()

    dataset = SimpleFusionDataset(
        samples,
        coedit_channel=coedit,
        tocsin_channel=tocsin,
        cache_scores=True
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 评估模型
    print("\n开始评估...")
    results = evaluate_model(model, dataloader, device)

    # 打印结果
    print_results(results)

    # 与固定权重对比
    if args.compare:
        compare_with_fixed_weights(args.data, model, device)

    # 保存结果
    if args.output:
        output_results = {
            'model': args.model,
            'data': args.data,
            'metrics': {
                'roc_auc': float(results['roc_auc']),
                'threshold': float(results['threshold']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1': float(results['f1']),
                'accuracy': float(results['accuracy']),
                'confusion_matrix': results['confusion_matrix']
            },
            'weights_stats': {
                'coedit_mean': float(results['weights'][:, 0].mean()),
                'coedit_std': float(results['weights'][:, 0].std()),
                'tocsin_mean': float(results['weights'][:, 1].mean()),
                'tocsin_std': float(results['weights'][:, 1].std())
            }
        }

        with open(args.output, 'w') as f:
            json.dump(output_results, f, indent=2)

        print(f"\n结果已保存到: {args.output}")

    print("\n" + "=" * 80)
    print("评估完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
