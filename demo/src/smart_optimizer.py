"""
智能优化器：使用贝叶斯优化快速找到最佳权重
"""

import numpy as np
from typing import Dict, List, Optional
import json

try:
    from channels.coedit_channel import CoEdITChannel
    from channels.tocsin_channel import TOCSINChannel
    from fusion.weighted_fusion import WeightedFusion
    from utils.metrics import get_roc_metrics
    from utils.data_loader import DataLoader
except ImportError:
    from demo.src.channels.coedit_channel import CoEdITChannel
    from demo.src.channels.tocsin_channel import TOCSINChannel
    from demo.src.fusion.weighted_fusion import WeightedFusion
    from demo.src.utils.metrics import get_roc_metrics
    from demo.src.utils.data_loader import DataLoader


class SmartOptimizer:
    """
    智能优化器：只评分一次，然后快速测试权重组合
    """

    def __init__(self,
                 coedit_model: str = "grammarly/coedit-large",
                 bart_model: str = "facebook/bart-base",
                 device: Optional[str] = None):
        """Initialize smart optimizer."""
        print("=== Smart Optimizer ===")
        print("This optimizer scores texts ONCE, then tests weight combinations quickly.\n")

        self.device = device

        print("Loading models (one-time operation)...")
        self.coedit_channel = CoEdITChannel(model_name=coedit_model, device=device)
        self.tocsin_channel = TOCSINChannel(bart_model=bart_model, device=device)
        print("Models loaded!\n")

    def score_texts(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Score texts using both channels."""
        coedit_scores = np.array(self.coedit_channel.score_texts(texts, show_progress=False))
        tocsin_scores = np.array(self.tocsin_channel.score_texts(texts, show_progress=False))
        return {
            'coedit': coedit_scores,
            'tocsin': tocsin_scores
        }

    def optimize_weights(self,
                        human_scores: Dict[str, np.ndarray],
                        llm_scores: Dict[str, np.ndarray],
                        n_steps: int = 21) -> Dict:
        """
        快速优化权重（已评分的数据）。

        Args:
            human_scores: 已评分的人类文本数据
            llm_scores: 已评分的LLM文本数据
            n_steps: 权重搜索步数

        Returns:
            优化结果
        """
        print(f"Testing {n_steps} weight combinations...")

        best_roc_auc = 0
        best_weights = {'coedit': 0.5, 'tocsin': 0.5}
        results = []

        # 只需要测试coedit_weight从0到1，tocsin_weight自动为1-coedit_weight
        for coedit_w in np.linspace(0, 1, n_steps):
            tocsin_w = 1 - coedit_w

            # 创建融合器
            fusion = WeightedFusion(weights={'coedit': coedit_w, 'tocsin': tocsin_w})

            # 融合分数
            human_fused, llm_fused = fusion.normalize_and_fuse(human_scores, llm_scores)

            # 计算ROC AUC
            roc_auc, _, _, _, _, _, _ = get_roc_metrics(
                human_fused.tolist(), llm_fused.tolist()
            )

            results.append({
                'coedit_weight': float(coedit_w),
                'tocsin_weight': float(tocsin_w),
                'roc_auc': float(roc_auc)
            })

            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_weights = {'coedit': float(coedit_w), 'tocsin': float(tocsin_w)}
                print(f"  ✓ New best: CoEdIT={coedit_w:.3f}, ROC AUC={roc_auc:.4f}")

        return {
            'best_weights': best_weights,
            'best_roc_auc': best_roc_auc,
            'all_results': results
        }

    def optimize_on_dataset(self,
                           dataset: str = 'xsum',
                           model: str = 'gpt-4',
                           n_samples: int = 50,
                           n_steps: int = 21) -> Dict:
        """
        在数据集上优化权重。

        Args:
            dataset: 数据集名称
            model: 模型名称
            n_samples: 样本数量
            n_steps: 权重搜索步数

        Returns:
            优化结果
        """
        print(f"Loading dataset: {dataset}, model: {model}")

        loader = DataLoader(base_dir="../")
        data = loader.load_combined_data(dataset, model)

        n = min(n_samples, len(data['original']), len(data['sampled']))
        human_texts = data['original'][:n]
        llm_texts = data['sampled'][:n]

        print(f"Using {n} human texts and {n} LLM texts\n")

        # 只评分一次！
        print("Scoring human texts (one-time)...")
        human_scores = self.score_texts(human_texts)

        print("Scoring LLM texts (one-time)...")
        llm_scores = self.score_texts(llm_texts)

        print(f"\nScores computed!")
        print(f"Human: CoEdIT={human_scores['coedit'].mean():.4f}, TOCSIN={human_scores['tocsin'].mean():.4f}")
        print(f"LLM:   CoEdIT={llm_scores['coedit'].mean():.4f}, TOCSIN={llm_scores['tocsin'].mean():.4f}")
        print(f"\nCoEdIT difference: {llm_scores['coedit'].mean() - human_scores['coedit'].mean():.4f}")
        print(f"TOCSIN difference: {llm_scores['tocsin'].mean() - human_scores['tocsin'].mean():.4f}\n")

        # 快速测试权重组合
        return self.optimize_weights(human_scores, llm_scores, n_steps)


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='Smart weight optimizer - scores once, tests many weights')
    parser.add_argument('--dataset', type=str, default='xsum', help='Dataset name')
    parser.add_argument('--model', type=str, default='gpt-4', help='Model name')
    parser.add_argument('--n-samples', type=int, default=50, help='Samples to use')
    parser.add_argument('--n-steps', type=int, default=21, help='Weight search steps (default: 21 for 0.0, 0.05, ..., 1.0)')
    parser.add_argument('--output', type=str, default=None, help='Output file')
    parser.add_argument('--device', type=str, default=None, help='Device')

    args = parser.parse_args()

    # 初始化优化器
    optimizer = SmartOptimizer(device=args.device)

    # 运行优化
    results = optimizer.optimize_on_dataset(
        dataset=args.dataset,
        model=args.model,
        n_samples=args.n_samples,
        n_steps=args.n_steps
    )

    # 打印结果
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"\n🎯 Best weights:")
    print(f"   CoEdIT: {results['best_weights']['coedit']:.4f}")
    print(f"   TOCSIN: {results['best_weights']['tocsin']:.4f}")
    print(f"\n📈 Best ROC AUC: {results['best_roc_auc']:.4f}")
    print(f"\n💡 To use these weights:")
    print(f"   python -m demo.src.detector --mode evaluate --dataset {args.dataset} --model {args.model}")
    print(f"       --coedit-weight {results['best_weights']['coedit']:.4f} --tocsin-weight {results['best_weights']['tocsin']:.4f}")

    # 保存结果
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\n✅ Results saved to {args.output}")

    return results


if __name__ == '__main__':
    main()
