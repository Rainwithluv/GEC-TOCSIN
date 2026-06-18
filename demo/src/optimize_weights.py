"""
权重优化器：寻找最佳的融合权重和参数
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import ParameterGrid
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


class WeightOptimizer:
    """
    融合权重优化器
    通过网格搜索寻找最佳权重组合
    """

    def __init__(self,
                 coedit_model: str = "grammarly/coedit-large",
                 bart_model: str = "facebook/bart-base",
                 device: Optional[str] = None):
        """
        Initialize weight optimizer.

        Args:
            coedit_model: CoEdIT model name
            bart_model: BART model name
            device: Device to use
        """
        print("Initializing Weight Optimizer...")

        # Initialize channels
        print("Loading CoEdIT model...")
        self.coedit_channel = CoEdITChannel(model_name=coedit_model, device=device)

        print("Loading TOCSIN model...")
        self.tocsin_channel = TOCSINChannel(bart_model=bart_model, device=device)

        print("Optimizer initialized!\n")

    def score_texts(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Score texts using both channels. TOCSIN scores are LLM-oriented (high = LLM)."""
        coedit_scores = np.array(self.coedit_channel.score_texts(texts, show_progress=False))
        tocsin_scores = np.array(self.tocsin_channel.score_texts(texts, show_progress=False, for_llm=True))
        return {
            'coedit': coedit_scores,
            'tocsin': tocsin_scores
        }

    def evaluate_weights(self,
                        human_scores: Dict[str, np.ndarray],
                        llm_scores: Dict[str, np.ndarray],
                        coedit_weight: float,
                        tocsin_weight: float) -> float:
        """
        Evaluate specific weight combination.

        Args:
            human_scores: Human text scores
            llm_scores: LLM text scores
            coedit_weight: Weight for CoEdIT channel
            tocsin_weight: Weight for TOCSIN channel

        Returns:
            ROC AUC score
        """
        weights = {'coedit': coedit_weight, 'tocsin': tocsin_weight}
        fusion = WeightedFusion(weights=weights)

        # Normalize and fuse
        human_fused, llm_fused = fusion.normalize_and_fuse(human_scores, llm_scores)

        # Calculate ROC AUC
        roc_auc, _, _, _, _, _, _ = get_roc_metrics(
            human_fused.tolist(), llm_fused.tolist()
        )

        return roc_auc

    def grid_search(self,
                   human_texts: List[str],
                   llm_texts: List[str],
                   n_steps: int = 11) -> Dict:
        """
        Grid search for optimal weights.

        Args:
            human_texts: Human text samples
            llm_texts: LLM text samples
            n_steps: Number of steps for each weight (default: 11 for 0.0, 0.1, ..., 1.0)

        Returns:
            Dictionary with optimization results
        """
        print(f"Grid search with {n_steps} steps per weight...")
        print(f"Testing {len(human_texts)} human texts and {len(llm_texts)} LLM texts\n")

        # Score texts
        print("Scoring human texts...")
        human_scores = self.score_texts(human_texts)

        print("Scoring LLM texts...")
        llm_scores = self.score_texts(llm_texts)

        print(f"\nHuman scores: CoEdIT={human_scores['coedit'].mean():.4f}, TOCSIN={human_scores['tocsin'].mean():.4f}")
        print(f"LLM scores:   CoEdIT={llm_scores['coedit'].mean():.4f}, TOCSIN={llm_scores['tocsin'].mean():.4f}\n")

        # Generate weight grid
        weight_values = np.linspace(0, 1, n_steps)
        best_roc_auc = 0
        best_weights = {'coedit': 0.5, 'tocsin': 0.5}
        results = []

        print("Testing weight combinations...")
        for coedit_w in weight_values:
            for tocsin_w in weight_values:
                # Normalize weights
                total = coedit_w + tocsin_w
                if total == 0:
                    continue

                coedit_w_norm = coedit_w / total
                tocsin_w_norm = tocsin_w / total

                # Evaluate
                try:
                    roc_auc = self.evaluate_weights(
                        human_scores, llm_scores,
                        coedit_w_norm, tocsin_w_norm
                    )

                    result = {
                        'coedit_weight': float(coedit_w_norm),
                        'tocsin_weight': float(tocsin_w_norm),
                        'roc_auc': float(roc_auc)
                    }
                    results.append(result)

                    if roc_auc > best_roc_auc:
                        best_roc_auc = roc_auc
                        best_weights = {
                            'coedit': float(coedit_w_norm),
                            'tocsin': float(tocsin_w_norm)
                        }
                        print(f"  New best: CoEdIT={coedit_w_norm:.2f}, TOCSIN={tocsin_w_norm:.2f}, ROC AUC={roc_auc:.4f}")

                except Exception as e:
                    print(f"  Error with weights {coedit_w_norm:.2f}/{tocsin_w_norm:.2f}: {e}")

        print(f"\n{'='*60}")
        print(f"BEST WEIGHTS FOUND:")
        print(f"  CoEdIT: {best_weights['coedit']:.4f}")
        print(f"  TOCSIN: {best_weights['tocsin']:.4f}")
        print(f"  ROC AUC: {best_roc_auc:.4f}")
        print(f"{'='*60}\n")

        return {
            'best_weights': best_weights,
            'best_roc_auc': best_roc_auc,
            'all_results': results
        }

    def optimize_on_dataset(self,
                           dataset: str = 'xsum',
                           model: str = 'gpt-4',
                           n_samples: int = 50,
                           n_steps: int = 11) -> Dict:
        """
        Optimize weights on a specific dataset.

        Args:
            dataset: Dataset name
            model: Model name
            n_samples: Number of samples to use
            n_steps: Grid search steps

        Returns:
            Optimization results
        """
        print(f"Loading dataset: {dataset}, model: {model}")

        loader = DataLoader(base_dir="../")
        data = loader.load_combined_data(dataset, model)

        n = min(n_samples, len(data['original']), len(data['sampled']))
        human_texts = data['original'][:n]
        llm_texts = data['sampled'][:n]

        print(f"Using {n} samples for optimization\n")

        return self.grid_search(human_texts, llm_texts, n_steps)


def main():
    """Main entry point for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Optimize fusion weights')
    parser.add_argument('--dataset', type=str, default='xsum',
                       help='Dataset to optimize on')
    parser.add_argument('--model', type=str, default='gpt-4',
                       help='Model name')
    parser.add_argument('--n-samples', type=int, default=50,
                       help='Number of samples for optimization')
    parser.add_argument('--n-steps', type=int, default=11,
                       help='Grid search steps (default: 11 for 0.0, 0.1, ..., 1.0)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use')

    args = parser.parse_args()

    # Initialize optimizer
    optimizer = WeightOptimizer(device=args.device)

    # Run optimization
    results = optimizer.optimize_on_dataset(
        dataset=args.dataset,
        model=args.model,
        n_samples=args.n_samples,
        n_steps=args.n_steps
    )

    # Print best weights
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"\nRecommended weights:")
    print(f"  --coedit-weight {results['best_weights']['coedit']:.4f}")
    print(f"  --tocsin-weight {results['best_weights']['tocsin']:.4f}")
    print(f"\nExpected ROC AUC: {results['best_roc_auc']:.4f}")

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\nResults saved to {args.output}")

    return results


if __name__ == '__main__':
    main()
