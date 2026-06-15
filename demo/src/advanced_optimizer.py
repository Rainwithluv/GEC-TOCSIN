"""
高级优化器：同时优化融合权重和通道参数
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
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


class AdvancedOptimizer:
    """
    高级优化器：同时优化多个参数
    """

    def __init__(self,
                 coedit_model: str = "grammarly/coedit-large",
                 bart_model: str = "facebook/bart-base",
                 device: Optional[str] = None):
        """Initialize advanced optimizer."""
        print("Initializing Advanced Optimizer...")
        print("Loading models (this may take a while)...")

        self.coedit_model_name = coedit_model
        self.bart_model_name = bart_model
        self.device = device

        # Will be initialized when needed
        self.coedit_channel = None
        self.tocsin_channel = None

        print("Advanced Optimizer initialized!\n")

    def _init_channels(self, tocsin_params: Dict = None):
        """Initialize channels with optional parameters."""
        if self.coedit_channel is None:
            print("Initializing channels...")

            self.coedit_channel = CoEdITChannel(
                model_name=self.coedit_model_name,
                device=self.device
            )

            tocsin_kwargs = {'device': self.device}
            if tocsin_params:
                tocsin_kwargs.update(tocsin_params)

            self.tocsin_channel = TOCSINChannel(
                bart_model=self.bart_model_name,
                **tocsin_kwargs
            )

            print("Channels initialized!\n")

    def score_texts(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Score texts using both channels."""
        coedit_scores = np.array(self.coedit_channel.score_texts(texts, show_progress=False))
        tocsin_scores = np.array(self.tocsin_channel.score_texts(texts, show_progress=False))
        return {
            'coedit': coedit_scores,
            'tocsin': tocsin_scores
        }

    def evaluate_config(self,
                       human_texts: List[str],
                       llm_texts: List[str],
                       config: Dict) -> float:
        """
        Evaluate a specific configuration.

        Args:
            human_texts: Human text samples
            llm_texts: LLM text samples
            config: Configuration dictionary

        Returns:
            ROC AUC score
        """
        # Reinitialize channels if TOCSIN params changed
        tocsin_params = config.get('tocsin_params')
        if tocsin_params:
            # Check if we need to reinitialize
            current_deletion_pct = getattr(self.tocsin_channel, 'deletion_pct', 0.015) if self.tocsin_channel else 0.015
            current_n_samples = getattr(self.tocsin_channel, 'n_samples', 10) if self.tocsin_channel else 10

            if (tocsin_params.get('deletion_pct', 0.015) != current_deletion_pct or
                tocsin_params.get('n_samples', 10) != current_n_samples):
                self.coedit_channel = None
                self.tocsin_channel = None
                self._init_channels(tocsin_params)

        # Score texts
        human_scores = self.score_texts(human_texts)
        llm_scores = self.score_texts(llm_texts)

        # Create fusion with specified weights
        weights = config.get('weights', {'coedit': 0.5, 'tocsin': 0.5})
        fusion = WeightedFusion(weights=weights)

        # Normalize and fuse
        human_fused, llm_fused = fusion.normalize_and_fuse(human_scores, llm_scores)

        # Calculate ROC AUC
        roc_auc, _, _, _, _, _, _ = get_roc_metrics(
            human_fused.tolist(), llm_fused.tolist()
        )

        return roc_auc

    def optimize_weights_only(self,
                             human_texts: List[str],
                             llm_texts: List[str],
                             n_steps: int = 11) -> Dict:
        """
        Optimize only fusion weights (fast, doesn't require reinitializing channels).

        Args:
            human_texts: Human text samples
            llm_texts: LLM text samples
            n_steps: Number of steps for weight grid

        Returns:
            Optimization results
        """
        print("Optimizing fusion weights only (fast mode)...\n")

        # Initialize channels once
        self._init_channels()

        # Score texts once
        print("Scoring human texts (one-time operation)...")
        human_scores = self.score_texts(human_texts)
        print(f"  Human texts scored: {len(human_texts)} samples")

        print("Scoring LLM texts (one-time operation)...")
        llm_scores = self.score_texts(llm_texts)
        print(f"  LLM texts scored: {len(llm_texts)} samples")

        print(f"\nHuman: CoEdIT={human_scores['coedit'].mean():.4f}, TOCSIN={human_scores['tocsin'].mean():.4f}")
        print(f"LLM:   CoEdIT={llm_scores['coedit'].mean():.4f}, TOCSIN={llm_scores['tocsin'].mean():.4f}\n")

        best_roc_auc = 0
        best_weights = {'coedit': 0.5, 'tocsin': 0.5}
        results = []

        # Test weight combinations
        weight_values = np.linspace(0, 1, n_steps)
        for coedit_w in weight_values:
            for tocsin_w in weight_values:
                total = coedit_w + tocsin_w
                if total == 0:
                    continue

                coedit_w_norm = coedit_w / total
                tocsin_w_norm = tocsin_w / total

                # Create fusion and evaluate
                weights = {'coedit': coedit_w_norm, 'tocsin': tocsin_w_norm}
                fusion = WeightedFusion(weights=weights)
                human_fused, llm_fused = fusion.normalize_and_fuse(human_scores, llm_scores)

                roc_auc, _, _, _, _, _, _ = get_roc_metrics(
                    human_fused.tolist(), llm_fused.tolist()
                )

                results.append({
                    'coedit_weight': float(coedit_w_norm),
                    'tocsin_weight': float(tocsin_w_norm),
                    'roc_auc': float(roc_auc)
                })

                if roc_auc > best_roc_auc:
                    best_roc_auc = roc_auc
                    best_weights = {
                        'coedit': float(coedit_w_norm),
                        'tocsin': float(tocsin_w_norm)
                    }
                    print(f"  ✓ New best: CoEdIT={coedit_w_norm:.2f}, TOCSIN={tocsin_w_norm:.2f}, ROC AUC={roc_auc:.4f}")

        return {
            'best_weights': best_weights,
            'best_roc_auc': best_roc_auc,
            'all_results': results
        }

    def optimize_full(self,
                     human_texts: List[str],
                     llm_texts: List[str],
                     weight_steps: int = 11,
                     deletion_pcts: List[float] = None,
                     n_samples_list: List[int] = None) -> Dict:
        """
        Full optimization including TOCSIN parameters.

        WARNING: This is very slow as it requires reinitializing channels.

        Args:
            human_texts: Human text samples
            llm_texts: LLM text samples
            weight_steps: Number of weight steps
            deletion_pcts: List of deletion percentages to test
            n_samples_list: List of n_samples values to test

        Returns:
            Full optimization results
        """
        print("Running FULL optimization (this will take a while)...\n")

        if deletion_pcts is None:
            deletion_pcts = [0.01, 0.015, 0.02]

        if n_samples_list is None:
            n_samples_list = [5, 10]

        best_config = None
        best_roc_auc = 0
        all_results = []

        for del_pct in deletion_pcts:
            for n_samp in n_samples_list:
                print(f"\n{'='*60}")
                print(f"Testing: deletion_pct={del_pct}, n_samples={n_samp}")
                print(f"{'='*60}")

                # Reinitialize channels with new parameters
                self.coedit_channel = None
                self.tocsin_channel = None
                self._init_channels({
                    'deletion_pct': del_pct,
                    'n_samples': n_samp
                })

                # Optimize weights for this configuration
                result = self.optimize_weights_only(human_texts, llm_texts, weight_steps)

                config_result = {
                    'tocsin_params': {
                        'deletion_pct': del_pct,
                        'n_samples': n_samp
                    },
                    'best_weights': result['best_weights'],
                    'best_roc_auc': result['best_roc_auc']
                }
                all_results.append(config_result)

                if result['best_roc_auc'] > best_roc_auc:
                    best_roc_auc = result['best_roc_auc']
                    best_config = config_result
                    print(f"\n  ★ NEW BEST CONFIGURATION ★")
                    print(f"    deletion_pct: {del_pct}")
                    print(f"    n_samples: {n_samp}")
                    print(f"    weights: {result['best_weights']}")
                    print(f"    ROC AUC: {best_roc_auc:.4f}")

        return {
            'best_config': best_config,
            'best_roc_auc': best_roc_auc,
            'all_results': all_results
        }

    def optimize_on_dataset(self,
                           dataset: str = 'xsum',
                           model: str = 'gpt-4',
                           n_samples: int = 50,
                           mode: str = 'weights',
                           **kwargs) -> Dict:
        """
        Optimize on a specific dataset.

        Args:
            dataset: Dataset name
            model: Model name
            n_samples: Number of samples
            mode: 'weights' (fast) or 'full' (slow)
            **kwargs: Additional arguments

        Returns:
            Optimization results
        """
        print(f"Loading dataset: {dataset}, model: {model}")

        loader = DataLoader(base_dir="../")
        data = loader.load_combined_data(dataset, model)

        n = min(n_samples, len(data['original']), len(data['sampled']))
        human_texts = data['original'][:n]
        llm_texts = data['sampled'][:n]

        print(f"Using {n} samples\n")

        if mode == 'weights':
            return self.optimize_weights_only(human_texts, llm_texts, kwargs.get('weight_steps', 11))
        else:
            return self.optimize_full(human_texts, llm_texts, **kwargs)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Advanced optimizer for MultiFusion-Detector')
    parser.add_argument('--dataset', type=str, default='xsum', help='Dataset name')
    parser.add_argument('--model', type=str, default='gpt-4', help='Model name')
    parser.add_argument('--n-samples', type=int, default=50, help='Samples for optimization')
    parser.add_argument('--mode', type=str, default='weights', choices=['weights', 'full'],
                       help='Optimization mode: weights (fast) or full (slow)')
    parser.add_argument('--weight-steps', type=int, default=11, help='Weight grid steps')
    parser.add_argument('--deletion-pcts', type=str, default='0.01,0.015,0.02',
                       help='Comma-separated deletion percentages (for full mode)')
    parser.add_argument('--n-samples-list', type=str, default='5,10',
                       help='Comma-separated n_samples values (for full mode)')
    parser.add_argument('--output', type=str, default=None, help='Output file')
    parser.add_argument('--device', type=str, default=None, help='Device')

    args = parser.parse_args()

    # Parse list arguments
    deletion_pcts = [float(x) for x in args.deletion_pcts.split(',')] if args.deletion_pcts else None
    n_samples_list = [int(x) for x in args.n_samples_list.split(',')] if args.n_samples_list else None

    # Initialize optimizer
    optimizer = AdvancedOptimizer(device=args.device)

    # Run optimization
    results = optimizer.optimize_on_dataset(
        dataset=args.dataset,
        model=args.model,
        n_samples=args.n_samples,
        mode=args.mode,
        weight_steps=args.weight_steps,
        deletion_pcts=deletion_pcts,
        n_samples_list=n_samples_list
    )

    # Print results
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")

    if args.mode == 'weights':
        print(f"\nBest weights found:")
        print(f"  CoEdIT: {results['best_weights']['coedit']:.4f}")
        print(f"  TOCSIN: {results['best_weights']['tocsin']:.4f}")
        print(f"  ROC AUC: {results['best_roc_auc']:.4f}")
    else:
        print(f"\nBest configuration:")
        print(f"  TOCSIN params: {results['best_config']['tocsin_params']}")
        print(f"  Weights: {results['best_config']['best_weights']}")
        print(f"  ROC AUC: {results['best_roc_auc']:.4f}")

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\nResults saved to {args.output}")

    return results


if __name__ == '__main__':
    main()
