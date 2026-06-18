"""
Mini MultiFusion-Detector for fast testing and debugging.
Uses minimal samples and quick evaluation.
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Literal

# Handle imports
try:
    from channels.coedit_channel import CoEdITChannel
    from channels.tocsin_channel import TOCSINChannel
    from fusion.weighted_fusion import WeightedFusion
    from fusion.voting_fusion import VotingFusion
    from fusion.improved_fusion import ImprovedFusion
    from utils.metrics import get_roc_metrics, get_metrics_with_threshold
    from utils.data_loader import DataLoader
except ImportError:
    from demo.src.channels.coedit_channel import CoEdITChannel
    from demo.src.channels.tocsin_channel import TOCSINChannel
    from demo.src.fusion.weighted_fusion import WeightedFusion
    from demo.src.fusion.voting_fusion import VotingFusion
    from demo.src.fusion.improved_fusion import ImprovedFusion
    from demo.src.utils.metrics import get_roc_metrics, get_metrics_with_threshold
    from demo.src.utils.data_loader import DataLoader


class MiniDetector:
    """
    Mini detector for fast testing with minimal samples.
    """

    # Default test texts for quick debugging
    # These are more realistic texts with actual differences
    DEFAULT_HUMAN_TEXTS = [
        "I went to the store yesterday and bought some groceries. The milk was on sale so I got two cartons. When I got home, I realized I forgot to buy eggs, so I had to go back.",
        "The conference was held in San Francisco last month. There were about 500 attendees from various companies. The keynote speaker discussed the future of artificial intelligence in healthcare.",
        "My dog likes to play fetch in the park. Every morning we go for a walk and he runs around chasing the ball. Sometimes he meets other dogs and they play together.",
        "I'm learning to play the guitar. It's been challenging but rewarding. I practice for about 30 minutes every day. My fingers hurt at first but now I'm getting used to the strings.",
        "The new restaurant downtown serves amazing Italian food. I went there last weekend with my family. We ordered pasta, pizza, and salad. The service was excellent and the prices were reasonable."
    ]

    DEFAULT_LLM_TEXTS = [
        "Proceeded to commercial establishment yesterday for grocery procurement transactions. Milk products featured promotional pricing, resulting in acquisition of dual containers. Upon residential arrival, recognized egg acquisition failure, necessitating return visitation.",
        "The symposium was convened within San Francisco municipal boundaries during the previous calendar month. Approximately five hundred participants representing diverse corporate entities were in attendance. The featured presenter delivered discourse regarding artificial intelligence trajectories within healthcare delivery systems.",
        "Canine companion exhibits recreational retrieval behavior within municipal park boundaries. Daily morning excursions occur accompanied by ball pursuit activities. Interspecies social interactions transpire during these temporal engagements with other quadruped entities.",
        "Currently engaged in musical instrument acquisition endeavors specifically targeting guitar proficiency development. The educational process presents challenges的同时 yielding positive outcomes. Daily half-hour practice sessions are maintained. Digital discomfort initially manifested but adaptation has occurred.",
        "The newly established dining facility located in the central business district specializes in Italian culinary preparation. Weekend familial patronage occurred recently. Comestible selections included pasta varieties, pizza preparations, and vegetable medleys. Service excellence was demonstrated alongside reasonable pricing structures."
    ]

    def __init__(self,
                 fusion_strategy: Literal['weighted', 'voting', 'improved'] = 'weighted',
                 coedit_model: str = "grammarly/coedit-large",
                 bart_model: str = "facebook/bart-base",
                 device: Optional[str] = None,
                 n_samples: int = 3,
                 **kwargs):
        """
        Initialize Mini Detector.

        Args:
            fusion_strategy: Fusion strategy to use
            coedit_model: CoEdIT model name
            bart_model: BART model name
            device: Device to use (auto-detect if None)
            n_samples: Number of samples to use for testing
            **kwargs: Additional arguments for fusion strategies
        """
        self.n_samples = n_samples
        self.fusion_strategy = fusion_strategy

        print("Initializing Mini Detector...")
        print(f"  Using {n_samples} samples for testing")
        print(f"  Fusion strategy: {fusion_strategy}")
        print(f"  Device: {device or 'auto'}")

        # Initialize channels
        print("  Loading CoEdIT model...")
        self.coedit_channel = CoEdITChannel(model_name=coedit_model, device=device)

        print("  Loading TOCSIN model...")
        self.tocsin_channel = TOCSINChannel(bart_model=bart_model, device=device)

        # Initialize fusion
        print(f"  Initializing {fusion_strategy} fusion...")
        if fusion_strategy == 'weighted':
            weights = kwargs.get('weights', {'coedit': 0.2, 'tocsin': 0.8})
            self.fusion = WeightedFusion(weights=weights)
        elif fusion_strategy == 'voting':
            voting_method = kwargs.get('voting_method', 'confidence')
            confidence_method = kwargs.get('confidence_method', 'distance')
            self.fusion = VotingFusion(
                voting_method=voting_method,
                confidence_method=confidence_method,
                calibration=True
            )
        elif fusion_strategy == 'improved':
            weight_method = kwargs.get('weight_method', 'adaptive')
            self.fusion = ImprovedFusion(weight_method=weight_method)
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")

        print("Mini Detector initialized!\n")

    def score_texts(self, texts: List[str]) -> Dict:
        """
        Score texts using both channels.

        IMPORTANT: TOCSIN scores are inverted for LLM detection (high = LLM)
        """
        # CoEdIT: high = more likely LLM (original)
        coedit_scores = np.array(self.coedit_channel.score_texts(texts, show_progress=False))

        # TOCSIN: high = more likely LLM (inverted from original)
        tocsin_scores = np.array(self.tocsin_channel.score_texts(texts, show_progress=False, for_llm=True))

        return {'coedit': coedit_scores, 'tocsin': tocsin_scores}

    def quick_test(self,
                   human_texts: Optional[List[str]] = None,
                   llm_texts: Optional[List[str]] = None) -> Dict:
        """
        Run a quick test with default or provided texts.

        Args:
            human_texts: Optional list of human texts (uses default if None)
            llm_texts: Optional list of LLM texts (uses default if None)

        Returns:
            Test results with scores and metrics
        """
        # Use default texts if not provided
        if human_texts is None:
            human_texts = self.DEFAULT_HUMAN_TEXTS[:self.n_samples]
        if llm_texts is None:
            llm_texts = self.DEFAULT_LLM_TEXTS[:self.n_samples]

        print("="*60)
        print("MINI DETECTOR - QUICK TEST")
        print("="*60)
        print(f"\nTesting with {len(human_texts)} human texts and {len(llm_texts)} LLM texts\n")

        # Score texts
        print("Scoring human texts...")
        human_scores = self.score_texts(human_texts)

        print("Scoring LLM texts...")
        llm_scores = self.score_texts(llm_texts)

        # Calculate statistics
        human_coedit = human_scores['coedit']
        human_tocsin = human_scores['tocsin']
        llm_coedit = llm_scores['coedit']
        llm_tocsin = llm_scores['tocsin']

        print("\n" + "="*60)
        print("CHANNEL SCORES (LLM-oriented: high = more likely LLM)")
        print("="*60)

        print(f"\nCoEdIT Channel (Grammar):")
        print(f"  Human:  mean={human_coedit.mean():.4f}, std={human_coedit.std():.4f}")
        print(f"  LLM:    mean={llm_coedit.mean():.4f}, std={llm_coedit.std():.4f}")
        print(f"  Diff:   {llm_coedit.mean() - human_coedit.mean():+.4f}")
        print(f"  Expected: LLM > Human (positive diff)")

        print(f"\nTOCSIN Channel (Inverted Cohesiveness):")
        print(f"  Human:  mean={human_tocsin.mean():.4f}, std={human_tocsin.std():.4f}")
        print(f"  LLM:    mean={llm_tocsin.mean():.4f}, std={llm_tocsin.std():.4f}")
        print(f"  Diff:   {llm_tocsin.mean() - human_tocsin.mean():+.4f}")
        print(f"  Expected: LLM > Human (positive diff)")

        # Normalize and fuse
        human_fused, llm_fused = self.fusion.normalize_and_fuse(human_scores, llm_scores)

        print(f"\nFused Scores:")
        print(f"  Human:  mean={human_fused.mean():.4f}, std={human_fused.std():.4f}")
        print(f"  LLM:    mean={llm_fused.mean():.4f}, std={llm_fused.std():.4f}")
        print(f"  Diff:   {llm_fused.mean() - human_fused.mean():+.4f}")
        print(f"  Expected: LLM > Human (positive diff)")

        # Calculate metrics
        roc_auc, opt_threshold, _, _, _, _, _ = get_roc_metrics(
            human_fused.tolist(), llm_fused.tolist()
        )

        _, _, conf_matrix, precision, recall, f1, accuracy = get_metrics_with_threshold(
            human_fused.tolist(), llm_fused.tolist(), opt_threshold
        )

        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        print(f"  ROC AUC:     {roc_auc:.4f}")
        print(f"  Threshold:   {opt_threshold:.4f}")
        print(f"  Accuracy:    {accuracy:.4f}")
        print(f"  Precision:   {precision:.4f}")
        print(f"  Recall:      {recall:.4f}")
        print(f"  F1 Score:    {f1:.4f}")
        print(f"  Conf Matrix: {conf_matrix}")

        # Detailed scores
        print("\n" + "="*60)
        print("DETAILED SCORES")
        print("="*60)

        for i in range(len(human_texts)):
            print(f"\nSample {i+1}:")
            print(f"  Human:  CoEdIT={human_coedit[i]:.4f}, TOCSIN={human_tocsin[i]:.4f}, Fused={human_fused[i]:.4f}")
            print(f"  LLM:    CoEdIT={llm_coedit[i]:.4f}, TOCSIN={llm_tocsin[i]:.4f}, Fused={llm_fused[i]:.4f}")

        return {
            'human_scores': human_scores,
            'llm_scores': llm_scores,
            'human_fused': human_fused.tolist(),
            'llm_fused': llm_fused.tolist(),
            'metrics': {
                'roc_auc': roc_auc,
                'threshold': opt_threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': conf_matrix
            }
        }

    def evaluate_dataset(self,
                         dataset: str = 'xsum',
                         model: str = 'gpt-4',
                         n_samples: int = 5) -> Dict:
        """
        Evaluate on real dataset with minimal samples.

        Args:
            dataset: Dataset name
            model: Model name
            n_samples: Number of samples to use

        Returns:
            Evaluation results
        """
        print(f"\nLoading dataset: {dataset}, model: {model}")
        print(f"Using {n_samples} samples for quick evaluation\n")

        try:
            loader = DataLoader(base_dir="../")
            data = loader.load_combined_data(dataset, model)

            n = min(n_samples, len(data['original']), len(data['sampled']))
            human_texts = data['original'][:n]
            llm_texts = data['sampled'][:n]

            return self.quick_test(human_texts, llm_texts)

        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Falling back to default texts...")
            return self.quick_test()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Mini Detector for fast testing')

    parser.add_argument('--n-samples', type=int, default=3,
                        help='Number of samples to test (default: 3)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (uses default texts if not specified)')
    parser.add_argument('--model', type=str, default='gpt-4',
                        help='Model name for dataset evaluation')
    parser.add_argument('--fusion', type=str, default='improved',
                        choices=['weighted', 'voting', 'improved'],
                        help='Fusion strategy to use')
    parser.add_argument('--weight-method', type=str, default='optimized',
                        choices=['adaptive', 'equal', 'optimized'],
                        help='Weight method for improved fusion')
    parser.add_argument('--voting-method', type=str, default='confidence',
                        choices=['soft', 'confidence', 'bayesian'],
                        help='Voting method for voting fusion')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results')

    args = parser.parse_args()

    # Initialize mini detector
    kwargs = {}
    if args.fusion == 'improved':
        kwargs['weight_method'] = args.weight_method
    elif args.fusion == 'voting':
        kwargs['voting_method'] = args.voting_method

    detector = MiniDetector(
        fusion_strategy=args.fusion,
        device=args.device,
        n_samples=args.n_samples,
        **kwargs
    )

    # Run evaluation
    if args.dataset:
        results = detector.evaluate_dataset(
            dataset=args.dataset,
            model=args.model,
            n_samples=args.n_samples
        )
    else:
        results = detector.quick_test()

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\nResults saved to {args.output}")

    print("\n" + "="*60)
    print("MINI TEST COMPLETED")
    print("="*60)

    return results


if __name__ == '__main__':
    main()
