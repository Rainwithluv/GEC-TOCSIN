"""
MultiFusion-Detector: Main detector implementation.
Combines CoEdIT and TOCSIN channels for LLM-generated text detection.
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Literal
from tqdm import tqdm

# Handle both relative and absolute imports
try:
    # Try relative imports first (when running as module)
    from channels.coedit_channel import CoEdITChannel
    from channels.tocsin_channel import TOCSINChannel
    from fusion.weighted_fusion import WeightedFusion
    from fusion.adaptive_fusion import AdaptiveFusion
    from fusion.cascade_fusion import CascadeFusion
    from fusion.voting_fusion import VotingFusion
    from fusion.improved_fusion import ImprovedFusion
    from fusion.advanced_fusion import AdvancedFusion
    from fusion.dynamic_attention_fusion import DynamicAttentionFusion
    from utils.metrics import get_roc_metrics, get_metrics_with_threshold
    from utils.data_loader import DataLoader
except ImportError:
    # Fall back to absolute imports
    from demo.src.channels.coedit_channel import CoEdITChannel
    from demo.src.channels.tocsin_channel import TOCSINChannel
    from demo.src.fusion.weighted_fusion import WeightedFusion
    from demo.src.fusion.adaptive_fusion import AdaptiveFusion
    from demo.src.fusion.cascade_fusion import CascadeFusion
    from demo.src.fusion.voting_fusion import VotingFusion
    from demo.src.fusion.improved_fusion import ImprovedFusion
    from demo.src.fusion.advanced_fusion import AdvancedFusion
    from demo.src.fusion.dynamic_attention_fusion import DynamicAttentionFusion
    from demo.src.utils.metrics import get_roc_metrics, get_metrics_with_threshold
    from demo.src.utils.data_loader import DataLoader


class MultiFusionDetector:
    """
    Main detector class combining CoEdIT and TOCSIN channels.
    """

    def __init__(self,
                 fusion_strategy: Literal['weighted', 'adaptive', 'cascade', 'voting', 'improved', 'advanced', 'dynamic'] = 'weighted',
                 coedit_model: str = "grammarly/coedit-large",
                 bart_model: str = "facebook/bart-base",
                 device: Optional[str] = None,
                 **kwargs):
        """
        Initialize MultiFusion Detector.

        Args:
            fusion_strategy: Fusion strategy to use
            coedit_model: CoEdIT model name
            bart_model: BART model name
            device: Device to use (auto-detect if None)
            **kwargs: Additional arguments for specific fusion strategies
        """
        self.fusion_strategy = fusion_strategy

        # Initialize channels
        print("Initializing channels...")
        self.coedit_channel = CoEdITChannel(model_name=coedit_model, device=device)
        self.tocsin_channel = TOCSINChannel(bart_model=bart_model, device=device)

        # Initialize fusion strategy
        print(f"Initializing {fusion_strategy} fusion...")
        if fusion_strategy == 'weighted':
            weights = kwargs.get('weights', {'coedit': 0.2, 'tocsin': 0.8})
            self.fusion = WeightedFusion(weights=weights)

        elif fusion_strategy == 'adaptive':
            base_weights = kwargs.get('base_weights', {'coedit': 0.5, 'tocsin': 0.5})
            adaptation_mode = kwargs.get('adaptation_mode', 'length')
            self.fusion = AdaptiveFusion(base_weights=base_weights, adaptation_mode=adaptation_mode)

        elif fusion_strategy == 'cascade':
            high_threshold = kwargs.get('high_threshold', 0.95)
            low_threshold = kwargs.get('low_threshold', 0.85)
            fusion_weight = kwargs.get('fusion_weight', 0.6)
            self.fusion = CascadeFusion(high_threshold=high_threshold,
                                        low_threshold=low_threshold,
                                        fusion_weight=fusion_weight)

        elif fusion_strategy == 'voting':
            voting_method = kwargs.get('voting_method', 'confidence')
            confidence_method = kwargs.get('confidence_method', 'distance')
            calibration = kwargs.get('calibration', True)
            self.fusion = VotingFusion(
                voting_method=voting_method,
                confidence_method=confidence_method,
                calibration=calibration
            )

        elif fusion_strategy == 'improved':
            weight_method = kwargs.get('weight_method', 'adaptive')
            self.fusion = ImprovedFusion(
                weight_method=weight_method
            )

        elif fusion_strategy == 'advanced':
            fusion_method = kwargs.get('fusion_method', 'ensemble')
            use_pca = kwargs.get('use_pca', True)
            pca_components = kwargs.get('pca_components', 5)
            self.fusion = AdvancedFusion(
                fusion_method=fusion_method,
                use_pca=use_pca,
                pca_components=pca_components
            )

        elif fusion_strategy == 'dynamic':
            dynamic_mode = kwargs.get('dynamic_mode', 'hybrid')
            temperature = kwargs.get('temperature', 1.0)
            min_weight = kwargs.get('min_weight', 0.1)
            max_weight = kwargs.get('max_weight', 0.9)
            self.fusion = DynamicAttentionFusion(
                mode=dynamic_mode,
                temperature=temperature,
                min_weight=min_weight,
                max_weight=max_weight
            )

        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")

        print("Detector initialized successfully!")

    def score_texts(self, texts: List[str], show_progress: bool = True) -> Dict:
        """
        Score a list of texts using both channels.

        Args:
            texts: List of input texts
            show_progress: Show progress bars

        Returns:
            Dictionary with channel scores and features
        """
        print(f"Scoring {len(texts)} texts...")

        # Score with CoEdIT (higher = more likely LLM)
        coedit_scores = self.coedit_channel.score_texts(texts, show_progress)

        # Score with TOCSIN (higher = more likely LLM after inversion)
        # CRITICAL FIX: Use for_llm=True to invert TOCSIN scores
        tocsin_scores = self.tocsin_channel.score_texts(texts, show_progress, for_llm=True)

        return {
            'coedit_scores': np.array(coedit_scores),
            'tocsin_scores': np.array(tocsin_scores)
        }

    def detect(self, texts: List[str], threshold: Optional[float] = None,
               show_progress: bool = True) -> Dict:
        """
        Detect LLM-generated texts.

        Args:
            texts: List of input texts
            threshold: Decision threshold (auto-calculate if None)
            show_progress: Show progress bars

        Returns:
            Dictionary with predictions, scores, and metrics
        """
        # Get channel scores
        scores = self.score_texts(texts, show_progress)
        coedit_scores = scores['coedit_scores']
        tocsin_scores = scores['tocsin_scores']

        # Apply fusion strategy
        if self.fusion_strategy == 'cascade':
            result = self.fusion.classify(coedit_scores, tocsin_scores, return_scores=True)
            final_scores = result['scores']
            predictions = result['predictions']
            confidences = result['confidences']

        else:
            # Weighted or adaptive fusion
            channel_scores = {
                'coedit': coedit_scores,
                'tocsin': tocsin_scores
            }

            if self.fusion_strategy == 'weighted':
                final_scores = self.fusion.fuse(channel_scores)

            elif self.fusion_strategy == 'adaptive':
                # For adaptive, we need to extract text features
                # This is a simplified version without full adaptation
                final_scores = self.fusion.fuse(channel_scores)

            # Apply threshold
            if threshold is None:
                threshold = 0.5  # Default threshold

            predictions = ['llm' if score >= threshold else 'human' for score in final_scores]
            confidences = ['medium'] * len(predictions)

        return {
            'predictions': predictions,
            'scores': final_scores,
            'confidences': confidences,
            'coedit_scores': coedit_scores,
            'tocsin_scores': tocsin_scores,
            'threshold': threshold
        }

    def evaluate(self, human_texts: List[str], llm_texts: List[str],
                 threshold: Optional[float] = None) -> Dict:
        """
        Evaluate detector performance.

        Args:
            human_texts: List of human-written texts
            llm_texts: List of LLM-generated texts
            threshold: Decision threshold (auto-calculate if None)

        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Evaluating on {len(human_texts)} human and {len(llm_texts)} LLM texts...")

        # Score texts
        human_scores = self.score_texts(human_texts, show_progress=True)
        llm_scores = self.score_texts(llm_texts, show_progress=True)

        human_coedit = human_scores['coedit_scores']
        human_tocsin = human_scores['tocsin_scores']
        llm_coedit = llm_scores['coedit_scores']
        llm_tocsin = llm_scores['tocsin_scores']

        # Calculate fused scores
        if self.fusion_strategy == 'cascade':
            # Cascade has its own evaluation
            all_coedit = np.concatenate([human_coedit, llm_coedit])
            all_tocsin = np.concatenate([human_tocsin, llm_tocsin])

            result = self.fusion.classify(all_coedit, all_tocsin)
            predictions = result['predictions']

            # Calculate metrics
            labels = ['human'] * len(human_texts) + ['llm'] * len(llm_texts)
            binary_preds = [1 if p == 'llm' else 0 for p in predictions]
            binary_labels = [1 if l == 'llm' else 0 for l in labels]

            # Simple accuracy calculation
            accuracy = sum(1 for p, l in zip(binary_preds, binary_labels) if p == l) / len(binary_preds)

            return {
                'accuracy': accuracy,
                'predictions': predictions,
                'stats': result['stats']
            }

        else:
            # Weighted, adaptive, voting, improved, advanced, or dynamic fusion
            if isinstance(self.fusion, (WeightedFusion, VotingFusion, ImprovedFusion, AdvancedFusion, DynamicAttentionFusion)):
                # Use proper normalization: normalize across all samples together
                human_channel_scores = {
                    'coedit': human_coedit,
                    'tocsin': human_tocsin
                }
                llm_channel_scores = {
                    'coedit': llm_coedit,
                    'tocsin': llm_tocsin
                }

                if threshold is None:
                    # Normalize across all samples, then fuse
                    human_fused, llm_fused = self.fusion.normalize_and_fuse(
                        human_channel_scores, llm_channel_scores
                    )

                    roc_auc, opt_threshold, _, _, _, _, _ = get_roc_metrics(
                        human_fused.tolist(), llm_fused.tolist()
                    )
                    threshold = opt_threshold
                else:
                    # Normalize across all samples, then fuse
                    human_fused, llm_fused = self.fusion.normalize_and_fuse(
                        human_channel_scores, llm_channel_scores
                    )

                    roc_auc, _, _, _, _, _, _ = get_roc_metrics(
                        human_fused.tolist(), llm_fused.tolist()
                    )
            else:
                # For adaptive fusion, use the old method (needs updating)
                if threshold is None:
                    channel_scores = {
                        'coedit': human_coedit,
                        'tocsin': human_tocsin
                    }
                    human_fused = self.fusion.fuse(channel_scores)

                    channel_scores = {
                        'coedit': llm_coedit,
                        'tocsin': llm_tocsin
                    }
                    llm_fused = self.fusion.fuse(channel_scores)

                    roc_auc, opt_threshold, _, _, _, _, _ = get_roc_metrics(
                        human_fused.tolist(), llm_fused.tolist()
                    )
                    threshold = opt_threshold
                else:
                    channel_scores = {
                        'coedit': human_coedit,
                        'tocsin': human_tocsin
                    }
                    human_fused = self.fusion.fuse(channel_scores)

                    channel_scores = {
                        'coedit': llm_coedit,
                        'tocsin': llm_tocsin
                    }
                    llm_fused = self.fusion.fuse(channel_scores)

                    roc_auc, _, _, _, _, _, _ = get_roc_metrics(
                        human_fused.tolist(), llm_fused.tolist()
                    )

                roc_auc, _, _, _, _, _, _ = get_roc_metrics(
                    human_fused.tolist(), llm_fused.tolist()
                )

            # Calculate metrics with threshold
            _, _, conf_matrix, precision, recall, f1, accuracy = get_metrics_with_threshold(
                human_fused.tolist(), llm_fused.tolist(), threshold
            )

            return {
                'roc_auc': roc_auc,
                'threshold': threshold,
                'confusion_matrix': conf_matrix,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy
            }


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(description='MultiFusion-Detector for LLM text detection')

    parser.add_argument('--mode', type=str, choices=['detect', 'evaluate'], default='evaluate',
                        help='Operation mode')
    parser.add_argument('--input', type=str, required=False,
                        help='Input data file (for detect mode)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results')
    parser.add_argument('--fusion', type=str, choices=['weighted', 'adaptive', 'cascade', 'voting', 'improved', 'advanced', 'dynamic'],
                        default='weighted', help='Fusion strategy')
    parser.add_argument('--voting-method', type=str, default='confidence',
                        choices=['soft', 'confidence', 'bayesian'],
                        help='Voting method (for voting fusion)')
    parser.add_argument('--confidence-method', type=str, default='distance',
                        choices=['distance', 'entropy', 'variance'],
                        help='Confidence calculation method (for voting fusion)')
    parser.add_argument('--weight-method', type=str, default='adaptive',
                        choices=['adaptive', 'equal', 'optimized'],
                        help='Weight calculation method (for improved fusion)')
    parser.add_argument('--normalization', type=str, default='minmax',
                        choices=['robust', 'minmax', 'zscore'],
                        help='Normalization method (deprecated, kept for compatibility)')
    parser.add_argument('--coedit-model', type=str, default='grammarly/coedit-large',
                        help='CoEdIT model name')
    parser.add_argument('--bart-model', type=str, default='facebook/bart-base',
                        help='BART model name')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--dataset', type=str, default='xsum',
                        help='Dataset name (for evaluation mode)')
    parser.add_argument('--model', type=str, default='gpt-4',
                        help='LLM model name (for evaluation mode)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Decision threshold')
    parser.add_argument('--n-samples', type=int, default=50,
                        help='Number of samples to evaluate (default: 50 for demo)')
    parser.add_argument('--coedit-weight', type=float, default=0.2,
                        help='Weight for CoEdIT channel (default: 0.2 - optimized)')
    parser.add_argument('--tocsin-weight', type=float, default=0.8,
                        help='Weight for TOCSIN channel (default: 0.8 - optimized)')
    parser.add_argument('--fusion-method', type=str, default='ensemble',
                        choices=['ensemble', 'weighted', 'stacking'],
                        help='Advanced fusion method (for advanced fusion)')
    parser.add_argument('--use-pca', action='store_true',
                        help='Use PCA for feature reduction (for advanced fusion)')
    parser.add_argument('--pca-components', type=int, default=5,
                        help='Number of PCA components (for advanced fusion)')
    parser.add_argument('--dynamic-mode', type=str, default='hybrid',
                        choices=['confidence', 'entropy', 'hybrid', 'attention'],
                        help='Dynamic fusion mode (for dynamic fusion)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature parameter for dynamic fusion weight smoothing (default: 1.0)')
    parser.add_argument('--min-weight', type=float, default=0.1,
                        help='Minimum weight for any channel in dynamic fusion (default: 0.1)')
    parser.add_argument('--max-weight', type=float, default=0.9,
                        help='Maximum weight for any channel in dynamic fusion (default: 0.9)')

    args = parser.parse_args()

    # Validate arguments based on mode
    if args.mode == 'detect' and args.input is None:
        parser.error("--input is required in detect mode")
    if args.mode == 'evaluate' and args.dataset is None:
        parser.error("--dataset is required in evaluate mode")

    # Initialize detector with fusion-specific parameters
    if args.fusion == 'voting':
        # Voting fusion parameters
        detector = MultiFusionDetector(
            fusion_strategy=args.fusion,
            coedit_model=args.coedit_model,
            bart_model=args.bart_model,
            device=args.device,
            voting_method=args.voting_method,
            confidence_method=args.confidence_method,
            calibration=True
        )
    elif args.fusion == 'improved':
        # Improved fusion parameters
        detector = MultiFusionDetector(
            fusion_strategy=args.fusion,
            coedit_model=args.coedit_model,
            bart_model=args.bart_model,
            device=args.device,
            weight_method=args.weight_method
        )
    elif args.fusion == 'advanced':
        # Advanced fusion parameters
        detector = MultiFusionDetector(
            fusion_strategy=args.fusion,
            coedit_model=args.coedit_model,
            bart_model=args.bart_model,
            device=args.device,
            fusion_method=args.fusion_method,
            use_pca=args.use_pca,
            pca_components=args.pca_components
        )
    elif args.fusion == 'dynamic':
        # Dynamic attention fusion parameters
        detector = MultiFusionDetector(
            fusion_strategy=args.fusion,
            coedit_model=args.coedit_model,
            bart_model=args.bart_model,
            device=args.device,
            dynamic_mode=args.dynamic_mode,
            temperature=args.temperature,
            min_weight=args.min_weight,
            max_weight=args.max_weight
        )
    else:
        # Weighted fusion parameters
        # Normalize weights
        total_weight = args.coedit_weight + args.tocsin_weight
        if total_weight > 0:
            coedit_w = args.coedit_weight / total_weight
            tocsin_w = args.tocsin_weight / total_weight
        else:
            coedit_w, tocsin_w = 0.5, 0.5

        detector = MultiFusionDetector(
            fusion_strategy=args.fusion,
            coedit_model=args.coedit_model,
            bart_model=args.bart_model,
            device=args.device,
            weights={'coedit': coedit_w, 'tocsin': tocsin_w}
        )

    if args.mode == 'evaluate':
        # Evaluation mode
        loader = DataLoader(base_dir="../")

        try:
            data = loader.load_combined_data(args.dataset, args.model)
            n_samples = min(args.n_samples, len(data['original']), len(data['sampled']))
            human_texts = data['original'][:n_samples]
            llm_texts = data['sampled'][:n_samples]

            print(f"Loaded {len(human_texts)} human texts and {len(llm_texts)} LLM texts")
            print(f"(limited to {n_samples} samples for demo, use --n-samples to adjust)")

            # Evaluate
            results = detector.evaluate(human_texts, llm_texts, threshold=args.threshold)

            print("\n=== Evaluation Results ===")
            print(f"ROC AUC: {results.get('roc_auc', 'N/A'):.4f}" if 'roc_auc' in results else f"Accuracy: {results.get('accuracy', 'N/A'):.4f}")
            print(f"Threshold: {results.get('threshold', 'N/A')}")
            if 'confusion_matrix' in results:
                print(f"Confusion Matrix: {results['confusion_matrix']}")
                print(f"Precision: {results['precision']:.4f}")
                print(f"Recall: {results['recall']:.4f}")
                print(f"F1: {results['f1']:.4f}")
                print(f"Accuracy: {results['accuracy']:.4f}")

            # Save results
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\nResults saved to {args.output}")

        except Exception as e:
            print(f"Error loading data: {e}")
            print("Please ensure TOCSIN and GECSore data directories are available")

    else:
        print("Detect mode - TODO: implement")


if __name__ == '__main__':
    main()
