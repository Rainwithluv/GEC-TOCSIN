"""
Quick test script to verify ROC AUC improvements.
Tests the TOCSIN inversion fix and advanced fusion.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.src.detector import MultiFusionDetector
from demo.src.utils.data_loader import DataLoader
import numpy as np


def test_fusion_strategies():
    """Test different fusion strategies with TOCSIN fix."""
    print("=" * 60)
    print("ROC AUC Improvement Test")
    print("=" * 60)

    # Load small dataset for quick testing
    loader = DataLoader(base_dir="../")
    data = loader.load_combined_data("xsum", "gpt-4")

    n_samples = 20  # Small sample for quick testing
    human_texts = data['original'][:n_samples]
    llm_texts = data['sampled'][:n_samples]

    print(f"\nTesting with {n_samples} human texts and {n_samples} LLM texts")
    print("-" * 60)

    # Test strategies
    strategies = [
        ('improved', {'weight_method': 'optimized'}),
        ('improved', {'weight_method': 'adaptive'}),
        ('advanced', {'fusion_method': 'ensemble'}),
        ('advanced', {'fusion_method': 'weighted'}),
    ]

    results = {}

    for fusion_name, kwargs in strategies:
        print(f"\nTesting: {fusion_name} ({kwargs})")
        print("-" * 40)

        try:
            detector = MultiFusionDetector(
                fusion_strategy=fusion_name,
                **kwargs
            )

            result = detector.evaluate(human_texts, llm_texts)

            roc_auc = result.get('roc_auc', 0)
            precision = result.get('precision', 0)
            recall = result.get('recall', 0)
            f1 = result.get('f1', 0)
            conf_matrix = result.get('confusion_matrix', [])

            print(f"ROC AUC: {roc_auc:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1: {f1:.4f}")
            print(f"Confusion Matrix: {conf_matrix}")

            results[f"{fusion_name}_{kwargs.get('weight_method', kwargs.get('fusion_method', 'default'))}"] = {
                'roc_auc': roc_auc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': conf_matrix
            }

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Strategy':<30} {'ROC AUC':<10} {'F1':<10} {'Recall':<10}")
    print("-" * 60)

    for name, metrics in results.items():
        print(f"{name:<30} {metrics['roc_auc']:<10.4f} {metrics['f1']:<10.4f} {metrics['recall']:<10.4f}")

    # Find best
    best_strategy = max(results.items(), key=lambda x: x[1]['roc_auc'])
    print(f"\nBest ROC AUC: {best_strategy[0]} with {best_strategy[1]['roc_auc']:.4f}")

    return results


if __name__ == '__main__':
    test_fusion_strategies()
