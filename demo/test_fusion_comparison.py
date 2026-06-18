"""
Compare fusion strategies with minimal samples for quick validation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.src.detector_mini import MiniDetector
import json


def compare_strategies(n_samples=3):
    """Compare different fusion strategies."""
    print("=" * 70)
    print("FUSION STRATEGY COMPARISON TEST")
    print("=" * 70)
    print(f"\nTesting with {n_samples} samples each (~3-5 minutes total)\n")

    strategies = [
        ('weighted', {}),
        ('improved', {'weight_method': 'optimized'}),
        ('improved', {'weight_method': 'adaptive'}),
        ('voting', {'voting_method': 'confidence'}),
    ]

    results = {}

    for fusion_name, kwargs in strategies:
        print("\n" + "=" * 70)
        print(f"Testing: {fusion_name} {kwargs}")
        print("=" * 70)

        try:
            detector = MiniDetector(
                fusion_strategy=fusion_name,
                n_samples=n_samples,
                **kwargs
            )

            result = detector.quick_test()
            metrics = result['metrics']

            key = f"{fusion_name}_{kwargs.get('weight_method', kwargs.get('voting_method', 'default'))}"
            results[key] = {
                'roc_auc': metrics['roc_auc'],
                'f1': metrics['f1'],
                'recall': metrics['recall'],
                'precision': metrics['precision'],
                'confusion_matrix': metrics['confusion_matrix']
            }

            print(f"\n✓ ROC AUC: {metrics['roc_auc']:.4f}")

        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<35} {'ROC AUC':<12} {'F1':<10} {'Recall':<10}")
    print("-" * 70)

    for name, metrics in results.items():
        print(f"{name:<35} {metrics['roc_auc']:<12.4f} {metrics['f1']:<10.4f} {metrics['recall']:<10.4f}")

    # Winner
    if results:
        best = max(results.items(), key=lambda x: x[1]['roc_auc'])
        print(f"\n🏆 Best ROC AUC: {best[0]} with {best[1]['roc_auc']:.4f}")

        # Check if fix is working
        if best[1]['roc_auc'] > 0.7:
            print("\n✅ Fix is working! ROC AUC is above 0.7")
            print("   You can now run the full evaluation with confidence.")
        elif best[1]['roc_auc'] > 0.5:
            print("\n⚠️  Partial improvement. ROC AUC is better than random but not great.")
            print("   The fix may need more work.")
        else:
            print("\n❌ Fix failed! ROC AUC is at or below random (0.5).")
            print("   There is still a bug that needs to be fixed.")

    return results


if __name__ == '__main__':
    compare_strategies(n_samples=3)
