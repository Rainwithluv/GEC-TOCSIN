"""
Analyze why XSum and Writing datasets have such different performance.
"""
import os
import sys
import numpy as np

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from channels.coedit_channel import CoEdITChannel
from channels.tocsin_channel import TOCSINChannel
from fusion.weighted_fusion import WeightedFusion
from utils.data_loader import DataLoader
from utils.metrics import get_roc_metrics, get_metrics_with_threshold

def analyze_dataset(dataset_name, model_name='gpt-4', n_samples=50):
    """Analyze detector performance on a specific dataset."""
    print("="*70)
    print(f"Dataset: {dataset_name} | Model: {model_name} | Samples: {n_samples}")
    print("="*70)

    # Load data
    loader = DataLoader(base_dir="../")
    data = loader.load_combined_data(dataset_name, model_name)

    n = min(n_samples, len(data['original']), len(data['sampled']))
    human_texts = data['original'][:n]
    llm_texts = data['sampled'][:n]

    print(f"\nLoaded {len(human_texts)} human texts and {len(llm_texts)} LLM texts")

    # Initialize channels
    print("\nInitializing channels...")
    coedit = CoEdITChannel()
    tocsin = TOCSINChannel()

    # Score texts
    print("Scoring human texts...")
    human_coedit = np.array(coedit.score_texts(human_texts, show_progress=False))
    human_tocsin = np.array(tocsin.score_texts(human_texts, show_progress=False, for_llm=True))

    print("Scoring LLM texts...")
    llm_coedit = np.array(coedit.score_texts(llm_texts, show_progress=False))
    llm_tocsin = np.array(tocsin.score_texts(llm_texts, show_progress=False, for_llm=True))

    # Channel statistics
    print("\n" + "-"*70)
    print("CHANNEL SCORES (High = LLM)")
    print("-"*70)

    print(f"\nCoEdIT Channel:")
    print(f"  Human:  mean={human_coedit.mean():.4f}, std={human_coedit.std():.4f}, min={human_coedit.min():.4f}, max={human_coedit.max():.4f}")
    print(f"  LLM:    mean={llm_coedit.mean():.4f}, std={llm_coedit.std():.4f}, min={llm_coedit.min():.4f}, max={llm_coedit.max():.4f}")
    print(f"  Diff:   {llm_coedit.mean() - human_coedit.mean():+.4f}")
    coedit_sep = abs(llm_coedit.mean() - human_coedit.mean())
    print(f"  Separation: {coedit_sep:.4f}")

    print(f"\nTOCSIN Channel:")
    print(f"  Human:  mean={human_tocsin.mean():.4f}, std={human_tocsin.std():.4f}, min={human_tocsin.min():.4f}, max={human_tocsin.max():.4f}")
    print(f"  LLM:    mean={llm_tocsin.mean():.4f}, std={llm_tocsin.std():.4f}, min={llm_tocsin.min():.4f}, max={llm_tocsin.max():.4f}")
    print(f"  Diff:   {llm_tocsin.mean() - human_tocsin.mean():+.4f}")
    tocsin_sep = abs(llm_tocsin.mean() - human_tocsin.mean())
    print(f"  Separation: {tocsin_sep:.4f}")

    # Check for overlap
    print(f"\nScore Overlap Analysis:")
    human_coedit_range = (human_coedit.min(), human_coedit.max())
    llm_coedit_range = (llm_coedit.min(), llm_coedit.max())
    human_tocsin_range = (human_tocsin.min(), human_tocsin.max())
    llm_tocsin_range = (llm_tocsin.min(), llm_tocsin.max())

    coedit_overlap = max(0, min(human_coedit_range[1], llm_coedit_range[1]) - max(human_coedit_range[0], llm_coedit_range[0]))
    tocsin_overlap = max(0, min(human_tocsin_range[1], llm_tocsin_range[1]) - max(human_tocsin_range[0], llm_tocsin_range[0]))

    print(f"  CoEdIT range overlap: {coedit_overlap:.4f}")
    print(f"  TOCSIN range overlap: {tocsin_overlap:.4f}")

    # Fused results
    fusion = WeightedFusion(weights={'coedit': 0.2, 'tocsin': 0.8})
    human_fused, llm_fused = fusion.normalize_and_fuse(
        {'coedit': human_coedit, 'tocsin': human_tocsin},
        {'coedit': llm_coedit, 'tocsin': llm_tocsin}
    )

    roc_auc, opt_threshold, _, _, _, _, _ = get_roc_metrics(
        human_fused.tolist(), llm_fused.tolist()
    )

    _, _, conf_matrix, precision, recall, f1, accuracy = get_metrics_with_threshold(
        human_fused.tolist(), llm_fused.tolist(), opt_threshold
    )

    print(f"\nFused Results (0.2/0.8 weights):")
    print(f"  ROC AUC:   {roc_auc:.4f}")
    print(f"  Threshold: {opt_threshold:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Confusion: {conf_matrix}")

    # Analysis
    print("\n" + "-"*70)
    print("ANALYSIS")
    print("-"*70)

    if roc_auc < 0.8:
        print("\n⚠️  LOW ROC AUC detected!")
        print("Possible causes:")
        print("  - LLM texts are very similar to human texts")
        print("  - Both channels struggle to separate")
        print("  - Dataset may have different characteristics")

    if recall < 0.7:
        print("\n⚠️  LOW RECALL detected!")
        print("Possible causes:")
        print("  - Many LLM texts score similarly to human texts")
        print("  - Threshold may be too high")
        print(f"  - Consider adjusting threshold from {opt_threshold:.4f} to improve recall")

    # Check which channel is better
    if coedit_sep > tocsin_sep:
        print(f"\n💡 CoEdIT has better separation ({coedit_sep:.4f} vs {tocsin_sep:.4f})")
        print("   Consider increasing CoEdIT weight for this dataset")
    else:
        print(f"\n💡 TOCSIN has better separation ({tocsin_sep:.4f} vs {coedit_sep:.4f})")
        print("   Current 0.8 weight is appropriate")

    return {
        'dataset': dataset_name,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'coedit_sep': coedit_sep,
        'tocsin_sep': tocsin_sep,
        'threshold': opt_threshold
    }

def main():
    """Compare XSum and Writing datasets."""
    print("\n" + "="*70)
    print("DATASET COMPARISON ANALYSIS")
    print("="*70)
    print("\nThis will analyze why XSum and Writing perform differently")
    print("Expected time: ~10-15 minutes per dataset\n")

    import time
    start = time.time()

    results = []

    # Analyze XSum
    print("\n" + "🔍 "*35)
    xsum_result = analyze_dataset('xsum', 'gpt-4', 50)
    results.append(xsum_result)

    print("\n\n" + "🔍 "*35)
    writing_result = analyze_dataset('writing', 'gpt-4', 50)
    results.append(writing_result)

    # Comparison
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    print(f"\n{'Metric':<20} {'XSum':>12} {'Writing':>12} {'Difference':>12}")
    print("-"*70)
    print(f"{'ROC AUC':<20} {results[0]['roc_auc']:>12.4f} {results[1]['roc_auc']:>12.4f} {results[0]['roc_auc'] - results[1]['roc_auc']:>+12.4f}")
    print(f"{'Precision':<20} {results[0]['precision']:>12.4f} {results[1]['precision']:>12.4f} {results[0]['precision'] - results[1]['precision']:>+12.4f}")
    print(f"{'Recall':<20} {results[0]['recall']:>12.4f} {results[1]['recall']:>12.4f} {results[0]['recall'] - results[1]['recall']:>+12.4f}")
    print(f"{'F1':<20} {results[0]['f1']:>12.4f} {results[1]['f1']:>12.4f} {results[0]['f1'] - results[1]['f1']:>+12.4f}")
    print(f"{'CoEdIT Separation':<20} {results[0]['coedit_sep']:>12.4f} {results[1]['coedit_sep']:>12.4f} {results[0]['coedit_sep'] - results[1]['coedit_sep']:>+12.4f}")
    print(f"{'TOCSIN Separation':<20} {results[0]['tocsin_sep']:>12.4f} {results[1]['tocsin_sep']:>12.4f} {results[0]['tocsin_sep'] - results[1]['tocsin_sep']:>+12.4f}")

    elapsed = time.time() - start
    print(f"\nAnalysis completed in {elapsed/60:.1f} minutes")

if __name__ == '__main__':
    main()
