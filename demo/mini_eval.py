"""
Mini evaluation script for quick debugging and tuning.
Uses only 10 samples for fast iteration.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from demo.src.detector import MultiFusionDetector
from demo.src.utils.data_loader import DataLoader

print("=== Mini Evaluation (10 samples) ===\n")

# Settings
DATASET = "xsum"
MODEL = "gpt-4"
N_SAMPLES = 10  # Use only 10 samples for quick testing

# Load data
print(f"Loading data: {DATASET} + {MODEL}...")
loader = DataLoader()

try:
    data = loader.load_combined_data(DATASET, MODEL)
    n_samples = min(N_SAMPLES, len(data['original']), len(data['sampled']))
    human_texts = data['original'][:n_samples]
    llm_texts = data['sampled'][:n_samples]

    print(f"Loaded {len(human_texts)} human texts and {len(llm_texts)} LLM texts\n")

    # Initialize detector
    print("Initializing detector...")
    detector = MultiFusionDetector(
        fusion_strategy='weighted',
        device='cpu'  # Use CPU for faster iteration
    )

    # Get raw channel scores (for debugging)
    print("Scoring texts...")
    human_scores = detector.score_texts(human_texts, show_progress=False)
    llm_scores = detector.score_texts(llm_texts, show_progress=False)

    print("\n=== Raw Channel Scores ===")
    print(f"CoEdIT - Human:   mean={human_scores['coedit_scores'].mean():.4f}, std={human_scores['coedit_scores'].std():.4f}")
    print(f"CoEdIT - LLM:     mean={llm_scores['coedit_scores'].mean():.4f}, std={llm_scores['coedit_scores'].std():.4f}")
    print(f"CoEdIT - Diff:    {llm_scores['coedit_scores'].mean() - human_scores['coedit_scores'].mean():.4f}")

    print(f"TOCSIN - Human:   mean={human_scores['tocsin_scores'].mean():.4f}, std={human_scores['tocsin_scores'].std():.4f}")
    print(f"TOCSIN - LLM:     mean={llm_scores['tocsin_scores'].mean():.4f}, std={llm_scores['tocsin_scores'].std():.4f}")
    print(f"TOCSIN - Diff:    {llm_scores['tocsin_scores'].mean() - human_scores['tocsin_scores'].mean():.4f}")

    # Evaluate with proper normalization
    print("\n=== Evaluating with proper normalization ===")
    results = detector.evaluate(human_texts, llm_texts, threshold=None)

    print("\n=== Results ===")
    print(f"ROC AUC: {results.get('roc_auc', 'N/A'):.4f}" if 'roc_auc' in results else f"Accuracy: {results.get('accuracy', 'N/A'):.4f}")
    print(f"Threshold: {results.get('threshold', 'N/A')}")

    if 'confusion_matrix' in results:
        print(f"Confusion Matrix: {results['confusion_matrix']}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1: {results['f1']:.4f}")
        print(f"Accuracy: {results['accuracy']:.4f}")

    print("\n=== Analysis ===")
    if 'roc_auc' in results:
        if results['roc_auc'] > 0.8:
            print("✓ Good performance!")
        elif results['roc_auc'] > 0.7:
            print("~ Moderate performance, may need tuning")
        else:
            print("✗ Poor performance, channels may need fixing")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
