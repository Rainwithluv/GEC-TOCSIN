"""
Improved test for MultiFusion-Detector with multiple samples.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from demo.src.detector import MultiFusionDetector


def test_with_multiple_samples():
    """Test with multiple human and LLM texts."""
    print("=== Testing MultiFusion-Detector (Multiple Samples) ===\n")

    # Sample human texts
    human_texts = [
        "The quick brown fox jumps over the lazy dog. This is a classic sentence.",
        "Machine learning has revolutionized the way we process and understand data.",
        "The weather today is quite pleasant, with clear skies and mild temperatures.",
        "Students should prepare thoroughly for their upcoming examinations.",
        "The new policy has received mixed reactions from various stakeholders."
    ]

    # Sample LLM texts (more formal/formulaic)
    llm_texts = [
        "The rapid brown canine leaps above the sluggish canine. This represents a traditional sentence.",
        "Artificial intelligence has transformed the methodology by which we analyze and comprehend information.",
        "Today's meteorological conditions are relatively favorable, featuring unclouded horizons and moderate thermal readings.",
        "Learners ought to study comprehensively for their impending academic assessments.",
        "The recently implemented regulation has garnered diverse responses from numerous interested parties."
    ]

    print(f"Testing with {len(human_texts)} human texts and {len(llm_texts)} LLM texts\n")

    # Initialize detector
    print("Initializing detector...")
    detector = MultiFusionDetector(
        fusion_strategy='weighted',
        device='cpu'
    )

    # Score texts
    print("\nScoring human texts...")
    human_scores = detector.score_texts(human_texts, show_progress=False)

    print("\nScoring LLM texts...")
    llm_scores = detector.score_texts(llm_texts, show_progress=False)

    # Calculate statistics
    human_coedit = human_scores['coedit_scores']
    human_tocsin = human_scores['tocsin_scores']
    llm_coedit = llm_scores['coedit_scores']
    llm_tocsin = llm_scores['tocsin_scores']

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    print(f"\nCoEdIT Channel (Grammar):")
    print(f"  Human texts:  mean={human_coedit.mean():.4f}, std={human_coedit.std():.4f}")
    print(f"  LLM texts:    mean={llm_coedit.mean():.4f}, std={llm_coedit.std():.4f}")
    print(f"  Difference:   {llm_coedit.mean() - human_coedit.mean():+.4f}")

    print(f"\nTOCSIN Channel (Cohesiveness):")
    print(f"  Human texts:  mean={human_tocsin.mean():.4f}, std={human_tocsin.std():.4f}")
    print(f"  LLM texts:    mean={llm_tocsin.mean():.4f}, std={llm_tocsin.std():.4f}")
    print(f"  Difference:   {llm_tocsin.mean() - human_tocsin.mean():+.4f}")

    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print("\nExpected patterns:")
    print("  CoEdIT: LLM texts should have HIGHER scores (more 'perfect')")
    print("  TOCSIN: LLM texts should have HIGHER scores (more cohesive)")
    print("\nActual results:")
    if llm_coedit.mean() > human_coedit.mean():
        print("  ✓ CoEdIT correctly identifies LLM texts")
    else:
        print("  ✗ CoEdIT does not distinguish (samples too similar?)")

    if llm_tocsin.mean() > human_tocsin.mean():
        print("  ✓ TOCSIN correctly identifies LLM texts")
    else:
        print("  ✗ TOCSIN does not distinguish (may need more samples)")

    print("\n" + "="*60)
    print("DETAILED SCORES")
    print("="*60)
    print("\nIndividual CoEdIT scores:")
    print(f"  Human: {[f'{x:.4f}' for x in human_coedit]}")
    print(f"  LLM:   {[f'{x:.4f}' for x in llm_coedit]}")

    print("\nIndividual TOCSIN scores:")
    print(f"  Human: {[f'{x:.4f}' for x in human_tocsin]}")
    print(f"  LLM:   {[f'{x:.4f}' for x in llm_tocsin]}")

    print("\n=== Test completed ===")


if __name__ == '__main__':
    test_with_multiple_samples()
