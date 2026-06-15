"""
Simple test for MultiFusion-Detector.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from demo.src.detector import MultiFusionDetector


def test_basic_detection():
    """Test basic detection functionality."""
    print("=== Testing MultiFusion-Detector ===\n")

    # Sample texts
    human_text = """
    The quick brown fox jumps over the lazy dog. This is a classic sentence
    that contains all the letters of the alphabet. People often use it for
    testing typography and font designs.
    """

    llm_text = """
    The rapid brown canine leaps above the sluggish canine. This represents
    a traditional sentence that encompasses every letter from the alphabet.
    Individuals frequently utilize it for examining typography and font layouts.
    """

    # Initialize detector (with minimal resources for testing)
    print("Initializing detector...")
    detector = MultiFusionDetector(
        fusion_strategy='weighted',
        coedit_model='grammarly/coedit-large',
        bart_model='facebook/bart-base',
        device='cpu'  # Use CPU for testing
    )

    # Score texts
    print("\nScoring texts...")
    human_scores = detector.score_texts([human_text], show_progress=False)
    llm_scores = detector.score_texts([llm_text], show_progress=False)

    print(f"\nHuman text scores:")
    print(f"  CoEdIT: {human_scores['coedit_scores'][0]:.4f}")
    print(f"  TOCSIN: {human_scores['tocsin_scores'][0]:.4f}")

    print(f"\nLLM text scores:")
    print(f"  CoEdIT: {llm_scores['coedit_scores'][0]:.4f}")
    print(f"  TOCSIN: {llm_scores['tocsin_scores'][0]:.4f}")

    print("\n=== Test completed ===")


if __name__ == '__main__':
    test_basic_detection()
