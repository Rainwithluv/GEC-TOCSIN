"""
Analyze the actual behavior of CoEdIT and TOCSIN channels.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.src.channels.coedit_channel import CoEdITChannel
from demo.src.channels.tocsin_channel import TOCSINChannel
from demo.src.utils.data_loader import DataLoader
import numpy as np


def analyze_channel_behavior():
    """Analyze what the channels are actually measuring."""
    print("=" * 70)
    print("CHANNEL BEHAVIOR ANALYSIS")
    print("=" * 70)

    # Use more samples to get better statistics
    loader = DataLoader(base_dir="../")
    data = loader.load_combined_data("xsum", "gpt-4")

    n = 20  # Use more samples
    human_texts = data['original'][:n]
    llm_texts = data['sampled'][:n]

    print(f"\nUsing {n} samples for analysis")
    print("-" * 70)

    # Initialize channels
    coedit = CoEdITChannel()
    tocsin = TOCSINChannel()

    # Score with CoEdIT
    print("\n--- CoEdIT Channel Analysis ---")
    human_coedit = np.array(coedit.score_texts(human_texts, show_progress=False))
    llm_coedit = np.array(coedit.score_texts(llm_texts, show_progress=False))

    print(f"Human: mean={human_coedit.mean():.4f}, std={human_coedit.std():.4f}")
    print(f"LLM:   mean={llm_coedit.mean():.4f}, std={llm_coedit.std():.4f}")
    print(f"Actual Diff: {llm_coedit.mean() - human_coedit.mean():+.4f}")

    # Check if difference is statistically significant
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(llm_coedit, human_coedit)
    print(f"T-test: t={t_stat:.4f}, p={p_value:.4f}")
    print(f"Significant: {'Yes' if p_value < 0.05 else 'No'}")

    # Score with TOCSIN (original)
    print("\n--- TOCSIN Channel Analysis (Original) ---")
    human_tocsin = np.array(tocsin.score_texts(human_texts, show_progress=False, for_llm=False))
    llm_tocsin = np.array(tocsin.score_texts(llm_texts, show_progress=False, for_llm=False))

    print(f"Human: mean={human_tocsin.mean():.4f}, std={human_tocsin.std():.4f}")
    print(f"LLM:   mean={llm_tocsin.mean():.4f}, std={llm_tocsin.std():.4f}")
    print(f"Actual Diff: {llm_tocsin.mean() - human_tocsin.mean():+.4f}")

    t_stat, p_value = stats.ttest_ind(llm_tocsin, human_tocsin)
    print(f"T-test: t={t_stat:.4f}, p={p_value:.4f}")
    print(f"Significant: {'Yes' if p_value < 0.05 else 'No'}")

    # Analysis conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    coedit_llm_higher = llm_coedit.mean() > human_coedit.mean()
    tocsin_human_higher = human_tocsin.mean() > llm_tocsin.mean()

    print(f"\nCoEdIT: {'LLM > Human' if coedit_llm_higher else 'Human > LLM'}")
    print(f"  - If LLM > Human: LLM texts are LESS grammatically standard (need more GEC)")
    print(f"  - If Human > LLM: Human texts are LESS grammatically standard (need more GEC)")

    print(f"\nTOCSIN: {'Human > LLM' if tocsin_human_higher else 'LLM > Human'}")
    print(f"  - If Human > LLM: Human texts are MORE cohesive (higher cohesiveness_score)")
    print(f"  - If LLM > Human: LLM texts are MORE cohesive (higher cohesiveness_score)")

    # Determine the actual behavior
    print("\n" + "=" * 70)
    print("ACTUAL BEHAVIOR")
    print("=" * 70)

    if not coedit_llm_higher:
        print("\n⚠️  CoEdIT: Human > LLM")
        print("   This means Human texts need MORE grammar corrections")
        print("   OR LLM texts are MORE grammatically standard")
        print("   → For LLM detection, we should invert CoEdIT scores!")

    if not tocsin_human_higher:
        print("\n⚠️  TOCSIN: LLM > Human (unexpected)")
        print("   This means LLM texts are MORE cohesive")
        print("   → This contradicts the TOCSIN paper assumption")

    # Suggest the fix
    print("\n" + "=" * 70)
    print("SUGGESTED FIX")
    print("=" * 70)

    if not coedit_llm_higher and not tocsin_human_higher:
        print("\nBoth channels have unexpected directions.")
        print("Suggested fusion strategy:")
        print("  - Invert CoEdIT scores: llm_score = 1 - coedit_score")
        print("  - Keep TOCSIN as-is for LLM detection (no inversion needed)")
        print("  - Or investigate why the behavior differs from papers")
    elif not coedit_llm_higher:
        print("\nOnly CoEdIT has unexpected direction.")
        print("Suggested fusion strategy:")
        print("  - Invert CoEdIT scores: llm_score = 1 - coedit_score")
        print("  - Invert TOCSIN scores: llm_score = 1 - tocsin_score")
    elif not tocsin_human_higher:
        print("\nOnly TOCSIN has unexpected direction.")
        print("Suggested fusion strategy:")
        print("  - Don't invert TOCSIN (it's already LLM-oriented)")
        print("  - Keep CoEdIT as-is")


if __name__ == '__main__':
    analyze_channel_behavior()
