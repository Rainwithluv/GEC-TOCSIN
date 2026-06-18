"""
Debug script to check raw channel scores before fusion.
This helps identify if the issue is with the channels or the fusion.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.src.channels.coedit_channel import CoEdITChannel
from demo.src.channels.tocsin_channel import TOCSINChannel
from demo.src.utils.data_loader import DataLoader
import numpy as np


def debug_channels():
    """Debug channel scoring with real data."""
    print("=" * 70)
    print("CHANNEL DEBUG - RAW SCORES")
    print("=" * 70)

    # Load minimal real data
    print("\nLoading 3 samples from xsum/gpt-4 dataset...")
    loader = DataLoader(base_dir="../")
    data = loader.load_combined_data("xsum", "gpt-4")

    n = 3
    human_texts = data['original'][:n]
    llm_texts = data['sampled'][:n]

    # Initialize channels
    print("\nInitializing channels...")
    coedit = CoEdITChannel()
    tocsin = TOCSINChannel()

    print("\n" + "=" * 70)
    print("SCORING")
    print("=" * 70)

    # Score with CoEdIT
    print("\n--- CoEdIT Channel ---")
    human_coedit = np.array(coedit.score_texts(human_texts, show_progress=False))
    llm_coedit = np.array(coedit.score_texts(llm_texts, show_progress=False))

    print(f"Human: {human_coedit}")
    print(f"LLM:   {llm_coedit}")
    print(f"Human mean: {human_coedit.mean():.4f}, std: {human_coedit.std():.4f}")
    print(f"LLM mean:   {llm_coedit.mean():.4f}, std: {llm_coedit.std():.4f}")
    print(f"Diff: {llm_coedit.mean() - human_coedit.mean():+.4f}")
    print(f"Expected: Positive (LLM > Human)")

    # Score with TOCSIN (original, not inverted)
    print("\n--- TOCSIN Channel (Original) ---")
    human_tocsin_orig = np.array(tocsin.score_texts(human_texts, show_progress=False, for_llm=False))
    llm_tocsin_orig = np.array(tocsin.score_texts(llm_texts, show_progress=False, for_llm=False))

    print(f"Human: {human_tocsin_orig}")
    print(f"LLM:   {llm_tocsin_orig}")
    print(f"Human mean: {human_tocsin_orig.mean():.4f}, std: {human_tocsin_orig.std():.4f}")
    print(f"LLM mean:   {llm_tocsin_orig.mean():.4f}, std: {llm_tocsin_orig.std():.4f}")
    print(f"Diff: {llm_tocsin_orig.mean() - human_tocsin_orig.mean():+.4f}")
    print(f"Expected: Negative (Human > LLM for original TOCSIN)")

    # Score with TOCSIN (inverted for LLM detection)
    print("\n--- TOCSIN Channel (Inverted for LLM) ---")
    human_tocsin_inv = np.array(tocsin.score_texts(human_texts, show_progress=False, for_llm=True))
    llm_tocsin_inv = np.array(tocsin.score_texts(llm_texts, show_progress=False, for_llm=True))

    print(f"Human: {human_tocsin_inv}")
    print(f"LLM:   {llm_tocsin_inv}")
    print(f"Human mean: {human_tocsin_inv.mean():.4f}, std: {human_tocsin_inv.std():.4f}")
    print(f"LLM mean:   {llm_tocsin_inv.mean():.4f}, std: {llm_tocsin_inv.std():.4f}")
    print(f"Diff: {llm_tocsin_inv.mean() - human_tocsin_inv.mean():+.4f}")
    print(f"Expected: Positive (LLM > Human after inversion)")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    coedit_ok = (llm_coedit.mean() - human_coedit.mean()) > 0
    tocsin_orig_ok = (llm_tocsin_orig.mean() - human_tocsin_orig.mean()) < 0
    tocsin_inv_ok = (llm_tocsin_inv.mean() - human_tocsin_inv.mean()) > 0

    print(f"\nCoEdIT working: {'✓' if coedit_ok else '✗'}")
    print(f"  Diff is {llm_coedit.mean() - human_coedit.mean():+.4f}, should be positive")

    print(f"\nTOCSIN original working: {'✓' if tocsin_orig_ok else '✗'}")
    print(f"  Diff is {llm_tocsin_orig.mean() - human_tocsin_orig.mean():+.4f}, should be negative")

    print(f"\nTOCSIN inverted working: {'✓' if tocsin_inv_ok else '✗'}")
    print(f"  Diff is {llm_tocsin_inv.mean() - human_tocsin_inv.mean():+.4f}, should be positive")

    if coedit_ok and tocsin_inv_ok:
        print("\n✅ Both channels are working correctly!")
        print("   The fusion should produce good results.")
    elif coedit_ok and tocsin_orig_ok and not tocsin_inv_ok:
        print("\n⚠️  TOCSIN inversion may have an issue.")
        print("   Original scores are correct, but inverted scores are wrong.")
    else:
        print("\n❌ Channel scoring has issues.")
        print("   Check the channel implementations.")

    # Show sample texts
    print("\n" + "=" * 70)
    print("SAMPLE TEXTS")
    print("=" * 70)
    for i in range(n):
        print(f"\n--- Sample {i+1} ---")
        print(f"Human: {human_texts[i][:100]}...")
        print(f"LLM:   {llm_texts[i][:100]}...")


if __name__ == '__main__':
    debug_channels()
