"""
Debug script to analyze channel scores and identify issues.
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from demo.src.channels.coedit_channel import CoEdITChannel
from demo.src.channels.tocsin_channel import TOCSINChannel

print("=== Debugging Channel Scores ===\n")

# Load some test data
data = json.load(open('demo/data/xsum_gpt-4.raw_data.json', 'r', encoding='utf-8'))

# Take first 5 samples for debugging
human_texts = data['original'][:5]
llm_texts = data['sampled'][:5]

print(f"Testing with {len(human_texts)} human texts and {len(llm_texts)} LLM texts\n")

# Initialize channels
print("Initializing channels...")
coedit = CoEdITChannel(device='cpu')
tocsin = TOCSINChannel(device='cpu')

# Get CoEdIT scores
print("\n=== CoEdIT Channel Scores ===")
print("Computing CoEdIT scores...")

human_coedit = []
for i, text in enumerate(human_texts):
    score = coedit.score_text(text)
    human_coedit.append(score)
    print(f"Human [{i}]: {score:.4f} - {text[:50]}...")

llm_coedit = []
for i, text in enumerate(llm_texts):
    score = coedit.score_text(text)
    llm_coedit.append(score)
    print(f"LLM   [{i}]: {score:.4f} - {text[:50]}...")

print(f"\nCoEdIT Statistics:")
print(f"  Human:  mean={sum(human_coedit)/len(human_coedit):.4f}")
print(f"  LLM:    mean={sum(llm_coedit)/len(llm_coedit):.4f}")
print(f"  Diff:   {sum(llm_coedit)/len(llm_coedit) - sum(human_coedit)/len(human_coedit):.4f}")

# Get TOCSIN scores
print("\n=== TOCSIN Channel Scores ===")
print("Computing TOCSIN scores...")

human_tocsin = []
for i, text in enumerate(human_texts):
    score = tocsin.score_text(text)
    human_tocsin.append(score)
    print(f"Human [{i}]: {score:.4f} - {text[:50]}...")

llm_tocsin = []
for i, text in enumerate(llm_texts):
    score = tocsin.score_text(text)
    llm_tocsin.append(score)
    print(f"LLM   [{i}]: {score:.4f} - {text[:50]}...")

print(f"\nTOCSIN Statistics:")
print(f"  Human:  mean={sum(human_tocsin)/len(human_tocsin):.4f}")
print(f"  LLM:    mean={sum(llm_tocsin)/len(llm_tocsin):.4f}")
print(f"  Diff:   {sum(llm_tocsin)/len(llm_tocsin) - sum(human_tocsin)/len(human_tocsin):.4f}")

print("\n=== Analysis ===")
print("Expected: LLM texts should have HIGHER scores in both channels")
print("Actual: Check if the patterns match the expectation")
