"""
Cascade Fusion Strategy for MultiFusion-Detector.
Two-stage detection: quick screening with CoEdIT, detailed check with TOCSIN.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Literal


class CascadeFusion:
    """
    Cascade fusion for efficient and accurate detection.
    First stage: CoEdIT quick screening
    Second stage: TOCSIN detailed check for uncertain cases
    """

    def __init__(self,
                 high_threshold: float = 0.95,
                 low_threshold: float = 0.85,
                 fusion_weight: float = 0.6):
        """
        Initialize cascade fusion.

        Args:
            high_threshold: Upper threshold for confident LLM classification
            low_threshold: Lower threshold for confident human classification
            fusion_weight: Weight for CoEdIT in fused score (second stage)
        """
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.fusion_weight = fusion_weight

        # Statistics
        self.stats = {
            'total': 0,
            'stage1_llm': 0,
            'stage1_human': 0,
            'stage2': 0
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = {
            'total': 0,
            'stage1_llm': 0,
            'stage1_human': 0,
            'stage2': 0
        }

    def get_stats(self) -> Dict[str, int]:
        """Get current statistics."""
        return self.stats.copy()

    def classify_stage1(self, coedit_score: float) -> Optional[Literal['llm', 'human']]:
        """
        First stage classification using CoEdIT score.

        Args:
            coedit_score: CoEdIT channel score

        Returns:
            'llm', 'human', or None (needs second stage)
        """
        if coedit_score >= self.high_threshold:
            return 'llm'
        elif coedit_score <= self.low_threshold:
            return 'human'
        else:
            return None  # Uncertain, needs second stage

    def classify_stage2(self, coedit_score: float, tocsin_score: float,
                       threshold: Optional[float] = None) -> Literal['llm', 'human']:
        """
        Second stage classification using fused scores.

        Args:
            coedit_score: CoEdIT channel score
            tocsin_score: TOCSIN channel score
            threshold: Decision threshold (default: midpoint)

        Returns:
            'llm' or 'human'
        """
        if threshold is None:
            threshold = (self.high_threshold + self.low_threshold) / 2

        # Weighted fusion
        fused_score = self.fusion_weight * coedit_score + (1 - self.fusion_weight) * tocsin_score

        if fused_score >= threshold:
            return 'llm'
        else:
            return 'human'

    def classify(self, coedit_scores: np.ndarray, tocsin_scores: np.ndarray,
                 return_scores: bool = False) -> Dict:
        """
        Classify samples using cascade approach.

        Args:
            coedit_scores: Array of CoEdIT scores
            tocsin_scores: Array of TOCSIN scores
            return_scores: Whether to return final scores

        Returns:
            Dictionary with predictions and optional scores
        """
        n_samples = len(coedit_scores)
        predictions = []
        confidences = []
        final_scores = []

        # Reset stats
        self.reset_stats()
        self.stats['total'] = n_samples

        for i in range(n_samples):
            coedit_score = coedit_scores[i]
            tocsin_score = tocsin_scores[i]

            # Stage 1: Quick screening
            stage1_result = self.classify_stage1(coedit_score)

            if stage1_result == 'llm':
                predictions.append('llm')
                confidences.append('high')
                final_scores.append(coedit_score)
                self.stats['stage1_llm'] += 1

            elif stage1_result == 'human':
                predictions.append('human')
                confidences.append('high')
                final_scores.append(coedit_score)
                self.stats['stage1_human'] += 1

            else:
                # Stage 2: Detailed check
                threshold = (self.high_threshold + self.low_threshold) / 2
                stage2_result = self.classify_stage2(coedit_score, tocsin_score, threshold)
                predictions.append(stage2_result)
                confidences.append('medium')

                # Calculate fused score
                fused_score = self.fusion_weight * coedit_score + (1 - self.fusion_weight) * tocsin_score
                final_scores.append(fused_score)
                self.stats['stage2'] += 1

        result = {
            'predictions': predictions,
            'confidences': confidences,
            'stats': self.get_stats()
        }

        if return_scores:
            result['scores'] = np.array(final_scores)

        return result

    def set_thresholds(self, high: float = None, low: float = None):
        """
        Update threshold values.

        Args:
            high: New high threshold
            low: New low threshold
        """
        if high is not None:
            self.high_threshold = high
        if low is not None:
            self.low_threshold = low

    def get_thresholds(self) -> Tuple[float, float]:
        """
        Get current threshold values.

        Returns:
            Tuple of (high_threshold, low_threshold)
        """
        return (self.high_threshold, self.low_threshold)

    def __repr__(self) -> str:
        return (f"CascadeFusion(high={self.high_threshold}, low={self.low_threshold}, "
                f"fusion_weight={self.fusion_weight})")
