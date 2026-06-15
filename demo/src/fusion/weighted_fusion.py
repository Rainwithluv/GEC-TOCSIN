"""
Weighted Fusion Strategy for MultiFusion-Detector.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class WeightedFusion:
    """
    Simple weighted fusion of channel scores.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize weighted fusion.

        Args:
            weights: Dictionary mapping channel names to weights
                    Default: {'coedit': 0.5, 'tocsin': 0.5}
        """
        if weights is None:
            weights = {'coedit': 0.5, 'tocsin': 0.5}

        # Normalize weights
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}

        self.channel_names = list(self.weights.keys())

    def normalize_scores(self, scores: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize scores to [0, 1] range.

        Args:
            scores: Input scores array
            method: Normalization method ('minmax' or 'zscore')

        Returns:
            Normalized scores
        """
        if method == 'minmax':
            min_val = np.min(scores)
            max_val = np.max(scores)
            if max_val == min_val:
                return np.ones_like(scores) * 0.5
            return (scores - min_val) / (max_val - min_val)

        elif method == 'zscore':
            mean = np.mean(scores)
            std = np.std(scores)
            if std == 0:
                return np.zeros_like(scores)
            return (scores - mean) / std

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def fuse(self, channel_scores: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Fuse channel scores using weighted combination.

        Args:
            channel_scores: Dictionary of channel name -> scores array

        Returns:
            Fused scores array
        """
        # Ensure all channels have same length
        lengths = [len(scores) for scores in channel_scores.values()]
        if len(set(lengths)) > 1:
            raise ValueError(f"Channel scores have different lengths: {lengths}")

        n_samples = lengths[0]
        fused_scores = np.zeros(n_samples)

        # Combine each channel
        for channel, scores in channel_scores.items():
            if channel not in self.weights:
                raise ValueError(f"Unknown channel: {channel}")
            fused_scores += self.weights[channel] * scores

        return fused_scores

    def normalize_and_fuse(self, human_channel_scores: Dict[str, np.ndarray],
                           llm_channel_scores: Dict[str, np.ndarray]) -> tuple:
        """
        Normalize scores across all samples (human + LLM together), then fuse.

        Args:
            human_channel_scores: Dictionary of channel -> human scores arrays
            llm_channel_scores: Dictionary of channel -> LLM scores arrays

        Returns:
            Tuple of (human_fused, llm_fused) scores
        """
        # Combine human and LLM scores for each channel
        normalized_human = {}
        normalized_llm = {}

        for channel in human_channel_scores.keys():
            # Concatenate human and LLM scores
            all_scores = np.concatenate([human_channel_scores[channel],
                                       llm_channel_scores[channel]])

            # Normalize combined scores
            norm_scores = self.normalize_scores(all_scores)

            # Split back into human and LLM
            n_human = len(human_channel_scores[channel])
            normalized_human[channel] = norm_scores[:n_human]
            normalized_llm[channel] = norm_scores[n_human:]

        # Fuse normalized scores
        human_fused = self.fuse(normalized_human)
        llm_fused = self.fuse(normalized_llm)

        return human_fused, llm_fused

    def fuse_single(self, channel_scores: Dict[str, float],
                   normalization_params: Optional[Dict[str, Dict]] = None) -> float:
        """
        Fuse single sample scores from all channels.

        Args:
            channel_scores: Dictionary of channel name -> score
            normalization_params: Optional parameters for normalization
                                 (min, max values for each channel)

        Returns:
            Fused score
        """
        fused_score = 0.0

        for channel, score in channel_scores.items():
            if channel not in self.weights:
                raise ValueError(f"Unknown channel: {channel}")

            normalized_score = score

            # Apply normalization if parameters provided
            if normalization_params and channel in normalization_params:
                params = normalization_params[channel]
                min_val, max_val = params['min'], params['max']
                if max_val != min_val:
                    normalized_score = (score - min_val) / (max_val - min_val)

            fused_score += self.weights[channel] * normalized_score

        return fused_score

    def set_weights(self, weights: Dict[str, float]):
        """
        Update fusion weights.

        Args:
            weights: New weight dictionary
        """
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}

    def get_weights(self) -> Dict[str, float]:
        """
        Get current weights.

        Returns:
            Dictionary of channel weights
        """
        return self.weights.copy()

    def __repr__(self) -> str:
        return f"WeightedFusion(weights={self.weights})"
