"""
Adaptive Fusion Strategy for MultiFusion-Detector.
Dynamically adjusts weights based on text features.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class AdaptiveFusion:
    """
    Adaptive fusion that adjusts weights based on text characteristics.
    """

    def __init__(self,
                 base_weights: Optional[Dict[str, float]] = None,
                 adaptation_mode: str = 'length'):
        """
        Initialize adaptive fusion.

        Args:
            base_weights: Base weights (default: equal weights)
            adaptation_mode: How to adapt weights
                - 'length': Adapt based on text length
                - 'variance': Adapt based on score variance
                - 'confidence': Adapt based on channel confidence
        """
        if base_weights is None:
            base_weights = {'coedit': 0.5, 'tocsin': 0.5}

        # Normalize base weights
        total = sum(base_weights.values())
        self.base_weights = {k: v / total for k, v in base_weights.items()}

        self.adaptation_mode = adaptation_mode
        self.channel_names = list(self.base_weights.keys())

    def _get_length_adaptation(self, text_features: Dict) -> Dict[str, float]:
        """
        Adapt weights based on text length.

        Longer texts rely more on grammar (CoEdIT),
        shorter texts rely more on cohesiveness (TOCSIN).

        Args:
            text_features: Dictionary containing text features

        Returns:
            Adjusted weights
        """
        text_length = text_features.get('text_length', 100)

        # Longer texts -> more weight on CoEdIT
        # Shorter texts -> more weight on TOCSIN
        if text_length > 200:
            w_coedit = 0.6
            w_tocsin = 0.4
        elif text_length < 50:
            w_coedit = 0.3
            w_tocsin = 0.7
        else:
            w_coedit = 0.5
            w_tocsin = 0.5

        return {'coedit': w_coedit, 'tocsin': w_tocsin}

    def _get_variance_adaptation(self, channel_features: Dict) -> Dict[str, float]:
        """
        Adapt weights based on channel score variance.

        Lower variance channels get higher weights (more stable).

        Args:
            channel_features: Dictionary of channel features

        Returns:
            Adjusted weights
        """
        variances = {}
        for channel, features in channel_features.items():
            if isinstance(features, dict) and 'bart_std' in features:
                variances[channel] = features['bart_std']
            else:
                variances[channel] = 1.0  # Default variance

        # Inverse variance weighting
        inv_var = {k: 1.0 / (v + 1e-8) for k, v in variances.items()}
        total = sum(inv_var.values())

        return {k: v / total for k, v in inv_var.items()}

    def _get_confidence_adaptation(self, channel_scores: Dict) -> Dict[str, float]:
        """
        Adapt weights based on channel confidence.

        Channels with extreme scores (high confidence) get higher weights.

        Args:
            channel_scores: Dictionary of channel scores

        Returns:
            Adjusted weights
        """
        confidences = {}
        for channel, score in channel_scores.items():
            # Distance from 0.5 (uncertainty point)
            conf = abs(score - 0.5)
            confidences[channel] = conf

        # Normalize
        total = sum(confidences.values())
        if total > 0:
            return {k: v / total for k, v in confidences.items()}
        else:
            return self.base_weights.copy()

    def get_adapted_weights(self, text_features: Dict = None,
                           channel_features: Dict = None,
                           channel_scores: Dict = None) -> Dict[str, float]:
        """
        Get adapted weights based on the specified mode.

        Args:
            text_features: Text-level features (for 'length' mode)
            channel_features: Channel features (for 'variance' mode)
            channel_scores: Channel scores (for 'confidence' mode)

        Returns:
            Adapted weight dictionary
        """
        if self.adaptation_mode == 'length':
            if text_features is None:
                return self.base_weights.copy()
            return self._get_length_adaptation(text_features)

        elif self.adaptation_mode == 'variance':
            if channel_features is None:
                return self.base_weights.copy()
            return self._get_variance_adaptation(channel_features)

        elif self.adaptation_mode == 'confidence':
            if channel_scores is None:
                return self.base_weights.copy()
            return self._get_confidence_adaptation(channel_scores)

        else:
            return self.base_weights.copy()

    def normalize_scores(self, scores: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize scores to [0, 1] range.

        Args:
            scores: Input scores array
            method: Normalization method

        Returns:
            Normalized scores
        """
        if method == 'minmax':
            min_val = np.min(scores)
            max_val = np.max(scores)
            if max_val == min_val:
                return np.ones_like(scores) * 0.5
            return (scores - min_val) / (max_val - min_val)
        else:
            return scores

    def fuse(self, channel_scores: Dict[str, np.ndarray],
             text_features_list: List[Dict] = None,
             channel_features_list: List[Dict] = None) -> np.ndarray:
        """
        Fuse channel scores with adaptive weights.

        Args:
            channel_scores: Dictionary of channel name -> scores array
            text_features_list: Optional list of text features for each sample
            channel_features_list: Optional list of channel features for each sample

        Returns:
            Fused scores array
        """
        n_samples = len(next(iter(channel_scores.values())))
        fused_scores = np.zeros(n_samples)

        for i in range(n_samples):
            # Get adapted weights for this sample
            text_feats = text_features_list[i] if text_features_list else None
            channel_feats = channel_features_list[i] if channel_features_list else None
            sample_scores = {k: v[i] for k, v in channel_scores.items()}

            weights = self.get_adapted_weights(
                text_features=text_feats,
                channel_features=channel_feats,
                channel_scores=sample_scores
            )

            # Normalize and combine
            sample_fused = 0.0
            for channel, score in sample_scores.items():
                sample_fused += weights[channel] * score

            fused_scores[i] = sample_fused

        return fused_scores

    def __repr__(self) -> str:
        return f"AdaptiveFusion(base_weights={self.base_weights}, mode={self.adaptation_mode})"
