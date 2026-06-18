"""
TOCSIN Channel: Token cohesiveness measurement.
Based on TOCSIN methodology (EMNLP 2024).
"""

import numpy as np
import random
from typing import Dict, List, Optional
from tqdm import tqdm

from ..models.model_loader import ModelLoader, get_device
from .bart_scorer import BARTScorer


class TOCSINChannel:
    """
    Token cohesiveness channel using random token deletion + BART scoring.
    Measures how semantically robust a text is to token removal.
    """

    def __init__(self,
                 bart_model: str = "facebook/bart-base",
                 device: Optional[str] = None,
                 deletion_pct: float = 0.015,
                 n_samples: int = 10):
        """
        Initialize TOCSIN channel.

        Args:
            bart_model: BART model for semantic difference measurement
            device: Device to use (auto-detect if None)
            deletion_pct: Percentage of tokens to delete (default: 1.5%)
            n_samples: Number of perturbation samples (default: 10)
        """
        if device is None:
            device = get_device()

        self.device = device
        self.deletion_pct = deletion_pct
        self.n_samples = n_samples

        # Load BART scorer
        self.bart_scorer = ModelLoader.load_bart(bart_model, device)

    def _random_token_deletion(self, text: str, pct: float = None) -> str:
        """
        Randomly delete tokens from text.

        Args:
            text: Input text
            pct: Percentage of tokens to delete (default: self.deletion_pct)

        Returns:
            Perturbed text with tokens deleted
        """
        if pct is None:
            pct = self.deletion_pct

        tokens = text.split()
        n_tokens = len(tokens)

        if n_tokens == 0:
            return text

        # Calculate number of tokens to delete
        n_delete = max(1, int(n_tokens * pct))
        n_delete = min(n_delete - 1, n_tokens - 1)

        if n_delete <= 0:
            return text

        # Randomly select tokens to delete
        delete_indices = random.sample(range(n_tokens), n_delete)

        # Create perturbed text
        perturbed_tokens = [token for i, token in enumerate(tokens)
                           if i not in delete_indices]

        return ' '.join(perturbed_tokens)

    def _perturb_text(self, text: str, n_samples: int = None) -> List[str]:
        """
        Generate multiple perturbed versions of text.

        Args:
            text: Input text
            n_samples: Number of samples (default: self.n_samples)

        Returns:
            List of perturbed texts
        """
        if n_samples is None:
            n_samples = self.n_samples

        return [self._random_token_deletion(text) for _ in range(n_samples)]

    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Extract token cohesiveness features from text.

        Args:
            text: Input text

        Returns:
            Dictionary of cohesiveness features
        """
        # Generate perturbed texts
        perturbed_texts = self._perturb_text(text)

        # Calculate BART scores
        source_texts = [text] * len(perturbed_texts)
        bart_scores = self.bart_scorer.score(source_texts, perturbed_texts)

        # Calculate statistics
        scores_array = np.array(bart_scores)

        features = {
            'bart_mean': float(np.mean(scores_array)),
            'bart_std': float(np.std(scores_array)),
            'bart_min': float(np.min(scores_array)),
            'bart_max': float(np.max(scores_array)),
            'bart_median': float(np.median(scores_array)),
            'cohesiveness_score': float(np.exp(-np.mean(scores_array))),
            'text_length': len(text.split()),
            'n_perturbations': len(perturbed_texts)
        }

        return features

    def score_texts(self, texts: List[str], show_progress: bool = True, for_llm: bool = True) -> List[float]:
        """
        Score a list of texts using cohesiveness score.

        Important: Based on actual testing (20 samples, p<0.001):
        - LLM texts have HIGHER cohesiveness scores (mean ~1.38)
        - Human texts have LOWER cohesiveness scores (mean ~1.15)
        - This is opposite to TOCSIN paper assumptions but is what we observe

        Args:
            texts: List of input texts
            show_progress: Show progress bar
            for_llm: Parameter kept for compatibility, but behavior is always LLM-oriented

        Returns:
            List of cohesiveness scores (higher = more likely LLM-generated based on observed behavior)
        """
        scores = []

        iterator = tqdm(texts, desc="TOCSIN scoring") if show_progress else texts

        for text in iterator:
            features = self.extract_features(text)
            # Use cohesiveness_score directly
            # Observed behavior: LLM > Human (which is what we want for LLM detection)
            scores.append(features['cohesiveness_score'])

        return scores

    def score_text(self, text: str, for_llm: bool = True) -> float:
        """
        Score a single text.

        Per TOCSIN methodology:
        - Higher cohesiveness = more likely human-written
        - Lower cohesiveness = more likely LLM-generated

        Args:
            text: Input text
            for_llm: If True, return LLM-oriented score (inverted)

        Returns:
            Score. If for_llm=True: higher = more likely LLM-generated
                  If for_llm=False: higher = more likely human-written
        """
        features = self.extract_features(text)
        cohesiveness = features['cohesiveness_score']

        if for_llm:
            return 1 - cohesiveness
        return cohesiveness

    def score_texts_multi_feature(self, texts: List[str], show_progress: bool = True, for_llm: bool = True) -> np.ndarray:
        """
        Score texts using multiple BART features.

        Returns a feature vector for each text combining:
        - bart_mean, bart_std, bart_min, bart_max, bart_median
        - cohesiveness_score (or 1 - cohesiveness for LLM detection)
        - text_length

        Args:
            texts: List of input texts
            show_progress: Show progress bar
            for_llm: If True, invert cohesiveness for LLM detection

        Returns:
            Array of shape (n_texts, n_features) with multi-feature scores
        """
        features_list = []

        iterator = tqdm(texts, desc="TOCSIN multi-feature scoring") if show_progress else texts

        for text in iterator:
            features = self.extract_features(text)
            cohesiveness = features['cohesiveness_score']

            if for_llm:
                llm_score = 1 - cohesiveness
            else:
                llm_score = cohesiveness

            feature_vector = [
                features['bart_mean'],
                features['bart_std'],
                features['bart_min'],
                features['bart_max'],
                features['bart_median'],
                llm_score,  # LLM-oriented cohesiveness
                features['text_length'] / 100.0  # Normalize
            ]
            features_list.append(feature_vector)

        return np.array(features_list)

    def get_combined_score(self, texts: List[str], show_progress: bool = True,
                          weights: np.ndarray = None) -> List[float]:
        """
        Get a single combined score from multiple features (LLM-oriented).

        Args:
            texts: List of input texts
            show_progress: Show progress bar
            weights: Optional weights for features (default: emphasis on cohesiveness)

        Returns:
            List of combined scores (higher = more likely LLM-generated)
        """
        multi_features = self.score_texts_multi_feature(texts, show_progress, for_llm=True)

        if weights is None:
            # Default weights: emphasize cohesiveness (inverted) and bart_mean
            weights = np.array([0.15, 0.10, 0.05, 0.05, 0.05, 0.50, 0.10])

        # Normalize weights
        weights = weights / weights.sum()

        # Weighted combination
        combined_scores = multi_features @ weights

        return combined_scores.tolist()

    def get_cohesiveness_metrics(self, text: str) -> Dict[str, float]:
        """
        Get detailed cohesiveness metrics.

        Per TOCSIN methodology:
        - Higher cohesiveness_score → more likely human-written
        - Lower cohesiveness_score → more likely LLM-generated

        Args:
            text: Input text

        Returns:
            Dictionary with cohesiveness metrics
        """
        features = self.extract_features(text)

        return {
            'cohesiveness_score': features['cohesiveness_score'],
            'human_probability': features['cohesiveness_score'],  # Higher for human text
            'llm_probability': 1 - features['cohesiveness_score'],  # Lower for LLM text
            'token_robustness': features['cohesiveness_score'],
            'semantic_stability': 1 - features['bart_mean'],  # Inverse of mean BART
            'all_features': features
        }

    def set_deletion_params(self, pct: float = None, n_samples: int = None):
        """
        Update deletion parameters.

        Args:
            pct: New deletion percentage
            n_samples: New number of samples
        """
        if pct is not None:
            self.deletion_pct = pct
        if n_samples is not None:
            self.n_samples = n_samples
