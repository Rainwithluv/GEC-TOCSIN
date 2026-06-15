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

    def score_texts(self, texts: List[str], show_progress: bool = True) -> List[float]:
        """
        Score a list of texts using cohesiveness score.

        Per TOCSIN methodology:
        - Human texts: perturbation causes large semantic drop → high score
        - LLM texts: perturbation causes small semantic drop → low score
        - Higher score = more likely human-written

        Args:
            texts: List of input texts
            show_progress: Show progress bar

        Returns:
            List of scores (higher = more likely human-written)
        """
        scores = []

        iterator = tqdm(texts, desc="TOCSIN scoring") if show_progress else texts

        for text in iterator:
            features = self.extract_features(text)
            # Use cohesiveness score: higher = more human-like
            # exp(-mean_bart): low mean (large semantic drop) → high score → human
            scores.append(features['cohesiveness_score'])

        return scores

    def score_text(self, text: str) -> float:
        """
        Score a single text.

        Per TOCSIN methodology:
        - Higher score = more likely human-written
        - Lower score = more likely LLM-generated

        Args:
            text: Input text

        Returns:
            Cohesiveness score (higher = more likely human-written)
        """
        features = self.extract_features(text)
        return features['cohesiveness_score']

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
