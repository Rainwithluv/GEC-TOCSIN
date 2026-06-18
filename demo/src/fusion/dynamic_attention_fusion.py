"""
Dynamic Attention Fusion Strategy for MultiFusion-Detector.

Uses cross-branch attention mechanism to dynamically weight CoEdIT and TOCSIN channels
based on input text characteristics and channel confidence.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union


class CrossBranchAttention(nn.Module):
    """
    Cross-branch attention mechanism for dynamic channel weighting.

    This module learns to attend between CoEdIT and TOCSIN branches
    to determine optimal fusion weights for each input sample.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 16, num_heads: int = 2):
        """
        Initialize cross-branch attention module.

        Args:
            input_dim: Dimension of input features (2 for coedit+tocsin scores)
            hidden_dim: Hidden dimension for attention computation
            num_heads: Number of attention heads
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Query, Key, Value projections
        self.q_projection = nn.Linear(input_dim, hidden_dim * num_heads)
        self.k_projection = nn.Linear(input_dim, hidden_dim * num_heads)
        self.v_projection = nn.Linear(input_dim, hidden_dim * num_heads)

        # Output projection
        self.out_projection = nn.Linear(hidden_dim * num_heads, input_dim)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
               Contains [coedit_score, tocsin_score] for each sample

        Returns:
            Tuple of (output_weights, attention_weights)
            - output_weights: Normalized fusion weights (batch_size, 2)
            - attention_weights: Raw attention scores (batch_size, num_heads)
        """
        batch_size = x.size(0)

        # Layer norm
        normalized_x = self.layer_norm(x)

        # Multi-head projections
        Q = self.q_projection(normalized_x)  # (batch, hidden_dim * num_heads)
        K = self.k_projection(normalized_x)
        V = self.v_projection(normalized_x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, self.num_heads, self.hidden_dim)
        K = K.view(batch_size, self.num_heads, self.hidden_dim)
        V = V.view(batch_size, self.num_heads, self.hidden_dim)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.hidden_dim)
        attention_weights = F.softmax(scores, dim=-1)  # (batch, num_heads, num_heads)

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)

        # Concatenate heads
        attended = attended.view(batch_size, self.num_heads * self.hidden_dim)

        # Output projection
        output = self.out_projection(attended)
        output = self.dropout(output)

        # Softmax to get normalized weights
        fusion_weights = F.softmax(output, dim=-1)

        return fusion_weights, attention_weights.mean(dim=1)  # Return mean attention per head


class DynamicAttentionFusion:
    """
    Dynamic fusion using cross-branch attention mechanism.

    This fusion strategy:
    1. Extracts features from both channels
    2. Uses attention mechanism to compute sample-specific weights
    3. Applies dynamic weights to fuse the scores

    Unlike fixed-weight fusion, this adapts to each input sample.
    """

    def __init__(self,
                 mode: str = 'attention',
                 temperature: float = 1.0,
                 min_weight: float = 0.1,
                 max_weight: float = 0.9):
        """
        Initialize dynamic attention fusion.

        Args:
            mode: Fusion mode
                - 'attention': Learned attention (requires training)
                - 'confidence': Score-based confidence weighting
                - 'entropy': Entropy-based adaptive weighting
                - 'hybrid': Hybrid of confidence + entropy
            temperature: Softmax temperature for weight smoothing
            min_weight: Minimum weight for any channel
            max_weight: Maximum weight for any channel
        """
        self.mode = mode
        self.temperature = temperature
        self.min_weight = min_weight
        self.max_weight = max_weight

        # Initialize attention module if using learned mode
        if mode == 'attention':
            self.attention_module = None  # Will be loaded if trained model exists
            self.device = 'cpu'
        else:
            self.attention_module = None

    def _compute_confidence_weights(self,
                                    coedit_scores: np.ndarray,
                                    tocsin_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute confidence-based dynamic weights.

        Channels with higher absolute scores (far from 0.5) get higher weights,
        as they are more confident in their predictions.

        Args:
            coedit_scores: CoEdIT channel scores
            tocsin_scores: TOCSIN channel scores

        Returns:
            Tuple of (coedit_weights, tocsin_weights)
        """
        # Compute confidence as distance from decision boundary (0.5)
        coedit_confidence = np.abs(coedit_scores - 0.5)
        tocsin_confidence = np.abs(tocsin_scores - 0.5)

        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        total_confidence = coedit_confidence + tocsin_confidence + epsilon

        # Normalize to get weights
        coedit_weight = coedit_confidence / total_confidence
        tocsin_weight = tocsin_confidence / total_confidence

        # Apply temperature scaling
        weights = np.stack([coedit_weight, tocsin_weight], axis=-1)
        weights = weights / self.temperature
        weights = np.exp(weights) / np.exp(weights).sum(axis=-1, keepdims=True)

        # Clip to bounds
        weights = np.clip(weights, self.min_weight, self.max_weight)
        weights = weights / weights.sum(axis=-1, keepdims=True)

        return weights[:, 0], weights[:, 1]

    def _compute_entropy_weights(self,
                                coedit_scores: np.ndarray,
                                tocsin_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute entropy-based adaptive weights.

        Channels with lower entropy (more certain predictions) get higher weights.

        Args:
            coedit_scores: CoEdIT channel scores
            tocsin_scores: TOCSIN channel scores

        Returns:
            Tuple of (coedit_weights, tocsin_weights)
        """
        # Compute entropy for each channel
        def entropy(scores):
            # P(positive) = score, P(negative) = 1 - score
            p_pos = np.clip(scores, 1e-8, 1 - 1e-8)
            p_neg = 1 - p_pos
            ent = -(p_pos * np.log(p_pos) + p_neg * np.log(p_neg))
            return ent

        coedit_entropy = entropy(coedit_scores)
        tocsin_entropy = entropy(tocsin_scores)

        # Inverse relationship: lower entropy -> higher weight
        coedit_inv_ent = 1 / (coedit_entropy + 1e-8)
        tocsin_inv_ent = 1 / (tocsin_entropy + 1e-8)

        total = coedit_inv_ent + tocsin_inv_ent
        coedit_weight = coedit_inv_ent / total
        tocsin_weight = tocsin_inv_ent / total

        # Apply temperature and clip
        weights = np.stack([coedit_weight, tocsin_weight], axis=-1)
        weights = weights / self.temperature
        weights = np.exp(weights) / np.exp(weights).sum(axis=-1, keepdims=True)
        weights = np.clip(weights, self.min_weight, self.max_weight)
        weights = weights / weights.sum(axis=-1, keepdims=True)

        return weights[:, 0], weights[:, 1]

    def _compute_hybrid_weights(self,
                               coedit_scores: np.ndarray,
                               tocsin_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute hybrid weights combining confidence and entropy.

        Args:
            coedit_scores: CoEdIT channel scores
            tocsin_scores: TOCSIN channel scores

        Returns:
            Tuple of (coedit_weights, tocsin_weights)
        """
        # Get confidence weights
        conf_coedit, conf_tocsin = self._compute_confidence_weights(coedit_scores, tocsin_scores)

        # Get entropy weights
        ent_coedit, ent_tocsin = self._compute_entropy_weights(coedit_scores, tocsin_scores)

        # Average the two weighting schemes
        coedit_weight = (conf_coedit + ent_coedit) / 2
        tocsin_weight = (conf_tocsin + ent_tocsin) / 2

        # Normalize
        total = coedit_weight + tocsin_weight
        coedit_weight = coedit_weight / total
        tocsin_weight = tocsin_weight / total

        return coedit_weight, tocsin_weight

    def _compute_learned_attention_weights(self,
                                         coedit_scores: np.ndarray,
                                         tocsin_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute learned attention weights using neural network.

        Args:
            coedit_scores: CoEdIT channel scores
            tocsin_scores: TOCSIN channel scores

        Returns:
            Tuple of (coedit_weights, tocsin_weights)
        """
        if self.attention_module is None:
            # Fall back to confidence-based if no trained model
            print("Warning: No trained attention model found, using confidence-based weights")
            return self._compute_confidence_weights(coedit_scores, tocsin_scores)

        # Prepare input tensor
        scores = np.stack([coedit_scores, tocsin_scores], axis=1)  # (n_samples, 2)

        # Convert to tensor
        scores_tensor = torch.FloatTensor(scores).to(self.device)

        # Get attention weights
        with torch.no_grad():
            fusion_weights, _ = self.attention_module(scores_tensor)

        # Convert back to numpy
        fusion_weights = fusion_weights.cpu().numpy()

        # Clip to bounds
        fusion_weights = np.clip(fusion_weights, self.min_weight, self.max_weight)
        fusion_weights = fusion_weights / fusion_weights.sum(axis=-1, keepdims=True)

        return fusion_weights[:, 0], fusion_weights[:, 1]

    def normalize_and_fuse(self,
                          human_channel_scores: Dict[str, np.ndarray],
                          llm_channel_scores: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize scores and apply dynamic fusion.

        This is the main interface for evaluation.

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

            # MinMax normalize combined scores
            min_val = all_scores.min()
            max_val = all_scores.max()
            if max_val > min_val:
                norm_scores = (all_scores - min_val) / (max_val - min_val)
            else:
                norm_scores = np.ones_like(all_scores) * 0.5

            # Split back
            n_human = len(human_channel_scores[channel])
            normalized_human[channel] = norm_scores[:n_human]
            normalized_llm[channel] = norm_scores[n_human:]

        # Get normalized scores
        human_coedit = normalized_human.get('coedit', normalized_human.get('coedit', np.zeros(len(normalized_human[list(normalized_human.keys())[0]]))))
        human_tocsin = normalized_human.get('tocsin', np.zeros(len(human_coedit)))
        llm_coedit = normalized_llm.get('coedit', np.zeros(len(normalized_llm[list(normalized_llm.keys())[0]])))
        llm_tocsin = normalized_llm.get('tocsin', np.zeros(len(llm_coedit)))

        # Compute dynamic weights
        if self.mode == 'confidence':
            human_coedit_w, human_tocsin_w = self._compute_confidence_weights(human_coedit, human_tocsin)
            llm_coedit_w, llm_tocsin_w = self._compute_confidence_weights(llm_coedit, llm_tocsin)
        elif self.mode == 'entropy':
            human_coedit_w, human_tocsin_w = self._compute_entropy_weights(human_coedit, human_tocsin)
            llm_coedit_w, llm_tocsin_w = self._compute_entropy_weights(llm_coedit, llm_tocsin)
        elif self.mode == 'hybrid':
            human_coedit_w, human_tocsin_w = self._compute_hybrid_weights(human_coedit, human_tocsin)
            llm_coedit_w, llm_tocsin_w = self._compute_hybrid_weights(llm_coedit, llm_tocsin)
        else:  # attention
            human_coedit_w, human_tocsin_w = self._compute_learned_attention_weights(human_coedit, human_tocsin)
            llm_coedit_w, llm_tocsin_w = self._compute_learned_attention_weights(llm_coedit, llm_tocsin)

        # Apply dynamic weights
        human_fused = human_coedit_w * human_coedit + human_tocsin_w * human_tocsin
        llm_fused = llm_coedit_w * llm_coedit + llm_tocsin_w * llm_tocsin

        return human_fused, llm_fused

    def fuse(self, channel_scores: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Fuse channel scores using dynamic weighting.

        Args:
            channel_scores: Dictionary of channel name -> scores array

        Returns:
            Fused scores array
        """
        coedit_scores = channel_scores.get('coedit', channel_scores.get('coedit', np.zeros(len(list(channel_scores.values())[0]))))
        tocsin_scores = channel_scores.get('tocsin', np.zeros(len(coedit_scores)))

        # Compute dynamic weights
        if self.mode == 'confidence':
            coedit_w, tocsin_w = self._compute_confidence_weights(coedit_scores, tocsin_scores)
        elif self.mode == 'entropy':
            coedit_w, tocsin_w = self._compute_entropy_weights(coedit_scores, tocsin_scores)
        elif self.mode == 'hybrid':
            coedit_w, tocsin_w = self._compute_hybrid_weights(coedit_scores, tocsin_scores)
        else:  # attention
            coedit_w, tocsin_w = self._compute_learned_attention_weights(coedit_scores, tocsin_scores)

        # Apply weights
        fused = coedit_w * coedit_scores + tocsin_w * tocsin_scores

        return fused

    def get_weights(self, coedit_scores: np.ndarray, tocsin_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the dynamic weights for a given pair of score arrays.

        Useful for analysis and visualization.

        Args:
            coedit_scores: CoEdIT channel scores
            tocsin_scores: TOCSIN channel scores

        Returns:
            Tuple of (coedit_weights, tocsin_weights) arrays
        """
        if self.mode == 'confidence':
            return self._compute_confidence_weights(coedit_scores, tocsin_scores)
        elif self.mode == 'entropy':
            return self._compute_entropy_weights(coedit_scores, tocsin_scores)
        elif self.mode == 'hybrid':
            return self._compute_hybrid_weights(coedit_scores, tocsin_scores)
        else:
            return self._compute_learned_attention_weights(coedit_scores, tocsin_scores)

    def __repr__(self) -> str:
        return f"DynamicAttentionFusion(mode={self.mode}, temp={self.temperature})"


# Convenience function for testing
def test_dynamic_fusion():
    """Test dynamic fusion with synthetic data."""
    print("Testing Dynamic Attention Fusion")

    # Create synthetic scores
    np.random.seed(42)
    n_samples = 100

    # Human scores: lower values
    human_coedit = np.random.beta(2, 5, n_samples) * 0.4
    human_tocsin = np.random.beta(2, 5, n_samples) * 0.4

    # LLM scores: higher values
    llm_coedit = np.random.beta(5, 2, n_samples) * 0.4 + 0.6
    llm_tocsin = np.random.beta(5, 2, n_samples) * 0.4 + 0.6

    # Test each mode
    for mode in ['confidence', 'entropy', 'hybrid']:
        print(f"\n--- Testing mode: {mode} ---")

        fusion = DynamicAttentionFusion(mode=mode)

        human_fused, llm_fused = fusion.normalize_and_fuse(
            {'coedit': human_coedit, 'tocsin': human_tocsin},
            {'coedit': llm_coedit, 'tocsin': llm_tocsin}
        )

        print(f"Human fused: mean={human_fused.mean():.4f}, std={human_fused.std():.4f}")
        print(f"LLM fused:   mean={llm_fused.mean():.4f}, std={llm_fused.std():.4f}")
        print(f"Separation:  {llm_fused.mean() - human_fused.mean():.4f}")

        # Get sample weights
        coedit_w, tocsin_w = fusion.get_weights(
            np.concatenate([human_coedit, llm_coedit]),
            np.concatenate([human_tocsin, llm_tocsin])
        )
        print(f"Weight range - CoEdIT: [{coedit_w.min():.3f}, {coedit_w.max():.3f}]")
        print(f"Weight range - TOCSIN: [{tocsin_w.min():.3f}, {tocsin_w.max():.3f}]")


if __name__ == '__main__':
    test_dynamic_fusion()
