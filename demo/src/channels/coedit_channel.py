"""
CoEdIT Channel: Grammar correction and Rouge scoring.
Based on GECScore methodology.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
from rouge import Rouge

from ..models.model_loader import ModelLoader, get_device


class CoEdITChannel:
    """
    Grammar correction channel using CoEdIT model.
    Measures grammatical standardness via GEC + Rouge scoring.
    """

    def __init__(self,
                 model_name: str = "grammarly/coedit-large",
                 device: Optional[str] = None):
        """
        Initialize CoEdIT channel.

        Args:
            model_name: Hugging Face model name
            device: Device to use (auto-detect if None)
        """
        if device is None:
            device = get_device()

        self.device = device
        self.model_name = model_name

        # Load CoEdIT model
        self.tokenizer, self.model = ModelLoader.load_coedit(model_name, device)

        # Initialize Rouge scorer
        self.rouge = Rouge()

    def correct_grammar(self, text: str, instruction: str = None) -> str:
        """
        Correct grammar errors in text using CoEdIT.

        Args:
            text: Input text
            instruction: Custom instruction (default: GEC instruction)

        Returns:
            Grammar-corrected text
        """
        if instruction is None:
            instruction = f"Fix grammatical errors in this sentence: {text}"

        inputs = self.tokenizer(
            instruction,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=512)

        corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected

    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Extract grammatical standardness features from text.

        Args:
            text: Input text

        Returns:
            Dictionary of features including Rouge scores
        """
        # Grammar correction
        gec_text = self.correct_grammar(text)

        # Calculate Rouge scores
        rouge_scores = self.rouge.get_scores(text, gec_text, avg=True)

        features = {
            'rouge_1_f': rouge_scores['rouge-1']['f'],
            'rouge_2_f': rouge_scores['rouge-2']['f'],
            'rouge_l_f': rouge_scores['rouge-l']['f'],
            'rouge_1_p': rouge_scores['rouge-1']['p'],
            'rouge_2_p': rouge_scores['rouge-2']['p'],
            'rouge_l_p': rouge_scores['rouge-l']['p'],
            'rouge_1_r': rouge_scores['rouge-1']['r'],
            'rouge_2_r': rouge_scores['rouge-2']['r'],
            'rouge_l_r': rouge_scores['rouge-l']['r'],
            'text_length': len(text.split()),
            'gec_length': len(gec_text.split()),
            'gec_length_change': len(gec_text.split()) - len(text.split()),
            'gec_text': gec_text  # Store corrected text
        }

        return features

    def score_texts(self, texts: List[str], show_progress: bool = True, for_llm: bool = True) -> List[float]:
        """
        Score a list of texts using Rouge-2 F1 score.

        Important: CoEdIT shows inconsistent behavior:
        - 3 samples: Human > LLM (wrong direction for LLM detection)
        - 20 samples: LLM > Human (correct direction)

        To ensure reliability, we INVERT CoEdIT scores:
        - High ROUGE → Low LLM score (more likely human)
        - Low ROUGE → High LLM score (more likely LLM)

        Args:
            texts: List of input texts
            show_progress: Show progress bar
            for_llm: Parameter kept for compatibility

        Returns:
            List of inverted ROUGE-2 F1 scores (higher = more likely LLM-generated)
        """
        scores = []

        iterator = tqdm(texts, desc="CoEdIT scoring") if show_progress else texts

        for text in iterator:
            features = self.extract_features(text)
            rouge_score = features['rouge_2_f']

            # INVERT CoEdIT scores for reliable LLM detection
            # Original: High ROUGE = Human-like
            # Inverted: High score = LLM-like
            llm_score = 1 - rouge_score
            scores.append(llm_score)

        return scores

    def score_texts_multi_feature(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Score texts using multiple ROUGE features.

        Returns a feature vector for each text combining:
        - rouge_1_f, rouge_2_f, rouge_l_f (F1 scores)
        - rouge_1_p, rouge_2_p, rouge_l_p (Precision scores)
        - rouge_1_r, rouge_2_r, rouge_l_r (Recall scores)
        - text_length, gec_length_change

        Args:
            texts: List of input texts
            show_progress: Show progress bar

        Returns:
            Array of shape (n_texts, n_features) with multi-feature scores
        """
        features_list = []

        iterator = tqdm(texts, desc="CoEdIT multi-feature scoring") if show_progress else texts

        for text in iterator:
            features = self.extract_features(text)
            feature_vector = [
                features['rouge_1_f'],
                features['rouge_2_f'],
                features['rouge_l_f'],
                features['rouge_1_p'],
                features['rouge_2_p'],
                features['rouge_l_p'],
                features['rouge_1_r'],
                features['rouge_2_r'],
                features['rouge_l_r'],
                features['text_length'] / 100.0,  # Normalize by dividing by 100
                features['gec_length_change'] / 50.0  # Normalize
            ]
            features_list.append(feature_vector)

        return np.array(features_list)

    def get_combined_score(self, texts: List[str], show_progress: bool = True,
                          weights: np.ndarray = None) -> List[float]:
        """
        Get a single combined score from multiple features.

        Args:
            texts: List of input texts
            show_progress: Show progress bar
            weights: Optional weights for features (default: equal weights)

        Returns:
            List of combined scores (higher = more likely LLM-generated)
        """
        multi_features = self.score_texts_multi_feature(texts, show_progress)

        if weights is None:
            # Default weights: emphasize rouge_2_f and rouge_l_f
            weights = np.array([0.15, 0.30, 0.25, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0, 0.0])

        # Normalize weights
        weights = weights / weights.sum()

        # Weighted combination
        combined_scores = multi_features @ weights

        return combined_scores.tolist()

    def score_text(self, text: str) -> float:
        """
        Score a single text.

        Args:
            text: Input text

        Returns:
            Score (higher = more likely LLM-generated)
        """
        features = self.extract_features(text)
        return features['rouge_2_f']

    def get_gec_score(self, text: str) -> Dict[str, float]:
        """
        Get GEC score (inverse of Rouge score for interpretability).

        Args:
            text: Input text

        Returns:
            Dictionary with GEC-related scores
        """
        features = self.extract_features(text)

        return {
            'gec_score': features['rouge_2_f'],  # Main score
            'llm_probability': features['rouge_2_f'],  # Higher for LLM text
            'grammar_error_score': 1 - features['rouge_2_f'],  # Higher for human text
            'all_features': features
        }
