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

    def score_texts(self, texts: List[str], show_progress: bool = True) -> List[float]:
        """
        Score a list of texts using Rouge-2 F1 score.

        Args:
            texts: List of input texts
            show_progress: Show progress bar

        Returns:
            List of scores (higher = more likely LLM-generated)
        """
        scores = []

        iterator = tqdm(texts, desc="CoEdIT scoring") if show_progress else texts

        for text in iterator:
            features = self.extract_features(text)
            # Use Rouge-2 F1 as the main score
            scores.append(features['rouge_2_f'])

        return scores

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
