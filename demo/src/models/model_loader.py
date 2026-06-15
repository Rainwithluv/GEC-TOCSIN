"""
Model loading utilities for CoEdIT and BART.
"""

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, BartForConditionalGeneration
from typing import Dict, Optional


class ModelLoader:
    """Load and manage detection models."""

    _models: Dict[str, Dict] = {}

    @classmethod
    def load_coedit(cls, model_name: str = "grammarly/coedit-large",
                    device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Load CoEdIT model for grammar error correction.

        Args:
            model_name: Hugging Face model name
            device: Device to load model on

        Returns:
            Tuple of (tokenizer, model)
        """
        cache_key = f"coedit_{model_name}"

        if cache_key in cls._models:
            return cls._models[cache_key]['tokenizer'], cls._models[cache_key]['model']

        print(f"Loading CoEdIT model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)

        model = model.to(device)
        model.eval()

        cls._models[cache_key] = {'tokenizer': tokenizer, 'model': model}

        print(f"CoEdIT model loaded on {device}")
        return tokenizer, model

    @classmethod
    def load_bart(cls, model_name: str = "facebook/bart-base",
                  device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Load BART model for semantic difference measurement.

        Args:
            model_name: Hugging Face model name
            device: Device to load model on

        Returns:
            BartScorer instance
        """
        cache_key = f"bart_{model_name}"

        if cache_key in cls._models:
            return cls._models[cache_key]['scorer']

        print(f"Loading BART model: {model_name}")

        try:
            from demo.src.channels.bart_scorer import BARTScorer
        except ImportError:
            from channels.bart_scorer import BARTScorer

        scorer = BARTScorer(device=device, checkpoint=model_name)

        cls._models[cache_key] = {'scorer': scorer}

        print(f"BART model loaded on {device}")
        return scorer

    @classmethod
    def unload_model(cls, model_type: str, model_name: str):
        """
        Unload a model to free memory.

        Args:
            model_type: Type of model ('coedit' or 'bart')
            model_name: Name of the model
        """
        cache_key = f"{model_type}_{model_name}"

        if cache_key in cls._models:
            del cls._models[cache_key]
            torch.cuda.empty_cache()
            print(f"Unloaded {model_type}: {model_name}")

    @classmethod
    def clear_cache(cls):
        """Clear all cached models."""
        cls._models.clear()
        torch.cuda.empty_cache()
        print("Cleared all model caches")


def get_device() -> str:
    """
    Get the best available device.

    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
