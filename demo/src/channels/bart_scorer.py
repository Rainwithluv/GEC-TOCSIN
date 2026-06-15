"""
BART Score implementation for semantic difference measurement.
Based on the BARTScore paper and TOCSIN implementation.
"""

import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration
from typing import List, Union
import numpy as np


class BARTScorer:
    """
    BART-based scorer for measuring semantic similarity between texts.
    """

    def __init__(self, device: str = 'cuda', max_length: int = 1024,
                 checkpoint: str = 'facebook/bart-base'):
        """
        Initialize BART Scorer.

        Args:
            device: Device to run model on
            max_length: Maximum sequence length
            checkpoint: Hugging Face model checkpoint
        """
        self.device = device
        self.max_length = max_length

        # Load model and tokenizer
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Loss function for scoring
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def score(self, srcs: List[str], tgts: List[str], batch_size: int = 4) -> List[float]:
        """
        Score a batch of (source, target) pairs.

        Args:
            srcs: Source texts list
            tgts: Target texts list
            batch_size: Batch size for processing

        Returns:
            List of scores (lower is better, negative log-likelihood)
        """
        score_list = []

        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i:i + batch_size]
            tgt_list = tgts[i:i + batch_size]

            try:
                with torch.no_grad():
                    # Encode source
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )

                    # Encode target
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )

                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    # Forward pass
                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )

                    # Calculate loss
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len

                    # Get negative scores (lower is better for BART)
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError as e:
                print(f"Error in BART scoring: {e}")
                print(f"Source: {src_list}")
                print(f"Target: {tgt_list}")
                raise

        return score_list

    def similarity_score(self, text1: str, text2: str) -> float:
        """
        Calculate similarity score between two texts.
        Higher score means more similar.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score
        """
        scores = self.score([text1], [text2], batch_size=1)
        # Convert to positive similarity score
        return -scores[0]
