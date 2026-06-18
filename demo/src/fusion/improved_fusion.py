"""
改进的融合策略 - 修复版
目标：ROC AUC 0.95+
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class ImprovedFusion:
    """
    改进的融合策略 - 修复版

    核心改进：
    1. 自适应权重：根据通道区分度自动调整权重
    2. 正确的归一化：与WeightedFusion相同的归一化逻辑
    """

    def __init__(self, weight_method: str = 'adaptive'):
        """
        初始化改进融合

        Args:
            weight_method: 权重方法
                - 'adaptive': 根据区分度自适应
                - 'equal': 等权重
                - 'optimized': 使用优化的固定权重
        """
        self.weight_method = weight_method
        self.is_fitted = False
        self.adaptive_weights = None

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """归一化分数到[0,1]范围"""
        min_val = np.min(scores)
        max_val = np.max(scores)

        if max_val == min_val:
            return np.ones_like(scores) * 0.5
        return (scores - min_val) / (max_val - min_val)

    def _compute_adaptive_weights(self,
                                  human_scores: Dict[str, np.ndarray],
                                  llm_scores: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        根据通道区分度计算自适应权重

        区分度 = |human_mean - llm_mean| / (human_std + llm_std)
        """
        separability = {}

        for channel in human_scores.keys():
            human_mean = human_scores[channel].mean()
            llm_mean = llm_scores[channel].mean()
            human_std = human_scores[channel].std()
            llm_std = llm_scores[channel].std()

            # 区分度：均值差异除以标准差之和
            diff = abs(human_mean - llm_mean)
            total_std = human_std + llm_std

            if total_std > 0:
                sep = diff / total_std
            else:
                sep = 0

            separability[channel] = sep

        # 根据区分度分配权重
        total_sep = sum(separability.values())

        if total_sep > 0:
            weights = {k: v / total_sep for k, v in separability.items()}
        else:
            # 等权重
            weights = {k: 1.0 / len(separability) for k in separability.keys()}

        return weights

    def fit(self, human_channel_scores: Dict[str, np.ndarray],
            llm_channel_scores: Dict[str, np.ndarray]) -> 'ImprovedFusion':
        """
        拟合融合器（学习自适应权重）

        Args:
            human_channel_scores: 人类文本的通道分数
            llm_channel_scores: LLM文本的通道分数

        Returns:
            self
        """
        if self.weight_method == 'adaptive':
            # 在归一化后的分数上计算区分度
            norm_human = {}
            norm_llm = {}

            for channel in human_channel_scores.keys():
                # 合并所有分数进行归一化
                all_scores = np.concatenate([human_channel_scores[channel],
                                           llm_channel_scores[channel]])
                norm_all = self._normalize_scores(all_scores)

                # 分离
                n_human = len(human_channel_scores[channel])
                norm_human[channel] = norm_all[:n_human]
                norm_llm[channel] = norm_all[n_human:]

            self.adaptive_weights = self._compute_adaptive_weights(norm_human, norm_llm)

        self.is_fitted = True
        return self

    def fuse(self, channel_scores: Dict[str, np.ndarray]) -> np.ndarray:
        """
        融合通道分数

        Args:
            channel_scores: 各通道的原始分数
                           IMPORTANT: 所有分数应该是 LLM-oriented (high = LLM)

        Returns:
            融合后的分数 (higher = more likely LLM-generated)
        """
        n_samples = len(next(iter(channel_scores.values())))
        fused = np.zeros(n_samples)

        # 确定权重
        if self.weight_method == 'adaptive' and self.adaptive_weights is not None:
            weights = self.adaptive_weights
        elif self.weight_method == 'equal':
            weights = {k: 1.0 / len(channel_scores) for k in channel_scores.keys()}
        else:  # optimized
            # 使用优化的权重（根据经验）
            weights = {'coedit': 0.6, 'tocsin': 0.4}

        # 加权融合
        # 直接使用传入的分数，假设已经是 LLM-oriented (high = LLM)
        for channel, scores in channel_scores.items():
            fused += weights.get(channel, 0.5) * scores

        return fused

    def normalize_and_fuse(self,
                           human_channel_scores: Dict[str, np.ndarray],
                           llm_channel_scores: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        归一化并融合（与WeightedFusion兼容的逻辑）

        首先在所有样本上归一化，然后融合

        Args:
            human_channel_scores: 人类文本的通道分数
            llm_channel_scores: LLM文本的通道分数

        Returns:
            (human_fused, llm_fused) 融合后的分数
        """
        # 首先拟合（学习权重）
        self.fit(human_channel_scores, llm_channel_scores)

        # 对每个通道分别归一化（与WeightedFusion相同的方法）
        normalized_human = {}
        normalized_llm = {}

        for channel in human_channel_scores.keys():
            # 合并所有分数进行归一化
            all_scores = np.concatenate([human_channel_scores[channel],
                                       llm_channel_scores[channel]])

            norm_scores = self._normalize_scores(all_scores)

            # 分离回人类和LLM
            n_human = len(human_channel_scores[channel])
            normalized_human[channel] = norm_scores[:n_human]
            normalized_llm[channel] = norm_scores[n_human:]

        # 融合归一化后的分数
        human_fused = self.fuse(normalized_human)
        llm_fused = self.fuse(normalized_llm)

        return human_fused, llm_fused

    def get_params(self) -> Dict:
        """获取当前参数"""
        params = {
            'weight_method': self.weight_method,
            'is_fitted': self.is_fitted
        }

        if self.adaptive_weights is not None:
            params['adaptive_weights'] = self.adaptive_weights

        return params

    def __repr__(self) -> str:
        return (f"ImprovedFusion(weight_method={self.weight_method}, "
                f"fitted={self.is_fitted})")
