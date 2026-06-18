"""
软投票融合策略 (Soft Voting Fusion)
用于MultiFusion-Detector，提升ROC AUC性能

核心思想：
1. 将通道分数转换为概率
2. 根据置信度动态调整权重
3. 使用软投票融合最终预测
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.special import expit as sigmoid
from sklearn.preprocessing import MinMaxScaler


class VotingFusion:
    """
    软投票融合策略

    相比传统加权融合的优势：
    1. 自适应权重：根据预测置信度动态调整
    2. 概率融合：使用概率而非原始分数
    3. 置信度感知：高置信度预测获得更大权重
    """

    def __init__(self,
                 voting_method: str = 'soft',
                 confidence_method: str = 'distance',
                 calibration: bool = True):
        """
        初始化软投票融合

        Args:
            voting_method: 投票方法
                - 'soft': 软投票（概率平均）
                - 'confidence': 置信度加权投票
                - 'bayesian': 贝叶斯融合
            confidence_method: 置信度计算方法
                - 'distance': 基于与阈值的距离
                - 'entropy': 基于信息熵
                - 'variance': 基于分数方差
            calibration: 是否进行概率校准
        """
        self.voting_method = voting_method
        self.confidence_method = confidence_method
        self.calibration = calibration

        # 用于归一化的scaler
        self.coedit_scaler = MinMaxScaler()
        self.tocsin_scaler = MinMaxScaler()

        # 存储校准参数
        self.is_fitted = False
        self.calibration_params = {}

    def _normalize_scores(self, scores: np.ndarray, channel: str) -> np.ndarray:
        """
        归一化分数到[0,1]范围

        Args:
            scores: 原始分数
            channel: 通道名称 ('coedit' 或 'tocsin')

        Returns:
            归一化后的分数
        """
        if self.is_fitted:
            # 使用已保存的scaler
            if channel == 'coedit':
                # 手动归一化
                min_val, max_val = self.calibration_params['coedit_min'], self.calibration_params['coedit_max']
                if max_val > min_val:
                    normalized = (scores - min_val) / (max_val - min_val)
                else:
                    normalized = np.ones_like(scores) * 0.5
            else:  # tocsin
                min_val, max_val = self.calibration_params['tocsin_min'], self.calibration_params['tocsin_max']
                if max_val > min_val:
                    normalized = (scores - min_val) / (max_val - min_val)
                else:
                    normalized = np.ones_like(scores) * 0.5
        else:
            # 简单归一化
            min_val, max_val = scores.min(), scores.max()
            if max_val > min_val:
                normalized = (scores - min_val) / (max_val - min_val)
            else:
                normalized = np.ones_like(scores) * 0.5

        return np.clip(normalized, 0, 1)

    def _to_probability(self, normalized_scores: np.ndarray) -> np.ndarray:
        """
        将归一化分数转换为LLM概率

        重要：实际数据中，高分数 = 更像LLM，低分数 = 更像人类
        这是基于实际测试结果，与理论预期相反

        Args:
            normalized_scores: 归一化分数 [0, 1]

        Returns:
            LLM概率 [0, 1], 高值表示更可能是LLM生成
        """
        # 直接使用归一化分数作为LLM概率
        # 高分数 = 更像LLM
        return normalized_scores

    def _compute_confidence(self, probabilities: np.ndarray, method: str = None) -> np.ndarray:
        """
        计算预测置信度

        Args:
            probabilities: LLM概率
            method: 置信度计算方法

        Returns:
            置信度分数 [0, 1]
        """
        if method is None:
            method = self.confidence_method

        if method == 'distance':
            # 基于与0.5的距离：越远离0.5，置信度越高
            confidence = 2 * np.abs(probabilities - 0.5)

        elif method == 'entropy':
            # 基于信息熵：熵越低，置信度越高
            # H(p) = -p*log(p) - (1-p)*log(1-p)
            eps = 1e-10
            entropy = -probabilities * np.log(probabilities + eps) - \
                     (1 - probabilities) * np.log(1 - probabilities + eps)
            # 最大熵为log(2)，归一化到[0,1]
            max_entropy = np.log(2)
            confidence = 1 - entropy / max_entropy

        elif method == 'variance':
            # 基于方差：对于伯努利分布，var = p*(1-p)
            # 方差越小，置信度越高
            variance = probabilities * (1 - probabilities)
            max_variance = 0.25  # p=0.5时最大
            confidence = 1 - variance / max_variance

        else:
            # 默认使用distance
            confidence = 2 * np.abs(probabilities - 0.5)

        return np.clip(confidence, 0, 1)

    def _soft_voting(self, proba_dict: Dict[str, np.ndarray],
                    weights_dict: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """
        软投票融合

        Args:
            proba_dict: 各通道的LLM概率 {'coedit': [...], 'tocsin': [...]}
            weights_dict: 各通道的置信度权重（可选）

        Returns:
            融合后的LLM概率
        """
        channels = list(proba_dict.keys())
        n_samples = len(proba_dict[channels[0]])

        fused_proba = np.zeros(n_samples)

        if weights_dict is None:
            # 等权重平均
            for channel in channels:
                fused_proba += proba_dict[channel]
            fused_proba /= len(channels)
        else:
            # 置信度加权
            total_weight = np.zeros(n_samples)

            for channel in channels:
                weight = weights_dict[channel]
                fused_proba += weight * proba_dict[channel]
                total_weight += weight

            # 避免除零
            total_weight = np.where(total_weight > 0, total_weight, 1)
            fused_proba /= total_weight

        return fused_proba

    def _bayesian_fusion(self, proba_dict: Dict[str, np.ndarray],
                        conf_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        贝叶斯融合

        将每个通道视为一个"专家"，根据其置信度进行贝叶斯更新

        Args:
            proba_dict: 各通道的LLM概率
            conf_dict: 各通道的置信度

        Returns:
            融合后的LLM概率
        """
        channels = list(proba_dict.keys())
        n_samples = len(proba_dict[channels[0]])

        # 初始化对数几率 (log odds)
        log_odds = np.zeros(n_samples)

        for channel in channels:
            # p = odds / (1 + odds) => odds = p / (1 - p)
            eps = 1e-10
            odds = proba_dict[channel] / (1 - proba_dict[channel] + eps)

            # 对数几率
            channel_log_odds = np.log(odds + eps)

            # 根据置信度加权
            confidence = conf_dict[channel]
            log_odds += confidence * channel_log_odds

        # 转换回概率
        fused_proba = 1 / (1 + np.exp(-log_odds))

        return fused_proba

    def fit(self, human_channel_scores: Dict[str, np.ndarray],
            llm_channel_scores: Dict[str, np.ndarray]) -> 'VotingFusion':
        """
        拟合融合器（学习归一化参数）

        Args:
            human_channel_scores: 人类文本的通道分数
            llm_channel_scores: LLM文本的通道分数

        Returns:
            self
        """
        # 合并人类和LLM分数来学习归一化参数
        for channel in human_channel_scores.keys():
            all_scores = np.concatenate([human_channel_scores[channel],
                                       llm_channel_scores[channel]])

            min_val = float(all_scores.min())
            max_val = float(all_scores.max())

            self.calibration_params[f'{channel}_min'] = min_val
            self.calibration_params[f'{channel}_max'] = max_val

        self.is_fitted = True
        return self

    def fuse(self, channel_scores: Dict[str, np.ndarray],
            return_proba: bool = False) -> np.ndarray:
        """
        融合通道分数

        Args:
            channel_scores: 各通道的原始分数
            return_proba: 是否返回概率而非分数

        Returns:
            融合后的分数/概率
        """
        # 转换为概率
        proba_dict = {}
        conf_dict = {}

        for channel, scores in channel_scores.items():
            # 归一化
            normalized = self._normalize_scores(scores, channel)

            # 转换为LLM概率
            proba = self._to_probability(normalized)
            proba_dict[channel] = proba

            # 计算置信度
            conf_dict[channel] = self._compute_confidence(proba)

        # 根据方法融合
        if self.voting_method == 'soft':
            fused = self._soft_voting(proba_dict)
        elif self.voting_method == 'confidence':
            fused = self._soft_voting(proba_dict, weights_dict=conf_dict)
        elif self.voting_method == 'bayesian':
            fused = self._bayesian_fusion(proba_dict, conf_dict)
        else:
            # 默认软投票
            fused = self._soft_voting(proba_dict)

        if return_proba:
            return fused
        else:
            # 转换回"人类概率"（高分数=更像人类）以保持与原系统一致
            # 但用户期望高分数=LLM，所以保持LLM概率
            return fused

    def normalize_and_fuse(self,
                           human_channel_scores: Dict[str, np.ndarray],
                           llm_channel_scores: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        归一化并融合（与原接口兼容）

        首先在所有样本上拟合归一化参数，然后融合

        Args:
            human_channel_scores: 人类文本的通道分数
            llm_channel_scores: LLM文本的通道分数

        Returns:
            (human_fused, llm_fused) 融合后的分数
        """
        # 首先拟合（学习归一化参数）
        self.fit(human_channel_scores, llm_channel_scores)

        # 融合
        human_fused = self.fuse(human_channel_scores)
        llm_fused = self.fuse(llm_channel_scores)

        return human_fused, llm_fused

    def set_params(self, voting_method: str = None, confidence_method: str = None):
        """更新参数"""
        if voting_method is not None:
            self.voting_method = voting_method
        if confidence_method is not None:
            self.confidence_method = confidence_method

    def get_params(self) -> Dict:
        """获取当前参数"""
        return {
            'voting_method': self.voting_method,
            'confidence_method': self.confidence_method,
            'calibration': self.calibration,
            'is_fitted': self.is_fitted
        }

    def __repr__(self) -> str:
        return (f"VotingFusion(method={self.voting_method}, "
                f"confidence={self.confidence_method}, "
                f"calibrated={self.calibration})")
