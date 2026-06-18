"""
测试软投票融合效果
快速验证VotingFusion是否正常工作
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from demo.src.fusion.voting_fusion import VotingFusion
    from demo.src.utils.metrics import get_roc_metrics
    print("✓ Imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

import numpy as np


def test_voting_fusion():
    """测试VotingFusion的基本功能"""
    print("\n=== Testing VotingFusion ===\n")

    # 创建模拟数据
    np.random.seed(42)
    n_samples = 50

    # 模拟人类和LLM文本的通道分数
    # 实际数据：LLM文本分数较高，人类文本分数较低

    # 人类文本：分数较低
    human_coedit = np.random.uniform(0.3, 0.6, n_samples)
    human_tocsin = np.random.uniform(0.4, 0.7, n_samples)

    # LLM文本：分数较高（更有区分度）
    llm_coedit = np.random.uniform(0.5, 0.9, n_samples)
    llm_tocsin = np.random.uniform(0.55, 0.95, n_samples)

    print(f"Generated {n_samples} human and {n_samples} LLM samples")
    print(f"\nHuman scores:")
    print(f"  CoEdIT: {human_coedit.mean():.4f} ± {human_coedit.std():.4f}")
    print(f"  TOCSIN: {human_tocsin.mean():.4f} ± {human_tocsin.std():.4f}")
    print(f"\nLLM scores:")
    print(f"  CoEdIT: {llm_coedit.mean():.4f} ± {llm_coedit.std():.4f}")
    print(f"  TOCSIN: {llm_tocsin.mean():.4f} ± {llm_tocsin.std():.4f}")

    # 测试不同的投票方法
    methods = ['soft', 'confidence', 'bayesian']

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Testing method: {method}")
        print(f"{'='*60}")

        fusion = VotingFusion(voting_method=method, confidence_method='distance')

        # 准备数据
        human_scores = {'coedit': human_coedit, 'tocsin': human_tocsin}
        llm_scores = {'coedit': llm_coedit, 'tocsin': llm_tocsin}

        # 融合
        human_fused, llm_fused = fusion.normalize_and_fuse(human_scores, llm_scores)

        # 计算ROC AUC
        roc_auc, opt_threshold, _, _, _, _, _ = get_roc_metrics(
            human_fused.tolist(), llm_fused.tolist()
        )

        print(f"Fused scores:")
        print(f"  Human:  {human_fused.mean():.4f} ± {human_fused.std():.4f}")
        print(f"  LLM:    {llm_fused.mean():.4f} ± {llm_fused.std():.4f}")
        print(f"  Diff:   {llm_fused.mean() - human_fused.mean():.4f}")
        print(f"\nROC AUC: {roc_auc:.4f}")
        print(f"Optimal Threshold: {opt_threshold:.4f}")

    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"{'='*60}")


if __name__ == '__main__':
    test_voting_fusion()
