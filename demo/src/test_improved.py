"""
测试ImprovedFusion效果
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from demo.src.fusion.improved_fusion import ImprovedFusion
    from demo.src.fusion.weighted_fusion import WeightedFusion
    from demo.src.utils.metrics import get_roc_metrics
    print("✓ Imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)


def test_improved_fusion():
    """测试ImprovedFusion"""
    print("\n=== 测试ImprovedFusion ===\n")

    # 创建模拟数据 - LLM分数更高
    np.random.seed(42)
    n_samples = 50

    # 人类文本：分数较低
    human_coedit = np.random.uniform(0.3, 0.6, n_samples)
    human_tocsin = np.random.uniform(0.4, 0.7, n_samples)

    # LLM文本：分数较高
    llm_coedit = np.random.uniform(0.5, 0.9, n_samples)
    llm_tocsin = np.random.uniform(0.55, 0.95, n_samples)

    print(f"Generated {n_samples} human and {n_samples} LLM samples")
    print(f"\nHuman scores:")
    print(f"  CoEdIT: {human_coedit.mean():.4f} ± {human_coedit.std():.4f}")
    print(f"  TOCSIN: {human_tocsin.mean():.4f} ± {human_tocsin.std():.4f}")
    print(f"\nLLM scores:")
    print(f"  CoEdIT: {llm_coedit.mean():.4f} ± {llm_coedit.std():.4f}")
    print(f"  TOCSIN: {llm_tocsin.mean():.4f} ± {llm_tocsin.std():.4f}")

    # 准备数据
    human_scores = {'coedit': human_coedit, 'tocsin': human_tocsin}
    llm_scores = {'coedit': llm_coedit, 'tocsin': llm_tocsin}

    # 测试不同的配置
    configs = [
        ('weighted', WeightedFusion(weights={'coedit': 0.5, 'tocsin': 0.5})),
        ('improved_adaptive', ImprovedFusion(weight_method='adaptive')),
        ('improved_equal', ImprovedFusion(weight_method='equal')),
        ('improved_optimized', ImprovedFusion(weight_method='optimized')),
    ]

    print(f"\n{'='*60}")
    print("测试不同融合方法")
    print(f"{'='*60}")

    for name, fusion in configs:
        if isinstance(fusion, ImprovedFusion):
            fusion.fit(human_scores, llm_scores)
            human_fused, llm_fused = fusion.normalize_and_fuse(human_scores, llm_scores)
        else:
            human_fused, llm_fused = fusion.normalize_and_fuse(human_scores, llm_scores)

        roc_auc, opt_threshold, _, _, _, _, _ = get_roc_metrics(
            human_fused.tolist(), llm_fused.tolist()
        )

        print(f"\n{name}:")
        print(f"  Fused: Human={human_fused.mean():.4f}, LLM={llm_fused.mean():.4f}, Diff={llm_fused.mean() - human_fused.mean():.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")

        if isinstance(fusion, ImprovedFusion):
            params = fusion.get_params()
            if 'adaptive_weights' in params:
                print(f"  自适应权重: {params['adaptive_weights']}")

    print(f"\n{'='*60}")
    print("测试完成")
    print(f"{'='*60}")


if __name__ == '__main__':
    test_improved_fusion()
