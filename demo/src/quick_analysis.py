"""
快速分析真实数据分数特征
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from demo.src.channels.coedit_channel import CoEdITChannel
    from demo.src.channels.tocsin_channel import TOCSINChannel
    from demo.src.utils.data_loader import DataLoader
    from demo.src.fusion.weighted_fusion import WeightedFusion
    from demo.src.utils.metrics import get_roc_metrics
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)


def quick_analysis():
    """快速分析真实数据"""
    print("\n=== 快速分析真实数据 ===\n")

    # 加载数据
    print("Loading dataset...")
    loader = DataLoader(base_dir="../")
    data = loader.load_combined_data('xsum', 'gpt-4')

    # 使用少量样本
    n = 10
    human_texts = data['original'][:n]
    llm_texts = data['sampled'][:n]

    print(f"Using {n} human texts and {n} LLM texts\n")

    # 初始化通道
    print("Initializing channels...")
    coedit_channel = CoEdITChannel(device='cpu')
    tocsin_channel = TOCSINChannel(device='cpu')

    # 评分
    print("Scoring...")
    human_coedit = np.array([coedit_channel.score_text(t) for t in human_texts])
    human_tocsin = np.array([tocsin_channel.score_text(t) for t in human_texts])
    llm_coedit = np.array([coedit_channel.score_text(t) for t in llm_texts])
    llm_tocsin = np.array([tocsin_channel.score_text(t) for t in llm_texts])

    print(f"\n{'='*60}")
    print("原始分数")
    print(f"{'='*60}")

    print(f"\nCoEdIT:")
    print(f"  Human:  min={human_coedit.min():.4f}, max={human_coedit.max():.4f}, mean={human_coedit.mean():.4f}")
    print(f"  LLM:    min={llm_coedit.min():.4f}, max={llm_coedit.max():.4f}, mean={llm_coedit.mean():.4f}")
    print(f"  Diff:   {llm_coedit.mean() - human_coedit.mean():+.4f}")

    print(f"\nTOCSIN:")
    print(f"  Human:  min={human_tocsin.min():.4f}, max={human_tocsin.max():.4f}, mean={human_tocsin.mean():.4f}")
    print(f"  LLM:    min={llm_tocsin.min():.4f}, max={llm_tocsin.max():.4f}, mean={llm_tocsin.mean():.4f}")
    print(f"  Diff:   {llm_tocsin.mean() - human_tocsin.mean():+.4f}")

    # 测试weighted fusion（作为对比）
    print(f"\n{'='*60}")
    print("Weighted Fusion (baseline)")
    print(f"{'='*60}")

    fusion = WeightedFusion(weights={'coedit': 0.5, 'tocsin': 0.5})
    human_scores = {'coedit': human_coedit, 'tocsin': human_tocsin}
    llm_scores = {'coedit': llm_coedit, 'tocsin': llm_tocsin}

    human_fused, llm_fused = fusion.normalize_and_fuse(human_scores, llm_scores)

    roc_auc, _, _, _, _, _, _ = get_roc_metrics(
        human_fused.tolist(), llm_fused.tolist()
    )

    print(f"\nFused scores:")
    print(f"  Human:  {human_fused.mean():.4f} ± {human_fused.std():.4f}")
    print(f"  LLM:    {llm_fused.mean():.4f} ± {llm_fused.std():.4f}")
    print(f"  Diff:   {llm_fused.mean() - human_fused.mean():+.4f}")
    print(f"\nROC AUC: {roc_auc:.4f}")

    # 分析最佳权重
    print(f"\n{'='*60}")
    print("寻找最佳权重")
    print(f"{'='*60}")

    best_auc = 0
    best_weights = (0.5, 0.5)

    for coedit_w in np.linspace(0, 1, 11):
        tocsin_w = 1 - coedit_w
        fusion = WeightedFusion(weights={'coedit': coedit_w, 'tocsin': tocsin_w})
        human_f, llm_f = fusion.normalize_and_fuse(human_scores, llm_scores)
        auc, _, _, _, _, _, _ = get_roc_metrics(human_f.tolist(), llm_f.tolist())

        if auc > best_auc:
            best_auc = auc
            best_weights = (coedit_w, tocsin_w)
            print(f"  ✓ New best: CoEdIT={coedit_w:.2f}, TOCSIN={tocsin_w:.2f}, AUC={auc:.4f}")

    print(f"\n最佳权重: CoEdIT={best_weights[0]:.2f}, TOCSIN={best_weights[1]:.2f}")
    print(f"最佳ROC AUC: {best_auc:.4f}")


if __name__ == '__main__':
    quick_analysis()
