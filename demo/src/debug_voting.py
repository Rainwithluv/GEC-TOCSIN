"""
调试真实数据的分数特征
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
    print("✓ Imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)


def analyze_real_data():
    """分析真实数据的分数特征"""
    print("\n=== 分析真实数据分数特征 ===\n")

    # 加载数据
    print("Loading dataset...")
    loader = DataLoader(base_dir="../")
    data = loader.load_combined_data('xsum', 'gpt-4')

    # 使用少量样本快速分析
    n = 20
    human_texts = data['original'][:n]
    llm_texts = data['sampled'][:n]

    print(f"Using {n} human texts and {n} LLM texts\n")

    # 初始化通道
    print("Initializing channels...")
    coedit_channel = CoEdITChannel(device='cpu')
    tocsin_channel = TOCSINChannel(device='cpu')
    print("Channels initialized\n")

    # 评分
    print("Scoring human texts...")
    human_coedit = []
    human_tocsin = []
    for text in human_texts:
        human_coedit.append(coedit_channel.score_text(text))
        human_tocsin.append(tocsin_channel.score_text(text))

    print("Scoring LLM texts...")
    llm_coedit = []
    llm_tocsin = []
    for text in llm_texts:
        llm_coedit.append(coedit_channel.score_text(text))
        llm_tocsin.append(tocsin_channel.score_text(text))

    human_coedit = np.array(human_coedit)
    human_tocsin = np.array(human_tocsin)
    llm_coedit = np.array(llm_coedit)
    llm_tocsin = np.array(llm_tocsin)

    print(f"\n{'='*60}")
    print("分数统计")
    print(f"{'='*60}")

    print(f"\nCoEdIT Channel (Grammar):")
    print(f"  Human:  min={human_coedit.min():.4f}, max={human_coedit.max():.4f}, mean={human_coedit.mean():.4f}, std={human_coedit.std():.4f}")
    print(f"  LLM:    min={llm_coedit.min():.4f}, max={llm_coedit.max():.4f}, mean={llm_coedit.mean():.4f}, std={llm_coedit.std():.4f}")
    print(f"  Diff:   {llm_coedit.mean() - human_coedit.mean():+.4f}")
    print(f"  Interpretation: {'LLM分数更高' if llm_coedit.mean() > human_coedit.mean() else 'LLM分数更低'}")

    print(f"\nTOCSIN Channel (Cohesiveness):")
    print(f"  Human:  min={human_tocsin.min():.4f}, max={human_tocsin.max():.4f}, mean={human_tocsin.mean():.4f}, std={human_tocsin.std():.4f}")
    print(f"  LLM:    min={llm_tocsin.min():.4f}, max={llm_tocsin.max():.4f}, mean={llm_tocsin.mean():.4f}, std={llm_tocsin.std():.4f}")
    print(f"  Diff:   {llm_tocsin.mean() - human_tocsin.mean():+.4f}")
    print(f"  Interpretation: {'LLM分数更高' if llm_tocsin.mean() > human_tocsin.mean() else 'LLM分数更低'}")

    print(f"\n{'='*60}")
    print("分数语义分析")
    print(f"{'='*60}")

    # 分析分数语义
    print(f"\n预期语义:")
    print(f"  高分数 = 更像人类")
    print(f"  低分数 = 更像LLM")

    print(f"\n实际数据:")
    if human_coedit.mean() > llm_coedit.mean():
        print(f"  ✓ CoEdIT: 人类分数更高 (符合预期)")
    else:
        print(f"  ✗ CoEdIT: LLM分数更高 (与预期相反)")

    if human_tocsin.mean() > llm_tocsin.mean():
        print(f"  ✓ TOCSIN: 人类分数更高 (符合预期)")
    else:
        print(f"  ✗ TOCSIN: LLM分数更高 (与预期相反)")

    # 测试归一化
    print(f"\n{'='*60}")
    print("测试归一化")
    print(f"{'='*60}")

    from sklearn.preprocessing import MinMaxScaler

    coedit_scaler = MinMaxScaler()
    tocsin_scaler = MinMaxScaler()

    # 合并所有分数进行fit
    all_coedit = np.concatenate([human_coedit, llm_coedit])
    all_tocsin = np.concatenate([human_tocsin, llm_tocsin])

    coedit_scaler.fit(all_coedit.reshape(-1, 1))
    tocsin_scaler.fit(all_tocsin.reshape(-1, 1))

    human_coedit_norm = coedit_scaler.transform(human_coedit.reshape(-1, 1)).flatten()
    llm_coedit_norm = coedit_scaler.transform(llm_coedit.reshape(-1, 1)).flatten()
    human_tocsin_norm = tocsin_scaler.transform(human_tocsin.reshape(-1, 1)).flatten()
    llm_tocsin_norm = tocsin_scaler.transform(llm_tocsin.reshape(-1, 1)).flatten()

    print(f"\n归一化后:")
    print(f"  CoEdIT - Human: {human_coedit_norm.mean():.4f}, LLM: {llm_coedit_norm.mean():.4f}")
    print(f"  TOCSIN - Human: {human_tocsin_norm.mean():.4f}, LLM: {llm_tocsin_norm.mean():.4f}")

    # 测试概率转换
    print(f"\n概率转换 (1 - normalized_score):")
    human_coedit_proba = 1 - human_coedit_norm
    llm_coedit_proba = 1 - llm_coedit_norm
    human_tocsin_proba = 1 - human_tocsin_norm
    llm_tocsin_proba = 1 - llm_tocsin_norm

    print(f"  CoEdIT LLM概率 - Human: {human_coedit_proba.mean():.4f}, LLM: {llm_coedit_proba.mean():.4f}")
    print(f"  TOCSIN LLM概率 - Human: {human_tocsin_proba.mean():.4f}, LLM: {llm_tocsin_proba.mean():.4f}")

    print(f"\n预期: LLM的LLM概率应该更高")
    if llm_coedit_proba.mean() > human_coedit_proba.mean():
        print(f"  ✓ CoEdIT: LLM概率更高 ({llm_coedit_proba.mean():.4f} > {human_coedit_proba.mean():.4f})")
    else:
        print(f"  ✗ CoEdIT: LLM概率更低 ({llm_coedit_proba.mean():.4f} < {human_coedit_proba.mean():.4f})")

    if llm_tocsin_proba.mean() > human_tocsin_proba.mean():
        print(f"  ✓ TOCSIN: LLM概率更高 ({llm_tocsin_proba.mean():.4f} > {human_tocsin_proba.mean():.4f})")
    else:
        print(f"  ✗ TOCSIN: LLM概率更低 ({llm_tocsin_proba.mean():.4f} < {human_tocsin_proba.mean():.4f})")


if __name__ == '__main__':
    analyze_real_data()
