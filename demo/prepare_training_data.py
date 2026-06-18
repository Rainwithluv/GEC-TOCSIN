"""
准备Attention Mode训练数据
"""
import os
import sys
import json
import argparse
from pathlib import Path

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from utils.data_loader import DataLoader
    from channels.coedit_channel import CoEdITChannel
    from channels.tocsin_channel import TOCSINChannel
except ImportError:
    from demo.src.utils.data_loader import DataLoader
    from demo.src.channels.coedit_channel import CoEdITChannel
    from demo.src.channels.tocsin_channel import TOCSINChannel


def prepare_training_data(dataset='xsum',
                         model='gpt-4',
                         n_samples=500,
                         output_file='training_data.json',
                         base_dir="../"):
    """
    准备训练数据

    Args:
        dataset: 数据集名称
        model: LLM模型名称
        n_samples: 每个类别的样本数（总样本数 = 2 * n_samples）
        output_file: 输出文件路径
        base_dir: 数据目录
    """
    print("="*70)
    print("准备 Attention Mode 训练数据")
    print("="*70)
    print(f"\n配置:")
    print(f"  数据集: {dataset}")
    print(f"  模型: {model}")
    print(f"  每类别样本数: {n_samples}")
    print(f"  总样本数: {n_samples * 2}")
    print(f"  输出文件: {output_file}")

    # 步骤1: 加载数据
    print(f"\n步骤1: 加载数据...")
    loader = DataLoader(base_dir=base_dir)

    try:
        data = loader.load_combined_data(dataset, model)
        human_texts = data['original']
        llm_texts = data['sampled']
    except Exception as e:
        print(f"错误: 无法加载数据: {e}")
        print("\n可用数据集:")
        print("  - xsum")
        print("  - writing")
        return None

    # 限制样本数
    n_samples = min(n_samples, len(human_texts), len(llm_texts))
    human_texts = human_texts[:n_samples]
    llm_texts = llm_texts[:n_samples]

    print(f"  ✓ 加载了 {len(human_texts)} 个人类文本")
    print(f"  ✓ 加载了 {len(llm_texts)} 个LLM文本")

    # 步骤2: 初始化通道
    print(f"\n步骤2: 初始化通道...")
    coedit = CoEdITChannel()
    tocsin = TOCSINChannel()
    print(f"  ✓ CoEdIT通道已初始化")
    print(f"  ✓ TOCSIN通道已初始化")

    # 步骤3: 计算人类文本分数
    print(f"\n步骤3: 计算人类文本分数...")
    human_coedit = coedit.score_texts(human_texts, show_progress=True)
    human_tocsin = tocsin.score_texts(human_texts, show_progress=True, for_llm=True)
    print(f"  ✓ 人类文本分数计算完成")

    # 步骤4: 计算LLM文本分数
    print(f"\n步骤4: 计算LLM文本分数...")
    llm_coedit = coedit.score_texts(llm_texts, show_progress=True)
    llm_tocsin = tocsin.score_texts(llm_texts, show_progress=True, for_llm=True)
    print(f"  ✓ LLM文本分数计算完成")

    # 步骤5: 组织数据
    print(f"\n步骤5: 组织训练数据...")

    samples = []

    # 添加人类样本
    for i, (text, coedit_score, tocsin_score) in enumerate(
        zip(human_texts, human_coedit, human_tocsin)
    ):
        samples.append({
            'id': f'{dataset}_human_{i:04d}',
            'text': text,
            'label': 'human',
            'coedit_score': float(coedit_score),
            'tocsin_score': float(tocsin_score),
            'metadata': {
                'dataset': dataset,
                'model': model,
                'source': 'human'
            }
        })

    # 添加LLM样本
    for i, (text, coedit_score, tocsin_score) in enumerate(
        zip(llm_texts, llm_coedit, llm_tocsin)
    ):
        samples.append({
            'id': f'{dataset}_llm_{i:04d}',
            'text': text,
            'label': 'llm',
            'coedit_score': float(coedit_score),
            'tocsin_score': float(tocsin_score),
            'metadata': {
                'dataset': dataset,
                'model': model,
                'source': model
            }
        })

    print(f"  ✓ 组织了 {len(samples)} 个样本")

    # 步骤6: 保存数据
    print(f"\n步骤6: 保存训练数据...")

    output_data = {
        'metadata': {
            'dataset': dataset,
            'model': model,
            'total_samples': len(samples),
            'human_samples': n_samples,
            'llm_samples': n_samples,
            'created_by': 'prepare_training_data.py'
        },
        'samples': samples
    }

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"  ✓ 数据已保存到 {output_path}")

    # 步骤7: 数据质量报告
    print(f"\n步骤7: 数据质量报告...")

    import numpy as np

    human_coedit_scores = [s['coedit_score'] for s in samples if s['label'] == 'human']
    human_tocsin_scores = [s['tocsin_score'] for s in samples if s['label'] == 'human']
    llm_coedit_scores = [s['coedit_score'] for s in samples if s['label'] == 'llm']
    llm_tocsin_scores = [s['tocsin_score'] for s in samples if s['label'] == 'llm']

    print(f"\n人类文本分数:")
    print(f"  CoEdIT: mean={np.mean(human_coedit_scores):.4f}, std={np.std(human_coedit_scores):.4f}")
    print(f"  TOCSIN: mean={np.mean(human_tocsin_scores):.4f}, std={np.std(human_tocsin_scores):.4f}")

    print(f"\nLLM文本分数:")
    print(f"  CoEdIT: mean={np.mean(llm_coedit_scores):.4f}, std={np.std(llm_coedit_scores):.4f}")
    print(f"  TOCSIN: mean={np.mean(llm_tocsin_scores):.4f}, std={np.std(llm_tocsin_scores):.4f}")

    coedit_sep = abs(np.mean(llm_coedit_scores) - np.mean(human_coedit_scores))
    tocsin_sep = abs(np.mean(llm_tocsin_scores) - np.mean(human_tocsin_scores))

    print(f"\n分离度:")
    print(f"  CoEdIT: {coedit_sep:.4f}")
    print(f"  TOCSIN: {tocsin_sep:.4f}")

    if coedit_sep > tocsin_sep:
        print(f"  → CoEdIT分离度更高")
    else:
        print(f"  → TOCSIN分离度更高")

    print(f"\n{'='*70}")
    print(f"训练数据准备完成！")
    print(f"{'='*70}")
    print(f"\n下一步:")
    print(f"  1. 检查数据质量:")
    print(f"     python check_data.py {output_path}")
    print(f"  2. 开始训练:")
    print(f"     python train_attention.py --data {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='准备Attention Mode训练数据')

    parser.add_argument('--dataset', type=str, default='xsum',
                        choices=['xsum', 'writing'],
                        help='数据集名称')
    parser.add_argument('--model', type=str, default='gpt-4',
                        help='LLM模型名称')
    parser.add_argument('--n-samples', type=int, default=500,
                        help='每个类别的样本数')
    parser.add_argument('--output', type=str, default='data/training_data.json',
                        help='输出文件路径')
    parser.add_argument('--base-dir', type=str, default='../',
                        help='数据目录')

    args = parser.parse_args()

    prepare_training_data(
        dataset=args.dataset,
        model=args.model,
        n_samples=args.n_samples,
        output_file=args.output,
        base_dir=args.base_dir
    )


if __name__ == '__main__':
    main()
