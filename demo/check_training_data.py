"""
检查训练数据质量
"""
import json
import numpy as np
import argparse
from pathlib import Path


def check_training_data(data_path):
    """检查训练数据"""
    print("="*70)
    print("训练数据质量检查")
    print("="*70)

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 检查格式
    if 'samples' in data:
        samples = data['samples']
    else:
        # 直接是样本列表
        samples = data

    print(f"\n数据文件: {data_path}")
    print(f"样本总数: {len(samples)}")

    # 1. 基本统计
    print("\n" + "-"*70)
    print("1. 基本统计")
    print("-"*70)

    human_count = sum(1 for s in samples if s['label'] == 'human')
    llm_count = sum(1 for s in samples if s['label'] == 'llm')
    total_count = len(samples)

    print(f"人类样本: {human_count} ({human_count/total_count*100:.1f}%)")
    print(f"LLM样本: {llm_count} ({llm_count/total_count*100:.1f}%)")
    print(f"平衡比例: {human_count/llm_count:.2f}")

    if abs(human_count - llm_count) / total_count > 0.1:
        print("⚠️  警告: 数据集不平衡!")
    else:
        print("✓ 数据集平衡良好")

    # 2. 检查必需字段
    print("\n" + "-"*70)
    print("2. 字段检查")
    print("-"*70)

    required_fields = ['text', 'label', 'coedit_score', 'tocsin_score']
    missing_fields = {field: 0 for field in required_fields}

    for sample in samples:
        for field in required_fields:
            if field not in sample:
                missing_fields[field] += 1

    all_present = True
    for field, count in missing_fields.items():
        if count > 0:
            print(f"⚠️  {field}: 缺失 {count} 个样本")
            all_present = False
        else:
            print(f"✓ {field}: 完整")

    if not all_present:
        print("\n⚠️  警告: 部分样本缺少必需字段!")
        return False

    # 3. 分数范围
    print("\n" + "-"*70)
    print("3. 分数范围")
    print("-"*70)

    coedit_scores = [s['coedit_score'] for s in samples]
    tocsin_scores = [s['tocsin_score'] for s in samples]

    print(f"CoEdIT分数:")
    print(f"  最小值: {min(coedit_scores):.4f}")
    print(f"  最大值: {max(coedit_scores):.4f}")
    print(f"  平均值: {np.mean(coedit_scores):.4f}")
    print(f"  标准差: {np.std(coedit_scores):.4f}")

    print(f"\nTOCSIN分数:")
    print(f"  最小值: {min(tocsin_scores):.4f}")
    print(f"  最大值: {max(tocsin_scores):.4f}")
    print(f"  平均值: {np.mean(tocsin_scores):.4f}")
    print(f"  标准差: {np.std(tocsin_scores):.4f}")

    # 检查方差
    if np.std(coedit_scores) < 0.01:
        print("⚠️  CoEdIT分数方差过小!")
    if np.std(tocsin_scores) < 0.01:
        print("⚠️  TOCSIN分数方差过小!")

    # 4. 分数分布（按标签）
    print("\n" + "-"*70)
    print("4. 分数分布（按标签）")
    print("-"*70)

    human_coedit = [s['coedit_score'] for s in samples if s['label'] == 'human']
    human_tocsin = [s['tocsin_score'] for s in samples if s['label'] == 'human']
    llm_coedit = [s['coedit_score'] for s in samples if s['label'] == 'llm']
    llm_tocsin = [s['tocsin_score'] for s in samples if s['label'] == 'llm']

    print(f"人类文本:")
    print(f"  CoEdIT: mean={np.mean(human_coedit):.4f}, std={np.std(human_coedit):.4f}")
    print(f"  TOCSIN: mean={np.mean(human_tocsin):.4f}, std={np.std(human_tocsin):.4f}")

    print(f"\nLLM文本:")
    print(f"  CoEdIT: mean={np.mean(llm_coedit):.4f}, std={np.std(llm_coedit):.4f}")
    print(f"  TOCSIN: mean={np.mean(llm_tocsin):.4f}, std={np.std(llm_tocsin):.4f}")

    # 5. 分离度
    print("\n" + "-"*70)
    print("5. 通道分离度")
    print("-"*70)

    coedit_separation = abs(np.mean(llm_coedit) - np.mean(human_coedit))
    tocsin_separation = abs(np.mean(llm_tocsin) - np.mean(human_tocsin))

    print(f"CoEdIT分离度: {coedit_separation:.4f}")
    print(f"TOCSIN分离度: {tocsin_separation:.4f}")

    if coedit_separation > tocsin_separation:
        print("→ CoEdIT具有更好的分离度")
    else:
        print("→ TOCSIN具有更好的分离度")

    # 评估分离度
    if coedit_separation < 0.1 and tocsin_separation < 0.1:
        print("⚠️  警告: 两个通道的分离度都很低!")
    elif coedit_separation > 0.3 or tocsin_separation > 0.3:
        print("✓ 分离度良好")
    else:
        print("✓ 分离度可接受")

    # 6. 分数相关性
    print("\n" + "-"*70)
    print("6. 分数相关性")
    print("-"*70)

    all_coedit = np.array(coedit_scores)
    all_tocsin = np.array(tocsin_scores)
    correlation = np.corrcoef(all_coedit, all_tocsin)[0, 1]

    print(f"CoEdIT与TOCSIN的相关系数: {correlation:.4f}")

    if correlation > 0.8:
        print("⚠️  警告: 两个通道高度相关，可能冗余!")
    elif correlation < 0.2:
        print("✓ 两个通道相对独立，互补性好")
    else:
        print("✓ 两个通道适度相关")

    # 7. 异常值检测
    print("\n" + "-"*70)
    print("7. 异常值检测")
    print("-"*70)

    # 使用IQR方法检测异常值
    def detect_outliers(scores):
        q1 = np.percentile(scores, 25)
        q3 = np.percentile(scores, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = [s for s in scores if s < lower or s > upper]
        return outliers

    coedit_outliers = detect_outliers(coedit_scores)
    tocsin_outliers = detect_outliers(tocsin_scores)

    print(f"CoEdIT异常值: {len(coedit_outliers)} 个")
    print(f"TOCSIN异常值: {len(tocsin_outliers)} 个")

    if len(coedit_outliers) > len(samples) * 0.05:
        print("⚠️  CoEdIT异常值过多!")
    if len(tocsin_outliers) > len(samples) * 0.05:
        print("⚠️  TOCSIN异常值过多!")

    # 8. 总体评估
    print("\n" + "="*70)
    print("总体评估")
    print("="*70)

    issues = []

    # 检查各项指标
    if abs(human_count - llm_count) / total_count > 0.1:
        issues.append("数据不平衡")

    if coedit_separation < 0.1 and tocsin_separation < 0.1:
        issues.append("分离度过低")

    if correlation > 0.8:
        issues.append("通道高度相关")

    if np.std(coedit_scores) < 0.01 or np.std(tocsin_scores) < 0.01:
        issues.append("分数方差过小")

    if len(coedit_outliers) > len(samples) * 0.05:
        issues.append("CoEdIT异常值过多")

    if len(tocsin_outliers) > len(samples) * 0.05:
        issues.append("TOCSIN异常值过多")

    if issues:
        print("⚠️  发现以下问题:")
        for issue in issues:
            print(f"  - {issue}")
        print(f"\n建议: 解决这些问题后再进行训练")
        return False
    else:
        print("✓ 数据质量良好，可以用于训练!")
        return True


def main():
    parser = argparse.ArgumentParser(description='检查训练数据质量')
    parser.add_argument('data_path', type=str, help='训练数据路径')

    args = parser.parse_args()

    if not Path(args.data_path).exists():
        print(f"错误: 文件不存在: {args.data_path}")
        return

    check_training_data(args.data_path)


if __name__ == '__main__':
    main()
