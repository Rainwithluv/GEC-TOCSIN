"""
训练Attention Mode模型 - 支持实时计算分数版本
适用于只有text和label的简单JSON格式
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from fusion.dynamic_attention_fusion import CrossBranchAttention
    from channels.coedit_channel import CoEdITChannel
    from channels.tocsin_channel import TOCSINChannel
except ImportError:
    from demo.src.fusion.dynamic_attention_fusion import CrossBranchAttention
    from demo.src.channels.coedit_channel import CoEdITChannel
    from demo.src.channels.tocsin_channel import TOCSINChannel


class SimpleFusionDataset(Dataset):
    """
    简单融合数据集 - 实时计算分数版本
    适用于只有text和label的JSON数据
    """

    def __init__(self, samples, coedit_channel=None, tocsin_channel=None, cache_scores=True):
        """
        Args:
            samples: 样本列表，每个样本包含text和label
            coedit_channel: CoEdIT通道（如果为None则创建新的）
            tocsin_channel: TOCSIN通道（如果为None则创建新的）
            cache_scores: 是否缓存分数（建议True，避免重复计算）
        """
        self.samples = samples
        self.cache_scores = cache_scores
        self.scores_cache = {}

        # 初始化通道
        if coedit_channel is None:
            print("初始化CoEdIT通道...")
            self.coedit_channel = CoEdITChannel()
        else:
            self.coedit_channel = coedit_channel

        if tocsin_channel is None:
            print("初始化TOCSIN通道...")
            self.tocsin_channel = TOCSINChannel()
        else:
            self.tocsin_channel = tocsin_channel

        # 预计算所有分数（如果启用缓存）
        if cache_scores:
            print("预计算分数...")
            self._precompute_scores()

    def _precompute_scores(self):
        """预计算所有样本的分数"""
        texts = [s['text'] for s in self.samples]

        print(f"  计算 {len(texts)} 个样本的CoEdIT分数...")
        coedit_scores = self.coedit_channel.score_texts(texts, show_progress=True)

        print(f"  计算 {len(texts)} 个样本的TOCSIN分数...")
        tocsin_scores = self.tocsin_channel.score_texts(texts, show_progress=True, for_llm=True)

        # 缓存分数
        for i, (c_score, t_score) in enumerate(zip(coedit_scores, tocsin_scores)):
            self.scores_cache[i] = {
                'coedit': float(c_score),
                'tocsin': float(t_score)
            }

        print(f"  ✓ 分数计算完成并缓存")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 获取分数
        if self.cache_scores and idx in self.scores_cache:
            scores = torch.tensor([
                self.scores_cache[idx]['coedit'],
                self.scores_cache[idx]['tocsin']
            ], dtype=torch.float32)
        else:
            # 实时计算（不推荐，会很慢）
            text = sample['text']
            coedit_score = self.coedit_channel.score_texts([text], show_progress=False)[0]
            tocsin_score = self.tocsin_channel.score_texts([text], show_progress=False, for_llm=True)[0]
            scores = torch.tensor([coedit_score, tocsin_score], dtype=torch.float32)

        # 标签：human=0, llm=1
        label = 1 if sample['label'] == 'llm' else 0
        label = torch.tensor(label, dtype=torch.long)

        return scores, label


def load_simple_json_data(data_path):
    """
    加载简单JSON格式的数据
    格式: [{"text": "...", "label": "human/llm"}, ...]
    """
    print(f"加载数据: {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    print(f"  ✓ 加载了 {len(samples)} 个样本")

    # 统计标签
    human_count = sum(1 for s in samples if s['label'] == 'human')
    llm_count = sum(1 for s in samples if s['label'] == 'llm')

    print(f"  人类样本: {human_count}")
    print(f"  LLM样本: {llm_count}")

    return samples


def analyze_data_quality(samples):
    """分析数据质量（不计算分数）"""
    print("\n数据质量分析:")

    human_count = sum(1 for s in samples if s['label'] == 'human')
    llm_count = sum(1 for s in samples if s['label'] == 'llm')
    total = len(samples)

    print(f"  总样本数: {total}")
    print(f"  平衡比例: {human_count/llm_count:.2f}")

    if abs(human_count - llm_count) / total > 0.1:
        print("  ⚠️  数据集不平衡")
    else:
        print("  ✓ 数据集平衡良好")

    # 文本长度统计
    human_texts = [s['text'] for s in samples if s['label'] == 'human']
    llm_texts = [s['text'] for s in samples if s['label'] == 'llm']

    human_lengths = [len(t.split()) for t in human_texts]
    llm_lengths = [len(t.split()) for t in llm_texts]

    print(f"\n  人类文本长度: 平均 {np.mean(human_lengths):.0f} 词")
    print(f"  LLM文本长度: 平均 {np.mean(llm_lengths):.0f} 词")


class AttentionFusionTrainer:
    """Attention融合训练器"""

    def __init__(self,
                 input_dim=2,
                 hidden_dim=16,
                 num_heads=2,
                 learning_rate=0.001,
                 device='cpu'):

        self.device = device
        self.model = CrossBranchAttention(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads
        ).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        print(f"\n模型配置:")
        print(f"  设备: {device}")
        print(f"  输入维度: {input_dim}")
        print(f"  隐藏维度: {hidden_dim}")
        print(f"  注意力头数: {num_heads}")
        print(f"  学习率: {learning_rate}")

    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for scores, labels in dataloader:
            scores = scores.to(self.device)
            labels = labels.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()

            # 获取融合权重
            fusion_weights, _ = self.model(scores)

            # 应用权重融合
            fused_scores = fusion_weights[:, 0] * scores[:, 0] + fusion_weights[:, 1] * scores[:, 1]

            # 转换为logits
            probs = torch.stack([1 - fused_scores, fused_scores], dim=1)
            logits = torch.log(probs + 1e-8)

            # 计算损失
            loss = self.criterion(logits, labels)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            predicted = (fused_scores > 0.5).long()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0

        return avg_loss, accuracy

    def validate(self, dataloader):
        """验证"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for scores, labels in dataloader:
                scores = scores.to(self.device)
                labels = labels.to(self.device)

                # 获取融合权重
                fusion_weights, _ = self.model(scores)

                # 应用权重融合
                fused_scores = fusion_weights[:, 0] * scores[:, 0] + fusion_weights[:, 1] * scores[:, 1]

                # 转换为logits
                probs = torch.stack([1 - fused_scores, fused_scores], dim=1)
                logits = torch.log(probs + 1e-8)

                # 计算损失
                loss = self.criterion(logits, labels)

                # 统计
                total_loss += loss.item()
                predicted = (fused_scores > 0.5).long()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0

        return avg_loss, accuracy

    def train(self, train_loader, val_loader, num_epochs, save_path='attention_model.pth'):
        """完整训练流程"""
        best_val_acc = 0
        patience = 10
        patience_counter = 0

        print(f"\n开始训练 ({num_epochs} epochs):")
        print("-" * 80)

        for epoch in range(num_epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)

            # 验证
            val_loss, val_acc = self.validate(val_loader)

            print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}", end='')

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, save_path)
                print(f" ✓ (best: {val_acc:.4f})")
            else:
                patience_counter += 1
                print()

            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        print(f"\n训练完成! 最佳验证准确率: {best_val_acc:.4f}")
        print(f"模型已保存到: {save_path}")

        return best_val_acc


def main():
    parser = argparse.ArgumentParser(
        description='训练Attention Mode模型 - 支持简单JSON格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 使用xsum数据训练
  python train_attention_simple.py --data data/xsum.GPT-4o.normal.test_data.json

  # 使用writing数据训练
  python train_attention_simple.py --data data/writing.GPT-4o.normal.test_data.json

  # 使用较少样本快速测试
  python train_attention_simple.py --data data/xsum.GPT-4o.normal.test_data.json --n-samples 100

  # 自定义输出路径
  python train_attention_simple.py --data data/xsum.GPT-4o.normal.test_data.json --output models/xsum_attention.pth
        """
    )

    parser.add_argument('data', type=str, help='训练数据路径 (JSON格式)')
    parser.add_argument('--output', type=str, default='models/attention_model.pth',
                        help='模型保存路径')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--hidden-dim', type=int, default=16,
                        help='隐藏层维度')
    parser.add_argument('--num-heads', type=int, default=2,
                        help='注意力头数')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='验证集比例 (0.0-1.0)')
    parser.add_argument('--n-samples', type=int, default=None,
                        help='使用样本数量 (None=使用全部)')
    parser.add_argument('--device', type=str, default=None,
                        help='设备 (cpu/cuda，默认自动选择)')
    parser.add_argument('--no-cache', action='store_true',
                        help='不缓存分数（训练会很慢）')

    args = parser.parse_args()

    # 选择设备
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("=" * 80)
    print("Attention Mode 模型训练 (简单JSON格式)")
    print("=" * 80)

    # 加载数据
    samples = load_simple_json_data(args.data)

    # 分析数据质量
    analyze_data_quality(samples)

    # 限制样本数
    if args.n_samples is not None:
        print(f"\n限制样本数为: {args.n_samples}")
        # 确保平衡采样
        human_samples = [s for s in samples if s['label'] == 'human']
        llm_samples = [s for s in samples if s['label'] == 'llm']

        n_each = min(args.n_samples // 2, len(human_samples), len(llm_samples))
        samples = human_samples[:n_each] + llm_samples[:n_each]
        print(f"  实际使用: {len(samples)} 个样本 ({n_each} each)")

    # 分割训练集和验证集
    if args.test_size > 0:
        train_samples, val_samples = train_test_split(
            samples,
            test_size=args.test_size,
            random_state=42,
            stratify=[s['label'] for s in samples]
        )
    else:
        train_samples = samples
        val_samples = []
        print("不使用验证集")

    print(f"\n训练样本: {len(train_samples)}")
    print(f"验证样本: {len(val_samples)}")

    # 创建数据集（共享通道以避免重复初始化）
    print("\n初始化通道...")
    coedit_channel = CoEdITChannel()
    tocsin_channel = TOCSINChannel()

    train_dataset = SimpleFusionDataset(
        train_samples,
        coedit_channel=coedit_channel,
        tocsin_channel=tocsin_channel,
        cache_scores=not args.no_cache
    )

    val_dataset = SimpleFusionDataset(
        val_samples,
        coedit_channel=coedit_channel,
        tocsin_channel=tocsin_channel,
        cache_scores=not args.no_cache
    ) if val_samples else None

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) if val_dataset else None

    # 创建训练器
    trainer = AttentionFusionTrainer(
        input_dim=2,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        learning_rate=args.lr,
        device=device
    )

    # 创建输出目录
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 训练
    if val_dataset:
        trainer.train(train_loader, val_loader, args.epochs, args.output)
    else:
        # 无验证集时的训练
        print(f"\n开始训练 ({args.epochs} epochs，无验证):")
        print("-" * 80)

        for epoch in range(args.epochs):
            train_loss, train_acc = trainer.train_epoch(train_loader)
            print(f"Epoch {epoch+1:3d}/{args.epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # 保存最终模型
        torch.save({
            'epoch': args.epochs,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
        }, args.output)
        print(f"\n训练完成! 模型已保存到: {args.output}")

    print("\n" + "=" * 80)
    print("训练完成!")
    print("=" * 80)
    print(f"\n下一步: 使用训练好的模型进行评估")
    print(f"  python evaluate_attention.py --data {args.data} --model {args.output}")


if __name__ == '__main__':
    main()
