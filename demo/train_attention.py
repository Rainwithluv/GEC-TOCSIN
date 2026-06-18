"""
训练Attention Mode模型
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from fusion.dynamic_attention_fusion import CrossBranchAttention
except ImportError:
    from demo.src.fusion.dynamic_attention_fusion import CrossBranchAttention


class FusionDataset(Dataset):
    """融合数据集"""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 分数
        scores = torch.tensor([
            sample['coedit_score'],
            sample['tocsin_score']
        ], dtype=torch.float32)

        # 标签：human=0, llm=1
        label = 1 if sample['label'] == 'llm' else 0
        label = torch.tensor(label, dtype=torch.long)

        return scores, label


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

        print(f"模型初始化完成:")
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

            # 转换为logits用于分类
            # 简单的阈值：fused_score > 0.5 → llm
            # 转换为概率分布
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
        print("-" * 70)

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
                print(f" ✓ (best)")
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


def load_data(data_path, test_size=0.2, random_state=42):
    """加载并分割数据"""
    print(f"加载数据: {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = data['samples']
    print(f"  总样本数: {len(samples)}")

    # 统计标签
    human_count = sum(1 for s in samples if s['label'] == 'human')
    llm_count = sum(1 for s in samples if s['label'] == 'llm')
    print(f"  人类样本: {human_count}")
    print(f"  LLM样本: {llm_count}")

    # 分割训练集和验证集
    from sklearn.model_selection import train_test_split

    train_samples, val_samples = train_test_split(
        samples,
        test_size=test_size,
        random_state=random_state,
        stratify=[s['label'] for s in samples]
    )

    print(f"  训练样本: {len(train_samples)}")
    print(f"  验证样本: {len(val_samples)}")

    return train_samples, val_samples


def main():
    parser = argparse.ArgumentParser(description='训练Attention Mode模型')

    parser.add_argument('--data', type=str, default='data/training_data.json',
                        help='训练数据路径')
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
                        help='验证集比例')
    parser.add_argument('--device', type=str, default=None,
                        help='设备 (cpu/cuda，默认自动选择)')

    args = parser.parse_args()

    # 选择设备
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("=" * 70)
    print("Attention Mode 模型训练")
    print("=" * 70)

    # 加载数据
    train_samples, val_samples = load_data(args.data, args.test_size)

    # 创建数据集
    train_dataset = FusionDataset(train_samples)
    val_dataset = FusionDataset(val_samples)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

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
    trainer.train(train_loader, val_loader, args.epochs, args.output)

    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    print(f"\n使用训练好的模型:")
    print(f"  python -m demo.src.detector --mode evaluate --dataset xsum --model gpt-4 \\")
    print(f"    --fusion dynamic --dynamic-mode attention \\")
    print(f"    --model-path {args.output}")


if __name__ == '__main__':
    main()
