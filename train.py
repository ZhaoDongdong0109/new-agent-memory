"""
训练脚本 - 记忆增强模型
支持：编码器训练、检索器训练、适配器训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import json
import os
from tqdm import tqdm
from dataclasses import dataclass

from memory_model import (
    MemoryAugmentedModel, 
    MemoryEncoder, 
    MemoryRetriever, 
    MemoryAdapter,
    ModelConfig,
    create_model
)


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础配置
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # 数据配置
    train_data_path: str = "training_data.json"
    retrieval_data_path: str = "retrieval_data.json"
    max_seq_length: int = 128
    
    # 保存配置
    output_dir: str = "./checkpoints"
    save_every: int = 1
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ConversationDataset(Dataset):
    """对话数据集"""
    
    def __init__(self, data_path: str, max_length: int = 128):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 简单的文本处理
        conversation = sample["conversation"]
        memories = sample["memories"]
        
        # 合并对话文本
        text = " ".join([turn["content"] for turn in conversation])
        
        # 简单的分词（实际应用中应使用真正的分词器）
        tokens = text.split()[:self.max_length]
        input_ids = torch.tensor([hash(t) % 50000 for t in tokens])
        
        # 填充
        if len(input_ids) < self.max_length:
            padding = torch.zeros(self.max_length - len(input_ids), dtype=torch.long)
            input_ids = torch.cat([input_ids, padding])
        
        # 重要性标签
        importance = torch.tensor([m["importance"] for m in memories])
        if len(importance) > 0:
            importance = importance.mean()
        else:
            importance = torch.tensor(0.5)
        
        return {
            "input_ids": input_ids,
            "importance": importance,
            "text": text
        }


class RetrievalDataset(Dataset):
    """检索数据集"""
    
    def __init__(self, data_path: str):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        query = sample["query"]
        positive = sample["positive"]
        negatives = sample["negatives"]
        
        # 简单的向量表示（实际应用中应使用编码器）
        def text_to_vector(text):
            tokens = text.split()
            vec = torch.tensor([hash(t) % 128 for t in tokens], dtype=torch.float)
            if len(vec) < 128:
                padding = torch.zeros(128 - len(vec))
                vec = torch.cat([vec, padding])
            return vec[:128]
        
        return {
            "query": text_to_vector(query),
            "positive": text_to_vector(positive),
            "negatives": [text_to_vector(neg) for neg in negatives]
        }


def train_encoder(
    model: MemoryAugmentedModel,
    train_loader: DataLoader,
    config: TrainingConfig
):
    """训练编码器"""
    print("\n" + "=" * 70)
    print("训练记忆编码器")
    print("=" * 70)
    
    model.encoder.train()
    optimizer = optim.AdamW(model.encoder.parameters(), lr=config.learning_rate)
    
    # 损失函数
    importance_loss_fn = nn.MSELoss()
    
    for epoch in range(config.num_epochs):
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(config.device)
            target_importance = batch["importance"].to(config.device)
            
            # 前向传播
            memory_vector, predicted_importance = model.encoder(input_ids)
            
            # 计算损失
            importance_loss = importance_loss_fn(predicted_importance.squeeze(), target_importance)
            
            # 对比损失（鼓励不同的记忆向量不同）
            contrastive_loss = torch.tensor(0.0, device=config.device)
            if memory_vector.size(0) > 1:
                similarity = torch.matmul(memory_vector, memory_vector.T)
                # 希望不同样本的记忆向量不同
                identity = torch.eye(similarity.size(0), device=config.device)
                contrastive_loss = ((similarity - identity) ** 2).mean()
            
            loss = importance_loss + 0.1 * contrastive_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), config.max_grad_norm)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # 保存检查点
        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(model, config, epoch + 1, "encoder")
    
    print("\n✅ 编码器训练完成！")


def train_retriever(
    model: MemoryAugmentedModel,
    train_loader: DataLoader,
    config: TrainingConfig
):
    """训练检索器"""
    print("\n" + "=" * 70)
    print("训练记忆检索器")
    print("=" * 70)
    
    model.retriever.train()
    optimizer = optim.AdamW(model.retriever.parameters(), lr=config.learning_rate)
    
    for epoch in range(config.num_epochs):
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch in progress_bar:
            query = batch["query"].to(config.device)
            positive = batch["positive"].to(config.device)
            
            # 前向传播
            # 正样本分数
            positive_scores = model.retriever(query, positive.unsqueeze(0).expand(query.size(0), -1))
            
            # InfoNCE 损失
            # 希望正样本分数高
            loss = -torch.log(torch.sigmoid(positive_scores.diag()) + 1e-9).mean()
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.retriever.parameters(), config.max_grad_norm)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(model, config, epoch + 1, "retriever")
    
    print("\n✅ 检索器训练完成！")


def train_adapter(
    model: MemoryAugmentedModel,
    train_loader: DataLoader,
    config: TrainingConfig
):
    """训练适配器"""
    print("\n" + "=" * 70)
    print("训练记忆适配器")
    print("=" * 70)
    
    model.adapter.train()
    optimizer = optim.AdamW(model.adapter.parameters(), lr=config.learning_rate)
    
    # 语言模型损失
    lm_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    
    for epoch in range(config.num_epochs):
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(config.device)
            
            # 获取记忆向量
            with torch.no_grad():
                memory_vector, _ = model.encoder(input_ids)
            
            # 前向传播
            logits = model.adapter(memory_vector.unsqueeze(1), max_length=50)
            
            # 计算损失（简化版）
            # 实际应用中应该使用真实的目标序列
            target = input_ids[:, :logits.size(1)]
            
            loss = lm_loss_fn(
                logits.view(-1, logits.size(-1)),
                target.view(-1)
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), config.max_grad_norm)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(model, config, epoch + 1, "adapter")
    
    print("\n✅ 适配器训练完成！")


def save_checkpoint(
    model: MemoryAugmentedModel,
    config: TrainingConfig,
    epoch: int,
    component: str
):
    """保存检查点"""
    os.makedirs(config.output_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(config.output_dir, f"{component}_epoch_{epoch}.pt")
    
    if component == "encoder":
        state_dict = model.encoder.state_dict()
    elif component == "retriever":
        state_dict = model.retriever.state_dict()
    elif component == "adapter":
        state_dict = model.adapter.state_dict()
    else:
        state_dict = model.state_dict()
    
    torch.save({
        "epoch": epoch,
        "state_dict": state_dict,
        "config": model.config.__dict__
    }, checkpoint_path)
    
    print(f"  检查点已保存: {checkpoint_path}")


def train_full_model(config: Optional[TrainingConfig] = None):
    """完整训练流程"""
    if config is None:
        config = TrainingConfig()
    
    print("=" * 70)
    print("记忆增强模型 - 完整训练流程")
    print("=" * 70)
    
    # 创建模型
    print("\n[1] 创建模型...")
    model = create_model()
    model = model.to(config.device)
    
    # 加载数据
    print("\n[2] 加载训练数据...")
    
    # 检查数据文件是否存在
    if not os.path.exists(config.train_data_path):
        print(f"  警告: {config.train_data_path} 不存在，请先运行 data_generator.py")
        print("  使用模拟数据进行演示...")
        # 创建模拟数据
        from data_generator import TrainingDataGenerator
        generator = TrainingDataGenerator()
        generator.generate_dataset(num_samples=100, output_file=config.train_data_path)
        generator.generate_retrieval_pairs(num_pairs=50, output_file=config.retrieval_data_path)
    
    train_dataset = ConversationDataset(config.train_data_path, config.max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    retrieval_dataset = RetrievalDataset(config.retrieval_data_path)
    retrieval_loader = DataLoader(retrieval_dataset, batch_size=config.batch_size, shuffle=True)
    
    print(f"  对话数据集大小: {len(train_dataset)}")
    print(f"  检索数据集大小: {len(retrieval_dataset)}")
    
    # 训练各个组件
    print("\n[3] 开始训练...")
    
    # 训练编码器
    train_encoder(model, train_loader, config)
    
    # 训练检索器
    train_retriever(model, retrieval_loader, config)
    
    # 训练适配器
    train_adapter(model, train_loader, config)
    
    # 保存最终模型
    print("\n[4] 保存最终模型...")
    final_path = os.path.join(config.output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_path)
    print(f"  最终模型已保存: {final_path}")
    
    print("\n" + "=" * 70)
    print("✅ 训练完成！")
    print("=" * 70)
    
    return model


if __name__ == "__main__":
    # 运行完整训练
    train_full_model()
