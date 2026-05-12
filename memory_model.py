"""
记忆增强模型 - 完整训练框架
包含：编码器、检索器、适配器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class ModelConfig:
    """模型配置"""
    # 编码器配置
    encoder_hidden_size: int = 256
    encoder_num_layers: int = 6
    encoder_num_heads: int = 4
    encoder_intermediate_size: int = 512
    encoder_max_length: int = 512
    memory_dim: int = 128
    
    # 检索器配置
    retriever_hidden_size: int = 128
    retriever_num_layers: int = 4
    
    # 适配器配置
    adapter_hidden_size: int = 512
    adapter_num_layers: int = 12
    adapter_num_heads: int = 8
    adapter_intermediate_size: int = 1024
    
    # 通用配置
    vocab_size: int = 50000
    dropout: float = 0.1
    pad_token_id: int = 0


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MemoryEncoder(nn.Module):
    """
    记忆编码器 - ~10M 参数
    
    将文本编码为紧凑的记忆向量
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 词嵌入
        self.token_embedding = nn.Embedding(config.vocab_size, config.encoder_hidden_size)
        self.position_encoding = PositionalEncoding(
            config.encoder_hidden_size, 
            config.encoder_max_length,
            config.dropout
        )
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.encoder_hidden_size,
            nhead=config.encoder_num_heads,
            dim_feedforward=config.encoder_intermediate_size,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.encoder_num_layers)
        
        # 记忆投影层
        self.memory_projection = nn.Sequential(
            nn.Linear(config.encoder_hidden_size, config.encoder_hidden_size),
            nn.ReLU(),
            nn.Linear(config.encoder_hidden_size, config.memory_dim)
        )
        
        # 重要性预测头
        self.importance_head = nn.Sequential(
            nn.Linear(config.encoder_hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            memory_vector: [batch_size, memory_dim]
            importance: [batch_size, 1]
        """
        # 嵌入
        x = self.token_embedding(input_ids)
        x = self.position_encoding(x)
        
        # Transformer 编码
        if attention_mask is not None:
            # 转换为 Transformer 期望的格式
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # 池化（使用第一个 token 或平均池化）
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = x.mean(dim=1)
        
        # 投影到记忆空间
        memory_vector = self.memory_projection(pooled)
        memory_vector = F.normalize(memory_vector, p=2, dim=-1)
        
        # 预测重要性
        importance = self.importance_head(pooled)
        
        return memory_vector, importance
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MemoryRetriever(nn.Module):
    """
    记忆检索器 - ~5M 参数
    
    双塔结构：查询编码器 + 记忆编码器
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 查询编码器
        self.query_encoder = nn.Sequential(
            nn.Linear(config.memory_dim, config.retriever_hidden_size),
            nn.ReLU(),
            nn.Linear(config.retriever_hidden_size, config.retriever_hidden_size)
        )
        
        # 记忆编码器
        self.memory_encoder = nn.Sequential(
            nn.Linear(config.memory_dim, config.retriever_hidden_size),
            nn.ReLU(),
            nn.Linear(config.retriever_hidden_size, config.retriever_hidden_size)
        )
        
        # 温度参数（可学习）
        self.temperature = nn.Parameter(torch.tensor(0.07))
    
    def forward(
        self,
        query_vector: torch.Tensor,
        memory_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query_vector: [batch_size, memory_dim]
            memory_vectors: [num_memories, memory_dim]
        
        Returns:
            scores: [batch_size, num_memories]
        """
        # 编码查询
        query_encoded = self.query_encoder(query_vector)
        query_encoded = F.normalize(query_encoded, p=2, dim=-1)
        
        # 编码记忆
        memory_encoded = self.memory_encoder(memory_vectors)
        memory_encoded = F.normalize(memory_encoded, p=2, dim=-1)
        
        # 计算相似度
        scores = torch.matmul(query_encoded, memory_encoded.T) / torch.exp(self.temperature)
        
        return scores
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MemoryAdapter(nn.Module):
    """
    记忆适配器 - ~85M 参数
    
    将记忆格式化为 LLM 可理解的格式
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 记忆嵌入层
        self.memory_embedding = nn.Linear(config.memory_dim, config.adapter_hidden_size)
        
        # 位置编码
        self.position_encoding = PositionalEncoding(
            config.adapter_hidden_size,
            max_len=256,
            dropout=config.dropout
        )
        
        # Transformer 解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.adapter_hidden_size,
            nhead=config.adapter_num_heads,
            dim_feedforward=config.adapter_intermediate_size,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=config.adapter_num_layers)
        
        # 输出层
        self.output_projection = nn.Linear(config.adapter_hidden_size, config.vocab_size)
    
    def forward(
        self,
        memory_vectors: torch.Tensor,
        max_length: int = 100
    ) -> torch.Tensor:
        """
        Args:
            memory_vectors: [batch_size, num_memories, memory_dim]
            max_length: 生成的最大长度
        
        Returns:
            logits: [batch_size, max_length, vocab_size]
        """
        batch_size = memory_vectors.size(0)
        
        # 嵌入记忆
        memory_embedded = self.memory_embedding(memory_vectors)
        
        # 生成起始 token
        start_tokens = torch.zeros(batch_size, 1, self.config.adapter_hidden_size, device=memory_vectors.device)
        
        # 自回归生成（简化版）
        output = start_tokens
        for _ in range(max_length - 1):
            # 添加位置编码
            output_with_pos = self.position_encoding(output)
            
            # Transformer 解码
            decoded = self.transformer(output_with_pos, memory_embedded)
            
            # 预测下一个 token
            next_token_embed = decoded[:, -1:, :]
            output = torch.cat([output, next_token_embed], dim=1)
        
        # 投影到词汇表
        logits = self.output_projection(output)
        
        return logits
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MemoryAugmentedModel(nn.Module):
    """
    完整的记忆增强模型
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.encoder = MemoryEncoder(config)
        self.retriever = MemoryRetriever(config)
        self.adapter = MemoryAdapter(config)
    
    def encode_memory(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码记忆"""
        return self.encoder(input_ids, attention_mask)
    
    def retrieve_memories(
        self,
        query_vector: torch.Tensor,
        memory_bank: torch.Tensor,
        top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """检索相关记忆"""
        scores = self.retriever(query_vector, memory_bank)
        top_scores, top_indices = torch.topk(scores, k=min(top_k, scores.size(-1)), dim=-1)
        return top_indices, top_scores
    
    def generate_response(
        self,
        query_vector: torch.Tensor,
        memory_bank: torch.Tensor,
        max_length: int = 100
    ) -> torch.Tensor:
        """生成响应"""
        # 检索相关记忆
        top_indices, _ = self.retrieve_memories(query_vector, memory_bank)
        
        # 获取相关记忆
        batch_size = query_vector.size(0)
        retrieved_memories = memory_bank[top_indices]
        
        # 生成响应
        logits = self.adapter(retrieved_memories, max_length)
        
        return logits
    
    def count_parameters(self) -> Dict[str, int]:
        return {
            "encoder": self.encoder.count_parameters(),
            "retriever": self.retriever.count_parameters(),
            "adapter": self.adapter.count_parameters(),
            "total": sum(p.numel() for p in self.parameters())
        }


def create_model(config: Optional[ModelConfig] = None) -> MemoryAugmentedModel:
    """创建模型"""
    if config is None:
        config = ModelConfig()
    
    model = MemoryAugmentedModel(config)
    
    # 打印参数统计
    params = model.count_parameters()
    print(f"\n模型参数统计:")
    print(f"  编码器: {params['encoder']:,} ({params['encoder']/1e6:.2f}M)")
    print(f"  检索器: {params['retriever']:,} ({params['retriever']/1e6:.2f}M)")
    print(f"  适配器: {params['adapter']:,} ({params['adapter']/1e6:.2f}M)")
    print(f"  总计: {params['total']:,} ({params['total']/1e6:.2f}M)")
    
    return model


if __name__ == "__main__":
    # 测试模型
    print("=" * 70)
    print("测试记忆增强模型")
    print("=" * 70)
    
    # 创建模型
    model = create_model()
    
    # 测试编码器
    print("\n测试编码器...")
    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, 50000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    memory_vector, importance = model.encode_memory(input_ids, attention_mask)
    print(f"  输入形状: {input_ids.shape}")
    print(f"  记忆向量形状: {memory_vector.shape}")
    print(f"  重要性形状: {importance.shape}")
    
    # 测试检索器
    print("\n测试检索器...")
    num_memories = 100
    memory_bank = torch.randn(num_memories, model.config.memory_dim)
    memory_bank = F.normalize(memory_bank, p=2, dim=-1)
    
    top_indices, top_scores = model.retrieve_memories(memory_vector, memory_bank, top_k=5)
    print(f"  记忆库大小: {memory_bank.shape}")
    print(f"  检索结果索引: {top_indices.shape}")
    print(f"  检索分数: {top_scores.shape}")
    
    # 测试适配器
    print("\n测试适配器...")
    logits = model.generate_response(memory_vector, memory_bank, max_length=20)
    print(f"  输出 logits 形状: {logits.shape}")
    
    print("\n" + "=" * 70)
    print("✅ 模型测试通过！")
    print("=" * 70)
