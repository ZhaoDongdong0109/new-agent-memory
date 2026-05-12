"""
VectorStore - 向量存储与相似度检索
这是新架构的核心模块，支持语义级别的记忆检索
"""

import math
import random
import hashlib
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time


class SimilarityMetric(Enum):
    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"


@dataclass
class VectorRecord:
    id: str
    vector: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0


@dataclass
class SearchResult:
    id: str
    similarity: float
    metadata: Dict[str, Any]
    vector: Optional[List[float]] = None


class VectorStore:
    """
    高效的向量存储与相似度检索引擎
    
    特性:
    - 支持自定义向量维度
    - 多种相似度计算方式
    - 批量搜索优化
    - 内存高效的存储方式
    """
    
    def __init__(
        self,
        dimension: int = 128,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
        index_file: Optional[str] = None
    ):
        self.dimension = dimension
        self.metric = metric
        self.vectors: Dict[str, VectorRecord] = {}
        
        self._index_by_tags: Dict[str, Set[str]] = {}
        self._index_by_keyword: Dict[str, Set[str]] = {}
        
        if index_file:
            self.load(index_file)
    
    def _random_vector(self) -> List[float]:
        """生成随机向量（用于演示）"""
        return [random.uniform(-1, 1) for _ in range(self.dimension)]
    
    def _text_to_vector(self, text: str) -> List[float]:
        """
        文本到向量的简单实现（生产环境应使用真实的 embedding 模型）
        
        这里使用哈希 + 随机化作为演示，实际项目中使用：
        - OpenAI Embeddings API
        - sentence-transformers
        - 本地部署的 embedding 模型
        """
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        random.seed(seed)
        
        vector = []
        for i in range(self.dimension):
            char_val = ord(text[i % len(text)]) if text else 0
            random_val = random.uniform(-1, 1)
            vector.append((char_val / 255 - 0.5) * 2 * 0.3 + random_val * 0.7)
        
        norm = math.sqrt(sum(v*v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector
    
    def add(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        vector: Optional[List[float]] = None
    ) -> str:
        """
        添加文本到向量存储
        
        Args:
            text: 文本内容
            metadata: 额外元数据
            tags: 标签列表（用于索引）
            vector: 可选的预计算向量
            
        Returns:
            记录 ID
        """
        record_id = f"vec_{hashlib.md5(text.encode()).hexdigest()[:12]}"
        
        if vector and len(vector) != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {len(vector)}")
        
        if vector is None:
            vector = self._text_to_vector(text)
        
        record = VectorRecord(
            id=record_id,
            vector=vector,
            metadata=metadata or {"text": text}
        )
        self.vectors[record_id] = record
        
        if tags:
            for tag in tags:
                if tag not in self._index_by_tags:
                    self._index_by_tags[tag] = set()
                self._index_by_tags[tag].add(record_id)
        
        return record_id
    
    def get(self, record_id: str) -> Optional[VectorRecord]:
        """获取记录"""
        record = self.vectors.get(record_id)
        if record:
            record.access_count += 1
            record.accessed_at = time.time()
        return record
    
    def remove(self, record_id: str) -> bool:
        """删除记录"""
        if record_id not in self.vectors:
            return False
        
        for tag_set in self._index_by_tags.values():
            tag_set.discard(record_id)
        
        del self.vectors[record_id]
        return True
    
    def similarity(self, v1: List[float], v2: List[float]) -> float:
        """计算两个向量的相似度"""
        if self.metric == SimilarityMetric.COSINE:
            dot = sum(a * b for a, b in zip(v1, v2))
            norm1 = math.sqrt(sum(a * a for a in v1))
            norm2 = math.sqrt(sum(b * b for b in v2))
            return dot / (norm1 * norm2 + 1e-10)
        
        elif self.metric == SimilarityMetric.DOT:
            return sum(a * b for a, b in zip(v1, v2))
        
        elif self.metric == SimilarityMetric.EUCLIDEAN:
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))
            return 1 / (1 + dist)
        
        return 0
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.0,
        tags: Optional[List[str]] = None,
        include_vector: bool = False
    ) -> List[SearchResult]:
        """
        搜索最相似的记录
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            threshold: 相似度阈值
            tags: 可选的标签过滤
            include_vector: 是否包含向量
            
        Returns:
            搜索结果列表，按相似度降序排列
        """
        query_vec = self._text_to_vector(query)
        
        candidate_ids = set(self.vectors.keys())
        if tags:
            tag_matches = set()
            for tag in tags:
                tag_matches.update(self._index_by_tags.get(tag, set()))
            if tag_matches:
                candidate_ids = candidate_ids.intersection(tag_matches)
        
        results = []
        for record_id in candidate_ids:
            record = self.vectors[record_id]
            sim = self.similarity(query_vec, record.vector)
            
            if sim >= threshold:
                record.access_count += 1
                record.accessed_at = time.time()
                
                result = SearchResult(
                    id=record.id,
                    similarity=sim,
                    metadata=record.metadata.copy(),
                    vector=record.vector if include_vector else None
                )
                results.append(result)
        
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:top_k]
    
    def search_by_vector(
        self,
        vector: List[float],
        top_k: int = 10,
        threshold: float = 0.0,
        tags: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """直接使用向量搜索"""
        query_text = " ".join([f"{i:.2f}" for i in vector[:5]])
        return self.search(query_text, top_k, threshold, tags, include_vector=True)
    
    def batch_add(
        self,
        items: List[Tuple[str, Optional[Dict[str, Any]], Optional[List[str]]]]
    ) -> List[str]:
        """批量添加"""
        ids = []
        for text, metadata, tags in items:
            ids.append(self.add(text, metadata, tags))
        return ids
    
    def batch_search(
        self,
        queries: List[str],
        top_k: int = 10
    ) -> Dict[str, List[SearchResult]]:
        """批量搜索"""
        results = {}
        for query in queries:
            results[query] = self.search(query, top_k)
        return results
    
    def save(self, filepath: str) -> None:
        """保存到文件"""
        import pickle
        data = {
            "dimension": self.dimension,
            "metric": self.metric.value,
            "vectors": {
                k: {
                    "id": v.id,
                    "vector": v.vector,
                    "metadata": v.metadata,
                    "created_at": v.created_at,
                    "accessed_at": v.accessed_at,
                    "access_count": v.access_count
                }
                for k, v in self.vectors.items()
            },
            "index_by_tags": {k: list(v) for k, v in self._index_by_tags.items()}
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str) -> None:
        """从文件加载"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.dimension = data["dimension"]
        self.metric = SimilarityMetric(data["metric"])
        self.vectors = {
            k: VectorRecord(**v)
            for k, v in data["vectors"].items()
        }
        self._index_by_tags = {
            k: set(v) for k, v in data["index_by_tags"].items()
        }
    
    def clear(self) -> None:
        """清空存储"""
        self.vectors.clear()
        self._index_by_tags.clear()
    
    def __len__(self) -> int:
        return len(self.vectors)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_vectors": len(self.vectors),
            "tags_count": len(self._index_by_tags),
            "avg_access_count": (
                sum(v.access_count for v in self.vectors.values()) / max(1, len(self.vectors))
            ),
            "memory_estimate_mb": (
                sum(len(v.vector) * 8 for v in self.vectors.values()) / (1024 * 1024)
            )
        }
