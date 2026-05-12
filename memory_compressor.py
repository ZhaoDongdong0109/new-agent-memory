"""
Memory Compressor - 记忆压缩与摘要
智能压缩不重要的记忆，节省存储空间和上下文
"""

import re
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class CompressedMemory:
    id: str
    original_ids: List[str]
    summary: str
    key_points: List[str]
    keywords: Set[str]
    compression_ratio: float
    created_at: float
    is_important: bool = False


class MemoryCompressor:
    """
    记忆压缩与摘要系统
    
    策略：
    - 抽取关键词和关键句
    - 按重要性分级压缩
    - 保持语义连贯性
    - 支持压缩记忆的唤醒
    """
    
    def __init__(
        self,
        compression_threshold: int = 1000,
        min_important_score: float = 0.3
    ):
        self.compression_threshold = compression_threshold
        self.min_important_score = min_important_score
        self.compressed: Dict[str, CompressedMemory] = {}
        
        self._keywords_cache = defaultdict(int)
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """简单关键词提取"""
        words = re.findall(r'[\w\u4e00-\u9fff]+', text.lower())
        stop_words = {'的', '了', '和', '是', '就', '都', '而', '及', '等', '这', '那', '有', '在', '我', '你', '他'}
        keywords = {
            w for w in words 
            if len(w) > 1 and w not in stop_words
        }
        
        for kw in keywords:
            self._keywords_cache[kw] += 1
        
        return keywords
    
    def _extract_key_points(self, text: str) -> List[str]:
        """提取关键句子/短语"""
        sentences = re.split(r'[。！？.!?]', text)
        key_points = []
        
        for s in sentences:
            s = s.strip()
            if len(s) > 10:
                key_points.append(s)
        
        return key_points[:5]
    
    def _calculate_importance(self, text: str, access_count: int = 0) -> float:
        """计算文本重要性分数"""
        score = 0.0
        
        keywords = self._extract_keywords(text)
        rare_kw_count = sum(
            1 for kw in keywords 
            if self._keywords_cache[kw] < 5
        )
        
        score += rare_kw_count * 0.1
        score += min(access_count * 0.1, 0.5)
        
        if any(kw in text.lower() for kw in ['重要', '关键', '必须', '记住', 'remember', 'important', 'must']):
            score += 0.3
        
        return min(1.0, score)
    
    def compress(
        self,
        original_text: str,
        original_ids: List[str],
        access_count: int = 0
    ) -> CompressedMemory:
        """
        压缩单个或多个记忆
        
        Returns:
            CompressedMemory 对象
        """
        importance = self._calculate_importance(original_text, access_count)
        keywords = self._extract_keywords(original_text)
        key_points = self._extract_key_points(original_text)
        
        if importance > self.min_important_score:
            summary = original_text[:500]
            if len(original_text) > 500:
                summary += "..."
        else:
            summary = " ".join(key_points[:2])
            if len(keywords) > 3:
                summary += f"（关键词: {', '.join(list(keywords)[:5])}）"
        
        original_length = len(original_text)
        compressed_length = len(summary)
        compression_ratio = (original_length - compressed_length) / max(original_length, 1)
        
        compressed_id = f"comp_{int(time.time())}_{hash(original_text) % 10000}"
        
        compressed = CompressedMemory(
            id=compressed_id,
            original_ids=original_ids,
            summary=summary,
            key_points=key_points,
            keywords=keywords,
            compression_ratio=compression_ratio,
            created_at=time.time(),
            is_important=importance > 0.5
        )
        
        self.compressed[compressed_id] = compressed
        return compressed
    
    def compress_batch(
        self,
        items: List[Dict[str, Any]]
    ) -> List[CompressedMemory]:
        """批量压缩记忆"""
        results = []
        
        for item in items:
            compressed = self.compress(
                item.get("content", ""),
                item.get("original_ids", []),
                item.get("access_count", 0)
            )
            results.append(compressed)
        
        return results
    
    def search_compressed(
        self,
        query: str,
        top_k: int = 5
    ) -> List[CompressedMemory]:
        """
        在压缩记忆中搜索
        
        策略：关键词匹配 + 重要性排序
        """
        query_kw = self._extract_keywords(query)
        scored = []
        
        for compressed in self.compressed.values():
            matches = len(query_kw.intersection(compressed.keywords))
            score = matches * 0.3
            if compressed.is_important:
                score += 0.4
            score += 0.3 / (1 + (time.time() - compressed.created_at) / (30 * 24 * 3600))
            
            scored.append((score, compressed))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取压缩统计"""
        if not self.compressed:
            return {"total_compressed": 0}
        
        total_ratio = sum(c.compression_ratio for c in self.compressed.values())
        important_count = sum(1 for c in self.compressed.values() if c.is_important)
        
        return {
            "total_compressed": len(self.compressed),
            "average_compression_ratio": total_ratio / len(self.compressed),
            "important_count": important_count,
            "total_keywords_cached": len(self._keywords_cache)
        }
