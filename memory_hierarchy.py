"""
MemoryHierarchy - 三层记忆系统
工作记忆 -> 近期记忆 -> 长期记忆，自动管理生命周期
"""

import time
import heapq
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict

from vector_store import VectorStore, SearchResult


class MemoryTier(Enum):
    WORKING = "working"
    RECENT = "recent"
    LONG_TERM = "long_term"


@dataclass
class MemoryItem:
    id: str
    content: str
    tier: MemoryTier
    priority: float = 0.5
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_summary: bool = False
    original_ids: List[str] = field(default_factory=list)


class LRUCache:
    """简单但高效的 LRU 缓存"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()
    
    def get(self, key: str) -> Optional[MemoryItem]:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: str, value: MemoryItem) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
    
    def __len__(self) -> int:
        return len(self.cache)


class MemoryHierarchy:
    """
    三层记忆层次系统
    
    Tier 1: Working Memory (高优先级，LRU)
    Tier 2: Recent Memory (向量检索)
    Tier 3: Long-Term Memory (压缩 + 分层索引)
    """
    
    def __init__(
        self,
        working_capacity: int = 20,
        recent_capacity: int = 500,
        vector_dimension: int = 128,
        decay_days: float = 7.0
    ):
        self.working_capacity = working_capacity
        self.recent_capacity = recent_capacity
        self.decay_seconds = decay_days * 24 * 3600
        
        self.working_memory: LRUCache = LRUCache(working_capacity)
        self.recent_memory: Dict[str, MemoryItem] = {}
        self.long_term_memory: Dict[str, MemoryItem] = {}
        
        self.vector_store = VectorStore(dimension=vector_dimension)
        
        self._id_to_tier: Dict[str, MemoryTier] = {}
    
    def _calculate_priority(self, item: MemoryItem) -> float:
        """计算记忆优先级（时间衰减 + 访问频率）"""
        age = time.time() - item.created_at
        recency = time.time() - item.accessed_at
        
        time_decay = max(0.1, 1.0 - age / self.decay_seconds)
        access_bonus = min(1.0, item.access_count * 0.1)
        recency_bonus = max(0.1, 1.0 - recency / (24 * 3600))
        
        priority = (
            item.priority * 0.4 +
            time_decay * 0.3 +
            access_bonus * 0.2 +
            recency_bonus * 0.1
        )
        return max(0.0, min(1.0, priority))
    
    def _should_promote(self, item: MemoryItem) -> bool:
        """判断是否应该升级记忆层级"""
        priority = self._calculate_priority(item)
        return priority > 0.6 or item.access_count > 5
    
    def _should_demote(self, item: MemoryItem) -> bool:
        """判断是否应该降级记忆层级"""
        priority = self._calculate_priority(item)
        return priority < 0.3 and item.access_count < 3
    
    def add(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        priority: float = 0.5
    ) -> str:
        """
        添加新记忆（默认从 Working Memory 开始）
        
        Returns:
            记忆 ID
        """
        memory_id = f"mem_{int(time.time())}_{hash(content) % 10000}"
        
        item = MemoryItem(
            id=memory_id,
            content=content,
            tier=MemoryTier.WORKING,
            priority=priority,
            metadata=metadata or {}
        )
        
        self.working_memory.put(memory_id, item)
        self._id_to_tier[memory_id] = MemoryTier.WORKING
        
        self.vector_store.add(content, metadata={"memory_id": memory_id})
        
        self._manage_lifecycle()
        
        return memory_id
    
    def get(self, memory_id: str) -> Optional[MemoryItem]:
        """获取记忆（自动提升访问热度）"""
        item = None
        
        if memory_id in self._id_to_tier:
            tier = self._id_to_tier[memory_id]
            
            if tier == MemoryTier.WORKING:
                item = self.working_memory.get(memory_id)
            elif tier == MemoryTier.RECENT:
                item = self.recent_memory.get(memory_id)
            elif tier == MemoryTier.LONG_TERM:
                item = self.long_term_memory.get(memory_id)
        
        if item:
            item.access_count += 1
            item.accessed_at = time.time()
            if self._should_promote(item):
                self._promote(item)
        
        return item
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        min_relevance: float = 0.2,
        tiers: Optional[List[MemoryTier]] = None
    ) -> List[SearchResult]:
        """
        智能搜索：先从 Working Memory 开始，逐步扩展
        
        Args:
            query: 查询文本
            top_k: 返回结果数
            min_relevance: 最小相关度
            tiers: 可选的层级限制
        """
        if tiers is None:
            tiers = [MemoryTier.WORKING, MemoryTier.RECENT, MemoryTier.LONG_TERM]
        
        results: List[SearchResult] = []
        
        for tier in tiers:
            tier_results = self._search_tier(query, tier, top_k, min_relevance)
            results.extend(tier_results)
            
            if len(results) >= top_k:
                break
        
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:top_k]
    
    def _search_tier(
        self,
        query: str,
        tier: MemoryTier,
        top_k: int,
        min_relevance: float
    ) -> List[SearchResult]:
        """搜索特定层级"""
        candidates: List[MemoryItem] = []
        
        if tier == MemoryTier.WORKING:
            candidates = list(self.working_memory.cache.values())
        elif tier == MemoryTier.RECENT:
            candidates = list(self.recent_memory.values())
        elif tier == MemoryTier.LONG_TERM:
            candidates = list(self.long_term_memory.values())
        
        results = self.vector_store.search(query, top_k=top_k, threshold=min_relevance)
        
        for result in results:
            memory_id = result.metadata.get("memory_id")
            item = self._get_item_by_id(memory_id)
            if item:
                result.metadata["text"] = item.content
                result.metadata["tier"] = tier.value
                result.metadata["access_count"] = item.access_count
        
        return results
    
    def _get_item_by_id(self, memory_id: str) -> Optional[MemoryItem]:
        """从任意层级获取记忆项"""
        if memory_id in self.working_memory.cache:
            return self.working_memory.get(memory_id)
        if memory_id in self.recent_memory:
            return self.recent_memory[memory_id]
        if memory_id in self.long_term_memory:
            return self.long_term_memory[memory_id]
        return None
    
    def _promote(self, item: MemoryItem) -> None:
        """将记忆升级到更高层级"""
        old_tier = item.tier
        
        if old_tier == MemoryTier.LONG_TERM:
            if item.id in self.long_term_memory:
                del self.long_term_memory[item.id]
            item.tier = MemoryTier.RECENT
            self.recent_memory[item.id] = item
            self._id_to_tier[item.id] = MemoryTier.RECENT
        
        elif old_tier == MemoryTier.RECENT:
            if item.id in self.recent_memory:
                del self.recent_memory[item.id]
            item.tier = MemoryTier.WORKING
            self.working_memory.put(item.id, item)
            self._id_to_tier[item.id] = MemoryTier.WORKING
    
    def _demote(self, item: MemoryItem) -> None:
        """将记忆降级到更低层级"""
        old_tier = item.tier
        
        if old_tier == MemoryTier.WORKING:
            if item.id in self.working_memory.cache:
                del self.working_memory.cache[item.id]
            item.tier = MemoryTier.RECENT
            self.recent_memory[item.id] = item
            self._id_to_tier[item.id] = MemoryTier.RECENT
        
        elif old_tier == MemoryTier.RECENT:
            if item.id in self.recent_memory:
                del self.recent_memory[item.id]
            item.tier = MemoryTier.LONG_TERM
            self.long_term_memory[item.id] = item
            self._id_to_tier[item.id] = MemoryTier.LONG_TERM
    
    def _manage_lifecycle(self) -> None:
        """自动管理记忆生命周期"""
        if len(self.recent_memory) > self.recent_capacity:
            sorted_recent = sorted(
                self.recent_memory.values(),
                key=lambda x: self._calculate_priority(x)
            )
            to_demote = sorted_recent[:len(self.recent_memory) - self.recent_capacity]
            for item in to_demote:
                self._demote(item)
    
    def summarize_old_memories(self, days: int = 30) -> int:
        """将长期记忆进行摘要压缩"""
        cutoff = time.time() - days * 24 * 3600
        to_summarize = [
            item for item in self.long_term_memory.values()
            if item.created_at < cutoff and not item.is_summary
        ]
        
        count = 0
        for item in to_summarize:
            summary_content = f"[摘要] {item.content[:200]}..."
            summary_id = f"sum_{item.id}"
            
            summary_item = MemoryItem(
                id=summary_id,
                content=summary_content,
                tier=MemoryTier.LONG_TERM,
                priority=item.priority * 0.7,
                is_summary=True,
                original_ids=[item.id]
            )
            
            self.long_term_memory[summary_id] = summary_item
            self._id_to_tier[summary_id] = MemoryTier.LONG_TERM
            self.vector_store.add(summary_content, metadata={"memory_id": summary_id})
            count += 1
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """获取层级系统统计"""
        return {
            "working_memory_count": len(self.working_memory),
            "recent_memory_count": len(self.recent_memory),
            "long_term_memory_count": len(self.long_term_memory),
            "total_memories": (
                len(self.working_memory) +
                len(self.recent_memory) +
                len(self.long_term_memory)
            ),
            "vector_store": self.vector_store.get_stats()
        }
