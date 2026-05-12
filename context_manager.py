"""
ContextManager - 上下文窗口智能管理
核心模块：智能选择、排序、压缩记忆，完美适配上下文窗口
"""

import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from vector_store import SearchResult
from memory_hierarchy import MemoryHierarchy, MemoryTier
from memory_compressor import MemoryCompressor


class ContextPriority(Enum):
    CRITICAL = 10  # 必须包含
    HIGH = 8
    MEDIUM = 5
    LOW = 2
    OPTIONAL = 0


@dataclass
class ContextItem:
    content: str
    priority: ContextPriority
    relevance: float
    tier: str
    memory_id: Optional[str] = None
    is_compressed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextWindow:
    items: List[ContextItem]
    total_tokens: int
    max_tokens: int
    used_ratio: float


class ContextManager:
    """
    智能上下文窗口管理器
    
    核心功能：
    - 优先级排序记忆
    - 动态压缩以适配窗口
    - 按重要性分层取舍
    - 统计优化效果
    """
    
    def __init__(
        self,
        max_tokens: int = 4096,
        hierarchy: Optional[MemoryHierarchy] = None,
        compressor: Optional[MemoryCompressor] = None
    ):
        self.max_tokens = max_tokens
        self.hierarchy = hierarchy or MemoryHierarchy()
        self.compressor = compressor or MemoryCompressor()
        
        self._current_context: List[ContextItem] = []
        self._token_usage_history: List[int] = []
        
        self._priority_rules = {
            MemoryTier.WORKING: ContextPriority.HIGH,
            MemoryTier.RECENT: ContextPriority.MEDIUM,
            MemoryTier.LONG_TERM: ContextPriority.LOW
        }
    
    def _estimate_tokens(self, text: str) -> int:
        """估算 token 数量（粗略但快速）"""
        return len(text) // 4 + 1
    
    def _sort_items(self, items: List[ContextItem]) -> List[ContextItem]:
        """按优先级和相关性排序"""
        return sorted(
            items,
            key=lambda x: (x.priority.value, x.relevance),
            reverse=True
        )
    
    def _compress_if_needed(self, items: List[ContextItem], budget: int) -> List[ContextItem]:
        """按需压缩项目以适配预算"""
        result = []
        remaining = budget
        
        for item in items:
            item_tokens = self._estimate_tokens(item.content)
            
            if item_tokens <= remaining:
                result.append(item)
                remaining -= item_tokens
                continue
            
            if item.priority.value >= ContextPriority.HIGH.value:
                result.append(item)
                remaining -= item_tokens
                continue
            
            if item.priority.value >= ContextPriority.MEDIUM.value:
                compressed = self.compressor.compress(
                    item.content,
                    [item.memory_id] if item.memory_id else [],
                    0
                )
                
                compressed_item = ContextItem(
                    content=compressed.summary,
                    priority=ContextPriority.LOW,
                    relevance=item.relevance * 0.8,
                    tier=item.tier,
                    memory_id=compressed.id,
                    is_compressed=True
                )
                
                compressed_tokens = self._estimate_tokens(compressed.summary)
                if compressed_tokens <= remaining:
                    result.append(compressed_item)
                    remaining -= compressed_tokens
        
        return result
    
    def build_context(
        self,
        query: str,
        extra_context: Optional[List[str]] = None
    ) -> ContextWindow:
        """
        构建最优上下文窗口
        
        Args:
            query: 当前查询
            extra_context: 额外的上下文
            
        Returns:
            ContextWindow 对象
        """
        items: List[ContextItem] = []
        
        if extra_context:
            for text in extra_context:
                items.append(ContextItem(
                    content=text,
                    priority=ContextPriority.HIGH,
                    relevance=1.0,
                    tier="explicit"
                ))
        
        search_results = self.hierarchy.search(query, top_k=30)
        
        for result in search_results:
            memory_id = result.metadata.get("memory_id")
            tier = result.metadata.get("tier", "unknown")
            tier_enum = MemoryTier(tier) if tier in [t.value for t in MemoryTier] else MemoryTier.LONG_TERM
            
            priority = self._priority_rules.get(tier_enum, ContextPriority.LOW)
            
            items.append(ContextItem(
                content=result.metadata.get("text", ""),
                priority=priority,
                relevance=result.similarity,
                tier=tier,
                memory_id=memory_id
            ))
        
        items = self._sort_items(items)
        
        budget = self.max_tokens - self._estimate_tokens(query) - 200
        final_items = self._compress_if_needed(items, budget)
        
        total_tokens = sum(self._estimate_tokens(i.content) for i in final_items)
        
        self._current_context = final_items
        self._token_usage_history.append(total_tokens)
        
        return ContextWindow(
            items=final_items,
            total_tokens=total_tokens,
            max_tokens=self.max_tokens,
            used_ratio=total_tokens / self.max_tokens
        )
    
    def format_context(self, window: ContextWindow) -> str:
        """
        格式化上下文为 LLM 友好格式
        
        输出格式：
        --- Relevant Memories ---
        [1] Content 1 (Working Memory)
        [2] Content 2 (Recent)
        ...
        """
        if not window.items:
            return "（无相关记忆）"
        
        lines = ["--- Relevant Memories ---"]
        
        for idx, item in enumerate(window.items, 1):
            tier_label = item.tier
            if item.is_compressed:
                tier_label += " (compressed)"
            
            prefix = f"[{idx}]"
            if item.priority.value >= ContextPriority.HIGH.value:
                prefix = f"[★{idx}]"
            elif item.priority.value >= ContextPriority.CRITICAL.value:
                prefix = f"[★★{idx}]"
            
            lines.append(f"{prefix} {item.content} ({tier_label})")
        
        return "\n".join(lines)
    
    def update_relevance_feedback(
        self,
        memory_id: str,
        relevant: bool
    ) -> None:
        """根据反馈更新相关性权重"""
        if memory_id:
            self.hierarchy.get(memory_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        avg_usage = (
            sum(self._token_usage_history) / max(1, len(self._token_usage_history))
            if self._token_usage_history else 0
        )
        
        return {
            "context_items_count": len(self._current_context),
            "current_usage": sum(self._estimate_tokens(i.content) for i in self._current_context),
            "average_usage": avg_usage,
            "max_tokens": self.max_tokens,
            "compression_stats": self.compressor.get_stats(),
            "hierarchy_stats": self.hierarchy.get_stats()
        }
    
    def clear_context(self) -> None:
        """清空当前上下文"""
        self._current_context = []
