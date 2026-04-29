"""
伪遗忘层管理 - 类人记忆系统

特点：
- 极低权重记忆的归档层
- 不参与主动检索
- 需要"信息锚点"才能唤醒
- 可以被删除（对系统无影响）
"""

from typing import Dict, List, Optional, Set, Any
import math
import time
import json
from dataclasses import dataclass

from memory_chunk import MemoryChunk, MemoryLayer


@dataclass
class ForgottenLayerStats:
    """伪遗忘层统计"""
    total_chunks: int = 0
    oldest_age_days: float = 0.0
    avg_weight: float = 0.0
    chunk_types: Dict[str, int] = None
    
    def __post_init__(self):
        self.chunk_types = self.chunk_types or {}


class ForgottenLayer:
    """
    伪遗忘层
    
    设计理念：
    - 不是"删除"，而是"归档"
    - 不主动检索，但可以通过信息锚点唤醒
    - 可以被清理，不影响系统运行
    """
    
    def __init__(
        self,
        # 唤醒参数
        wake_threshold: float = 0.3,        # 需要多强的锚点匹配才能唤醒
        min_match_tags: int = 2,            # 最少匹配几个标签才能唤醒
        
        # 清理参数
        auto_cleanup: bool = True,          # 是否自动清理
        cleanup_age_days: float = 365,       # 超过多少天自动清理
        cleanup_weight_max: float = 0.05,   # 权重低于此值且超过年龄才清理
        
        # 唤醒后参数
        wake_weight_boost: float = 0.2,     # 唤醒时临时权重提升
    ):
        self.wake_threshold = wake_threshold
        self.min_match_tags = min_match_tags
        self.auto_cleanup = auto_cleanup
        self.cleanup_age_days = cleanup_age_days
        self.cleanup_weight_max = cleanup_weight_max
        self.wake_weight_boost = wake_weight_boost
        
        # 存储
        self.chunks: Dict[str, MemoryChunk] = {}
        
        # 唤醒记录
        self.total_wake_attempts = 0
        self.total_wake_success = 0
    
    # ============ 归档操作 ============
    
    def archive(self, chunk: MemoryChunk) -> str:
        """将记忆归档到伪遗忘层"""
        chunk.layer = MemoryLayer.FORGOTTEN
        chunk.updated_at = time.time()
        self.chunks[chunk.id] = chunk
        return chunk.id
    
    def get(self, chunk_id: str) -> Optional[MemoryChunk]:
        """获取记忆"""
        return self.chunks.get(chunk_id)
    
    def remove(self, chunk_id: str) -> Optional[MemoryChunk]:
        """彻底删除记忆"""
        return self.chunks.pop(chunk_id, None)
    
    # ============ 唤醒机制 ============
    
    def calc_wake_score(self, chunk: MemoryChunk, query_tags: Dict[str, Any]) -> float:
        """
        计算唤醒得分
        
        锚点匹配越强，得分越高
        """
        score = 0.0
        matched_tags = 0
        
        # 时间标签匹配（最重要）
        if "time_absolute" in query_tags:
            if chunk.time_absolute == query_tags["time_absolute"]:
                score += 0.3
                matched_tags += 1
            elif chunk.time_relative and chunk.time_relative == query_tags.get("time_relative"):
                score += 0.2
                matched_tags += 1
        
        # 地点匹配
        if "location" in query_tags:
            if chunk.location == query_tags["location"]:
                score += 0.2
                matched_tags += 1
        
        # 人物匹配
        if "persons" in query_tags:
            matched_persons = query_tags["persons"] & chunk.persons
            if matched_persons:
                score += 0.15 * (len(matched_persons) / max(len(query_tags["persons"]), 1))
                matched_tags += 1
        
        # 主题匹配
        if "topics" in query_tags:
            matched_topics = query_tags["topics"] & chunk.topics
            if matched_topics:
                score += 0.15 * (len(matched_topics) / max(len(query_tags["topics"]), 1))
                matched_tags += 1
        
        # 情绪方向匹配
        if "emotion_valence" in query_tags:
            if (chunk.emotion_valence > 0) == (query_tags["emotion_valence"] > 0):
                score += 0.1
                matched_tags += 1
        
        # 重要性加成（重要的记忆更容易被唤醒）
        score += chunk.importance * 0.1
        
        return score, matched_tags
    
    def try_wake(
        self,
        query_tags: Dict[str, Any],
        limit: int = 5,
    ) -> List[tuple]:
        """
        尝试唤醒伪遗忘层的记忆
        
        query_tags: 信息锚点（来自外部输入，如照片、问句等）
        
        返回：[(碎片, 唤醒得分), ...]
        """
        self.total_wake_attempts += 1
        
        candidates = []
        
        for chunk in self.chunks.values():
            score, matched_tags = self.calc_wake_score(chunk, query_tags)
            
            # 必须匹配足够多的标签
            if matched_tags >= self.min_match_tags and score >= self.wake_threshold:
                # 计算临时权重（唤醒时提升）
                temp_weight = min(1.0, score + self.wake_weight_boost)
                candidates.append((chunk, score, temp_weight))
        
        # 按得分降序
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        result = [(c, tw) for c, s, tw in candidates[:limit]]
        
        if result:
            self.total_wake_success += 1
        
        return result
    
    def wake_and_promote(
        self,
        query_tags: Dict[str, Any],
        promotion_weight_threshold: float = 0.4,
    ) -> List[MemoryChunk]:
        """
        唤醒记忆，并提升回核心层
        
        如果唤醒后得分足够高（可能是有价值的记忆），放回核心层
        
        返回：被提升到核心层的记忆列表
        """
        candidates = self.try_wake(query_tags, limit=10)
        
        promoted = []
        for chunk, temp_weight in candidates:
            chunk.successful_recall()
            
            # 如果唤醒得分很高（记忆很有价值），提升回核心层
            # 但目前我们无法在伪遗忘层准确计算权重，所以用临时权重代替
            if temp_weight >= promotion_weight_threshold:
                promoted.append(chunk)
        
        return promoted
    
    # ============ 审阅 ============
    
    def review(
        self,
        chunk_id: str,
        decision: str,  # "approve", "questionable", "reject"
        note: Optional[str] = None,
    ):
        """
        审阅伪遗忘层的记忆
        
        decision:
        - approve: 记忆合理，提升权重
        - questionable: 记忆存疑
        - reject: 记忆不可信，标记
        """
        chunk = self.chunks.get(chunk_id)
        if not chunk:
            return
        
        chunk.review_status = decision
        chunk.review_note = note
        
        if decision == "approve":
            # 合理的记忆，提升一些权重
            chunk.importance = min(1.0, chunk.importance + 0.1)
        elif decision == "reject":
            # 标记为不可信，降低重要性
            chunk.importance = max(0.0, chunk.importance - 0.2)
    
    # ============ 清理 ============
    
    def cleanup(self) -> List[str]:
        """
        清理伪遗忘层
        
        清理条件：
        1. 权重极低
        2. 超过一定年龄
        3. 审阅结果为 reject
        
        返回：被清理的记忆ID列表
        """
        if not self.auto_cleanup:
            return []
        
        now = time.time()
        to_remove = []
        
        for chunk_id, chunk in self.chunks.items():
            age_days = (now - chunk.created_at) / (24 * 3600)
            
            # 条件1：年龄超过阈值
            if age_days < self.cleanup_age_days:
                continue
            
            # 条件2：重要性极低
            if chunk.importance > self.cleanup_weight_max:
                continue
            
            # 条件3：不是被标记为有价值的
            if chunk.review_status == "approve":
                continue
            
            # 条件4：唤醒次数很少（几乎没被触发过）
            if chunk.successful_recall_count > 2:
                continue
            
            to_remove.append(chunk_id)
        
        # 执行删除
        for chunk_id in to_remove:
            self.chunks.pop(chunk_id, None)
        
        return to_remove
    
    def get_stats(self) -> ForgottenLayerStats:
        """获取统计信息"""
        if not self.chunks:
            return ForgottenLayerStats()
        
        now = time.time()
        ages = [(now - c.created_at) / (24 * 3600) for c in self.chunks.values()]
        importances = [c.importance for c in self.chunks.values()]
        
        type_counts: Dict[str, int] = {}
        for chunk in self.chunks.values():
            type_counts[chunk.memory_type] = type_counts.get(chunk.memory_type, 0) + 1
        
        return ForgottenLayerStats(
            total_chunks=len(self.chunks),
            oldest_age_days=max(ages) if ages else 0.0,
            avg_weight=sum(importances) / len(importances) if importances else 0.0,
            chunk_types=type_counts,
        )
    
    # ============ 持久化 ============
    
    def save(self, filepath: str):
        """保存到文件"""
        data = {
            "chunks": {cid: c.to_dict() for cid, c in self.chunks.items()},
            "stats": {
                "total_wake_attempts": self.total_wake_attempts,
                "total_wake_success": self.total_wake_success,
            },
            "timestamp": time.time(),
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str) -> bool:
        """从文件加载"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.chunks = {
                cid: MemoryChunk.from_dict(cdata)
                for cid, cdata in data.get("chunks", {}).items()
            }
            
            stats = data.get("stats", {})
            self.total_wake_attempts = stats.get("total_wake_attempts", 0)
            self.total_wake_success = stats.get("total_wake_success", 0)
            
            return True
        except FileNotFoundError:
            return False
    
    def __len__(self) -> int:
        return len(self.chunks)
