"""
伪遗忘层管理 - 类人记忆系统

特点：
- 极低权重记忆的归档层
- 不参与主动检索
- 需要"信息锚点"才能唤醒
- 可以被删除（对系统无影响）

重构：使用可插拔 MemoryStore 后端
"""

from typing import Dict, List, Optional, Tuple, Any
import time
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

    重构后使用可插拔 MemoryStore 后端：
    - 默认使用 JsonMemoryStore（向后兼容）
    - 可切换为 SqliteMemoryStore 等
    """

    def __init__(
        self,
        store=None,
        # 唤醒参数
        wake_threshold: float = 0.3,        # 需要多强的锚点匹配才能唤醒
        min_match_tags: int = 2,            # 最少匹配几个标签才能唤醒

        # 清理参数
        auto_cleanup: bool = True,          # 是否自动清理
        cleanup_age_days: float = 365,       # 超过多少天自动清理
        cleanup_weight_max: float = 0.05,   # 权重低于此值且超过年龄才清理

        # 唤醒后参数
        wake_weight_boost: float = 0.2,     # 唤醒时临时权重提升

        # 向后兼容：旧代码可能传 filepath
        filepath: Optional[str] = None,
    ):
        self.wake_threshold = wake_threshold
        self.min_match_tags = min_match_tags
        self.auto_cleanup = auto_cleanup
        self.cleanup_age_days = cleanup_age_days
        self.cleanup_weight_max = cleanup_weight_max
        self.wake_weight_boost = wake_weight_boost

        # 存储后端
        if store is not None:
            self._store = store
        else:
            # 向后兼容：如果没有传 store，使用 JsonMemoryStore
            from core.json_store import JsonMemoryStore
            self._store = JsonMemoryStore(filepath or "memory_data/forgotten.json")

        # 唤醒记录
        self.total_wake_attempts = 0
        self.total_wake_success = 0

    # ============ 向后兼容属性 ============

    @property
    def chunks(self) -> Dict[str, MemoryChunk]:
        """向后兼容：返回所有 chunks 的 dict 视图"""
        return self._store.get_all()

    # ============ 归档操作 ============

    def archive(self, chunk: MemoryChunk) -> str:
        """将记忆归档到伪遗忘层"""
        chunk.layer = MemoryLayer.FORGOTTEN
        chunk.updated_at = time.time()
        self._store.put(chunk)
        return chunk.id

    def get(self, chunk_id: str) -> Optional[MemoryChunk]:
        """获取记忆"""
        return self._store.get(chunk_id)

    def remove(self, chunk_id: str) -> Optional[MemoryChunk]:
        """彻底删除记忆"""
        chunk = self._store.get(chunk_id)
        if chunk:
            self._store.delete(chunk_id)
        return chunk

    # ============ 唤醒机制 ============

    def calc_wake_score(self, chunk: MemoryChunk, query_tags: Dict[str, Any]) -> Tuple[float, int]:
        """
        计算唤醒得分

        锚点匹配越强，得分越高。返回 (得分, 匹配锚点数)。
        """
        score = 0.0
        matched_tags = 0

        # 绝对时间锚点（最重要）
        if "time_absolute" in query_tags:
            if chunk.time_absolute == query_tags["time_absolute"]:
                score += 0.3
                matched_tags += 1

        # 相对时间 / 时间上下文锚点：独立生效，互相宽松匹配
        # （"昨天"、"中午"这类线索不应该依赖查询同时给出绝对时间）
        relative_anchor = query_tags.get("time_relative")
        context_anchor = query_tags.get("time_context")
        chunk_times = {t for t in (chunk.time_relative, chunk.time_context) if t}
        if relative_anchor and relative_anchor in chunk_times:
            score += 0.2
            matched_tags += 1
        if context_anchor and context_anchor != relative_anchor and context_anchor in chunk_times:
            score += 0.15
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
        # 按"概念组"计算覆盖率：查询主题经过同义扩展后标签数会膨胀，
        # 直接用标签数做分母会稀释匹配强度
        if "topics" in query_tags:
            matched_topics = query_tags["topics"] & chunk.topics
            if matched_topics:
                from core.topic_vocab import count_topic_groups
                query_groups = max(count_topic_groups(query_tags["topics"]), 1)
                matched_groups = min(count_topic_groups(matched_topics), query_groups)
                score += 0.15 * (matched_groups / query_groups)
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

        for chunk in self._store.get_all().values():
            score, matched_tags = self.calc_wake_score(chunk, query_tags)

            # 必须匹配足够多的标签
            if matched_tags >= self.min_match_tags and score >= self.wake_threshold:
                # 计算临时权重（唤醒时提升）
                temp_weight = min(1.0, score + self.wake_weight_boost)
                candidates.append((chunk, score, temp_weight))

        # 按得分降序
        candidates.sort(key=lambda x: x[1], reverse=True)

        woken = candidates[:limit]
        result = [(c, tw) for c, s, tw in woken]

        if result:
            self.total_wake_success += 1
            # 唤醒即留痕：记录成功唤醒并写回存储，
            # 让"这段记忆被线索唤醒过"成为持久事实（影响清理与后续提升决策）
            for chunk, _score, _tw in woken:
                chunk.successful_recall()
                chunk.updated_at = time.time()
                self._store.put(chunk)

        return result

    def record_wake(self, chunk_id: str) -> None:
        """记录一次成功唤醒并持久化（供联想唤醒等外部线索路径使用）"""
        chunk = self._store.get(chunk_id)
        if not chunk:
            return
        chunk.successful_recall()
        chunk.updated_at = time.time()
        self._store.put(chunk)

    def promote(self, chunk_ids: List[str]) -> List[MemoryChunk]:
        """
        把被唤醒的记忆从伪遗忘层移出，交还给核心层。

        返回被移出的记忆（layer 已置回 CORE，并带有一次"再巩固奖励"：
        recall_bias 小幅上浮，给重新唤醒的记忆一个存活窗口——
        如果之后继续被使用它会留下，不用则会再次自然衰减降级）。

        调用方（检索层）负责将返回的 chunk 通过 core.add() 放回核心层。
        """
        promoted = []
        for chunk_id in chunk_ids:
            chunk = self._store.get(chunk_id)
            if not chunk:
                continue
            self._store.delete(chunk_id)
            chunk.layer = MemoryLayer.CORE
            chunk.updated_at = time.time()
            # 再巩固奖励：唤醒后的记忆获得短期权重支撑
            chunk.recall_bias = min(0.25, chunk.recall_bias + 0.05)
            promoted.append(chunk)
        return promoted

    def wake_and_promote(
        self,
        query_tags: Dict[str, Any],
        promotion_weight_threshold: float = 0.4,
    ) -> List[MemoryChunk]:
        """
        唤醒记忆，并把得分足够高的移出伪遗忘层。

        返回：被移出（待放回核心层）的记忆列表。
        注意：与 promote() 相同，调用方负责 core.add()。
        """
        candidates = self.try_wake(query_tags, limit=10)

        to_promote = [
            chunk.id for chunk, temp_weight in candidates
            if temp_weight >= promotion_weight_threshold
        ]
        return self.promote(to_promote)

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
        chunk = self._store.get(chunk_id)
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

        self._store.put(chunk)

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

        for chunk_id, chunk in self._store.get_all().items():
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
            self._store.delete(chunk_id)

        return to_remove

    def get_stats(self) -> ForgottenLayerStats:
        """获取统计信息"""
        all_chunks = self._store.get_all()
        if not all_chunks:
            return ForgottenLayerStats()

        now = time.time()
        ages = [(now - c.created_at) / (24 * 3600) for c in all_chunks.values()]
        importances = [c.importance for c in all_chunks.values()]

        type_counts: Dict[str, int] = {}
        for chunk in all_chunks.values():
            type_key = getattr(chunk.memory_type, "value", chunk.memory_type)
            type_counts[type_key] = type_counts.get(type_key, 0) + 1

        return ForgottenLayerStats(
            total_chunks=len(all_chunks),
            oldest_age_days=max(ages) if ages else 0.0,
            avg_weight=sum(importances) / len(importances) if importances else 0.0,
            chunk_types=type_counts,
        )

    # ============ 持久化 ============

    def save(self, filepath: str = None):
        """
        保存到后端

        filepath 参数仅用于向后兼容
        """
        self._store.save()

    def load(self, filepath: str = None) -> bool:
        """
        从后端加载

        filepath 参数仅用于向后兼容
        """
        return self._store.load()

    def __len__(self) -> int:
        return self._store.count()
