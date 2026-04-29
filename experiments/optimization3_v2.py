"""
优化方案v2：真正有效的性能优化

问题诊断：
1. 分桶索引在随机查询下无效（没有命中特定桶时返回全量）
2. 稀疏关联的开销超过收益
3. 核心瓶颈：遍历所有chunk + 每次都计算权重

解决方案：
1. 多级索引：时间 > 主题 > 地点 > 人物
2. 权重缓存：检索时批量计算并缓存
3. 早期退出：找到足够候选就停止
4. 候选集限制：最多扫描N个chunk
"""

from typing import Dict, List, Optional, Set, Any, Tuple
import math
import time
from dataclasses import dataclass

from memory_chunk import MemoryChunk, MemoryLayer


@dataclass
class CachedWeight:
    """缓存的权重"""
    factors: 'WeightFactors'
    calculated_at: float


@dataclass 
class WeightFactors:
    time_decay: float = 0.0
    frequency: float = 0.0
    recency: float = 0.0
    emotion_boost: float = 0.0
    association_density: float = 0.0
    importance_base: float = 0.0
    final: float = 0.0


class OptimizedV2MemoryLayerCore:
    """
    优化版本v2
    
    核心优化：
    1. 多级索引（时间 > 主题 > 地点）
    2. 权重缓存（批量计算，只算一次）
    3. 早期退出（找到top-K就停）
    4. 候选集限制（最多扫描N个）
    """
    
    def __init__(
        self,
        decay_half_life: float = 7 * 24 * 3600,
        decay_rate: float = 0.1,
        freq_half_life: float = 10,
        recency_window: float = 24 * 3600,
        degrade_threshold: float = 0.15,
        
        # 扫描限制
        max_scan_candidates: int = 200,  # 最多扫描这么多候选
        early_exit_k: int = 20,  # 找到20个就停止
        
        weights: Optional[Dict[str, float]] = None,
    ):
        self.decay_half_life = decay_half_life
        self.decay_rate = decay_rate
        self.freq_half_life = freq_half_life
        self.recency_window = recency_window
        self.degrade_threshold = degrade_threshold
        self.max_scan_candidates = max_scan_candidates
        self.early_exit_k = early_exit_k
        
        self.coeffs = weights or {
            'time_decay': 0.20,
            'frequency': 0.15,
            'recency': 0.15,
            'emotion': 0.15,
            'association': 0.10,
            'importance': 0.10,
            'connection': 0.05,
        }
        
        # 存储
        self.chunks: Dict[str, MemoryChunk] = {}
        
        # 多级索引
        self.time_index: Dict[str, Set[str]] = {}   # "2026-04" -> {ids}
        self.topic_index: Dict[str, Set[str]] = {}  # "food" -> {ids}
        self.location_index: Dict[str, Set[str]] = {}  # "北京" -> {ids}
        
        # 全量列表（用于没有特定索引的扫描）
        self.all_ids: List[str] = []
        
        # 权重缓存
        self.weight_cache: Dict[str, CachedWeight] = {}
        self.cache_ttl: float = 60.0  # 缓存60秒
        
        # 统计
        self.total_recall_success = 0
        self.total_recall_fail = 0
    
    # ============ 权重计算 ============
    
    def calc_weight(self, chunk: MemoryChunk, force: bool = False) -> float:
        """计算单条记忆的权重"""
        # 检查缓存
        if not force and chunk.id in self.weight_cache:
            cached = self.weight_cache[chunk.id]
            if time.time() - cached.calculated_at < self.cache_ttl:
                return cached.factors.final
        
        age = time.time() - chunk.created_at
        
        # 时间衰减
        effective_decay = self.decay_rate * (1 - len(chunk.associations) * 0.02)
        effective_decay = max(effective_decay, 0.01)
        time_decay = math.exp(-effective_decay * age / self.decay_half_life)
        time_decay = 0.1 + 0.9 * time_decay
        
        # 使用频率
        if chunk.access_count == 0:
            frequency = 0.0
        else:
            frequency = math.log(1 + chunk.access_count) / math.log(1 + self.freq_half_life)
        
        # 近因效应
        time_since_access = time.time() - chunk.last_accessed
        if time_since_access <= self.recency_window:
            recency = 1.0 - (time_since_access / self.recency_window) * 0.5
        else:
            excess = time_since_access - self.recency_window
            recency = 0.5 * math.exp(-excess / (7 * 24 * 3600))
        
        # 情绪增强
        emotion_boost = chunk.emotion_valence * chunk.emotion_intensity
        
        # 关联密度（简化）
        assoc_count = len(chunk.associations)
        association_density = min(math.log(1 + assoc_count) / math.log(11), 1.0) if assoc_count else 0.0
        
        # 重要性
        importance_base = chunk.importance
        
        # 综合
        final = (
            self.coeffs['time_decay'] * time_decay +
            self.coeffs['frequency'] * frequency +
            self.coeffs['recency'] * recency +
            self.coeffs['emotion'] * (0.5 + 0.5 * emotion_boost) +
            self.coeffs['association'] * association_density +
            self.coeffs['importance'] * importance_base +
            self.coeffs['connection'] * chunk.connection_value
        )
        final = max(0.0, min(1.0, final))
        
        # 缓存
        self.weight_cache[chunk.id] = CachedWeight(
            factors=WeightFactors(
                time_decay=time_decay,
                frequency=frequency,
                recency=recency,
                emotion_boost=emotion_boost,
                association_density=association_density,
                importance_base=importance_base,
                final=final,
            ),
            calculated_at=time.time(),
        )
        
        return final
    
    def batch_calc_weights(self, chunk_ids: List[str]) -> Dict[str, float]:
        """批量计算权重（优先使用缓存）"""
        results = {}
        now = time.time()
        
        for chunk_id in chunk_ids:
            chunk = self.chunks.get(chunk_id)
            if not chunk:
                continue
            
            if chunk_id in self.weight_cache:
                cached = self.weight_cache[chunk_id]
                if now - cached.calculated_at < self.cache_ttl:
                    results[chunk_id] = cached.factors.final
                    continue
            
            # 需要重新计算
            results[chunk_id] = self.calc_weight(chunk)
        
        return results
    
    # ============ 索引 ============
    
    def _add_to_index(self, chunk: MemoryChunk):
        """添加chunk到索引"""
        # 时间
        if chunk.time_absolute:
            year_month = chunk.time_absolute[:7]
            if year_month not in self.time_index:
                self.time_index[year_month] = set()
            self.time_index[year_month].add(chunk.id)
        
        # 主题
        for topic in chunk.topics:
            if topic not in self.topic_index:
                self.topic_index[topic] = set()
            self.topic_index[topic].add(chunk.id)
        
        # 地点
        if chunk.location:
            if chunk.location not in self.location_index:
                self.location_index[chunk.location] = set()
            self.location_index[chunk.location].add(chunk.id)
        
        self.all_ids.append(chunk.id)
    
    def _remove_from_index(self, chunk: MemoryChunk):
        """从索引移除chunk"""
        if chunk.time_absolute:
            year_month = chunk.time_absolute[:7]
            if year_month in self.time_index:
                self.time_index[year_month].discard(chunk.id)
        
        for topic in chunk.topics:
            if topic in self.topic_index:
                self.topic_index[topic].discard(chunk.id)
        
        if chunk.location and chunk.location in self.location_index:
            self.location_index[chunk.location].discard(chunk.id)
        
        if chunk.id in self.all_ids:
            self.all_ids.remove(chunk.id)
    
    # ============ 记忆操作 ============
    
    def add(self, chunk: MemoryChunk) -> str:
        chunk.layer = MemoryLayer.CORE
        self.chunks[chunk.id] = chunk
        self._add_to_index(chunk)
        # 添加时立即计算并缓存权重
        self.calc_weight(chunk, force=True)
        return chunk.id
    
    def get(self, chunk_id: str) -> Optional[MemoryChunk]:
        return self.chunks.get(chunk_id)
    
    def access(self, chunk_id: str) -> Optional[Tuple[MemoryChunk, float]]:
        chunk = self.chunks.get(chunk_id)
        if not chunk:
            return None
        chunk.access()
        weight = self.calc_weight(chunk, force=True)  # 强制重新计算
        return chunk, weight
    
    def remove(self, chunk_id: str) -> Optional[MemoryChunk]:
        if chunk_id in self.chunks:
            chunk = self.chunks.pop(chunk_id)
            self._remove_from_index(chunk)
            self.weight_cache.pop(chunk_id, None)
            return chunk
        return None
    
    # ============ 检索（优化核心）===========
    
    def _select_candidates(self, query_tags: Dict[str, Any]) -> List[str]:
        """
        选择候选集（多级索引 + 限制）
        """
        candidates: Optional[Set[str]] = None
        selector_name = ""
        
        # 时间（最高优先，区分度最大）
        if "time_absolute" in query_tags:
            year_month = query_tags["time_absolute"][:7]
            bucket = self.time_index.get(year_month, set())
            if bucket:
                candidates = bucket.copy()
                selector_name = f"time:{year_month}"
        
        # 主题（第二优先）
        if "topics" in query_tags and candidates is not None:
            topic_bucket: Set[str] = set()
            for topic in query_tags["topics"]:
                if topic in self.topic_index:
                    topic_bucket |= self.topic_index[topic]
            if topic_bucket:
                candidates &= topic_bucket
                selector_name += f" + topics:{len(query_tags['topics'])}"
        
        # 地点
        if "location" in query_tags and candidates is not None:
            loc_bucket = self.location_index.get(query_tags["location"], set())
            if loc_bucket:
                candidates &= loc_bucket
                selector_name += f" + location:{query_tags['location']}"
        
        # 如果候选集太大或为空，使用全量
        if candidates is None or len(candidates) > self.max_scan_candidates:
            candidates = set(self.all_ids[:self.max_scan_candidates])
            selector_name = f"all_limited:{len(candidates)}"
        
        # 如果候选集为空，使用全量
        if not candidates:
            candidates = set(self.all_ids)
            selector_name = "all"
        
        return list(candidates)
    
    def retrieve(
        self,
        query_tags: Dict[str, Any],
        min_weight: float = 0.0,
        limit: int = 10,
    ) -> List[Tuple[MemoryChunk, float]]:
        """
        检索（优化版）
        1. 用索引选择候选集（限制扫描范围）
        2. 批量计算权重
        3. 早期退出
        """
        # 选择候选
        candidate_ids = self._select_candidates(query_tags)
        
        # 过滤和匹配
        matched = []
        for chunk_id in candidate_ids:
            chunk = self.chunks.get(chunk_id)
            if not chunk:
                continue
            if not chunk.matches_query(query_tags):
                continue
            matched.append(chunk_id)
            
            # 早期退出：找到足够多就停止扫描
            if len(matched) >= self.early_exit_k:
                break
        
        # 批量计算权重
        weights = self.batch_calc_weights(matched)
        
        # 构建结果
        results = []
        for chunk_id in matched:
            chunk = self.chunks.get(chunk_id)
            if not chunk:
                continue
            w = weights.get(chunk_id, 0.0)
            if w >= min_weight:
                results.append((chunk, w))
        
        # 排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def get_top(self, limit: int = 20) -> List[Tuple[MemoryChunk, float]]:
        # 使用缓存的权重排序
        weighted = []
        for chunk_id in self.all_ids:
            chunk = self.chunks.get(chunk_id)
            if not chunk:
                continue
            w = self.calc_weight(chunk)
            weighted.append((chunk, w))
        
        weighted.sort(key=lambda x: x[1], reverse=True)
        return weighted[:limit]
    
    # ============ Hebbian ============
    
    def strengthen_association(self, chunk_id_a: str, chunk_id_b: str, strength: float = 0.1):
        if chunk_id_a not in self.chunks or chunk_id_b not in self.chunks:
            return
        
        chunk_a = self.chunks[chunk_id_a]
        chunk_b = self.chunks[chunk_id_b]
        
        old_a = chunk_a.associations.get(chunk_id_b, 0.0)
        old_b = chunk_b.associations.get(chunk_id_a, 0.0)
        
        new_a = min(1.0, old_a + strength * (1 - old_a))
        new_b = min(1.0, old_b + strength * (1 - old_b))
        
        chunk_a.associations[chunk_id_b] = new_a
        chunk_b.associations[chunk_id_a] = new_b
        
        # 清除缓存（下次检索会重新计算）
        self.weight_cache.pop(chunk_id_a, None)
        self.weight_cache.pop(chunk_id_b, None)
    
    def access_together(self, chunk_ids: List[str]):
        for chunk_id in chunk_ids:
            if chunk_id in self.chunks:
                self.chunks[chunk_id].access()
        
        for i, id_a in enumerate(chunk_ids):
            for id_b in chunk_ids[i+1:]:
                self.strengthen_association(id_a, id_b)
    
    # ============ 反馈 ============
    
    def adjust_after_recall(self, chunk_id: str, success: bool, feedback_emotion: float = 0.0):
        chunk = self.chunks.get(chunk_id)
        if not chunk:
            return
        
        if success:
            chunk.successful_recall()
            chunk.importance = min(1.0, chunk.importance + 0.01)
            self.total_recall_success += 1
        else:
            chunk.importance = max(0.0, chunk.importance - 0.05)
            self.total_recall_fail += 1
        
        # 清除缓存
        self.weight_cache.pop(chunk_id, None)
    
    # ============ 降级 ============
    
    def check_degrade(self) -> List[str]:
        to_degrade = []
        for chunk_id in self.all_ids:
            chunk = self.chunks.get(chunk_id)
            if not chunk:
                continue
            w = self.calc_weight(chunk)
            if w < self.degrade_threshold:
                to_degrade.append(chunk_id)
        return to_degrade
    
    def degrade_chunks(self, chunk_ids: List[str]) -> List[MemoryChunk]:
        degraded = []
        for chunk_id in chunk_ids:
            if chunk_id in self.chunks:
                chunk = self.chunks.pop(chunk_id)
                self._remove_from_index(chunk)
                self.weight_cache.pop(chunk_id, None)
                chunk.layer = MemoryLayer.FORGOTTEN
                degraded.append(chunk)
        return degraded
    
    def decay_all_unused(self):
        now = time.time()
        for chunk in self.chunks.values():
            if now - chunk.last_accessed > 7 * 24 * 3600:
                chunk.access_count = max(0, chunk.access_count - 1)
        # 清除所有缓存（强制重算）
        self.weight_cache.clear()
    
    # ============ 统计 ============
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "chunks": len(self.chunks),
            "time_index_buckets": len(self.time_index),
            "topic_index_buckets": len(self.topic_index),
            "location_index_buckets": len(self.location_index),
            "weight_cache_size": len(self.weight_cache),
            "avg_assocs": sum(len(c.associations) for c in self.chunks.values()) / max(len(self.chunks), 1),
        }
    
    def __len__(self) -> int:
        return len(self.chunks)
