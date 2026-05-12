"""
核心记忆层管理 - 类人记忆系统

负责：
- 记忆的核心层存储
- 动态权重计算
- 记忆的增删改查
- Hebbian关联更新
- 向伪遗忘层的降级
"""

from typing import Dict, List, Optional, Set, Tuple, Any
import math
import time
import json
from dataclasses import dataclass

from memory_chunk import MemoryChunk, MemoryLayer


@dataclass
class WeightFactors:
    """权重因子分解"""
    time_decay: float = 0.0
    frequency: float = 0.0
    recency: float = 0.0
    emotion_boost: float = 0.0
    association_density: float = 0.0
    importance_base: float = 0.0
    connection_boost: float = 0.0
    final: float = 0.0


class MemoryLayerCore:
    """
    核心记忆层
    
    特点：
    - 高权重记忆常驻
    - 主动检索命中率高
    - 权重低于阈值时降级到伪遗忘层
    """
    
    def __init__(
        self,
        # 权重参数
        decay_half_life: float = 7 * 24 * 3600,      # 半衰期7天
        decay_rate: float = 0.1,
        freq_half_life: float = 10,                  # 频率饱和点
        recency_window: float = 24 * 3600,          # 24小时近因窗口
        assoc_stability: float = 0.05,               # 关联减缓衰减
        
        # 降级参数
        degrade_threshold: float = 0.15,             # 权重低于此值降级到伪遗忘层
        
        # 权重组合
        weights: Optional[Dict[str, float]] = None,
    ):
        self.decay_half_life = decay_half_life
        self.decay_rate = decay_rate
        self.freq_half_life = freq_half_life
        self.recency_window = recency_window
        self.assoc_stability = assoc_stability
        self.degrade_threshold = degrade_threshold
        
        self.coeffs = weights or {
            'time_decay': 0.20,
            'frequency': 0.15,
            'recency': 0.15,
            'emotion': 0.15,
            'association': 0.15,
            'importance': 0.10,
            'connection': 0.10,
        }
        
        # 存储
        self.chunks: Dict[str, MemoryChunk] = {}
        
        # 权重计算缓存：{chunk_id: (weight_factors, cache_time)}
        self._weight_cache: Dict[str, Tuple[WeightFactors, float]] = {}
        self._cache_ttl: float = 60.0  # 缓存有效期60秒
        
        # 多维索引（倒排索引）
        self._index_by_location: Dict[str, Set[str]] = {}
        self._index_by_person: Dict[str, Set[str]] = {}
        self._index_by_topic: Dict[str, Set[str]] = {}
        self._index_by_time_relative: Dict[str, Set[str]] = {}
        self._index_by_time_context: Dict[str, Set[str]] = {}
        self._index_by_memory_type: Dict[str, Set[str]] = {}
        
        # Hebbian 关联索引（邻接表）
        self._association_index: Dict[str, Dict[str, float]] = {}  # chunk_id -> {associated_id -> strength}
        
        # 性能统计
        self._stats = {
            "total_weight_calculations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "index_lookups": 0,
            "full_scans": 0,
        }
        
        # 统计
        self.total_recall_success = 0
        self.total_recall_fail = 0
    
    # ============ 权重计算 ============
    
    def _get_cached_weight(self, chunk: MemoryChunk) -> Optional[WeightFactors]:
        """获取缓存的权重"""
        self._stats["total_weight_calculations"] += 1
        if chunk.id not in self._weight_cache:
            self._stats["cache_misses"] += 1
            return None
        wf, cache_time = self._weight_cache[chunk.id]
        if time.time() - cache_time > self._cache_ttl:
            del self._weight_cache[chunk.id]
            self._stats["cache_misses"] += 1
            return None
        self._stats["cache_hits"] += 1
        return wf
    
    def _set_cached_weight(self, chunk: MemoryChunk, wf: WeightFactors):
        """设置权重缓存"""
        self._weight_cache[chunk.id] = (wf, time.time())
    
    def _invalidate_cache(self, chunk_id: str):
        """使缓存失效"""
        self._weight_cache.pop(chunk_id, None)
    
    def clear_cache(self):
        """清除所有权重缓存"""
        self._weight_cache.clear()
    
    def prune_indexes(self, min_usage: int = 0):
        """清理未使用的索引条目"""
        all_chunk_ids = set(self.chunks.keys())
        
        for index in [self._index_by_location, self._index_by_person, 
                       self._index_by_topic, self._index_by_time_relative,
                       self._index_by_time_context, self._index_by_memory_type]:
            empty_keys = [k for k, v in index.items() if not v or not any(cid in all_chunk_ids for cid in v)]
            for k in empty_keys:
                del index[k]
        
        empty_associations = [k for k in self._association_index if k not in all_chunk_ids]
        for k in empty_associations:
            del self._association_index[k]
        
        return len(empty_keys) + len(empty_associations)
    
    def calc_weight(self, chunk: MemoryChunk) -> WeightFactors:
        """计算记忆碎片权重"""
        cached = self._get_cached_weight(chunk)
        if cached:
            return cached
        
        age = time.time() - chunk.created_at
        
        # 时间衰减（指数衰减，关联减缓）
        assoc_count = len(chunk.associations)
        effective_decay = self.decay_rate * (1 - assoc_count * self.assoc_stability)
        effective_decay = max(effective_decay, 0.01)
        time_decay = math.exp(-effective_decay * age / self.decay_half_life)
        time_decay = 0.1 + 0.9 * time_decay  # 归一化到0.1~1.0
        
        # 使用频率（对数增长）
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
        
        # 关联密度
        if assoc_count == 0:
            association_density = 0.0
        else:
            association_density = min(math.log(1 + assoc_count) / math.log(11), 1.0)
        
        # 重要性基础
        importance_base = chunk.importance
        
        # 连接价值
        connection_boost = chunk.connection_value
        
        # 综合权重
        final = (
            self.coeffs['time_decay'] * time_decay +
            self.coeffs['frequency'] * frequency +
            self.coeffs['recency'] * recency +
            self.coeffs['emotion'] * (0.5 + 0.5 * emotion_boost) +
            self.coeffs['association'] * association_density +
            self.coeffs['importance'] * importance_base +
            self.coeffs['connection'] * connection_boost
        )
        final = max(0.0, min(1.0, final))
        
        wf = WeightFactors(
            time_decay=time_decay,
            frequency=frequency,
            recency=recency,
            emotion_boost=emotion_boost,
            association_density=association_density,
            importance_base=importance_base,
            connection_boost=connection_boost,
            final=final,
        )
        
        self._set_cached_weight(chunk, wf)
        return wf
    
    # ============ 索引管理 ============
    
    def _update_indexes(self, chunk: MemoryChunk):
        """更新记忆的索引"""
        chunk_id = chunk.id
        
        if chunk.location:
            if chunk.location not in self._index_by_location:
                self._index_by_location[chunk.location] = set()
            self._index_by_location[chunk.location].add(chunk_id)
        
        for person in chunk.persons:
            if person not in self._index_by_person:
                self._index_by_person[person] = set()
            self._index_by_person[person].add(chunk_id)
        
        for topic in chunk.topics:
            if topic not in self._index_by_topic:
                self._index_by_topic[topic] = set()
            self._index_by_topic[topic].add(chunk_id)
        
        if chunk.time_relative:
            if chunk.time_relative not in self._index_by_time_relative:
                self._index_by_time_relative[chunk.time_relative] = set()
            self._index_by_time_relative[chunk.time_relative].add(chunk_id)
        
        if chunk.time_context:
            if chunk.time_context not in self._index_by_time_context:
                self._index_by_time_context[chunk.time_context] = set()
            self._index_by_time_context[chunk.time_context].add(chunk_id)
        
        mem_type_key = chunk.memory_type.value if hasattr(chunk.memory_type, 'value') else str(chunk.memory_type)
        if mem_type_key not in self._index_by_memory_type:
            self._index_by_memory_type[mem_type_key] = set()
        self._index_by_memory_type[mem_type_key].add(chunk_id)
    
    def _remove_from_indexes(self, chunk: MemoryChunk):
        """从索引中移除记忆"""
        chunk_id = chunk.id
        
        if chunk.location and chunk_id in self._index_by_location.get(chunk.location, set()):
            self._index_by_location[chunk.location].discard(chunk_id)
        
        for person in chunk.persons:
            self._index_by_person.get(person, set()).discard(chunk_id)
        
        for topic in chunk.topics:
            self._index_by_topic.get(topic, set()).discard(chunk_id)
        
        if chunk.time_relative:
            self._index_by_time_relative.get(chunk.time_relative, set()).discard(chunk_id)
        
        if chunk.time_context:
            self._index_by_time_context.get(chunk.time_context, set()).discard(chunk_id)
        
        mem_type_key = chunk.memory_type.value if hasattr(chunk.memory_type, 'value') else str(chunk.memory_type)
        self._index_by_memory_type.get(mem_type_key, set()).discard(chunk_id)
        
        self._association_index.pop(chunk_id, None)
        for other_id in self._association_index:
            self._association_index[other_id].pop(chunk_id, None)
    
    def get_associated_memories(
        self,
        chunk_id: str,
        min_strength: float = 0.1,
        limit: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        快速获取与指定记忆关联的记忆（使用索引）
        
        返回：[(关联记忆ID, 关联强度), ...]，按强度降序
        """
        if chunk_id not in self._association_index:
            chunk = self.chunks.get(chunk_id)
            if chunk and chunk.associations:
                associations = [(aid, strength) for aid, strength in chunk.associations.items() if strength >= min_strength]
                associations.sort(key=lambda x: x[1], reverse=True)
                return associations[:limit]
            return []
        
        associations = [(aid, strength) for aid, strength in self._association_index[chunk_id].items() if strength >= min_strength]
        associations.sort(key=lambda x: x[1], reverse=True)
        return associations[:limit]
    
    def _get_candidates_from_index(self, query_tags: Dict[str, Any]) -> Set[str]:
        """使用索引获取候选记忆ID"""
        self._stats["index_lookups"] += 1
        candidate_sets: List[Set[str]] = []
        
        if "location" in query_tags and query_tags["location"] in self._index_by_location:
            candidate_sets.append(self._index_by_location[query_tags["location"]])
        
        if "persons" in query_tags:
            for person in query_tags["persons"]:
                if person in self._index_by_person:
                    candidate_sets.append(self._index_by_person[person])
        
        if "topics" in query_tags:
            for topic in query_tags["topics"]:
                if topic in self._index_by_topic:
                    candidate_sets.append(self._index_by_topic[topic])
        
        if "time_relative" in query_tags:
            tr = query_tags["time_relative"]
            if tr in self._index_by_time_relative:
                candidate_sets.append(self._index_by_time_relative[tr])
            if tr in self._index_by_time_context:
                candidate_sets.append(self._index_by_time_context[tr])
        
        if "time_context" in query_tags:
            tc = query_tags["time_context"]
            if tc in self._index_by_time_context:
                candidate_sets.append(self._index_by_time_context[tc])
            if tc in self._index_by_time_relative:
                candidate_sets.append(self._index_by_time_relative[tc])
        
        if not candidate_sets:
            self._stats["full_scans"] += 1
            return set(self.chunks.keys())
        
        if len(candidate_sets) == 1:
            return candidate_sets[0]
        
        result = candidate_sets[0].intersection(*candidate_sets[1:])
        return result if result else set(self.chunks.keys())
    
    # ============ 记忆操作 ============
    
    def add(self, chunk: MemoryChunk) -> str:
        """添加记忆"""
        if chunk.layer != MemoryLayer.CORE:
            chunk.layer = MemoryLayer.CORE
        self.chunks[chunk.id] = chunk
        self._invalidate_cache(chunk.id)
        self._update_indexes(chunk)
        return chunk.id
    
    def get(self, chunk_id: str) -> Optional[MemoryChunk]:
        """获取记忆"""
        return self.chunks.get(chunk_id)
    
    def access(self, chunk_id: str) -> Optional[Tuple[MemoryChunk, WeightFactors]]:
        """访问记忆，返回碎片和权重"""
        chunk = self.chunks.get(chunk_id)
        if not chunk:
            return None
        chunk.access()
        self._invalidate_cache(chunk_id)
        return chunk, self.calc_weight(chunk)
    
    def remove(self, chunk_id: str) -> Optional[MemoryChunk]:
        """删除记忆"""
        chunk = self.chunks.get(chunk_id)
        if chunk:
            self._remove_from_indexes(chunk)
        self._invalidate_cache(chunk_id)
        return self.chunks.pop(chunk_id, None)
    
    # ============ 检索 ============
    
    def retrieve(
        self,
        query_tags: Dict[str, Any],
        min_weight: float = 0.0,
        limit: int = 10,
    ) -> List[Tuple[MemoryChunk, WeightFactors]]:
        """
        基于标签检索记忆（使用索引优化）
        
        返回：[(碎片, 权重), ...]，按权重降序
        """
        candidate_ids = self._get_candidates_from_index(query_tags)
        candidates = []
        
        for chunk_id in candidate_ids:
            chunk = self.chunks.get(chunk_id)
            if not chunk:
                continue
            if not chunk.matches_query(query_tags):
                continue
            
            wf = self.calc_weight(chunk)
            if wf.final >= min_weight:
                candidates.append((chunk, wf))
        
        candidates.sort(key=lambda x: x[1].final, reverse=True)
        return candidates[:limit]
    
    def get_top(self, limit: int = 20) -> List[Tuple[MemoryChunk, WeightFactors]]:
        """获取当前权重最高的记忆"""
        weighted = [(c, self.calc_weight(c)) for c in self.chunks.values()]
        weighted.sort(key=lambda x: x[1].final, reverse=True)
        return weighted[:limit]
    
    # ============ Hebbian关联 ============
    
    def strengthen_association(self, chunk_id_a: str, chunk_id_b: str, strength: float = 0.1):
        """Hebbian增强：一起使用的记忆互相增强"""
        chunk_a = self.chunks.get(chunk_id_a)
        chunk_b = self.chunks.get(chunk_id_b)
        if not chunk_a or not chunk_b:
            return
        
        current_a = chunk_a.associations.get(chunk_id_b, 0.0)
        current_b = chunk_b.associations.get(chunk_id_a, 0.0)
        
        chunk_a.associations[chunk_id_b] = min(1.0, current_a + strength * (1 - current_a))
        chunk_b.associations[chunk_id_a] = min(1.0, current_b + strength * (1 - current_b))
        
        if chunk_id_a not in self._association_index:
            self._association_index[chunk_id_a] = {}
        if chunk_id_b not in self._association_index:
            self._association_index[chunk_id_b] = {}
        self._association_index[chunk_id_a][chunk_id_b] = chunk_a.associations[chunk_id_b]
        self._association_index[chunk_id_b][chunk_id_a] = chunk_b.associations[chunk_id_a]
    
    def weaken_association(self, chunk_id_a: str, chunk_id_b: str, strength: float = 0.05):
        """Hebbian减弱：长期不一起使用则衰减"""
        chunk_a = self.chunks.get(chunk_id_a)
        chunk_b = self.chunks.get(chunk_id_b)
        if not chunk_a or not chunk_b:
            return
        
        if chunk_id_b in chunk_a.associations:
            chunk_a.associations[chunk_id_b] = max(0.0, chunk_a.associations[chunk_id_b] - strength)
        if chunk_id_a in chunk_b.associations:
            chunk_b.associations[chunk_id_a] = max(0.0, chunk_b.associations[chunk_id_a] - strength)
        
        if chunk_id_a in self._association_index and chunk_id_b in self._association_index[chunk_id_a]:
            self._association_index[chunk_id_a][chunk_id_b] = chunk_a.associations.get(chunk_id_b, 0.0)
        if chunk_id_b in self._association_index and chunk_id_a in self._association_index[chunk_id_b]:
            self._association_index[chunk_id_b][chunk_id_a] = chunk_b.associations.get(chunk_id_a, 0.0)
    
    def access_together(self, chunk_ids: List[str]):
        """同时访问多个记忆（触发Hebbian增强）"""
        for chunk_id in chunk_ids:
            if chunk_id in self.chunks:
                self.chunks[chunk_id].access()
        
        for i, id_a in enumerate(chunk_ids):
            for id_b in chunk_ids[i+1:]:
                self.strengthen_association(id_a, id_b)
    
    # ============ 反馈调整 ============
    
    def adjust_after_recall(
        self,
        chunk_id: str,
        success: bool,
        feedback_emotion: float = 0.0,
    ):
        """
        回忆后的权重调整
        
        success=True: 这次回忆被确认是正确的 → 权重提升
        success=False: 这次回忆是错误的 → 权重降低
        feedback_emotion: 反馈的情绪强度（影响调整幅度）
        """
        chunk = self.chunks.get(chunk_id)
        if not chunk:
            return
        
        current_weight = self.calc_weight(chunk).final
        
        if success:
            # 成功回忆，权重提升
            # 情绪强度影响提升幅度
            boost = 0.05 * (1 + feedback_emotion)
            new_weight = min(1.0, current_weight + boost)
            chunk.successful_recall()
            self.total_recall_success += 1
        else:
            # 错误回忆，权重降低
            penalty = 0.08 * (1 + abs(feedback_emotion))
            new_weight = max(0.0, current_weight - penalty)
            self.total_recall_fail += 1
        
        # 更新基础重要性（间接影响权重）
        # 如果持续成功，重要性缓慢上升
        if success:
            chunk.importance = min(1.0, chunk.importance + 0.01)
    
    # ============ 降级检查 ============
    
    def check_degrade(self) -> List[str]:
        """
        检查需要降级到伪遗忘层的记忆
        返回降级记忆的ID列表
        """
        to_degrade = []
        
        for chunk_id, chunk in self.chunks.items():
            wf = self.calc_weight(chunk)
            if wf.final < self.degrade_threshold:
                to_degrade.append(chunk_id)
        
        return to_degrade
    
    def degrade_chunks(self, chunk_ids: List[str]) -> List[MemoryChunk]:
        """将记忆降级到伪遗忘层"""
        degraded = []
        for chunk_id in chunk_ids:
            if chunk_id in self.chunks:
                chunk = self.chunks.pop(chunk_id)
                chunk.layer = MemoryLayer.FORGOTTEN
                self._invalidate_cache(chunk_id)
                degraded.append(chunk)
        return degraded
    
    # ============ 持久化 ============
    
    def save(self, filepath: str):
        """保存到文件"""
        data = {
            "chunks": {cid: c.to_dict() for cid, c in self.chunks.items()},
            "stats": {
                "total_recall_success": self.total_recall_success,
                "total_recall_fail": self.total_recall_fail,
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
            self.total_recall_success = stats.get("total_recall_success", 0)
            self.total_recall_fail = stats.get("total_recall_fail", 0)
            
            self._rebuild_indexes()
            return True
        except FileNotFoundError:
            return False
    
    def _rebuild_indexes(self):
        """重建所有索引"""
        self._index_by_location.clear()
        self._index_by_person.clear()
        self._index_by_topic.clear()
        self._index_by_time_relative.clear()
        self._index_by_time_context.clear()
        self._index_by_memory_type.clear()
        self._association_index.clear()
        
        for chunk in self.chunks.values():
            self._update_indexes(chunk)
            if chunk.associations:
                self._association_index[chunk.id] = dict(chunk.associations)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        total = self._stats["total_weight_calculations"]
        return {
            "weight_calculations": total,
            "cache_hits": self._stats["cache_hits"],
            "cache_misses": self._stats["cache_misses"],
            "cache_hit_rate": self._stats["cache_hits"] / max(1, total),
            "index_lookups": self._stats["index_lookups"],
            "full_scans": self._stats["full_scans"],
        }
    
    def reset_performance_stats(self):
        """重置性能统计"""
        self._stats = {
            "total_weight_calculations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "index_lookups": 0,
            "full_scans": 0,
        }
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计"""
        import sys
        total_size = sys.getsizeof(self.chunks)
        
        chunk_sizes = []
        for chunk in self.chunks.values():
            chunk_size = (
                sys.getsizeof(chunk.content) +
                sys.getsizeof(chunk.associations) +
                sys.getsizeof(chunk.persons) +
                sys.getsizeof(chunk.topics) +
                sys.getsizeof(chunk.metadata)
            )
            chunk_sizes.append(chunk_size)
        
        index_sizes = {
            "location": sys.getsizeof(self._index_by_location),
            "person": sys.getsizeof(self._index_by_person),
            "topic": sys.getsizeof(self._index_by_topic),
            "association": sys.getsizeof(self._association_index),
        }
        
        return {
            "total_chunks": len(self.chunks),
            "estimated_memory_mb": total_size / 1024 / 1024,
            "avg_chunk_size_bytes": sum(chunk_sizes) / max(1, len(chunk_sizes)),
            "index_memory_mb": sum(index_sizes.values()) / 1024 / 1024,
            "cache_entries": len(self._weight_cache),
            "cache_memory_bytes": len(self._weight_cache) * sys.getsizeof(WeightFactors()),
        }
