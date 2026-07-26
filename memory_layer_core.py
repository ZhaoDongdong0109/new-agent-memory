"""
核心记忆层管理 - 类人记忆系统

负责：
- 记忆的核心层存储
- 动态权重计算
- 记忆的增删改查
- Hebbian关联更新
- 向伪遗忘层的降级

重构：使用可插拔 MemoryStore 后端
"""

from typing import Dict, List, Optional, Set, Tuple, Any
import math
import time
from dataclasses import dataclass

from memory_chunk import MemoryChunk, MemoryLayer


@dataclass
class WeightFactors:
    """权重因子分解（完整可解释：每个数字都能溯源）"""
    # ACT-R 基线激活 B = ln(Σ t^-d)，以及映射后的保持率 P
    activation: float = 0.0
    retention: float = 0.0
    # 解释性指标（不直接参与 final，供审计展示）
    time_decay: float = 0.0     # 等于 retention（历史字段名，保持兼容）
    frequency: float = 0.0
    recency: float = 0.0
    # 静态因子（被 retention 门控）
    emotion_boost: float = 0.0
    association_density: float = 0.0
    importance_base: float = 0.0
    connection_boost: float = 0.0
    recall_bias: float = 0.0
    final: float = 0.0


@dataclass
class CachedWeight:
    """带时间戳的权重缓存"""
    factors: WeightFactors
    calculated_at: float


class MemoryLayerCore:
    """
    核心记忆层

    特点：
    - 高权重记忆常驻
    - 主动检索命中率高
    - 权重低于阈值时降级到伪遗忘层

    重构后使用可插拔 MemoryStore 后端：
    - 默认使用 JsonMemoryStore（向后兼容）
    - 可切换为 SqliteMemoryStore 等
    """

    def __init__(
        self,
        store=None,
        # 权重参数
        decay_half_life: float = 7 * 24 * 3600,      # 半衰期7天
        decay_rate: float = 0.1,
        freq_half_life: float = 10,                  # 频率饱和点
        recency_window: float = 24 * 3600,          # 24小时近因窗口
        assoc_stability: float = 0.05,               # 关联减缓衰减

        # 降级参数
        degrade_threshold: float = 0.15,             # 权重低于此值降级到伪遗忘层

        # 检索优化参数
        max_scan_candidates: int = 200,
        early_exit_k: int = 20,
        cache_ttl: float = 60.0,

        # 权重组合
        weights: Optional[Dict[str, float]] = None,

        # 向后兼容：旧代码可能传 filepath
        filepath: Optional[str] = None,
    ):
        self.decay_half_life = decay_half_life
        self.decay_rate = decay_rate
        self.freq_half_life = freq_half_life
        self.recency_window = recency_window
        self.assoc_stability = assoc_stability
        self.degrade_threshold = degrade_threshold
        self.max_scan_candidates = max_scan_candidates
        self.early_exit_k = early_exit_k
        self.cache_ttl = cache_ttl

        self.coeffs = weights or {
            'time_decay': 0.20,
            'frequency': 0.15,
            'recency': 0.15,
            'emotion': 0.15,
            'association': 0.15,
            'importance': 0.10,
            'connection': 0.10,
        }

        # 存储后端
        if store is not None:
            self._store = store
        else:
            # 向后兼容：如果没有传 store，使用 JsonMemoryStore
            from core.json_store import JsonMemoryStore
            self._store = JsonMemoryStore(filepath or "memory_data/core.json")

        # 多级倒排索引（保持在内存中）
        self.all_ids: List[str] = []
        self.time_index: Dict[str, Set[str]] = {}
        self.time_relative_index: Dict[str, Set[str]] = {}
        self.time_context_index: Dict[str, Set[str]] = {}
        self.topic_index: Dict[str, Set[str]] = {}
        self.location_index: Dict[str, Set[str]] = {}
        self.person_index: Dict[str, Set[str]] = {}

        # 权重计算会频繁触发，短期缓存能避免重复扫描时反复计算
        self.weight_cache: Dict[str, CachedWeight] = {}

        # 索引键快照：id -> 实际写入索引的键列表。
        # 移除时按快照精确回滚，而不是按 chunk 当前状态推断——
        # 否则调用方原地修改 topics 后再 add()，旧键会永远残留在索引里。
        self._index_keys: Dict[str, List[Tuple[str, str]]] = {}

        # 统计
        self.total_recall_success = 0
        self.total_recall_fail = 0

    # ============ 向后兼容属性 ============

    @property
    def chunks(self) -> Dict[str, MemoryChunk]:
        """向后兼容：返回所有 chunks 的 dict 视图"""
        return self._store.get_all()

    # ============ 权重计算 ============

    # ACT-R 激活计算中 t 的最小值：一分钟粒度，
    # 避免"一秒前刚访问"产生病态激活尖峰
    _MIN_EVENT_AGE = 60.0

    def _base_level_activation(self, chunk: MemoryChunk, now: float) -> float:
        """
        ACT-R 基线激活 B = ln(Σ_j t_j^-d)，Petrov O(k) 混合近似。

        - t_j: 距第 j 次使用的秒数（access_log 精确保留最近 k 次，
          编码事件算第一次使用）
        - 更早的 (n-k) 次使用用均匀分布积分近似，落在
          [创建时间, 最旧保留时间戳] 区间上
        - d 按记忆类型分层（故事最持久），关联强度进一步减缓衰减
          （原 assoc_stability 思想，如今作用于 d 而不是独立通道）
        """
        from core.weight_system import actr_decay

        d = actr_decay(chunk.memory_type)
        # 关联减缓衰减，但必须渐进且有上限（10%）：
        # 上限过大时，强关联记忆的保持率一年后仍在阈值之上，
        # 关联又变回了"永久权重下限"（此前已修复过一次的缺陷类别）
        assoc_strength = sum(chunk.associations.values())
        d = max(0.2, d * (1.0 - min(0.10, assoc_strength * 0.01)))

        # 一致性钳制：使用事件不可能晚于 last_accessed。
        # 正常路径下 access() 同步更新两者；测试/迁移直接回拨
        # created_at/last_accessed 模拟老化时，这个钳制让 access_log
        # 一起"老化"，而不是留下一个未来时间戳撑高激活。
        anchor = max(chunk.last_accessed, chunk.created_at)
        retained = [min(t, anchor) for t in chunk.access_log] or [chunk.created_at]

        # Pavlik 间隔效应：每次使用事件用自己的衰减速率（复习时
        # 激活越高衰减越快）。access_decays 与 access_log 右对齐；
        # 缺失/None（老数据、直接调用）回退到类型基线 d。关联减缓
        # 因子对逐事件衰减同样生效（等比缩放）。
        assoc_factor = d / actr_decay(chunk.memory_type) if actr_decay(chunk.memory_type) else 1.0
        decays = list(getattr(chunk, "access_decays", []) or [])
        if len(decays) < len(retained):
            decays = [None] * (len(retained) - len(decays)) + decays
        else:
            decays = decays[-len(retained):]
        activation_sum = sum(
            max(self._MIN_EVENT_AGE, now - t)
            ** (-(max(0.2, dj * assoc_factor) if dj is not None else d))
            for t, dj in zip(retained, decays)
        )

        # Petrov 尾部近似：未保留的更早使用。
        # 尾部指数用被裁剪事件的衰减均值而不是类型基线 d——否则
        # 密集复习把事件推入尾部即逃脱间隔惩罚（对抗审查实测：
        # 20 次突击后 91% 的激活和来自未受罚的尾部）。没有逐事件
        # 记录的老事件按基线 d 记（与 None 回退语义一致）。
        total_events = max(chunk.access_count + 1, len(retained))
        older = total_events - len(retained)
        if older > 0:
            evicted_n = getattr(chunk, "evicted_decay_count", 0)
            evicted_sum = getattr(chunk, "evicted_decay_sum", 0.0)
            known = min(evicted_n, older)
            known_mean = (evicted_sum / evicted_n) if evicted_n else d
            tail_d = (known * known_mean * assoc_factor + (older - known) * d) / older
            tail_d = min(0.95, max(0.2, tail_d))
            lifetime = max(now - chunk.created_at, self._MIN_EVENT_AGE)
            oldest_kept = max(now - retained[0], self._MIN_EVENT_AGE)
            if lifetime > oldest_kept + 1.0:
                activation_sum += older * (
                    (lifetime ** (1 - tail_d) - oldest_kept ** (1 - tail_d))
                    / ((1 - tail_d) * (lifetime - oldest_kept))
                )
            else:
                activation_sum += older * oldest_kept ** (-tail_d)

        return math.log(max(activation_sum, 1e-12))

    def calc_weight(self, chunk: MemoryChunk) -> WeightFactors:
        """
        计算记忆碎片权重。

        记忆强度核心是 ACT-R 基线激活（30 年认知科学验证的方程，
        取代此前手调的 time_decay/frequency/recency 三因子）：

            B = ln(Σ t^-d)                     幂律遗忘 + 频率 + 近因
            P = 1/(1+exp(-(B-τ)/s))            保持率（0~1）
            final = W_r * P
                  + P * (情绪 + 重要性 + 连接 + 关联密度)   # 门控
                  + recall_bias                             # 反馈

        静态因子被 P 门控：不被使用的记忆无论多"重要"，保持率
        趋零后权重也趋零——遗忘可达是生命周期的前提。
        校准锚点见 core/weight_system.py。
        """
        cached = self.weight_cache.get(chunk.id)
        if cached and time.time() - cached.calculated_at < self.cache_ttl:
            return cached.factors

        from core.weight_system import ACTR_NOISE_SCALE, ACTR_THRESHOLD

        now = time.time()

        # ACT-R 激活 -> 保持率
        activation = self._base_level_activation(chunk, now)
        retention = 1.0 / (1.0 + math.exp(
            -(activation - ACTR_THRESHOLD) / ACTR_NOISE_SCALE
        ))

        # 解释性指标（供审计展示，不直接参与 final）
        if chunk.access_count == 0:
            frequency = 0.0
        else:
            frequency = math.log(1 + chunk.access_count) / math.log(1 + self.freq_half_life)
        time_since_access = now - chunk.last_accessed
        recency = math.exp(-time_since_access / max(self.recency_window, 1.0))

        # 情绪增强
        emotion_boost = chunk.emotion_valence * chunk.emotion_intensity

        # 关联密度（按强度加权：弱边贡献按比例缩小）
        assoc_strength = sum(chunk.associations.values())
        if assoc_strength <= 0:
            association_density = 0.0
        else:
            association_density = min(math.log(1 + assoc_strength) / math.log(11), 1.0)

        # 重要性基础
        importance_base = chunk.importance

        # 连接价值
        connection_boost = chunk.connection_value

        # 保持率权重 = 原三个使用类因子的系数之和（保留自定义
        # coeffs 的兼容性：调过 time_decay/frequency/recency 的
        # 部署，其总投入比例不变）
        retention_weight = (
            self.coeffs['time_decay']
            + self.coeffs['frequency']
            + self.coeffs['recency']
        )

        # 静态因子必须被保持率门控：否则它们构成不随时间衰减的
        # 权重下限，记忆永远无法降级，遗忘-唤醒生命周期失效
        final = (
            retention_weight * retention
            + retention * (
                self.coeffs['emotion'] * (0.5 + 0.5 * emotion_boost) +
                self.coeffs['importance'] * importance_base +
                self.coeffs['connection'] * connection_boost +
                self.coeffs['association'] * association_density
            )
        )
        # 回忆反馈偏置：被确认正确的记忆权重上浮，被纠错的下沉
        final += chunk.recall_bias
        final = max(0.0, min(1.0, final))

        factors = WeightFactors(
            activation=activation,
            retention=retention,
            time_decay=retention,  # 历史字段名，保持兼容
            frequency=frequency,
            recency=recency,
            emotion_boost=emotion_boost,
            association_density=association_density,
            importance_base=importance_base,
            connection_boost=connection_boost,
            recall_bias=chunk.recall_bias,
            final=final,
        )
        self.weight_cache[chunk.id] = CachedWeight(factors=factors, calculated_at=time.time())
        return factors

    def _invalidate_weight(self, chunk_id: str):
        """清除单条记忆的权重缓存"""
        self.weight_cache.pop(chunk_id, None)

    # 索引名 -> 索引 dict 的映射（快照回滚时使用）
    def _index_map(self) -> Dict[str, Dict[str, Set[str]]]:
        return {
            "time": self.time_index,
            "time_relative": self.time_relative_index,
            "time_context": self.time_context_index,
            "topic": self.topic_index,
            "location": self.location_index,
            "person": self.person_index,
        }

    def _add_to_index(self, chunk: MemoryChunk):
        """把记忆加入倒排索引，并记录键快照"""
        if chunk.id not in self.all_ids:
            self.all_ids.append(chunk.id)

        keys: List[Tuple[str, str]] = []

        if chunk.time_absolute:
            keys.append(("time", chunk.time_absolute[:7]))
        if chunk.time_relative:
            keys.append(("time_relative", chunk.time_relative))
        if chunk.time_context:
            keys.append(("time_context", chunk.time_context))
        for topic in chunk.topics:
            keys.append(("topic", topic))
        if chunk.location:
            keys.append(("location", chunk.location))
        for person in chunk.persons:
            keys.append(("person", person))

        index_map = self._index_map()
        for index_name, key in keys:
            index_map[index_name].setdefault(key, set()).add(chunk.id)

        self._index_keys[chunk.id] = keys

    def _remove_from_index(self, chunk: MemoryChunk):
        """从倒排索引移除记忆（按加入时的键快照精确回滚）"""
        if chunk.id in self.all_ids:
            self.all_ids.remove(chunk.id)

        index_map = self._index_map()
        for index_name, key in self._index_keys.pop(chunk.id, []):
            bucket = index_map[index_name].get(key)
            if bucket is not None:
                bucket.discard(chunk.id)
                if not bucket:
                    del index_map[index_name][key]

        self._invalidate_weight(chunk.id)

    def _rebuild_indexes(self):
        """加载持久化数据后重建所有索引"""
        self.all_ids = []
        self.time_index.clear()
        self.time_relative_index.clear()
        self.time_context_index.clear()
        self.topic_index.clear()
        self.location_index.clear()
        self.person_index.clear()
        self.weight_cache.clear()
        self._index_keys.clear()
        for chunk in self._store.get_all().values():
            self._add_to_index(chunk)

    def _select_candidates(
        self,
        query_tags: Dict[str, Any],
        allow_scan_fallback: bool = True,
    ) -> List[str]:
        """
        基于索引选择候选集。

        多个索引命中时取交集；没有索引可用时退回受限扫描，避免全量遍历。

        allow_scan_fallback=False 时，索引零命中直接返回空列表——
        供混合检索的元数据腿使用：锚点存在但没有命中时，不能把
        "整个存储的前 200 条"当作命中结果灌进 RRF 融合。
        """
        buckets: List[Set[str]] = []

        if "time_absolute" in query_tags:
            year_month = query_tags["time_absolute"][:7]
            buckets.append(set(self.time_index.get(year_month, set())))

        if "time_relative" in query_tags:
            buckets.append(set(self.time_relative_index.get(query_tags["time_relative"], set())))

        if "time_context" in query_tags:
            buckets.append(set(self.time_context_index.get(query_tags["time_context"], set())))

        if "topics" in query_tags:
            topic_bucket: Set[str] = set()
            for topic in query_tags["topics"]:
                topic_bucket.update(self.topic_index.get(topic, set()))
            buckets.append(topic_bucket)

        if "location" in query_tags:
            buckets.append(set(self.location_index.get(query_tags["location"], set())))

        if "persons" in query_tags:
            person_bucket: Set[str] = set()
            for person in query_tags["persons"]:
                person_bucket.update(self.person_index.get(person, set()))
            buckets.append(person_bucket)

        non_empty = [bucket for bucket in buckets if bucket]
        if non_empty:
            candidates = set.intersection(*non_empty)
            if not candidates:
                candidates = set.union(*non_empty)
            return list(candidates)[:self.max_scan_candidates]

        if not allow_scan_fallback:
            return []

        return self.all_ids[:self.max_scan_candidates]

    # ============ 记忆操作 ============

    def add(self, chunk: MemoryChunk) -> str:
        """添加记忆"""
        existing = self._store.get(chunk.id)
        if existing:
            self._remove_from_index(existing)

        if chunk.layer != MemoryLayer.CORE:
            chunk.layer = MemoryLayer.CORE
        self._store.put(chunk)
        self._add_to_index(chunk)
        return chunk.id

    def get(self, chunk_id: str) -> Optional[MemoryChunk]:
        """获取记忆"""
        return self._store.get(chunk_id)

    def access(self, chunk_id: str) -> Optional[Tuple[MemoryChunk, WeightFactors]]:
        """访问记忆，返回碎片和权重"""
        chunk = self._store.get(chunk_id)
        if not chunk:
            return None
        chunk.access(decay=self._rehearsal_decay(chunk))
        # 必须写回：SQLite 后端的 get() 返回副本，不写回则访问统计静默丢失
        self._store.put(chunk)
        self._invalidate_weight(chunk.id)
        return chunk, self.calc_weight(chunk)

    def _rehearsal_decay(self, chunk: MemoryChunk) -> float:
        """本次使用事件的衰减速率（Pavlik 间隔效应）

        以访问瞬间的激活水平定衰减：刚用过就再用（激活高）的
        痕迹衰减快，快忘了才复习（激活近阈值）的痕迹最耐久。
        """
        from core.weight_system import actr_decay, pavlik_event_decay
        import time as _time
        activation = self._base_level_activation(chunk, _time.time())
        return pavlik_event_decay(activation, actr_decay(chunk.memory_type))

    def remove(self, chunk_id: str) -> Optional[MemoryChunk]:
        """删除记忆"""
        chunk = self._store.get(chunk_id)
        if chunk:
            self._store.delete(chunk_id)
            self._remove_from_index(chunk)
        return chunk

    # ============ 检索 ============

    def retrieve(
        self,
        query_tags: Dict[str, Any],
        min_weight: float = 0.0,
        limit: int = 10,
    ) -> List[Tuple[MemoryChunk, WeightFactors]]:
        """
        基于标签检索记忆

        使用 top-k heap 优化，避免全排序。

        返回：[(碎片, 权重), ...]，按权重降序
        """
        import heapq
        import itertools

        # min-heap，存储 (-weight, seq, chunk, wf) 以便快速获取 top-k。
        # seq 是单调递增的平局打破器：权重相同时避免比较 MemoryChunk（不可比较，会 TypeError）
        heap = []
        seq = itertools.count()

        matched_count = 0
        for chunk_id in self._select_candidates(query_tags):
            chunk = self._store.get(chunk_id)
            if not chunk:
                continue
            if not chunk.matches_query(query_tags):
                continue

            matched_count += 1
            if matched_count > self.early_exit_k:
                break

            wf = self.calc_weight(chunk)
            if wf.final >= min_weight:
                if len(heap) < limit:
                    heapq.heappush(heap, (-wf.final, next(seq), chunk, wf))
                elif -wf.final < heap[0][0]:
                    # 当前分数比堆顶高，替换
                    heapq.heapreplace(heap, (-wf.final, next(seq), chunk, wf))

        # 按权重降序返回
        result = [(chunk, wf) for _, _, chunk, wf in sorted(heap)]
        return result

    def get_top(self, limit: int = 20) -> List[Tuple[MemoryChunk, WeightFactors]]:
        """获取当前权重最高的记忆"""
        import heapq
        import itertools

        heap = []
        seq = itertools.count()
        for chunk in self._store.get_all().values():
            wf = self.calc_weight(chunk)
            if len(heap) < limit:
                heapq.heappush(heap, (-wf.final, next(seq), chunk, wf))
            elif -wf.final < heap[0][0]:
                heapq.heapreplace(heap, (-wf.final, next(seq), chunk, wf))

        return [(chunk, wf) for _, _, chunk, wf in sorted(heap)]

    # ============ Hebbian关联 ============

    def strengthen_association(self, chunk_id_a: str, chunk_id_b: str, strength: float = 0.1):
        """Hebbian增强：一起使用的记忆互相增强"""
        chunk_a = self._store.get(chunk_id_a)
        chunk_b = self._store.get(chunk_id_b)
        if not chunk_a or not chunk_b:
            return

        # 双向增强
        current_a = chunk_a.associations.get(chunk_id_b, 0.0)
        current_b = chunk_b.associations.get(chunk_id_a, 0.0)

        chunk_a.associations[chunk_id_b] = min(1.0, current_a + strength * (1 - current_a))
        chunk_b.associations[chunk_id_a] = min(1.0, current_b + strength * (1 - current_b))
        self._store.put(chunk_a)
        self._store.put(chunk_b)
        self._invalidate_weight(chunk_id_a)
        self._invalidate_weight(chunk_id_b)

    def weaken_association(self, chunk_id_a: str, chunk_id_b: str, strength: float = 0.05):
        """Hebbian减弱：长期不一起使用则衰减"""
        chunk_a = self._store.get(chunk_id_a)
        chunk_b = self._store.get(chunk_id_b)
        if not chunk_a or not chunk_b:
            return

        # 衰减到接近零的边直接删除：零强度条目没有任何信息价值，
        # 却会永久占据关联表（Hebbian 突触修剪）
        for owner, other in ((chunk_a, chunk_id_b), (chunk_b, chunk_id_a)):
            if other in owner.associations:
                new_strength = max(0.0, owner.associations[other] - strength)
                if new_strength <= 0.01:
                    del owner.associations[other]
                else:
                    owner.associations[other] = new_strength
        self._store.put(chunk_a)
        self._store.put(chunk_b)
        self._invalidate_weight(chunk_id_a)
        self._invalidate_weight(chunk_id_b)

    def access_together(self, chunk_ids: List[str]):
        """同时访问多个记忆（触发Hebbian增强）"""
        for chunk_id in chunk_ids:
            chunk = self._store.get(chunk_id)
            if chunk:
                chunk.access(decay=self._rehearsal_decay(chunk))
                self._store.put(chunk)
                self._invalidate_weight(chunk_id)

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
        chunk = self._store.get(chunk_id)
        if not chunk:
            return

        if success:
            # 成功回忆：累积正向偏置（情绪强度影响提升幅度），
            # 偏置直接叠加进 calc_weight 的最终权重，持久生效。
            boost = 0.05 * (1 + feedback_emotion)
            chunk.recall_bias = min(0.25, chunk.recall_bias + boost)
            chunk.successful_recall()
            self.total_recall_success += 1
            # 持续成功的记忆，长期重要性缓慢上升
            chunk.importance = min(1.0, chunk.importance + 0.01)
        else:
            # 错误回忆：累积负向偏置，把不可靠的记忆推向降级阈值
            penalty = 0.08 * (1 + abs(feedback_emotion))
            chunk.recall_bias = max(-0.25, chunk.recall_bias - penalty)
            self.total_recall_fail += 1

        self._store.put(chunk)
        self._invalidate_weight(chunk_id)

    # ============ 降级检查 ============

    def check_degrade(self) -> List[str]:
        """
        检查需要降级到伪遗忘层的记忆
        返回降级记忆的ID列表
        """
        to_degrade = []

        for chunk_id, chunk in self._store.get_all().items():
            wf = self.calc_weight(chunk)
            if wf.final < self.degrade_threshold:
                to_degrade.append(chunk_id)

        return to_degrade

    def degrade_chunks(self, chunk_ids: List[str]) -> List[MemoryChunk]:
        """将记忆降级到伪遗忘层"""
        degraded = []
        for chunk_id in chunk_ids:
            chunk = self._store.get(chunk_id)
            if chunk:
                self._store.delete(chunk_id)
                self._remove_from_index(chunk)
                chunk.layer = MemoryLayer.FORGOTTEN
                degraded.append(chunk)
        return degraded

    def decay_all_unused(self, idle_seconds: float = 7 * 24 * 3600) -> int:
        """
        衰减长期未访问记忆的使用频率。

        时间和近因衰减在 calc_weight 中动态计算；这里负责让访问频率也逐步回落，
        这样 maintain() 可以安全运行并推动低价值记忆降级。
        """
        now = time.time()
        changed = 0
        for chunk in self._store.get_all().values():
            if now - chunk.last_accessed <= idle_seconds:
                continue
            if chunk.access_count <= 0:
                continue
            chunk.access_count -= 1
            chunk.updated_at = now
            self._store.put(chunk)
            self._invalidate_weight(chunk.id)
            changed += 1
        return changed

    # ============ 持久化 ============

    def save(self, filepath: str = None):
        """
        保存到后端

        filepath 参数仅用于向后兼容（JsonMemoryStore 会在构造时指定路径）
        """
        self._store.save()

    def load(self, filepath: str = None) -> bool:
        """
        从后端加载

        filepath 参数仅用于向后兼容
        """
        success = self._store.load()
        if success:
            self._rebuild_indexes()
        return success

    def __len__(self) -> int:
        return self._store.count()
