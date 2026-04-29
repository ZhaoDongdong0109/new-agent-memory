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
        
        # 统计
        self.total_recall_success = 0
        self.total_recall_fail = 0
    
    # ============ 权重计算 ============
    
    def calc_weight(self, chunk: MemoryChunk) -> WeightFactors:
        """计算记忆碎片权重"""
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
        
        return WeightFactors(
            time_decay=time_decay,
            frequency=frequency,
            recency=recency,
            emotion_boost=emotion_boost,
            association_density=association_density,
            importance_base=importance_base,
            connection_boost=connection_boost,
            final=final,
        )
    
    # ============ 记忆操作 ============
    
    def add(self, chunk: MemoryChunk) -> str:
        """添加记忆"""
        if chunk.layer != MemoryLayer.CORE:
            chunk.layer = MemoryLayer.CORE
        self.chunks[chunk.id] = chunk
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
        return chunk, self.calc_weight(chunk)
    
    def remove(self, chunk_id: str) -> Optional[MemoryChunk]:
        """删除记忆"""
        return self.chunks.pop(chunk_id, None)
    
    # ============ 检索 ============
    
    def retrieve(
        self,
        query_tags: Dict[str, Any],
        min_weight: float = 0.0,
        limit: int = 10,
    ) -> List[Tuple[MemoryChunk, WeightFactors]]:
        """
        基于标签检索记忆
        
        返回：[(碎片, 权重), ...]，按权重降序
        """
        candidates = []
        
        for chunk in self.chunks.values():
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
        
        # 双向增强
        current_a = chunk_a.associations.get(chunk_id_b, 0.0)
        current_b = chunk_b.associations.get(chunk_id_a, 0.0)
        
        chunk_a.associations[chunk_id_b] = min(1.0, current_a + strength * (1 - current_a))
        chunk_b.associations[chunk_id_a] = min(1.0, current_b + strength * (1 - current_b))
    
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
            
            return True
        except FileNotFoundError:
            return False
    
    def __len__(self) -> int:
        return len(self.chunks)
