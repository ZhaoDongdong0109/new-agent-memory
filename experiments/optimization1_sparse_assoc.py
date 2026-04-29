"""
优化1：关联稀疏化

问题：每个记忆存储所有关联（即使强度0.01），导致存储和计算线性增长
解决：只存储强度 > 阈值的关联

原理：
- Hebbian规则：一起放电则连接加强
- 但大部分关联是弱的（噪音），可以忽略
- 只保留强关联（> 阈值），减少90%的存储和计算
"""

from typing import Dict, Set, Optional
from dataclasses import dataclass


@dataclass
class SparseAssociation:
    """稀疏关联"""
    target_id: str
    strength: float  # 0.0 ~ 1.0


class SparseAssociationStore:
    """
    稀疏关联存储
    
    设计：
    - 关联强度 < threshold 时不存储（视为0）
    - 关联强度 < min_storage 时删除（防止无限积累）
    - 增量更新，避免每次重新计算
    """
    
    def __init__(
        self,
        storage_threshold: float = 0.1,   # 低于此值不存储
        min_storage_threshold: float = 0.05,  # 低于此值且在队列中则删除
        initial_strength: float = 0.1,    # 新关联初始强度
    ):
        self.storage_threshold = storage_threshold
        self.min_storage_threshold = min_storage_threshold
        self.initial_strength = initial_strength
        
        # 存储强关联：chunk_id -> {target_id -> strength}
        self.strong_assocs: Dict[str, Dict[str, float]] = {}
        
        # 待增强队列（记录最近被一起访问的pair）
        self.pending_boosts: Set[tuple] = set()
    
    def get_associations(self, chunk_id: str) -> Dict[str, float]:
        """获取某记忆的所有强关联"""
        return self.strong_assocs.get(chunk_id, {}).copy()
    
    def get_strength(self, chunk_id_a: str, chunk_id_b: str) -> float:
        """获取两个记忆之间的关联强度"""
        return self.strong_assocs.get(chunk_id_a, {}).get(chunk_id_b, 0.0)
    
    def _ensure_chunk(self, chunk_id: str):
        """确保chunk有存储空间"""
        if chunk_id not in self.strong_assocs:
            self.strong_assocs[chunk_id] = {}
    
    def add_pending_boost(self, chunk_id_a: str, chunk_id_b: str):
        """
        添加待增强的关联到队列
        实际增强在 flush_pending_boosts() 时执行
        """
        # 保持有序，避免 (a,b) 和 (b,a) 重复
        pair = tuple(sorted([chunk_id_a, chunk_id_b]))
        self.pending_boosts.add(pair)
    
    def flush_pending_boosts(self, boost_strength: float = 0.1):
        """
        批量执行待增强的关联
        减少频繁更新带来的开销
        """
        if not self.pending_boosts:
            return
        
        for chunk_id_a, chunk_id_b in self.pending_boosts:
            self.boost(chunk_id_a, chunk_id_b, boost_strength)
        
        self.pending_boosts.clear()
    
    def boost(self, chunk_id_a: str, chunk_id_b: str, strength: float = 0.1):
        """
        增强两个记忆之间的关联（Hebbian规则）
        
        new_strength = old_strength + η × (1 - old_strength)
        
        只有超过阈值时才存储
        """
        self._ensure_chunk(chunk_id_a)
        self._ensure_chunk(chunk_id_b)
        
        # Hebbian 增强
        old_a = self.strong_assocs[chunk_id_a].get(chunk_id_b, self.initial_strength)
        old_b = self.strong_assocs[chunk_id_b].get(chunk_id_a, self.initial_strength)
        
        new_a = min(1.0, old_a + strength * (1 - old_a))
        new_b = min(1.0, old_b + strength * (1 - old_b))
        
        # 只有超过阈值才存储
        if new_a >= self.storage_threshold:
            self.strong_assocs[chunk_id_a][chunk_id_b] = new_a
        elif chunk_id_b in self.strong_assocs[chunk_id_a]:
            # 低于阈值，删除
            del self.strong_assocs[chunk_id_a][chunk_id_b]
        
        if new_b >= self.storage_threshold:
            self.strong_assocs[chunk_id_b][chunk_id_a] = new_b
        elif chunk_id_a in self.strong_assocs[chunk_id_b]:
            del self.strong_assocs[chunk_id_b][chunk_id_a]
        
        # 清理空桶
        if not self.strong_assocs[chunk_id_a]:
            del self.strong_assocs[chunk_id_a]
        if not self.strong_assocs[chunk_id_b]:
            del self.strong_assocs[chunk_id_b]
    
    def decay(self, chunk_id_a: str, chunk_id_b: str, strength: float = 0.02):
        """
        衰减两个记忆之间的关联
        """
        if chunk_id_a not in self.strong_assocs:
            return
        
        old = self.strong_assocs[chunk_id_a].get(chunk_id_b, 0.0)
        if old <= 0:
            return
        
        new = max(0.0, old - strength)
        
        if new >= self.storage_threshold:
            self.strong_assocs[chunk_id_a][chunk_id_b] = new
            self.strong_assocs[chunk_id_b][chunk_id_a] = new
        else:
            # 低于阈值，删除
            if chunk_id_b in self.strong_assocs[chunk_id_a]:
                del self.strong_assocs[chunk_id_a][chunk_id_b]
            if chunk_id_a in self.strong_assocs[chunk_id_b]:
                del self.strong_assocs[chunk_id_b][chunk_id_a]
        
        # 清理空桶
        if not self.strong_assocs.get(chunk_id_a):
            self.strong_assocs.pop(chunk_id_a, None)
        if not self.strong_assocs.get(chunk_id_b):
            self.strong_assocs.pop(chunk_id_b, None)
    
    def get_connected_chunks(self, chunk_id: str, min_strength: float = 0.1) -> Dict[str, float]:
        """
        获取与某记忆关联的所有记忆（强度 >= min_strength）
        """
        assocs = self.strong_assocs.get(chunk_id, {})
        return {k: v for k, v in assocs.items() if v >= min_strength}
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        total_chunks = len(self.strong_assocs)
        total_assocs = sum(len(v) for v in self.strong_assocs.values()) // 2
        avg_assocs = total_assocs / max(total_chunks, 1)
        
        strengths = []
        for assocs in self.strong_assocs.values():
            strengths.extend(assocs.values())
        
        return {
            "total_chunks": total_chunks,
            "total_assocs": total_assocs,
            "avg_assocs_per_chunk": avg_assocs,
            "avg_strength": sum(strengths) / len(strengths) if strengths else 0,
            "min_strength": min(strengths) if strengths else 0,
            "max_strength": max(strengths) if strengths else 0,
        }
    
    def remove_chunk(self, chunk_id: str):
        """删除某记忆的所有关联"""
        if chunk_id in self.strong_assocs:
            # 删除所有指向该chunk的关联
            for target_id in self.strong_assocs[chunk_id]:
                if target_id in self.strong_assocs:
                    self.strong_assocs[target_id].pop(chunk_id, None)
                    if not self.strong_assocs[target_id]:
                        del self.strong_assocs[target_id]
            del self.strong_assocs[chunk_id]


# ============ 测试 ============

if __name__ == "__main__":
    print("=" * 60)
    print("关联稀疏化测试")
    print("=" * 60)
    
    store = SparseAssociationStore(storage_threshold=0.15)
    
    # 测试 Hebbian 增强
    print("\n[1] Hebbian 增强测试")
    store.add_pending_boost("A", "B")
    store.add_pending_boost("A", "B")  # 重复
    store.add_pending_boost("A", "C")
    store.flush_pending_boosts(boost_strength=0.1)
    
    print(f"  A-B 强度: {store.get_strength('A', 'B'):.3f}")
    print(f"  A-C 强度: {store.get_strength('A', 'C'):.3f}")
    
    # 多次增强
    print("\n[2] 多次增强后")
    for _ in range(10):
        store.add_pending_boost("A", "B")
    store.flush_pending_boosts(boost_strength=0.1)
    
    print(f"  A-B 强度: {store.get_strength('A', 'B'):.3f}")
    
    # 衰减测试
    print("\n[3] 衰减测试")
    for _ in range(5):
        store.decay("A", "B", 0.1)
    
    print(f"  A-B 强度: {store.get_strength('A', 'B'):.3f}")
    
    # 弱关联过滤测试
    print("\n[4] 弱关联过滤测试")
    store.add_pending_boost("X", "Y")  # 只有0.1的初始强度
    store.flush_pending_boosts(boost_strength=0.1)
    print(f"  X-Y 强度（低于阈值0.15）: {store.get_strength('X', 'Y'):.3f}")
    print(f"  X 的关联数: {len(store.get_associations('X'))}")
    
    # 统计
    print("\n[5] 统计信息")
    stats = store.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
