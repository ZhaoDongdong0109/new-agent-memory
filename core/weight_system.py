"""
动态权重系统 - 类人记忆系统核心

模拟生物大脑的权重机制：
- 时间衰减：越久远的记忆权重越低
- 使用频率：越常用的记忆权重越高
- 情绪强度：情绪强烈的事件记得更牢
- 关联密度：与其他记忆连接越多越稳定
- 重要性：生存/情感价值影响权重
- Hebbian Rule：一起使用的记忆互相增强
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Set, Optional
from enum import Enum


class EmotionType(Enum):
    """情绪类型（简化版）"""
    NEUTRAL = 0.0
    MILD_POSITIVE = 0.3      # 轻微正面
    POSITIVE = 0.6           # 正面
    STRONG_POSITIVE = 1.0    # 强烈正面（如奖励）
    MILD_NEGATIVE = -0.3     # 轻微负面
    NEGATIVE = -0.6          # 负面
    STRONG_NEGATIVE = -1.0   # 强烈负面（如危险）


@dataclass
class MemoryItem:
    """记忆项"""
    id: str
    content: str  # 实际内容（可以是切片ID或原文）
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    
    # 多维索引
    time_tags: Set[str] = field(default_factory=set)      # 时间标签：["2026-04", "morning"]
    person_tags: Set[str] = field(default_factory=set)     # 人物标签
    location_tags: Set[str] = field(default_factory=set)   # 地点标签
    topic_tags: Set[str] = field(default_factory=set)       # 主题标签
    emotion_tags: Set[str] = field(default_factory=set)    # 情绪标签
    
    # 动态权重因子（0.0 ~ 1.0）
    base_importance: float = 0.5  # 基础重要性
    emotion_intensity: float = 0.0  # 情绪强度 -1.0 ~ 1.0
    
    # 关联记忆（Hebbian 关联）
    associated_memories: Dict[str, float] = field(default_factory=dict)  # memory_id -> association_weight
    
    def access(self):
        """记录一次访问"""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class WeightFactors:
    """权重因子计算结果"""
    time_decay: float = 0.0        # 时间衰减因子 (0~1, 越老越低)
    usage_frequency: float = 0.0   # 使用频率因子 (0~1, 越常用越高)
    recency: float = 0.0           # 近因效应 (0~1, 最近访问的更高)
    emotion_boost: float = 0.0     # 情绪增强因子 (负值也会降低)
    association_density: float = 0.0  # 关联密度 (连接越多越稳定)
    importance_base: float = 0.0   # 重要性基础值
    
    final_weight: float = 0.0      # 最终权重
    
    def summary(self) -> str:
        return (
            f"final={self.final_weight:.3f} "
            f"(time={self.time_decay:.2f} "
            f"freq={self.usage_frequency:.2f} "
            f"recency={self.recency:.2f} "
            f"emotion={self.emotion_boost:+.2f} "
            f"assoc={self.association_density:.2f} "
            f"imp={self.importance_base:.2f})"
        )


class DynamicWeightSystem:
    """
    动态权重系统
    
    核心公式:
    W_final = f(time_decay, usage, recency, emotion, association, importance)
    
    其中：
    - 时间衰减: 基于艾宾浩斯遗忘曲线，指数衰减
    - 使用频率: log scale，避免无限增长
    - 近因效应: 最近访问的权重更高
    - 情绪增强: 情绪强度直接加到权重上
    - 关联密度: 关联越多，权重衰减越慢（更稳定）
    - 重要性: 基础权重系数
    """
    
    def __init__(
        self,
        # 时间衰减参数
        decay_half_life: float = 7.0 * 24 * 3600,  # 半衰期7天
        decay_rate: float = 0.1,
        
        # 使用频率参数
        freq_half_life: float = 10.0,  # 多少次访问后频率趋于饱和
        
        # 近因效应参数
        recency_window: float = 24 * 3600,  # 24小时内算"最近"
        
        # 关联密度参数
        assoc_stability_factor: float = 0.05,  # 每个关联减少的衰减率
        
        # 权重组合参数
        weights: Optional[Dict[str, float]] = None,
    ):
        self.decay_half_life = decay_half_life
        self.decay_rate = decay_rate
        self.freq_half_life = freq_half_life
        self.recency_window = recency_window
        self.assoc_stability_factor = assoc_stability_factor
        
        # 权重组合系数（可调）
        self.coeffs = weights or {
            'time_decay': 0.25,
            'usage_frequency': 0.20,
            'recency': 0.15,
            'emotion': 0.20,
            'association': 0.10,
            'importance': 0.10,
        }
        
        # 存储的记忆
        self.memories: Dict[str, MemoryItem] = {}
    
    # ============ 核心计算方法 ============
    
    def calc_time_decay(self, memory: MemoryItem) -> float:
        """
        计算时间衰减因子
        基于指数衰减: e^(-λt)，其中半衰期为 T_half = ln2/λ
        关联密度会减缓衰减
        """
        age = time.time() - memory.created_at
        
        # 关联数量会影响有效衰减率
        assoc_count = len(memory.associated_memories)
        effective_decay_rate = self.decay_rate * (1 - assoc_count * self.assoc_stability_factor)
        effective_decay_rate = max(effective_decay_rate, 0.01)  # 最小衰减率
        
        decay = math.exp(-effective_decay_rate * age / self.decay_half_life)
        return decay
    
    def calc_usage_frequency(self, memory: MemoryItem) -> float:
        """
        计算使用频率因子
        使用对数函数，避免无限增长
        """
        if memory.access_count == 0:
            return 0.0
        return math.log(1 + memory.access_count) / math.log(1 + self.freq_half_life)
    
    def calc_recency(self, memory: MemoryItem) -> float:
        """
        计算近因效应
        越接近现在访问的，权重越高
        """
        time_since_access = time.time() - memory.last_accessed
        
        if time_since_access <= self.recency_window:
            # 在窗口期内，线性接近1
            return 1.0 - (time_since_access / self.recency_window) * 0.5
        else:
            # 超出窗口，按指数衰减到0.1
            excess = time_since_access - self.recency_window
            return 0.5 * math.exp(-excess / (7 * 24 * 3600))  # 7天后基本为0
    
    def calc_emotion_boost(self, memory: MemoryItem) -> float:
        """
        计算情绪增强因子
        情绪强度直接加到权重上（负值也会降低）
        """
        return memory.emotion_intensity
    
    def calc_association_density(self, memory: MemoryItem) -> float:
        """
        计算关联密度因子
        关联越多，权重越高且越稳定
        """
        assoc_count = len(memory.associated_memories)
        if assoc_count == 0:
            return 0.0
        
        # 对数增长，有天花板
        density = math.log(1 + assoc_count) / math.log(11)  # 10个关联趋于饱和
        return min(density, 1.0)
    
    def calc_importance_base(self, memory: MemoryItem) -> float:
        """计算基础重要性"""
        return memory.base_importance
    
    def calculate_weight(self, memory: MemoryItem) -> WeightFactors:
        """
        计算记忆的综合权重
        """
        factors = WeightFactors(
            time_decay=self.calc_time_decay(memory),
            usage_frequency=self.calc_usage_frequency(memory),
            recency=self.calc_recency(memory),
            emotion_boost=self.calc_emotion_boost(memory),
            association_density=self.calc_association_density(memory),
            importance_base=self.calc_importance_base(memory),
        )
        
        # 归一化 time_decay 到 0.1~1.0（记忆再老也有基本权重）
        time_norm = 0.1 + 0.9 * factors.time_decay
        
        # 综合计算
        raw_weight = (
            self.coeffs['time_decay'] * time_norm +
            self.coeffs['usage_frequency'] * factors.usage_frequency +
            self.coeffs['recency'] * factors.recency +
            self.coeffs['emotion'] * (0.5 + 0.5 * factors.emotion_boost) +  # 情绪映射到0~1
            self.coeffs['association'] * factors.association_density +
            self.coeffs['importance'] * factors.importance_base
        )
        
        factors.final_weight = max(0.0, min(1.0, raw_weight))
        return factors
    
    # ============ Hebbian 关联更新 ============
    
    def strengthen_association(self, memory_a: str, memory_b: str, strength: float = 0.1):
        """
        Hebbian 更新：一起使用的记忆互相增强
        如果记忆A和记忆B同时被访问，它们的关联权重增加
        """
        if memory_a not in self.memories or memory_b not in self.memories:
            return
        
        a = self.memories[memory_a]
        b = self.memories[memory_b]
        
        # 双向增强
        current_ab = a.associated_memories.get(memory_b, 0.0)
        current_ba = b.associated_memories.get(memory_a, 0.0)
        
        #Hebbian: "一起放电则连接加强"
        new_ab = min(1.0, current_ab + strength * (1 - current_ab))
        new_ba = min(1.0, current_ba + strength * (1 - current_ba))
        
        a.associated_memories[memory_b] = new_ab
        b.associated_memories[memory_a] = new_ba
    
    def weaken_association(self, memory_a: str, memory_b: str, strength: float = 0.05):
        """
        减弱关联（长期不一起使用则衰减）
        """
        if memory_a not in self.memories or memory_b not in self.memories:
            return
        
        a = self.memories[memory_a]
        b = self.memories[memory_b]
        
        current_ab = a.associated_memories.get(memory_b, 0.0)
        current_ba = b.associated_memories.get(memory_a, 0.0)
        
        if current_ab > 0:
            a.associated_memories[memory_b] = max(0.0, current_ab - strength)
        if current_ba > 0:
            b.associated_memories[memory_a] = max(0.0, current_ba - strength)
    
    # ============ 记忆操作 ============
    
    def add_memory(
        self,
        content: str,
        importance: float = 0.5,
        emotion: float = 0.0,
        time_tags: Optional[Set[str]] = None,
        person_tags: Optional[Set[str]] = None,
        location_tags: Optional[Set[str]] = None,
        topic_tags: Optional[Set[str]] = None,
    ) -> str:
        """添加新记忆"""
        memory_id = f"mem_{len(self.memories)}_{int(time.time() * 1000)}"
        
        memory = MemoryItem(
            id=memory_id,
            content=content,
            base_importance=importance,
            emotion_intensity=emotion,
            time_tags=time_tags or set(),
            person_tags=person_tags or set(),
            location_tags=location_tags or set(),
            topic_tags=topic_tags or set(),
        )
        
        self.memories[memory_id] = memory
        return memory_id
    
    def access_memory(self, memory_id: str) -> Optional[WeightFactors]:
        """访问记忆，返回权重"""
        if memory_id not in self.memories:
            return None
        
        memory = self.memories[memory_id]
        memory.access()
        return self.calculate_weight(memory)
    
    def access_together(self, memory_ids: list):
        """
        同时访问多个记忆（触发Hebbian增强）
        用于检索时一起使用的记忆互相强化
        """
        for memory_id in memory_ids:
            if memory_id in self.memories:
                self.memories[memory_id].access()
        
        # Hebbian 关联更新
        for i, id_a in enumerate(memory_ids):
            for id_b in memory_ids[i+1:]:
                self.strengthen_association(id_a, id_b)
    
    def decay_all_unused(self):
        """
        对长期未使用的记忆进行时间衰减更新
        （不需要每次调用，在合适时机定期调用即可）
        """
        now = time.time()
        for memory in self.memories.values():
            # 如果很久没访问，减少其 access_count 的影响
            if now - memory.last_accessed > 7 * 24 * 3600:  # 7天未访问
                memory.access_count = max(0, memory.access_count - 1)
    
    def get_top_memories(self, limit: int = 10) -> list:
        """获取当前权重最高的记忆"""
        weighted = []
        for memory in self.memories.values():
            w = self.calculate_weight(memory)
            weighted.append((memory, w))
        
        weighted.sort(key=lambda x: x[1].final_weight, reverse=True)
        return weighted[:limit]
    
    # ============ 检索（基于权重） ============
    
    def retrieve_by_tags(
        self,
        time_tags: Optional[Set[str]] = None,
        person_tags: Optional[Set[str]] = None,
        location_tags: Optional[Set[str]] = None,
        topic_tags: Optional[Set[str]] = None,
        min_weight: float = 0.0,
        limit: int = 10,
    ) -> list:
        """
        基于多维标签检索记忆，返回符合条件且权重较高的记忆
        """
        candidates = []
        
        for memory in self.memories.values():
            # 标签匹配
            matched = True
            if time_tags and not (time_tags & memory.time_tags):
                matched = False
            if person_tags and not (person_tags & memory.person_tags):
                matched = False
            if location_tags and not (location_tags & memory.location_tags):
                matched = False
            if topic_tags and not (topic_tags & memory.topic_tags):
                matched = False
            
            if not matched:
                continue
            
            weight = self.calculate_weight(memory)
            if weight.final_weight >= min_weight:
                candidates.append((memory, weight))
        
        # 按权重排序
        candidates.sort(key=lambda x: x[1].final_weight, reverse=True)
        return candidates[:limit]


# ============ 测试示例 ============

if __name__ == "__main__":
    print("=" * 60)
    print("动态权重系统 - 测试")
    print("=" * 60)
    
    system = DynamicWeightSystem()
    
    # 创建一些测试记忆
    print("\n[1] 创建测试记忆...")
    
    # 日常记忆
    mem1 = system.add_memory(
        content="今天早餐吃了包子",
        importance=0.3,
        emotion=0.1,
        time_tags={"2026-04", "morning"},
        topic_tags={"daily", "food"}
    )
    print(f"  创建记忆1: {mem1}")
    
    # 情绪强烈的记忆
    mem2 = system.add_memory(
        content="今天获得了大奖！非常开心！",
        importance=0.8,
        emotion=0.9,
        time_tags={"2026-04"},
        topic_tags={"achievement", "emotion"}
    )
    print(f"  创建记忆2: {mem2}")
    
    # 重要且多次访问的记忆
    mem3 = system.add_memory(
        content="项目核心架构设计方案",
        importance=0.9,
        emotion=0.2,
        time_tags={"2026-04"},
        topic_tags={"project", "architecture"}
    )
    print(f"  创建记忆3: {mem3}")
    
    # 模拟多次访问mem3
    for _ in range(5):
        system.access_memory(mem3)
    print(f"  访问mem3 5次")
    
    # 创建关联
    print("\n[2] 建立记忆关联...")
    system.strengthen_association(mem2, mem3, strength=0.3)
    print(f"  mem2 <-> mem3 关联增强")
    
    # 测试同时访问
    print("\n[3] 同时访问mem2和mem3（Hebbian增强）...")
    system.access_together([mem2, mem3])
    
    # 获取权重
    print("\n[4] 各记忆权重详情:")
    for mid in [mem1, mem2, mem3]:
        if mid in system.memories:
            mem = system.memories[mid]
            w = system.calculate_weight(mem)
            print(f"\n  记忆: {mem.content[:20]}...")
            print(f"    权重: {w.summary()}")
    
    # 检索测试
    print("\n[5] 按标签检索 [topic_tags={'project'}]:")
    results = system.retrieve_by_tags(topic_tags={"project"})
    for mem, w in results:
        print(f"  - {mem.content[:30]}... [weight={w.final_weight:.3f}]")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
