"""
自适应权重系统 - 类人记忆系统核心

核心理念：
- 权重 = 基准关注度 × 情绪系数 × 时间衰减因子
- 情绪系数由概率分布采样，带随机性和不确定性
- 时间衰减基于互动量而非真实时间
- 基准关注度会随使用情况动态漂移

权重更新原则：
- 非线性：不是简单的+0.1累加
- 情绪敏感：强烈情绪带来更大的权重增量
- 时效感知：基于互动量驱动衰减，而非挂钟时间
- 自适应：基准关注度会随用户使用模式缓慢变化
"""

import math
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Set, Optional, List, Any
from enum import Enum

# 导入情绪引擎
from core.emotion_engine import EmotionEngine, EmotionResult


class MemoryType(Enum):
    """记忆类型，影响衰减节奏"""
    STORY = "story"           # 用户故事/经历，衰减最慢
    IDEA = "idea"             # 想法/观点，中慢
    PREFERENCE = "preference"  # 用户偏好，中等
    FACT = "fact"             # 事实信息，中快
    INTERACTION = "interaction" # 我们之间发生的事，衰减最快


# 各类型记忆的半衰期（基于互动次数）
HALFLIFE_BY_TYPE = {
    MemoryType.STORY: 500,
    MemoryType.IDEA: 400,
    MemoryType.PREFERENCE: 300,
    MemoryType.FACT: 200,
    MemoryType.INTERACTION: 150,
}


@dataclass
class MemoryItem:
    """记忆项"""
    id: str = field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:12]}")
    content: str = ""
    summary: str = ""  # 摘要/索引描述
    
    # 类型
    memory_type: MemoryType = MemoryType.INTERACTION
    
    # 多维标签
    time_tags: Set[str] = field(default_factory=set)
    person_tags: Set[str] = field(default_factory=set)
    topic_tags: Set[str] = field(default_factory=set)
    emotion_tags: Set[str] = field(default_factory=set)  # 这条记忆携带的情绪标签
    
    # 基准关注度（用户对这件事的长期重视程度，0~1）
    # 这是会随使用情况缓慢漂移的值
    base_attention: float = 0.5
    
    # 上次被强化时的互动计数
    last_strengthened_at_interaction: int = 0
    
    # 最后一次强化时的情绪标签（用于情绪系数再激活）
    last_emotion: str = "平静"
    
    # 时间戳
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    
    # 访问统计
    access_count: int = 0
    
    # 关联记忆（Hebbian关联）
    associated_memories: Dict[str, float] = field(default_factory=dict)
    
    # 版本追踪
    version: int = 1
    parent_id: Optional[str] = None  # 如果是重建的，记录原记忆ID
    
    def access(self):
        """记录一次访问"""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass 
class WeightResult:
    """权重计算结果"""
    # 各因子
    effective_weight: float = 0.0   # 当前有效权重
    base_attention: float = 0.5     # 基准关注度
    emotion_modifier: float = 1.0   # 情绪修正系数
    recency_decay: float = 0.0      # 时效衰减因子
    
    # 半衰期信息
    halflife_interactions: int = 0  # 当前半衰期（互动次数）
    interactions_since_last: int = 0  # 距上次强化的互动数
    
    def summary(self) -> str:
        return (
            f"effective={self.effective_weight:.3f} "
            f"(base={self.base_attention:.2f} "
            f"emotion={self.emotion_modifier:+.2f} "
            f"recency={self.recency_decay:.2f})"
        )


class AdaptiveWeightSystem:
    """
    自适应权重系统
    
    核心公式：
    effective_weight = base_attention × emotion_modifier × recency_decay
    
    其中：
    - base_attention: 基准关注度，随使用情况漂移
    - emotion_modifier: 上次强化的情绪系数（让情绪继续影响）
    - recency_decay: 基于互动量的时效衰减
    
    设计原则：
    - 不追求精确，允许不确定性
    - 遗忘是feature，不是bug
    - 系统随使用越来越懂用户
    """
    
    # 权重组合参数
    LEARNING_RATE = 0.3      # 新权重增量对基准的贡献比例
    DRIFT_RATE = 0.05        # 基准关注度每月向当前值漂移的比例
    DRIFT_INTERVAL = 1000   # 每多少次互动触发一次漂移
    
    # 阈值
    CORE_THRESHOLD = 0.5    # 进入/留在核心层需要达到的有效权重
    PSEUDO_THRESHOLD = 0.2   # 低于这个值进入伪遗忘层
    DELETE_THRESHOLD = 0.05  # 低于这个值彻底删除
    
    def __init__(self):
        # 存储的记忆
        self.memories: Dict[str, MemoryItem] = {}
        
        # 全局互动计数器
        self.global_interaction_count = 0
        
        # 情绪引擎
        self.emotion_engine = EmotionEngine()
        
        # 漂移计数器
        self.drift_counter = 0
    
    # ============ 核心权重计算 ============
    
    def _get_halflife(self, memory: MemoryItem) -> int:
        """获取某条记忆的半衰期（基于类型）"""
        return HALFLIFE_BY_TYPE.get(memory.memory_type, 200)
    
    def _calculate_recency_decay(self, memory: MemoryItem) -> float:
        """
        计算时效衰减因子
        基于距离上次强化经历了多少次互动
        """
        delta = self.global_interaction_count - memory.last_strengthened_at_interaction
        halflife = self._get_halflife(memory)
        
        # 指数衰减：exp(-delta / halflife)
        # 刚强化的记忆 → decay≈1.0
        # 经过一个半衰期 → decay≈0.5
        decay = math.exp(-delta / halflife)
        return max(0.0, decay)
    
    def _calculate_effective_weight(self, memory: MemoryItem) -> WeightResult:
        """
        计算记忆在当前时刻的有效权重
        """
        halflife = self._get_halflife(memory)
        delta = self.global_interaction_count - memory.last_strengthened_at_interaction
        recency_decay = self._calculate_recency_decay(memory)
        
        # 情绪修正系数（从上次强化时的情绪标签再激活采样）
        emotion_result = self.emotion_engine.infer_emotion(
            "",  # 不需要文本，用上次保存的情绪标签
            context={"topic_repeat_count": 0},
            user_override=memory.last_emotion if memory.last_emotion != "平静" else None,
        )
        emotion_modifier = emotion_result.coefficient
        
        # 有效权重 = 基准 × 情绪 × 时效
        effective = (
            memory.base_attention 
            * emotion_modifier 
            * recency_decay
        )
        
        return WeightResult(
            effective_weight=min(1.0, max(0.0, effective)),
            base_attention=memory.base_attention,
            emotion_modifier=emotion_modifier,
            recency_decay=recency_decay,
            halflife_interactions=halflife,
            interactions_since_last=delta,
        )
    
    def calculate_weight(self, memory: MemoryItem) -> WeightResult:
        """外部接口：计算记忆权重"""
        return self._calculate_effective_weight(memory)
    
    # ============ 权重更新（核心操作） ============
    
    def strengthen_memory(
        self,
        memory_id: str,
        text: str = "",
        emotion_tag: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[WeightResult]:
        """
        强化一条记忆
        
        这是记忆系统最核心的操作之一。
        当用户提到某事、讨论某事、或某事被检索到时调用。
        
        Args:
            memory_id: 记忆ID
            text: 用户说的原文（用于情绪推断）
            emotion_tag: 用户手动指定的情绪标签（可选）
            context: 上下文，包含topic_repeat_count等
        """
        if memory_id not in self.memories:
            return None
        
        memory = self.memories[memory_id]
        context = context or {}
        
        # 1. 推断情绪
        emotion_result = self.emotion_engine.infer_emotion(
            text, 
            context,
            user_override=emotion_tag,
        )
        
        # 2. 计算基准关注度增量
        # delta = interaction_weight × emotion_coefficient
        # interaction_weight默认是1.0，可调
        interaction_weight = context.get("interaction_weight", 1.0)
        delta = interaction_weight * emotion_result.coefficient
        
        # 3. 更新基准关注度（漂移机制）
        # new_base = old_base × (1 - drift_rate) + delta × learning_rate
        old_base = memory.base_attention
        new_base = (
            old_base * (1 - self.DRIFT_RATE)
            + delta * self.LEARNING_RATE
        )
        memory.base_attention = min(1.0, max(0.0, new_base))
        
        # 4. 记录本次强化
        memory.last_strengthened_at_interaction = self.global_interaction_count
        memory.last_emotion = emotion_result.emotion_tag
        memory.last_accessed = time.time()
        
        # 5. 全局互动计数+1
        self.global_interaction_count += 1
        self.drift_counter += 1
        
        # 6. 触发漂移检查
        if self.drift_counter >= self.DRIFT_INTERVAL:
            self._apply_drift()
            self.drift_counter = 0
        
        return self._calculate_effective_weight(memory)
    
    def _apply_drift(self):
        """
        对所有记忆应用漂移
        基准关注度向当前值方向缓慢回归
        很久没被强化的记忆，基准会缓慢下降
        """
        for memory in self.memories.values():
            # 计算距离上次强化的互动数
            delta = self.global_interaction_count - memory.last_strengthened_at_interaction
            
            # 如果很久没被强化，基准缓慢向0漂移
            if delta > self.DRIFT_INTERVAL * 2:
                drift_factor = self.DRIFT_RATE * (delta / self.DRIFT_INTERVAL)
                memory.base_attention *= (1 - drift_factor)
    
    # ============ 记忆操作 ============
    
    def add_memory(
        self,
        content: str,
        summary: str = "",
        memory_type: MemoryType = MemoryType.INTERACTION,
        time_tags: Optional[Set[str]] = None,
        person_tags: Optional[Set[str]] = None,
        topic_tags: Optional[Set[str]] = None,
        emotion_tags: Optional[Set[str]] = None,
        initial_base_attention: float = 0.5,
    ) -> str:
        """
        添加新记忆
        """
        memory_id = f"mem_{len(self.memories)}_{int(time.time() * 1000)}"
        
        memory = MemoryItem(
            id=memory_id,
            content=content,
            summary=summary or content[:100],
            memory_type=memory_type,
            time_tags=time_tags or set(),
            person_tags=person_tags or set(),
            topic_tags=topic_tags or set(),
            emotion_tags=emotion_tags or set(),
            base_attention=initial_base_attention,
            last_strengthened_at_interaction=self.global_interaction_count,
        )
        
        self.memories[memory_id] = memory
        return memory_id
    
    def access_memory(
        self, 
        memory_id: str, 
        text: str = "",
        emotion_tag: Optional[str] = None,
    ) -> Optional[WeightResult]:
        """
        访问记忆，同时强化其权重
        """
        if memory_id not in self.memories:
            return None
        
        self.memories[memory_id].access()
        return self.strengthen_memory(memory_id, text, emotion_tag)
    
    def access_together(
        self, 
        memory_ids: List[str], 
        text: str = "",
    ):
        """
        同时访问多条记忆（触发Hebbian增强）
        同时用文本强化每一条
        """
        for memory_id in memory_ids:
            if memory_id in self.memories:
                self.memories[memory_id].access()
                self.strengthen_memory(memory_id, text)
        
        # Hebbian关联更新
        for i, id_a in enumerate(memory_ids):
            for id_b in memory_ids[i+1:]:
                self.strengthen_association(id_a, id_b)
    
    # ============ Hebbian关联 ============
    
    def strengthen_association(self, memory_a: str, memory_b: str, strength: float = 0.1):
        """Hebbian更新：一起使用的记忆互相增强"""
        if memory_a not in self.memories or memory_b not in self.memories:
            return
        
        a = self.memories[memory_a]
        b = self.memories[memory_b]
        
        current_ab = a.associated_memories.get(memory_b, 0.0)
        current_ba = b.associated_memories.get(memory_a, 0.0)
        
        new_ab = min(1.0, current_ab + strength * (1 - current_ab))
        new_ba = min(1.0, current_ba + strength * (1 - current_ba))
        
        a.associated_memories[memory_b] = new_ab
        b.associated_memories[memory_a] = new_ba
    
    # ============ 层级判断 ============
    
    def get_memory_layer(self, memory_id: str) -> str:
        """
        获取记忆当前应该在的层
        根据有效权重判断
        """
        if memory_id not in self.memories:
            return "none"
        
        memory = self.memories[memory_id]
        weight = self._calculate_effective_weight(memory)
        
        if weight.effective_weight >= self.CORE_THRESHOLD:
            return "core"
        elif weight.effective_weight >= self.PSEUDO_THRESHOLD:
            return "pseudo"
        else:
            return "delete"  # 应该删除
    
    def get_top_memories(self, limit: int = 10, layer: str = "all") -> List[tuple]:
        """
        获取当前权重最高的记忆
        
        Args:
            limit: 返回数量
            layer: 筛选层 ("core" / "pseudo" / "all")
        """
        weighted = []
        for memory in self.memories.values():
            if layer != "all":
                mem_layer = self.get_memory_layer(memory.id)
                if mem_layer != layer:
                    continue
            
            w = self._calculate_effective_weight(memory)
            weighted.append((memory, w))
        
        weighted.sort(key=lambda x: x[1].effective_weight, reverse=True)
        return weighted[:limit]
    
    # ============ 检索（基于权重+标签） ============
    
    def retrieve_by_tags(
        self,
        time_tags: Optional[Set[str]] = None,
        person_tags: Optional[Set[str]] = None,
        topic_tags: Optional[Set[str]] = None,
        emotion_tags: Optional[Set[str]] = None,
        min_weight: float = 0.0,
        layer: str = "core",
        limit: int = 10,
    ) -> List[tuple]:
        """
        基于多维标签检索记忆
        """
        candidates = []
        
        for memory in self.memories.values():
            # 层级过滤
            if layer != "all":
                mem_layer = self.get_memory_layer(memory.id)
                if mem_layer != layer:
                    continue
            
            # 标签匹配
            matched = True
            if time_tags and not (time_tags & memory.time_tags):
                matched = False
            if person_tags and not (person_tags & memory.person_tags):
                matched = False
            if topic_tags and not (topic_tags & memory.topic_tags):
                matched = False
            if emotion_tags and not (emotion_tags & memory.emotion_tags):
                matched = False
            
            if not matched:
                continue
            
            weight = self._calculate_effective_weight(memory)
            if weight.effective_weight >= min_weight:
                candidates.append((memory, weight))
        
        candidates.sort(key=lambda x: x[1].effective_weight, reverse=True)
        return candidates[:limit]
    
    # ============ 统计信息 ============
    
    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        total = len(self.memories)
        core_count = sum(1 for m in self.memories.values() if self.get_memory_layer(m.id) == "core")
        pseudo_count = sum(1 for m in self.memories.values() if self.get_memory_layer(m.id) == "pseudo")
        
        avg_weight = sum(
            self._calculate_effective_weight(m).effective_weight 
            for m in self.memories.values()
        ) / max(1, total)
        
        return {
            "total_memories": total,
            "core_layer": core_count,
            "pseudo_layer": pseudo_count,
            "avg_effective_weight": avg_weight,
            "global_interaction_count": self.global_interaction_count,
        }


# ============ 测试 ============

if __name__ == "__main__":
    print("=" * 60)
    print("自适应权重系统 - 测试")
    print("=" * 60)
    
    system = AdaptiveWeightSystem()
    
    # 创建不同类型的记忆
    print("\n[1] 创建测试记忆...")
    
    story_id = system.add_memory(
        content="小时候在乡下的奶奶家过暑假，那里有稻田和萤火虫",
        memory_type=MemoryType.STORY,
        topic_tags={"童年", "乡村", "奶奶"},
        emotion_tags={"怀念", "温暖"},
        initial_base_attention=0.6,
    )
    print(f"  创建 STORY 记忆: {story_id}")
    
    pref_id = system.add_memory(
        content="用户喜欢在晚上工作，白天效率比较低",
        memory_type=MemoryType.PREFERENCE,
        topic_tags={"工作习惯"},
        emotion_tags={"平静"},
        initial_base_attention=0.5,
    )
    print(f"  创建 PREFERENCE 记忆: {pref_id}")
    
    inter_id = system.add_memory(
        content="用户问我类人记忆系统是什么，我解释了动态权重",
        memory_type=MemoryType.INTERACTION,
        topic_tags={"AI", "记忆系统"},
        emotion_tags={"好奇"},
        initial_base_attention=0.5,
    )
    print(f"  创建 INTERACTION 记忆: {inter_id}")
    
    # 测试强化操作
    print("\n[2] 强化记忆（情绪采样）...")
    for i in range(3):
        result = system.strengthen_memory(
            story_id, 
            text="这件事对用户很重要，他提了很多次",
            emotion_tag="怀念",
        )
        print(f"  第{i+1}次强化 STORY: {result.summary()}")
    
    for i in range(2):
        result = system.strengthen_memory(
            inter_id,
            text="用户对这个话题很好奇，一直在追问",
            emotion_tag="好奇",
        )
        print(f"  第{i+1}次强化 INTERACTION: {result.summary()}")
    
    # 测试权重随互动衰减
    print("\n[3] 权重与半衰期（基于互动量）...")
    
    # STORY 半衰期500，INTERACTION 半衰期150
    for memory_id, mtype in [(story_id, "STORY"), (inter_id, "INTERACTION")]:
        memory = system.memories[memory_id]
        halflife = system._get_halflife(memory)
        print(f"  {mtype} 半衰期: {halflife} 次互动")
        
        # 模拟多次互动后的衰减
        for delta in [0, 50, 150, 500]:
            # 临时调整计数器来模拟
            original = system.global_interaction_count
            system.global_interaction_count = memory.last_strengthened_at_interaction + delta
            w = system._calculate_effective_weight(memory)
            print(f"    delta={delta}: recency_decay={w.recency_decay:.3f}, effective={w.effective_weight:.3f}")
            system.global_interaction_count = original
    
    # 情绪不确定性测试
    print("\n[4] 情绪系数不确定性测试...")
    print("  同一情绪标签'焦虑'采样10次:")
    for i in range(10):
        result = system.emotion_engine.infer_emotion("我很焦虑", user_override="焦虑")
        print(f"    第{i+1}次: {result.coefficient:.3f}")
    
    # 获取统计
    print("\n[5] 系统统计...")
    stats = system.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
