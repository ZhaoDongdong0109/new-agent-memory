"""
HumanMemorySystem - 极致模仿人类记忆系统
基于认知心理学最前沿研究成果

包含完整的人类记忆特性：
1. 感官记忆 (Iconic/Echoic)
2. 工作记忆 (Working Memory - 7±2 组块)
3. 长时记忆 (情景/语义/程序/情绪)
4. 记忆编码 (精细/联想/图像/组织)
5. 记忆提取 (再认/回想/线索依赖)
6. 遗忘机制 (衰退/干扰/提取失败)
7. 认知偏差 (首因/近因/雷斯多夫)
8. 睡眠巩固
"""

import time
import math
import random
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import hashlib


class MemoryType(Enum):
    SENSORY_ICONIC = "iconic"      # 图像记忆
    SENSORY_ECHOIC = "echoic"      # 声音记忆
    WORKING = "working"            # 工作记忆
    EPISODIC = "episodic"          # 情景记忆
    SEMANTIC = "semantic"          # 语义记忆
    PROCEDURAL = "procedural"      # 程序记忆
    EMOTIONAL = "emotional"       # 情绪记忆


@dataclass
class SensoryBuffer:
    """感官记忆缓冲器"""
    content: str
    modality: str  # visual, auditory
    timestamp: float
    intensity: float  # 感官强度
    decay_rate: float  # 衰减率
    
    def get_freshness(self) -> float:
        """获取新鲜度 (0-1)"""
        age = time.time() - self.timestamp
        return max(0.0, 1.0 - age * self.decay_rate)


@dataclass
class WorkingMemorySlot:
    """工作记忆槽位"""
    content: Any
    chunk_complexity: float  # 组块复杂度
    rehearsal_count: int = 0
    last_rehearsal: float = field(default_factory=time.time)
    
    def strengthen(self):
        """强化（复述）"""
        self.rehearsal_count += 1
        self.last_rehearsal = time.time()
        self.chunk_complexity = min(1.0, self.chunk_complexity * 1.1)


@dataclass
class EpisodicMemory:
    """情景记忆 - 个人经历"""
    id: str
    content: str
    context: Dict[str, Any]  # 时间、地点、情境
    emotions: List[Tuple[str, float]]  # 情绪标签和强度
    importance: float
    encoding_strength: float = 0.5
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    
    # 时间标记
    when: Optional[str] = None  # "今天上午"
    where: Optional[str] = None  # "在公司"
    who: List[str] = field(default_factory=list)  # 相关人物
    
    def get_temporal_weight(self, current_time: float) -> float:
        """时间距离权重（Ribot 定律：近期记忆更易提取）"""
        days_ago = (current_time - self.last_accessed) / 86400
        return math.exp(-0.1 * days_ago)


@dataclass
class SemanticMemory:
    """语义记忆 - 概念知识"""
    concept: str
    definition: str
    related_concepts: Dict[str, float]  # 关联概念及强度
    examples: List[str]
    confidence: float = 1.0
    abstraction_level: int = 1  # 抽象层次
    
    def get_activation(self, query_context: Dict) -> float:
        """基于上下文激活"""
        base = self.confidence
        context_match = sum(
            self.related_concepts.get(k, 0) 
            for k in query_context.keys()
        ) / max(1, len(query_context))
        return base * (0.7 + 0.3 * context_match)


@dataclass
class ProceduralMemory:
    """程序记忆 - 技能和习惯"""
    skill_name: str
    steps: List[str]
    automaticity: float  # 自动化程度
    last_practiced: float = field(default_factory=time.time)
    practice_count: int = 0
    
    def needs_review(self) -> bool:
        """是否需要复习"""
        days_since = (time.time() - self.last_practiced) / 86400
        return days_since > (30 / (self.automaticity + 0.1))


@dataclass
class EmotionalMemory:
    """情绪记忆 - 情感印记"""
    trigger: str
    emotion: str
    intensity: float
    associated_memories: List[str]  # 关联记忆ID
    valence: float  # -1 负面, +1 正面
    created_at: float = field(default_factory=time.time)
    
    def get_persistence_factor(self) -> float:
        """情绪记忆持久性因子"""
        intensity_boost = self.intensity * 0.5
        valence_effect = abs(self.valence) * 0.3
        return min(1.0, 0.5 + intensity_boost + valence_effect)


@dataclass
class MemoryTrace:
    """记忆痕迹 - 记忆的神经表征"""
    memory_id: str
    activation_level: float  # 激活水平
    accessibility: float  # 可及性
    consolidation_state: float  # 巩固程度 (0-1)
    emotional_tag: Optional[str] = None
    
    def strengthen(self, amount: float):
        self.activation_level = min(1.0, self.activation_level + amount)
        self.consolidation_state = min(1.0, self.consolidation_state + amount * 0.1)
    
    def weaken(self, amount: float):
        self.activation_level = max(0.0, self.activation_level - amount)


class HumanMemorySystem:
    """
    极致模仿人类记忆系统
    
    理论基础：
    - Atkinson-Shiffrin 模型 (1968)
    - Baddeley 工作记忆模型 (1974, 2000)
    - Ebbinghaus 遗忘曲线 (1885)
    - 情绪记忆研究 (LeDoux, 1996)
    - 睡眠巩固理论 (Walker & Stickgold, 2004)
    
    记忆流程：
    感知输入 → 感官缓冲 → 工作记忆 → 编码 → 长时记忆
                                            ↓
              遗忘 ← 衰退 ← 提取 ← 检索 ←────────
    """
    
    def __init__(
        self,
        working_memory_slots: int = 7,
        sensory_decay_rate: float = 0.1,
        long_term_capacity: int = 10000
    ):
        # === 感官记忆 ===
        self.iconic_memory: Dict[str, SensoryBuffer] = {}  # 图像
        self.echoic_memory: Dict[str, SensoryBuffer] = {}  # 声音
        
        # === 工作记忆 ===
        self.working_memory_slots = working_memory_slots
        self.working_memory: deque = deque(maxlen=working_memory_slots)
        self.phonological_loop: List[str] = []  # 语音环
        self.visuospatial_sketchpad: List[Any] = []  # 视觉空间模板
        self.central_executive_load: float = 0.5  # 中央执行器负荷
        
        # === 长时记忆 ===
        self.episodic_memory: Dict[str, EpisodicMemory] = {}  # 情景
        self.semantic_memory: Dict[str, SemanticMemory] = {}   # 语义
        self.procedural_memory: Dict[str, ProceduralMemory] = {}  # 程序
        self.emotional_memory: Dict[str, EmotionalMemory] = {}  # 情绪
        
        # === 记忆痕迹网络 ===
        self.memory_traces: Dict[str, MemoryTrace] = {}
        self.association_graph: Dict[str, Set[str]] = defaultdict(set)  # 联想图
        
        # === 遗忘机制 ===
        self.forgetting_rate = 0.1
        self.interference_buffer: List[str] = []  # 干扰项
        
        # === 认知偏差状态 ===
        self.recency_buffer: deque = deque(maxlen=10)  # 近因效应缓冲
        self.primacy_buffer: List[str] = []  # 首因效应缓冲
        self.serial_position_effect: bool = True
        
        # === 睡眠巩固 ===
        self.sleep_cycles: List[Dict] = []
        self.pending_consolidation: List[str] = []  # 待巩固记忆
        self.last_sleep_time: Optional[float] = None
        
        # === 统计 ===
        self.stats = {
            "total_encodings": 0,
            "total_retrievals": 0,
            "successful_retrievals": 0,
            "forgotten_items": 0,
            "false_memories": 0
        }
    
    # ==================== 感官记忆 ====================
    
    def encode_sensory(
        self,
        content: str,
        modality: str = "visual",
        intensity: float = 1.0
    ) -> str:
        """编码到感官记忆"""
        buffer_id = f"sensory_{hashlib.md5(content.encode()).hexdigest()[:8]}"
        
        buffer = SensoryBuffer(
            content=content,
            modality=modality,
            timestamp=time.time(),
            intensity=intensity,
            decay_rate=0.1 if modality == "visual" else 0.05  # 图像衰减更快
        )
        
        if modality == "visual":
            self.iconic_memory[buffer_id] = buffer
        else:
            self.echoic_memory[buffer_id] = buffer
        
        return buffer_id
    
    def decay_sensory_memory(self) -> int:
        """感官记忆自然衰减"""
        decayed = 0
        
        for memory in [self.iconic_memory, self.echoic_memory]:
            expired = [
                bid for bid, buf in memory.items()
                if buf.get_freshness() < 0.1
            ]
            for bid in expired:
                del memory[bid]
                decayed += 1
        
        return decayed
    
    # ==================== 工作记忆 ====================
    
    def add_to_working(self, item: Any, chunkify: bool = True) -> bool:
        """添加到工作记忆（容量限制：7±2 组块）"""
        if len(self.working_memory) >= self.working_memory_slots:
            # 工作记忆满了，需要替换或提升
            if not self._manage_working_overflow(item):
                return False
        
        complexity = self._calculate_chunk_complexity(item) if chunkify else 0.5
        
        slot = WorkingMemorySlot(
            content=item,
            chunk_complexity=complexity
        )
        self.working_memory.append(slot)
        
        # 更新系列位置效应
        self.recency_buffer.append(str(item)[:20])
        if len(self.recency_buffer) > self.working_memory_slots:
            removed = self.recency_buffer.popleft()
            if len(self.primacy_buffer) < self.working_memory_slots:
                self.primacy_buffer.append(removed)
        
        return True
    
    def _calculate_chunk_complexity(self, item: Any) -> float:
        """计算组块复杂度"""
        if isinstance(item, str):
            # 简单组块：7±2 法则
            chars = len(item)
            return min(1.0, chars / 7)
        return 0.5
    
    def _manage_working_overflow(self, new_item: Any) -> bool:
        """管理工作记忆溢出"""
        # 移除最不重要的项
        weakest_idx = 0
        weakest_strength = float('inf')
        
        for i, slot in enumerate(self.working_memory):
            # 考虑复述次数和新鲜度
            freshness = 1.0 - (time.time() - slot.last_rehearsal) / 60
            importance = slot.rehearsal_count * 0.3 + freshness * 0.7
            
            if importance < weakest_strength:
                weakest_strength = importance
                weakest_idx = i
        
        # 如果新项比最弱的更重要，替换
        new_complexity = self._calculate_chunk_complexity(new_item)
        if new_complexity > weakest_strength:
            self.working_memory[weakest_idx] = WorkingMemorySlot(
                content=new_item,
                chunk_complexity=new_complexity
            )
            return True
        
        return False
    
    def rehearsal(self, item: Any) -> bool:
        """复述强化（维持工作记忆）"""
        for slot in self.working_memory:
            if str(slot.content) == str(item):
                slot.strengthen()
                return True
        return False
    
    def clear_working_memory(self):
        """清除工作记忆"""
        self.working_memory.clear()
        self.phonological_loop.clear()
        self.visuospatial_sketchpad.clear()
    
    # ==================== 长时记忆编码 ====================
    
    def encode_to_long_term(
        self,
        content: str,
        memory_type: MemoryType,
        context: Optional[Dict] = None,
        emotions: Optional[List[Tuple[str, float]]] = None,
        importance: float = 0.5
    ) -> str:
        """
        编码到长时记忆
        
        编码策略（精细编码）：
        1. 深层次加工 > 浅层次加工
        2. 联想编码（与已有知识关联）
        3. 图像编码
        4. 组织编码（分类结构）
        """
        memory_id = f"{memory_type.value}_{hashlib.md5(content.encode()).hexdigest()[:12]}"
        
        # 创建记忆痕迹
        emotional_tag = emotions[0][0] if emotions else None
        trace = MemoryTrace(
            memory_id=memory_id,
            activation_level=importance,
            accessibility=0.5,
            consolidation_state=0.0,  # 初始未巩固
            emotional_tag=emotional_tag
        )
        self.memory_traces[memory_id] = trace
        
        # 分类存储
        if memory_type == MemoryType.EPISODIC:
            self._encode_episodic(memory_id, content, context, emotions, importance)
        
        elif memory_type == MemoryType.SEMANTIC:
            self._encode_semantic(memory_id, content, context, importance)
        
        elif memory_type == MemoryType.PROCEDURAL:
            self._encode_procedural(memory_id, content, context, importance)
        
        elif memory_type == MemoryType.EMOTIONAL:
            self._encode_emotional(memory_id, content, emotions, importance)
        
        # 建立联想连接
        self._create_associations(memory_id, content)
        
        # 加入待巩固队列
        self.pending_consolidation.append(memory_id)
        
        self.stats["total_encodings"] += 1
        
        return memory_id
    
    def _encode_episodic(
        self,
        memory_id: str,
        content: str,
        context: Optional[Dict],
        emotions: Optional[List[Tuple[str, float]]],
        importance: float
    ):
        """编码情景记忆"""
        episodic = EpisodicMemory(
            id=memory_id,
            content=content,
            context=context or {},
            emotions=emotions or [],
            importance=importance,
            when=context.get("time") if context else None,
            where=context.get("location") if context else None,
            who=context.get("people", []) if context else []
        )
        
        # 雷斯多夫效应：情绪强烈的记忆更深刻
        if emotions:
            max_intensity = max(e[1] for e in emotions)
            episodic.encoding_strength *= (1 + max_intensity * 0.5)
        
        self.episodic_memory[memory_id] = episodic
    
    def _encode_semantic(
        self,
        memory_id: str,
        content: str,
        context: Optional[Dict],
        importance: float
    ):
        """编码语义记忆"""
        semantic = SemanticMemory(
            concept=content[:100],  # 概念名称
            definition=content,
            related_concepts={},
            examples=context.get("examples", []) if context else []
        )
        self.semantic_memory[memory_id] = semantic
    
    def _encode_procedural(
        self,
        memory_id: str,
        content: str,
        context: Optional[Dict],
        importance: float
    ):
        """编码程序记忆"""
        procedural = ProceduralMemory(
            skill_name=content[:50],
            steps=context.get("steps", []) if context else [],
            automaticity=0.3  # 初始自动化程度低
        )
        self.procedural_memory[memory_id] = procedural
    
    def _encode_emotional(
        self,
        memory_id: str,
        content: str,
        emotions: Optional[List[Tuple[str, float]]],
        importance: float
    ):
        """编码情绪记忆"""
        if not emotions:
            return
        
        primary_emotion, intensity = emotions[0]
        
        emotional = EmotionalMemory(
            trigger=content,
            emotion=primary_emotion,
            intensity=intensity,
            associated_memories=[],
            valence=1.0 if primary_emotion in ["joy", "trust", "anticipation"]
                    else -1.0 if primary_emotion in ["anger", "fear", "sadness"]
                    else 0.0
        )
        self.emotional_memory[memory_id] = emotional
    
    def _create_associations(self, memory_id: str, content: str):
        """创建记忆联想"""
        # 与相关记忆建立连接
        for existing_id, episodic in self.episodic_memory.items():
            if existing_id == memory_id:
                continue
            
            # 检查语义相似度（简化版）
            if self._semantic_similarity(content, episodic.content) > 0.3:
                self.association_graph[memory_id].add(existing_id)
                self.association_graph[existing_id].add(memory_id)
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """优化版语义相似度计算"""
        if not text1 or not text2:
            return 0.0
        
        # 预处理文本
        text1 = text1.lower()
        text2 = text2.lower()
        
        # 子字符串匹配（更宽松）
        if text1 in text2 or text2 in text1:
            return 0.9
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        # 基础Jaccard
        jaccard = intersection / union if union > 0 else 0.0
        
        # 加上部分匹配的奖励
        partial_match = 0
        for word1 in words1:
            for word2 in words2:
                if len(word1) > 2 and len(word2) > 2:
                    if word1 in word2 or word2 in word1:
                        partial_match += 1
        
        return min(1.0, jaccard + (partial_match * 0.1))
    
    def quick_retrieve(self, query: str, top_k: int = 5) -> List[Any]:
        """快速检索 - 更简单直接的API"""
        return self.retrieve(query, retrieval_type="recognition")[:top_k]
    
    def get_recent_memories(self, n: int = 10) -> List[Any]:
        """获取最近的n条记忆"""
        all_memories = []
        for memory in self.episodic_memory.values():
            all_memories.append(memory)
        
        all_memories.sort(key=lambda x: x.last_accessed, reverse=True)
        return all_memories[:n]
    
    # ==================== 记忆提取 ====================
    
    def retrieve(
        self,
        query: str,
        retrieval_type: str = "recall",
        context: Optional[Dict] = None
    ) -> List[Any]:
        """
        记忆提取 - 优化版
        
        提取类型：
        - recall: 自由回忆
        - recognition: 再认
        - cued: 线索回忆
        """
        self.stats["total_retrievals"] += 1
        
        candidates = self._search_candidates(query)
        
        if not candidates:
            # 更宽松的搜索
            candidates = self._search_candidates(query, strict=False)
        
        # 应用提取机制
        if retrieval_type == "recognition":
            results = self._recognition(candidates, query)
        elif retrieval_type == "cued":
            results = self._cued_recall(candidates, query, context)
        else:  # free recall
            results = self._free_recall(candidates, query)
        
        # 更新统计
        if results:
            self.stats["successful_retrievals"] += 1
        
        # 应用系列位置效应
        if self.serial_position_effect and results:
            results = self._apply_serial_position_effect(results)
        
        return results
    
    def _search_candidates(self, query: str, strict: bool = True) -> List[Tuple[str, float, Any]]:
        """搜索候选记忆 - 优化版"""
        candidates = []
        
        # 搜索情景记忆
        for eid, episodic in self.episodic_memory.items():
            similarity = self._semantic_similarity(query, episodic.content)
            # 调整阈值
            threshold = 0.2 if strict else 0.1
            if similarity > threshold:
                candidates.append((eid, similarity, episodic))
        
        # 搜索语义记忆
        for sid, semantic in self.semantic_memory.items():
            similarity = self._semantic_similarity(query, semantic.concept)
            similarity = max(similarity, self._semantic_similarity(query, semantic.definition))
            threshold = 0.3 if strict else 0.15
            if similarity > threshold:
                candidates.append((sid, similarity, semantic))
        
        # 按相似度排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前20个候选
        return candidates[:20]
    
    def _recognition(self, candidates: List, query: str) -> List[Any]:
        """再认 - 优化版"""
        results = []
        threshold = 0.15
        
        for cid, similarity, memory in candidates:
            if similarity > threshold:
                trace = self.memory_traces.get(cid)
                if trace:
                    trace.strengthen(0.15)
                    # 安全更新属性
                    if hasattr(memory, 'last_accessed'):
                        memory.last_accessed = time.time()
                    if hasattr(memory, 'access_count'):
                        memory.access_count += 1
                    results.append(memory)
        
        return results
    
    def _free_recall(self, candidates: List, query: str) -> List[Any]:
        """自由回忆 - 优化版"""
        results = []
        
        for cid, similarity, memory in candidates:
            trace = self.memory_traces.get(cid)
            
            if not trace:
                continue
            
            # 计算提取概率 - 简化逻辑
            extraction_prob = trace.activation_level * trace.accessibility * (0.5 + similarity)
            
            # 提高成功率
            if extraction_prob > 0.2:
                results.append(memory)
                trace.strengthen(0.1)
                
                if hasattr(memory, 'last_accessed'):
                    memory.last_accessed = time.time()
                if hasattr(memory, 'access_count'):
                    memory.access_count += 1
        
        # 按时间排序（近因效应）
        results.sort(key=lambda x: getattr(x, 'last_accessed', 0), reverse=True)
        
        return results
    
    def _cued_recall(self, candidates: List, query: str, context: Optional[Dict]) -> List[Any]:
        """线索回忆 - 优化版"""
        if not context:
            return self._free_recall(candidates, query)
        
        results = []
        
        for cid, similarity, memory in candidates:
            trace = self.memory_traces.get(cid)
            
            if not trace:
                continue
            
            # 情境匹配加成
            context_bonus = 0.0
            if hasattr(memory, 'context') and memory.context:
                match_count = sum(
                    1 for k, v in context.items()
                    if k in memory.context and memory.context[k] == v
                )
                context_bonus = match_count * 0.15
            
            extraction_prob = trace.activation_level + context_bonus
            
            if extraction_prob > 0.3:
                results.append(memory)
                trace.strengthen(context_bonus)
                
                # 安全更新属性
                if hasattr(memory, 'last_accessed'):
                    memory.last_accessed = time.time()
                if hasattr(memory, 'access_count'):
                    memory.access_count += 1
        
        return results
    
    def _apply_serial_position_effect(self, results: List) -> List:
        """应用系列位置效应"""
        if len(results) < 3:
            return results
        
        # 首因效应：开头的记得更清楚
        primacy_boost = results[:len(results)//3]
        recency_boost = results[-len(results)//3:]
        middle = results[len(results)//3:-len(results)//3]
        
        return primacy_boost + middle + recency_boost
    
    # ==================== 遗忘机制 ====================
    
    def forget(self, memory_id: Optional[str] = None, reason: str = "decay") -> int:
        """
        遗忘机制
        
        遗忘原因：
        - decay: 时间衰退
        - interference: 干扰（前摄/后摄）
        - retrieval_failure: 提取失败
        - motivated: 动机性遗忘
        """
        forgotten_count = 0
        
        if memory_id:
            # 遗忘特定记忆
            self._forget_single(memory_id, reason)
            forgotten_count = 1
        else:
            # 批量遗忘
            for mid in list(self.episodic_memory.keys()):
                if random.random() < self.forgetting_rate:
                    self._forget_single(mid, reason)
                    forgotten_count += 1
        
        self.stats["forgotten_items"] += forgotten_count
        return forgotten_count
    
    def _forget_single(self, memory_id: str, reason: str):
        """遗忘单个记忆"""
        trace = self.memory_traces.get(memory_id)
        
        if not trace:
            return
        
        if reason == "decay":
            # 指数衰减
            decay = math.exp(-0.01) * 0.1
            trace.weaken(decay)
        
        elif reason == "interference":
            # 干扰导致弱化
            trace.weaken(0.15)
        
        elif reason == "retrieval_failure":
            # 提取失败降低可及性
            trace.accessibility *= 0.8
        
        # 如果激活水平过低，标记为遗忘
        if trace.activation_level < 0.1:
            # 延迟删除（可能有恢复机会）
            self.interference_buffer.append(memory_id)
    
    def apply_interference(self, new_memories: List[str]):
        """
        应用干扰效应
        
        前摄干扰：旧记忆干扰新记忆
        后摄干扰：新记忆干扰旧记忆
        """
        # 后摄干扰：新记忆影响旧记忆
        for old_id in list(self.episodic_memory.keys()):
            trace = self.memory_traces.get(old_id)
            if trace:
                # 新记忆越多，旧记忆越容易被干扰
                interference_strength = len(new_memories) * 0.02
                trace.weaken(interference_strength)
        
        # 清空干扰缓冲区
        self.interference_buffer.clear()
    
    # ==================== 睡眠巩固 ====================
    
    def simulate_sleep(self, duration_hours: float = 8.0):
        """
        模拟睡眠巩固
        
        睡眠阶段：
        - NREM Stage 1-3: 记忆整合
        - REM: 情绪记忆加工、创造性联想
        """
        cycles = int(duration_hours / 1.5)  # 每 1.5 小时一个周期
        
        for cycle in range(cycles):
            is_rem = (cycle % 2 == 1)  # 奇数周期为 REM
            
            # 选择待巩固的记忆
            to_consolidate = self.pending_consolidation[:min(5, len(self.pending_consolidation))]
            
            for memory_id in to_consolidate:
                self._consolidate_memory(memory_id, is_rem)
            
            self.sleep_cycles.append({
                "cycle": cycle + 1,
                "is_rem": is_rem,
                "memories_consolidated": len(to_consolidate),
                "timestamp": time.time()
            })
        
        self.pending_consolidation.clear()
        self.last_sleep_time = time.time()
        
        return len(self.sleep_cycles)
    
    def _consolidate_memory(self, memory_id: str, is_rem: bool):
        """巩固单个记忆"""
        trace = self.memory_traces.get(memory_id)
        
        if not trace:
            return
        
        if is_rem:
            # REM 睡眠：增强情绪记忆，创造联想
            trace.strengthen(0.15)
            
            # 创造新联想
            episodic = self.episodic_memory.get(memory_id)
            if episodic:
                self._create_novel_associations(memory_id, episodic)
        else:
            # NREM 睡眠：一般巩固
            trace.strengthen(0.08)
        
        # 提升巩固程度
        trace.consolidation_state = min(1.0, trace.consolidation_state + 0.2)
    
    def _create_novel_associations(self, memory_id: str, episodic: EpisodicMemory):
        """REM 睡眠中创造新联想"""
        # 随机与旧记忆建立连接（类比创造性）
        if len(self.episodic_memory) > 5:
            candidates = random.sample(
                list(self.episodic_memory.keys()),
                min(3, len(self.episodic_memory) - 1)
            )
            
            for cid in candidates:
                if cid != memory_id:
                    self.association_graph[memory_id].add(cid)
                    self.association_graph[cid].add(memory_id)
    
    # ==================== 统计和诊断 ====================
    
    def get_memory_report(self) -> Dict[str, Any]:
        """获取记忆系统状态报告"""
        return {
            "sensory": {
                "iconic": len(self.iconic_memory),
                "echoic": len(self.echoic_memory)
            },
            "working_memory": {
                "slots_used": len(self.working_memory),
                "slots_available": self.working_memory_slots,
                "utilization": len(self.working_memory) / self.working_memory_slots
            },
            "long_term": {
                "episodic": len(self.episodic_memory),
                "semantic": len(self.semantic_memory),
                "procedural": len(self.procedural_memory),
                "emotional": len(self.emotional_memory),
                "pending_consolidation": len(self.pending_consolidation)
            },
            "forgetting": {
                "forgotten_items": self.stats["forgotten_items"],
                "interference_buffer": len(self.interference_buffer)
            },
            "retrieval": {
                "total_retrievals": self.stats["total_retrievals"],
                "successful_retrievals": self.stats["successful_retrievals"],
                "success_rate": (
                    self.stats["successful_retrievals"] / max(1, self.stats["total_retrievals"])
                )
            },
            "consolidation": {
                "sleep_cycles": len(self.sleep_cycles),
                "last_sleep": (
                    time.time() - self.last_sleep_time
                    if self.last_sleep_time else None
                )
            }
        }
    
    def get_forgetting_curve(self, memory_id: str) -> List[Tuple[float, float]]:
        """
        获取遗忘曲线
        
        Returns: [(时间点, 保留率), ...]
        """
        if memory_id not in self.memory_traces:
            return []
        
        trace = self.memory_traces[memory_id]
        curve = []
        
        # 模拟不同时间点的记忆保留
        for hours in [0, 1, 6, 12, 24, 48, 72, 168]:  # 0小时到7天
            retention = trace.activation_level * math.exp(-0.1 * hours)
            curve.append((hours, retention))
        
        return curve
    
    def get_ribot_curve(self) -> Dict[str, float]:
        """
        获取 Ribot 曲线（记忆的时间层级）
        
        Returns: {时期: 平均强度}
        """
        now = time.time()
        
        def time_category(ts: float) -> str:
            days = (now - ts) / 86400
            if days < 1:
                return "recent"  # 最近
            elif days < 7:
                return "short_term"  # 短时
            elif days < 30:
                return "medium_term"  # 中期
            else:
                return "long_term"  # 长期
        
        categories = defaultdict(list)
        
        for eid, episodic in self.episodic_memory.items():
            cat = time_category(episodic.last_accessed)
            categories[cat].append(episodic.encoding_strength)
        
        return {
            cat: sum(strengths) / len(strengths) if strengths else 0.0
            for cat, strengths in categories.items()
        }
