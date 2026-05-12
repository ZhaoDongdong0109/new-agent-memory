"""
CognitiveMemorySystem - 认知记忆系统
基于情绪-意图-遗忘的统一架构

核心颠覆点：
1. 不存储对话，存储"认知状态"
2. 情感作为第一优先级
3. 意图驱动主动学习
4. 遗忘作为feature而非bug
"""

import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import hashlib


class EmotionType(Enum):
    JOY = "joy"
    SADNESS = "sadness"
    FEAR = "fear"
    ANGER = "anger"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    NEUTRAL = "neutral"


@dataclass
class EmotionState:
    primary: EmotionType
    intensity: float
    valence: float  # -1 (negative) to 1 (positive)
    
    def to_memory_weight(self) -> float:
        """情感强度转化为记忆权重"""
        base = 0.5
        intensity_bonus = self.intensity * 0.3
        valence_bonus = self.valence * 0.2
        return min(1.0, max(0.0, base + intensity_bonus + valence_bonus))


@dataclass
class Intention:
    goal: str
    target: Optional[str]
    confidence: float
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class CognitiveNode:
    """认知节点 - 存储理解而非对话"""
    id: str
    concept: str  # 核心概念
    relationships: Dict[str, float]  # 关系图: concept -> strength
    emotional_tags: Set[EmotionType]
    importance: float  # 基于情感计算
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    
    # 元认知
    confidence: float = 1.0  # 这个理解的置信度
    verified: bool = False  # 是否被验证过


@dataclass 
class PersonaProfile:
    """人物认知画像"""
    identity: Dict[str, Any]  # 身份特征
    preferences: Dict[str, float]  # 偏好权重
    communication_style: str  # 沟通风格
    emotional_patterns: Dict[EmotionType, float]  # 情绪模式
    
    def get_relevance(self, concept: str) -> float:
        """判断某个概念与该人物的相关度"""
        base = 0.5
        preference_boost = self.preferences.get(concept, 0.0) * 0.3
        return min(1.0, base + preference_boost)


class CognitiveMemorySystem:
    """
    认知记忆系统 - 颠覆 Transformer 的新范式
    
    核心原理：
    1. 不存储对话，存储"认知状态图"
    2. 情感作为第一优先级
    3. 意图驱动主动理解
    4. 遗忘作为优化机制
    """
    
    def __init__(self):
        # 核心：认知节点图
        self.cognitive_map: Dict[str, CognitiveNode] = {}
        
        # 情感状态追踪
        self.emotion_state: Optional[EmotionState] = None
        self.emotion_history: List[EmotionState] = []
        
        # 意图追踪
        self.active_intentions: List[Intention] = []
        self.achieved_goals: List[str] = []
        
        # 人物画像
        self.persona = PersonaProfile(
            identity={},
            preferences={},
            communication_style="friendly",
            emotional_patterns={}
        )
        
        # 遗忘阈值
        self.forgetting_threshold = 0.2
        self.max_nodes = 1000
    
    def _generate_node_id(self, concept: str) -> str:
        """生成节点ID"""
        return f"cog_{hashlib.md5(concept.encode()).hexdigest()[:12]}"
    
    def _extract_concepts(self, text: str) -> List[str]:
        """从文本中提取核心概念（简化版）"""
        # 实际应用中应该用 NLP 提取实体和关键词
        words = text.replace(",", " ").replace("。", " ").replace("！", " ").split()
        # 过滤停用词
        stop_words = {"的", "了", "和", "是", "在", "我", "你", "他", "她", "它", "我们", "你们"}
        concepts = [w for w in words if len(w) > 1 and w not in stop_words]
        return list(set(concepts))
    
    def process_emotion(self, emotion: EmotionState) -> None:
        """处理情感状态"""
        self.emotion_state = emotion
        self.emotion_history.append(emotion)
        
        # 情感强度影响记忆保留
        if emotion.intensity > 0.7:
            self.forgetting_threshold = max(0.1, self.forgetting_threshold - 0.05)
    
    def add_intention(self, goal: str, target: Optional[str] = None) -> None:
        """添加意图"""
        intention = Intention(goal=goal, target=target, confidence=0.8)
        self.active_intentions.append(intention)
    
    def update_cognitive_node(
        self,
        concept: str,
        relationships: Dict[str, float],
        emotion: Optional[EmotionState] = None,
        importance_override: Optional[float] = None
    ) -> CognitiveNode:
        """
        更新或创建认知节点
        
        核心方法：将对话转化为"认知理解"
        """
        node_id = self._generate_node_id(concept)
        
        if node_id in self.cognitive_map:
            node = self.cognitive_map[node_id]
            
            # 更新关系
            for related_concept, strength in relationships.items():
                if related_concept in node.relationships:
                    node.relationships[related_concept] = max(
                        node.relationships[related_concept],
                        strength
                    )
                else:
                    node.relationships[related_concept] = strength
            
            # 增加访问
            node.access_count += 1
            node.last_accessed = time.time()
            
            # 增强置信度
            node.confidence = min(1.0, node.confidence + 0.1)
            
        else:
            # 计算重要性
            if importance_override is not None:
                importance = importance_override
            elif emotion:
                importance = emotion.to_memory_weight()
            else:
                importance = 0.5
            
            # 考虑人物相关性
            persona_relevance = self.persona.get_relevance(concept)
            importance = importance * 0.7 + persona_relevance * 0.3
            
            # 创建新节点
            node = CognitiveNode(
                id=node_id,
                concept=concept,
                relationships=relationships,
                emotional_tags={emotion.primary} if emotion else set(),
                importance=importance,
            )
            
            self.cognitive_map[node_id] = node
            
            # 触发遗忘
            self._trigger_forgetting()
        
        return node
    
    def process_conversation(
        self,
        user_input: str,
        assistant_output: str,
        emotion: Optional[EmotionState] = None
    ) -> List[CognitiveNode]:
        """
        处理对话，转化为认知节点
        
        这是核心：将对话流转化为结构化认知
        """
        # 提取概念
        user_concepts = self._extract_concepts(user_input)
        assistant_concepts = self._extract_concepts(assistant_output)
        
        # 识别意图（简化版）
        if "想" in user_input or "要" in user_input:
            parts = user_input.replace("想", "要").split("要")
            if len(parts) > 1:
                goal = parts[-1].strip()
                self.add_intention(goal)
        
        # 处理情感
        if emotion:
            self.process_emotion(emotion)
        
        # 更新认知图
        updated_nodes = []
        
        for concept in set(user_concepts + assistant_concepts):
            relationships = {}
            
            # 建立概念间关系
            other_concepts = [c for c in user_concepts + assistant_concepts if c != concept]
            for other in other_concepts:
                relationships[other] = 0.6  # 默认关系强度
            
            node = self.update_cognitive_node(
                concept=concept,
                relationships=relationships,
                emotion=emotion
            )
            updated_nodes.append(node)
        
        return updated_nodes
    
    def _trigger_forgetting(self) -> List[str]:
        """
        触发遗忘机制
        
        核心：不是存储所有，而是保留核心
        """
        if len(self.cognitive_map) <= self.max_nodes:
            return []
        
        # 计算每个节点的"活跃度"
        now = time.time()
        for node in self.cognitive_map.values():
            time_since_access = now - node.last_accessed
            # 遗忘曲线
            decay = 0.1 * (time_since_access / (24 * 3600))  # 每天衰减10%
            node.importance -= decay
            node.importance = max(0.0, node.importance)
        
        # 标记要删除的节点
        to_remove = [
            node_id for node_id, node in self.cognitive_map.items()
            if node.importance < self.forgetting_threshold and not node.verified
        ]
        
        # 只删除超出的部分
        excess = len(self.cognitive_map) - self.max_nodes
        for node_id in to_remove[:excess]:
            del self.cognitive_map[node_id]
        
        return to_remove[:excess]
    
    def query(
        self,
        query: str,
        top_k: int = 5
    ) -> List[CognitiveNode]:
        """
        查询相关认知
        
        使用认知图 + 意图追踪
        """
        query_concepts = self._extract_concepts(query)
        
        # 意图推断
        inferred_concepts = set(query_concepts)
        if "谁" in query or "什么" in query:
            for intention in self.active_intentions:
                inferred_concepts.update(self._extract_concepts(intention.goal))
        
        # 找到匹配的节点
        matched_nodes = []
        for node in self.cognitive_map.values():
            if node.concept in inferred_concepts:
                matched_nodes.append(node)
                continue
            
            # 通过关系匹配
            for qc in inferred_concepts:
                if qc in node.relationships or any(qc in r for r in node.relationships):
                    matched_nodes.append(node)
                    break
        
        # 按重要性排序
        matched_nodes.sort(key=lambda n: n.importance, reverse=True)
        
        # 更新访问
        for node in matched_nodes[:top_k]:
            node.access_count += 1
            node.last_accessed = time.time()
        
        return matched_nodes[:top_k]
    
    def get_cognitive_summary(self) -> Dict[str, Any]:
        """获取认知状态摘要"""
        return {
            "total_concepts": len(self.cognitive_map),
            "cognitive_density": len(self.cognitive_map) / max(1, self.max_nodes),
            "active_intentions": len(self.active_intentions),
            "current_emotion": self.emotion_state.primary.value if self.emotion_state else None,
            "persona_profile": {
                "preferences_count": len(self.persona.preferences),
                "communication_style": self.persona.communication_style
            },
            "average_importance": (
                sum(n.importance for n in self.cognitive_map.values()) / max(1, len(self.cognitive_map))
            )
        }
    
    def verify_understanding(self, concept: str, correct: bool) -> None:
        """验证理解是否正确"""
        node_id = self._generate_node_id(concept)
        if node_id in self.cognitive_map:
            node = self.cognitive_map[node_id]
            node.verified = True
            if correct:
                node.confidence = min(1.0, node.confidence + 0.2)
            else:
                node.confidence = max(0.0, node.confidence - 0.3)
