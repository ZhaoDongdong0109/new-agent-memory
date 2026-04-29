"""
记忆碎片模块 - 类人记忆系统核心数据结构

核心设计：
- 每个记忆是一个碎片（chunk），携带多维标签
- 标签是检索的入口，类似C语言头文件
- 内容与标签分离，支持高效的检索
"""

from dataclasses import dataclass, field
from typing import Set, Dict, Optional, Any
from enum import Enum
import time
import uuid


class MemoryLayer(Enum):
    """记忆所在层级"""
    CORE = "core"           # 核心层，高权重
    FORGOTTEN = "forgotten"  # 伪遗忘层，极低权重


@dataclass
class MemoryChunk:
    """
    记忆碎片
    
    设计原则：
    - content 是实际记忆内容
    - tags 是元信息（头文件），用于检索
    - layer 标记当前所在层级
    """
    
    # 唯一标识
    id: str = field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:12]}")
    
    # 记忆内容（可以是原始文本、事件描述、或指向更大碎片的引用）
    content: str = ""
    
    # 记忆类型（影响权重策略）
    memory_type: str = "general"  # general / emotion / fact / procedure
    
    # 多维标签（头文件）
    tags: Dict[str, Any] = field(default_factory=dict)
    
    # 时间维度
    time_absolute: Optional[str] = None   # 绝对时间："2026-04-29"
    time_relative: Optional[str] = None   # 相对时间："10年前", "上周", "中午"
    time_context: Optional[str] = None    # 时间上下文："工作日", "假期", "出差"
    
    # 空间维度
    location: Optional[str] = None        # 地点标签
    location_detail: Optional[str] = None # 地点细节
    
    # 人物维度
    persons: Set[str] = field(default_factory=set)  # 涉及的人物
    person_count: int = 1                      # 涉及人数
    
    # 主题/语义维度
    topics: Set[str] = field(default_factory=set)  # 主题标签
    keywords: Set[str] = field(default_factory=set)  # 关键词
    
    # 情绪维度
    emotion_valence: float = 0.0   # 情绪效价 -1.0(负面) ~ +1.0(正面)
    emotion_intensity: float = 0.0 # 情绪强度 0.0 ~ 1.0
    
    # 重要性（主观）
    importance: float = 0.5        # 0.0 ~ 1.0
    
    # 连接价值（能触发多少其他记忆）
    connection_value: float = 0.5  # 0.0 ~ 1.0
    
    # 层级
    layer: MemoryLayer = MemoryLayer.CORE
    
    # 时间戳
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    
    # 访问统计
    access_count: int = 0
    successful_recall_count: int = 0  # 成功被唤醒次数
    
    # 关联记忆（Hebbian关联）
    associations: Dict[str, float] = field(default_factory=dict)  # chunk_id -> weight
    
    # 审阅标记
    review_status: str = "pending"  # pending / approved / questionable / rejected
    review_note: Optional[str] = None
    
    # 元数据（扩展字段，存储额外信息）
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def access(self):
        """记录一次访问"""
        self.last_accessed = time.time()
        self.updated_at = time.time()
        self.access_count += 1
    
    def successful_recall(self):
        """记录一次成功唤醒"""
        self.successful_recall_count += 1
    
    def get_tag_signature(self) -> str:
        """
        获取标签签名，用于检索匹配
        类似C语言头文件的函数声明
        """
        parts = []
        
        # 时间标签
        if self.time_absolute:
            parts.append(f"T:{self.time_absolute}")
        if self.time_relative:
            parts.append(f"TR:{self.time_relative}")
        if self.time_context:
            parts.append(f"TC:{self.time_context}")
        
        # 地点标签
        if self.location:
            parts.append(f"L:{self.location}")
        
        # 人物标签
        for p in sorted(self.persons):
            parts.append(f"P:{p}")
        
        # 主题标签
        for t in sorted(self.topics):
            parts.append(f"TOP:{t}")
        
        return " | ".join(parts)
    
    def matches_query(self, query_tags: Dict[str, Any]) -> bool:
        """
        检查当前碎片是否匹配查询标签
        类似 #include 时的编译检查
        """
        # 时间匹配
        if "time_absolute" in query_tags:
            if self.time_absolute != query_tags["time_absolute"]:
                return False
        
        if "time_relative" in query_tags:
            if self.time_relative != query_tags["time_relative"]:
                return False
        
        if "time_context" in query_tags:
            if self.time_context != query_tags["time_context"]:
                return False
        
        # 地点匹配
        if "location" in query_tags:
            if self.location != query_tags["location"]:
                return False
        
        # 人物匹配
        if "persons" in query_tags:
            if not (query_tags["persons"] & self.persons):
                return False
        
        # 主题匹配
        if "topics" in query_tags:
            if not (query_tags["topics"] & self.topics):
                return False
        
        # 情绪匹配
        if "emotion_valence_min" in query_tags:
            if self.emotion_valence < query_tags["emotion_valence_min"]:
                return False
        if "emotion_valence_max" in query_tags:
            if self.emotion_valence > query_tags["emotion_valence_max"]:
                return False
        
        # 重要性匹配
        if "importance_min" in query_tags:
            if self.importance < query_tags["importance_min"]:
                return False
        
        return True
    
    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "tags": self.tags,
            "time_absolute": self.time_absolute,
            "time_relative": self.time_relative,
            "time_context": self.time_context,
            "location": self.location,
            "location_detail": self.location_detail,
            "persons": list(self.persons),
            "person_count": self.person_count,
            "topics": list(self.topics),
            "keywords": list(self.keywords),
            "emotion_valence": self.emotion_valence,
            "emotion_intensity": self.emotion_intensity,
            "importance": self.importance,
            "connection_value": self.connection_value,
            "layer": self.layer.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "successful_recall_count": self.successful_recall_count,
            "associations": self.associations,
            "review_status": self.review_status,
            "review_note": self.review_note,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "MemoryChunk":
        """从字典反序列化"""
        data = data.copy()
        data["persons"] = set(data.get("persons", []))
        data["topics"] = set(data.get("topics", []))
        data["keywords"] = set(data.get("keywords", []))
        data["layer"] = MemoryLayer(data.get("layer", "core"))
        return cls(**data)
    
    def __repr__(self) -> str:
        content_preview = self.content[:30] + "..." if len(self.content) > 30 else self.content
        return f"<MemoryChunk {self.id} [{self.layer.value}] '{content_preview}'>"
