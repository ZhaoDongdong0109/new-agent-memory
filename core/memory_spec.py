"""
MemorySpec - 抽取器输出的结构化记忆规格

用于 MemoryExtractor 输出，ConsolidationEngine 输入。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from core.weight_system import MemoryType


@dataclass
class MemorySpec:
    """
    抽取器输出的结构化记忆规格

    用于从 episode 提取的结构化记忆描述，
    由 MemoryExtractor 输出，ConsolidationEngine 消费。
    """

    # 核心内容
    content: str = ""
    summary: str = ""

    # 记忆类型
    memory_type: MemoryType = MemoryType.INTERACTION

    # 时间维度
    time_absolute: Optional[str] = None   # "2026-06-18"
    time_relative: Optional[str] = None   # "10年前", "昨天"
    time_context: Optional[str] = None    # "中午", "出差时"

    # 空间维度
    location: Optional[str] = None
    location_detail: Optional[str] = None

    # 人物维度
    persons: Set[str] = field(default_factory=set)
    person_count: int = 0

    # 主题维度
    topics: Set[str] = field(default_factory=set)
    keywords: Set[str] = field(default_factory=set)

    # 情绪维度
    emotion_valence: float = 0.0     # -1.0 到 1.0
    emotion_intensity: float = 0.0   # 0.0 到 1.0
    emotion_tags: Set[str] = field(default_factory=set)

    # 重要性
    importance: float = 0.5

    # 关联
    connection_value: float = 0.0

    # 元信息
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 来源信息
    source_episode_id: Optional[str] = None
    source_type: str = "extraction"  # extraction / manual / consolidation

    def __post_init__(self):
        """初始化后处理"""
        if self.persons:
            self.person_count = len(self.persons)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "content": self.content,
            "summary": self.summary,
            "memory_type": self.memory_type.value if hasattr(self.memory_type, 'value') else str(self.memory_type),
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
            "emotion_tags": list(self.emotion_tags),
            "importance": self.importance,
            "connection_value": self.connection_value,
            "metadata": self.metadata,
            "source_episode_id": self.source_episode_id,
            "source_type": self.source_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemorySpec':
        """从字典创建"""
        return cls(
            content=data.get("content", ""),
            summary=data.get("summary", ""),
            memory_type=MemoryType(data.get("memory_type", "interaction")),
            time_absolute=data.get("time_absolute"),
            time_relative=data.get("time_relative"),
            time_context=data.get("time_context"),
            location=data.get("location"),
            location_detail=data.get("location_detail"),
            persons=set(data.get("persons", [])),
            person_count=data.get("person_count", 0),
            topics=set(data.get("topics", [])),
            keywords=set(data.get("keywords", [])),
            emotion_valence=data.get("emotion_valence", 0.0),
            emotion_intensity=data.get("emotion_intensity", 0.0),
            emotion_tags=set(data.get("emotion_tags", [])),
            importance=data.get("importance", 0.5),
            connection_value=data.get("connection_value", 0.0),
            metadata=data.get("metadata", {}),
            source_episode_id=data.get("source_episode_id"),
            source_type=data.get("source_type", "extraction"),
        )


@dataclass
class ExtractionResult:
    """
    抽取结果

    包含从一个 episode 抽取的多个 MemorySpec。
    """

    episode_id: str
    specs: List[MemorySpec] = field(default_factory=list)
    extraction_time: float = 0.0
    extractor_type: str = "rule"  # rule / llm

    def __len__(self) -> int:
        return len(self.specs)

    def __iter__(self):
        return iter(self.specs)

    def add_spec(self, spec: MemorySpec):
        """添加一个 MemorySpec"""
        spec.source_episode_id = self.episode_id
        self.specs.append(spec)
