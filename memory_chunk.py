"""
记忆碎片模块 - 类人记忆系统核心数据结构

核心设计：
- 每个记忆是一个碎片（chunk），携带多维标签
- 标签是检索的入口，类似C语言头文件
- 内容与标签分离，支持高效的检索
- 支持记忆类型区分（影响权重策略）
"""

from dataclasses import dataclass, field
from typing import Set, Dict, Optional, Any
from enum import Enum
import time
import uuid

# 导入记忆类型枚举
from core.weight_system import MemoryType


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
    - memory_type 影响权重衰减策略
    """

    # 唯一标识
    id: str = field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:12]}")

    # 记忆内容
    content: str = ""

    # 摘要（用于索引和快速检索）
    summary: str = ""

    # 记忆类型（影响衰减节奏）
    memory_type: MemoryType = MemoryType.INTERACTION

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
    person_count: int = 1

    # 主题/语义维度
    topics: Set[str] = field(default_factory=set)  # 主题标签
    keywords: Set[str] = field(default_factory=set)  # 关键词

    # 情绪维度
    emotion_valence: float = 0.0   # 情绪效价 -1.0(负面) ~ +1.0(正面)
    emotion_intensity: float = 0.0 # 情绪强度 0.0 ~ 1.0
    emotion_tags: Set[str] = field(default_factory=set)  # 情绪标签列表

    # 连接价值
    connection_value: float = 0.5  # 能触发多少其他记忆

    # 重要性（用户对这件事的长期重视程度，0~1）
    importance: float = 0.5

    # 层级
    layer: MemoryLayer = MemoryLayer.CORE

    # 时间戳
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    # 访问统计
    access_count: int = 0
    successful_recall_count: int = 0  # 成功被唤醒次数

    # 回忆反馈偏置（-0.25 ~ +0.25）
    # 由 adjust_after_recall 累积：确认正确的回忆提升，错误回忆降低。
    # 直接叠加进 calc_weight 的最终权重，让反馈真正影响记忆去留。
    recall_bias: float = 0.0

    # 使用时间戳日志（ACT-R 基线激活的输入，Petrov O(k) 混合近似：
    # 精确保留最近 ACCESS_LOG_SIZE 次，更早的次数由 access_count
    # 统计近似）。编码事件（创建）算第一次使用。
    access_log: list = field(default_factory=list)

    # 关联记忆（Hebbian关联）
    associations: Dict[str, float] = field(default_factory=dict)  # chunk_id -> weight

    # 审阅标记
    review_status: str = "pending"  # pending / approved / questionable / rejected
    review_note: Optional[str] = None

    # 重建相关
    reconstruction_count: int = 0   # 被重建过的次数
    parent_id: Optional[str] = None  # 如果是从旧版本重建的，记录原记忆ID

    # ===== 新增：统一 schema 字段 =====

    # 记忆来源
    source: str = "user"  # user / system_extract / import / consolidation

    # 置信度（这条记忆有多可信，0~1）
    confidence: float = 0.8

    # 有效期（事实可能过时）
    valid_at: Optional[float] = None    # 有效期开始时间戳
    invalid_at: Optional[float] = None  # 有效期结束时间戳（None 表示永不过期）

    # 用户与会话标识
    user_id: str = "default"      # 用户ID（多用户隔离）
    session_id: Optional[str] = None  # 会话ID（跨会话追踪）

    # 版本控制
    version: int = 1  # 版本号，更新时递增

    # 元数据（扩展字段）
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Petrov 混合近似精确保留的时间戳条数
    ACCESS_LOG_SIZE = 8

    # 访问去抖窗口：窗口内的重复检索视为同一次"使用"。
    # 狗粮期实测：一轮评测（19 个查询）就把访问计数刷到 40+，
    # ACT-R 频率效应随即让"热门"记忆霸榜——人类一次回忆
    # 不会让记忆强度暴涨，一分钟内的重复命中也不应该。
    ACCESS_DEBOUNCE_SECONDS = 60.0

    def __post_init__(self):
        # 编码事件算第一次使用（旧数据没有 access_log 时同样成立）
        if not self.access_log:
            self.access_log = [self.created_at]

    def access(self):
        """记录一次访问（60 秒窗口内去抖）"""
        now = time.time()
        debounced = (
            self.access_count > 0
            and now - self.last_accessed < self.ACCESS_DEBOUNCE_SECONDS
        )
        self.last_accessed = now
        self.updated_at = now
        if debounced:
            return
        self.access_count += 1
        self.access_log.append(now)
        if len(self.access_log) > self.ACCESS_LOG_SIZE:
            self.access_log = self.access_log[-self.ACCESS_LOG_SIZE:]
    
    def successful_recall(self):
        """记录一次成功唤醒"""
        self.successful_recall_count += 1
    
    def record_reconstruction(self, parent_id: str):
        """记录一次重建"""
        self.reconstruction_count += 1
        self.parent_id = parent_id
    
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
        
        # 情绪标签
        for e in sorted(self.emotion_tags):
            parts.append(f"EM:{e}")
        
        return " | ".join(parts)
    
    def matches_query(self, query_tags: Dict[str, Any]) -> bool:
        """
        检查当前碎片是否匹配查询标签
        
        时间字段（time_absolute, time_relative, time_context）使用宽松匹配：
        - 如果查询指定了 time_relative，同时也匹配 time_context
        - 如果查询指定了 time_context，同时也匹配 time_relative
        这样可以处理"昨天"这类语义重叠的时间词
        """
        # 时间匹配
        if "time_absolute" in query_tags:
            if self.time_absolute != query_tags["time_absolute"]:
                return False
        
        # 相对时间和上下文使用宽松匹配（两者可以互通）
        if "time_relative" in query_tags:
            query_val = query_tags["time_relative"]
            # 同时检查 time_relative 和 time_context
            if self.time_relative != query_val and self.time_context != query_val:
                return False
        
        if "time_context" in query_tags:
            query_val = query_tags["time_context"]
            # 同时检查 time_context 和 time_relative
            if self.time_context != query_val and self.time_relative != query_val:
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
        
        return True
    
    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            "id": self.id,
            "content": self.content,
            "summary": self.summary,
            "memory_type": self.memory_type.value if isinstance(self.memory_type, MemoryType) else self.memory_type,
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
            "emotion_tags": list(self.emotion_tags),
            "connection_value": self.connection_value,
            "importance": self.importance,
            "layer": self.layer.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "successful_recall_count": self.successful_recall_count,
            "recall_bias": self.recall_bias,
            "access_log": list(self.access_log),
            "associations": self.associations,
            "review_status": self.review_status,
            "review_note": self.review_note,
            "reconstruction_count": self.reconstruction_count,
            "parent_id": self.parent_id,
            # 新增字段
            "source": self.source,
            "confidence": self.confidence,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "version": self.version,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "MemoryChunk":
        """从字典反序列化"""
        data = data.copy()
        data["persons"] = set(data.get("persons", []))
        data["topics"] = set(data.get("topics", []))
        data["keywords"] = set(data.get("keywords", []))
        data["emotion_tags"] = set(data.get("emotion_tags", []))
        data["layer"] = MemoryLayer(data.get("layer", "core"))

        memory_type_val = data.get("memory_type", "interaction")
        if isinstance(memory_type_val, str):
            data["memory_type"] = MemoryType(memory_type_val)

        # 忽略未知字段：新版本写入的数据可以被旧字段集合安全加载
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls)}
        data = {k: v for k, v in data.items() if k in known}

        return cls(**data)
    
    def __repr__(self) -> str:
        content_preview = self.content[:30] + "..." if len(self.content) > 30 else self.content
        type_str = self.memory_type.value if isinstance(self.memory_type, MemoryType) else self.memory_type
        return f"<MemoryChunk {self.id} [{type_str}] '{content_preview}'>"
