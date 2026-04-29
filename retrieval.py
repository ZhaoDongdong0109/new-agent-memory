"""
检索与重建模块 - 类人记忆系统

核心流程：
1. 解析输入（问句/照片/外部信息）→ 信息锚点
2. 核心层检索
3. 若失败 → 伪遗忘层唤醒（信息锚点触发）
4. 碎片重组
5. 审阅（判断记忆是否合理）
6. 输出
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import time
import re

from memory_chunk import MemoryChunk, MemoryLayer
from memory_layer_core import MemoryLayerCore, WeightFactors
from forgotten_layer import ForgottenLayer


class ReviewResult(Enum):
    """审阅结果"""
    APPROVED = "approved"           # 合理，直接输出
    MODIFIED = "modified"           # 需要修正
    QUESTIONABLE = "questionable"   # 存疑，谨慎输出
    REJECTED = "rejected"           # 不合理，标记


@dataclass
class QueryContext:
    """查询上下文（信息锚点）"""
    # 原始输入
    raw_query: str = ""
    
    # 时间维度
    time_absolute: Optional[str] = None   # "2026-04-29"
    time_relative: Optional[str] = None   # "10年前", "昨天"
    time_context: Optional[str] = None    # "中午", "出差时"
    
    # 地点维度
    location: Optional[str] = None
    
    # 人物维度
    persons: Set[str] = None
    
    # 主题/语义维度
    topics: Set[str] = None
    keywords: Set[str] = None
    
    # 情绪维度
    emotion_valence: Optional[float] = None
    
    # 元信息
    source_type: str = "query"  # query / photo / audio / ...
    
    def __post_init__(self):
        self.persons = self.persons or set()
        self.topics = self.topics or set()
        self.keywords = self.keywords or set()
    
    def to_tags(self) -> Dict[str, Any]:
        """转换为检索标签"""
        tags = {}
        if self.time_absolute:
            tags["time_absolute"] = self.time_absolute
        if self.time_relative:
            tags["time_relative"] = self.time_relative
        if self.time_context:
            tags["time_context"] = self.time_context
        if self.location:
            tags["location"] = self.location
        if self.persons:
            tags["persons"] = self.persons
        if self.topics:
            tags["topics"] = self.topics
        if self.emotion_valence is not None:
            tags["emotion_valence"] = self.emotion_valence
        return tags


@dataclass
class ReconstructionResult:
    """重建结果"""
    success: bool
    chunks: List[MemoryChunk]          # 参与的碎片
    assembled_content: str             # 组装后的内容
    review_result: ReviewResult        # 审阅结果
    review_note: str = ""              # 审阅备注
    retrieval_path: str = ""          # 检索路径：core / forgotten / both
    confidence: float = 0.0           # 置信度 0~1
    
    def summary(self) -> str:
        return (
            f"[{self.review_result.value}] "
            f"path={self.retrieval_path} "
            f"chunks={len(self.chunks)} "
            f"confidence={self.confidence:.2f} | "
            f"{self.assembled_content[:100]}"
        )


class MemoryRetrieval:
    """
    记忆检索与重建系统
    
    工作流程：
    1. 解析输入 → QueryContext
    2. 核心层检索
    3. 若失败 → 伪遗忘层唤醒
    4. 碎片组装
    5. 审阅
    6. 输出
    """
    
    def __init__(
        self,
        core_layer: MemoryLayerCore,
        forgotten_layer: ForgottenLayer,
        
        # 检索参数
        core_min_weight: float = 0.2,
        core_limit: int = 10,
        forgotten_min_match: int = 2,
        
        # 组装参数
        assembly_method: str = "chronological",  # chronological / relevance / hybrid
        
        # 审阅参数
        review_confidence_threshold: float = 0.5,  # 低于此值标记为 questionable
    ):
        self.core = core_layer
        self.forgotten = forgotten_layer
        self.core_min_weight = core_min_weight
        self.core_limit = core_limit
        self.forgotten_min_match = forgotten_min_match
        self.assembly_method = assembly_method
        self.review_confidence_threshold = review_confidence_threshold
        
        # 统计
        self.total_retrievals = 0
        self.core_hit = 0
        self.forgotten_hit = 0
        self.both_hit = 0
    
    # ============ 查询解析 ============
    
    def parse_query(self, query: str) -> QueryContext:
        """
        解析自然语言查询为 QueryContext
        
        目前是简化版规则解析，未来可以换成LLM
        """
        ctx = QueryContext(raw_query=query)
        
        # 相对时间解析
        time_relative_patterns = {
            r"(\d+)年前": lambda m: f"{2026 - int(m.group(1))}年",
            r"(\d+)年前.*中午": lambda m: f"{2026 - int(m.group(1))}年",
            r"昨天": lambda _: "昨天",
            r"上周": lambda _: "上周",
            r"上个月": lambda _: "上个月",
            r"去年": lambda _: "去年",
        }
        
        for pattern, handler in time_relative_patterns.items():
            match = re.search(pattern, query)
            if match:
                ctx.time_relative = handler(match)
                if "中午" in query or "午饭" in query or "午餐" in query:
                    ctx.time_context = "中午"
                break
        
        # 时间上下文解析
        if "中午" in query or "午饭" in query or "午餐" in query:
            ctx.time_context = ctx.time_context or "中午"
        
        # 地点解析（简化）
        locations = ["北京", "上海", "家里", "公司", "餐厅", "酒店", "机场"]
        for loc in locations:
            if loc in query:
                ctx.location = loc
                break
        
        # 人物解析（简化）
        person_pattern = r"和(.+?)(一起|吃的|去的|见的)"
        match = re.search(person_pattern, query)
        if match:
            ctx.persons.add(match.group(1))
        
        # 主题解析
        topic_keywords = {
            "吃": {"food", "dining", "meal"},
            "饭": {"food", "dining", "meal"},
            "旅行": {"travel", "trip"},
            "出差": {"business", "work"},
            "会议": {"meeting", "work"},
            "项目": {"project", "work"},
        }
        
        for keyword, topics in topic_keywords.items():
            if keyword in query:
                ctx.topics.update(topics)
        
        # 情绪解析（简化）
        positive_words = ["开心", "高兴", "快乐", "愉快", "棒", "好"]
        negative_words = ["难过", "伤心", "痛苦", "糟糕", "差"]
        
        for pw in positive_words:
            if pw in query:
                ctx.emotion_valence = 0.5
                break
        for nw in negative_words:
            if nw in query:
                ctx.emotion_valence = -0.5
                break
        
        return ctx
    
    def parse_photo_info(self, photo_info: Dict[str, Any]) -> QueryContext:
        """
        解析照片信息为 QueryContext
        
        photo_info 可能包含：
        - 时间：photo_info.get("timestamp")
        - 地点：photo_info.get("location")
        - 人物：photo_info.get("faces", [])
        - 内容标签：photo_info.get("labels", [])
        """
        ctx = QueryContext()
        ctx.source_type = "photo"
        
        # 时间
        timestamp = photo_info.get("timestamp")
        if timestamp:
            # 假设是时间戳
            ctx.time_absolute = time.strftime("%Y-%m-%d", time.localtime(timestamp))
            ctx.time_context = time.strftime("%H:%M", time.localtime(timestamp))
        
        # 地点
        location = photo_info.get("location")
        if location:
            ctx.location = location
        
        # 人物
        faces = photo_info.get("faces", [])
        ctx.persons.update(faces)
        
        # 标签
        labels = photo_info.get("labels", [])
        ctx.topics.update(labels)
        
        return ctx
    
    # ============ 检索 ============
    
    def retrieve(
        self,
        query: str,
        allow_forgotten: bool = True,
    ) -> ReconstructionResult:
        """
        主检索入口
        
        流程：
        1. 解析查询
        2. 核心层检索
        3. 伪遗忘层唤醒（如需要）
        4. 组装 + 审阅
        5. 返回结果
        """
        self.total_retrievals += 1
        
        # Step 1: 解析
        ctx = self.parse_query(query)
        
        # Step 2: 核心层检索
        core_results = self.core.retrieve(
            ctx.to_tags(),
            min_weight=self.core_min_weight,
            limit=self.core_limit,
        )
        
        retrieval_path = ""
        all_chunks = []
        
        if core_results:
            # 核心层命中
            retrieval_path = "core"
            self.core_hit += 1
            all_chunks = [chunk for chunk, _ in core_results]
            
            # Hebbian 关联扩展（把相关记忆也拉进来）
            expanded = self._expand_via_associations(all_chunks)
            
            # 用ID去重
            seen_ids = {c.id for c in all_chunks}
            for c in expanded:
                if c.id not in seen_ids:
                    all_chunks.append(c)
                    seen_ids.add(c.id)
            
        elif allow_forgotten:
            # Step 3: 核心层没命中，尝试伪遗忘层唤醒
            retrieval_path = "forgotten"
            forgotten_results = self.forgotten.try_wake(
                ctx.to_tags(),
                limit=5,
            )
            
            if forgotten_results:
                self.forgotten_hit += 1
                all_chunks = [chunk for chunk, _ in forgotten_results]
                retrieval_path = "forgotten"
            else:
                retrieval_path = "none"
        else:
            retrieval_path = "none"
        
        # 如果都没有命中
        if not all_chunks:
            return ReconstructionResult(
                success=False,
                chunks=[],
                assembled_content="",
                review_result=ReviewResult.REJECTED,
                retrieval_path="none",
                confidence=0.0,
                review_note="没有找到相关记忆",
            )
        
        # Step 4: 组装
        assembled = self._assemble(all_chunks, ctx)
        
        # Step 5: 审阅
        review_result, confidence = self._review(all_chunks, assembled, ctx)
        
        result = ReconstructionResult(
            success=True,
            chunks=all_chunks,
            assembled_content=assembled,
            review_result=review_result,
            retrieval_path=retrieval_path,
            confidence=confidence,
        )
        
        # 反馈给核心层
        for chunk in all_chunks:
            if chunk.layer == MemoryLayer.CORE:
                self.core.access(chunk.id)
        
        return result
    
    def _expand_via_associations(self, chunks: List[MemoryChunk]) -> List[MemoryChunk]:
        """
        通过 Hebbian 关联扩展候选碎片
        """
        if not chunks:
            return []
        
        expanded = []
        for chunk in chunks:
            # 获取关联最强的记忆
            if not chunk.associations:
                continue
            
            sorted_assocs = sorted(
                chunk.associations.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            
            # 取前3个关联最强的
            for assoc_id, assoc_weight in sorted_assocs[:3]:
                if assoc_weight > 0.3:  # 阈值
                    assoc_chunk = self.core.get(assoc_id)
                    if assoc_chunk:
                        expanded.append(assoc_chunk)
        
        return expanded
    
    def _assemble(
        self,
        chunks: List[MemoryChunk],
        ctx: QueryContext,
    ) -> str:
        """
        将碎片组装成连贯的叙述
        
        目前是简化版，按时间/重要性排序后拼接
        未来可以换成LLM生成
        """
        if not chunks:
            return ""
        
        if len(chunks) == 1:
            return chunks[0].content
        
        # 按时间排序
        if self.assembly_method == "chronological":
            sorted_chunks = sorted(
                chunks,
                key=lambda c: c.created_at,
                reverse=True,
            )
        elif self.assembly_method == "relevance":
            # 按重要性排序
            sorted_chunks = sorted(
                chunks,
                key=lambda c: c.importance,
                reverse=True,
            )
        else:  # hybrid
            # 综合排序
            sorted_chunks = sorted(
                chunks,
                key=lambda c: c.importance * 0.5 + c.access_count * 0.3 + c.successful_recall_count * 0.2,
                reverse=True,
            )
        
        # 拼接内容（简化版）
        # 未来用LLM根据上下文生成连贯叙述
        parts = [c.content for c in sorted_chunks if c.content]
        
        if len(parts) == 1:
            return parts[0]
        
        # 简单的拼接
        assembled = "；".join(parts)
        
        # 如果上下文提示了具体时间，加上
        if ctx.time_relative:
            assembled = f"关于{ctx.time_relative}的记忆：{assembled}"
        
        return assembled
    
    def _review(
        self,
        chunks: List[MemoryChunk],
        assembled: str,
        ctx: QueryContext,
    ) -> Tuple[ReviewResult, float]:
        """
        审阅组装的记忆是否合理
        
        简化版审阅：
        1. 时间线检查（如果多个碎片，时间线是否矛盾）
        2. 情绪一致性检查
        3. 重要性加权置信度
        """
        if not chunks:
            return ReviewResult.REJECTED, 0.0
        
        confidence = 0.0
        issues = []
        
        # 1. 基础置信度（基于碎片数量和质量）
        if len(chunks) == 1:
            confidence += 0.3
        elif len(chunks) <= 3:
            confidence += 0.4
        else:
            confidence += 0.5
        
        # 2. 基于平均重要性
        avg_importance = sum(c.importance for c in chunks) / len(chunks)
        confidence += avg_importance * 0.3
        
        # 3. 基于成功回忆次数
        total_recalls = sum(c.successful_recall_count for c in chunks)
        confidence += min(0.2, total_recalls * 0.05)
        
        # 4. 检查情绪一致性（如果有上下文的话）
        if ctx.emotion_valence is not None:
            avg_emotion = sum(c.emotion_valence for c in chunks) / len(chunks)
            emotion_diff = abs(avg_emotion - ctx.emotion_valence)
            if emotion_diff > 0.5:
                issues.append("情绪不一致")
                confidence -= 0.15
        
        # 5. 检查时间合理性
        time_contexts = [c.time_context for c in chunks if c.time_context]
        if ctx.time_context and ctx.time_context not in time_contexts:
            # 问的是中午，但记忆没有中午的上下文
            issues.append("时间上下文可能不匹配")
            confidence -= 0.1
        
        # 6. 审阅状态检查
        statuses = [c.review_status for c in chunks]
        if "rejected" in statuses:
            return ReviewResult.REJECTED, confidence * 0.3
        if "questionable" in statuses:
            issues.append("部分记忆被标记为存疑")
            confidence *= 0.7
        
        # 最终置信度
        confidence = max(0.0, min(1.0, confidence))
        
        # 审阅结论
        if confidence >= self.review_confidence_threshold:
            if issues:
                return ReviewResult.MODIFIED, confidence
            return ReviewResult.APPROVED, confidence
        else:
            return ReviewResult.QUESTIONABLE, confidence
    
    # ============ 反馈 ============
    
    def feedback(
        self,
        query: str,
        accepted: bool,
        corrected_content: Optional[str] = None,
    ):
        """
        用户反馈
        
        accepted=True: 输出被接受了
        accepted=False: 输出被纠正了，corrected_content是正确的内容
        """
        result = self.retrieve(query, allow_forgotten=False)
        
        if not result.success:
            return
        
        for chunk in result.chunks:
            if chunk.layer == MemoryLayer.CORE:
                self.core.adjust_after_recall(
                    chunk.id,
                    success=accepted,
                    feedback_emotion=result.confidence,
                )
        
        # 如果被纠正，可能需要更新内容
        if not accepted and corrected_content:
            # 这是更高级的功能，涉及记忆修正
            pass
    
    # ============ 统计 ============
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_retrievals": self.total_retrievals,
            "core_hit_rate": self.core_hit / max(1, self.total_retrievals),
            "forgotten_hit_rate": self.forgotten_hit / max(1, self.total_retrievals),
            "core_chunks": len(self.core),
            "forgotten_chunks": len(self.forgotten),
        }
