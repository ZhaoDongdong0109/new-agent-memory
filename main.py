"""
类人记忆系统 - 主入口

整合所有模块，提供统一的API
"""

from typing import Dict, List, Optional, Any
import time

from memory_chunk import MemoryChunk, MemoryLayer
from memory_layer_core import MemoryLayerCore
from forgotten_layer import ForgottenLayer
from retrieval import MemoryRetrieval, QueryContext, ReconstructionResult, ReviewResult
from core.weight_system import MemoryType
from core.persona_layer import PersonaLayer, BehaviorType


class HumanLikeMemorySystem:
    """
    类人记忆系统
    
    使用示例：
    
    system = HumanLikeMemorySystem()
    
    # 添加记忆
    system.add_memory(
        content="今天中午和客户在北京餐厅吃了烤鸭",
        time_absolute="2026-04-29",
        time_context="中午",
        location="北京",
        persons=["客户"],
        topics={"food", "business"},
        emotion_valence=0.3,
        importance=0.7,
    )
    
    # 检索记忆
    result = system.retrieve("10年前中午吃了什么")
    
    print(result.assembled_content)
    """
    
    def __init__(
        self,
        data_dir: str = "./memory_data",
        
        # 核心层参数
        core_decay_half_life: float = 7 * 24 * 3600,
        core_degrade_threshold: float = 0.15,
        
        # 伪遗忘层参数
        forgotten_cleanup_age_days: float = 365,
        
        # 检索参数
        retrieval_confidence_threshold: float = 0.5,
    ):
        self.data_dir = data_dir
        
        # 初始化各层
        self.core = MemoryLayerCore(
            decay_half_life=core_decay_half_life,
            degrade_threshold=core_degrade_threshold,
        )
        
        self.forgotten = ForgottenLayer(
            cleanup_age_days=forgotten_cleanup_age_days,
        )
        
        self.retrieval = MemoryRetrieval(
            core_layer=self.core,
            forgotten_layer=self.forgotten,
            review_confidence_threshold=retrieval_confidence_threshold,
        )
        
        # 人格适应层
        self.persona = PersonaLayer()
        
        # 定时任务
        self.last_maintenance = time.time()
        self.maintenance_interval = 6 * 3600  # 每6小时维护一次
    
    # ============ 记忆操作 ============
    
    def add_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.INTERACTION,
        
        # 时间维度
        time_absolute: Optional[str] = None,
        time_relative: Optional[str] = None,
        time_context: Optional[str] = None,
        
        # 空间维度
        location: Optional[str] = None,
        location_detail: Optional[str] = None,
        
        # 人物维度
        persons: Optional[List[str]] = None,
        
        # 主题维度
        topics: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        
        # 情绪维度
        emotion_valence: float = 0.0,
        emotion_intensity: float = 0.0,
        
        # 重要性
        importance: float = 0.5,
        
        # 元数据
        metadata: Optional[Dict[str, Any]] = None,
        
        # 直接指定层级
        target_layer: Optional[MemoryLayer] = None,
    ) -> str:
        """
        添加记忆
        
        返回记忆ID
        """
        chunk = MemoryChunk(
            content=content,
            memory_type=memory_type,
            
            time_absolute=time_absolute,
            time_relative=time_relative,
            time_context=time_context,
            
            location=location,
            location_detail=location_detail,
            
            persons=set(persons) if persons else set(),
            topics=set(topics) if topics else set(),
            keywords=set(keywords) if keywords else set(),
            
            emotion_valence=emotion_valence,
            emotion_intensity=emotion_intensity,
            
            importance=importance,
            
            metadata=metadata or {},
        )
        
        if target_layer == MemoryLayer.FORGOTTEN:
            self.forgotten.archive(chunk)
        else:
            self.core.add(chunk)
        
        return chunk.id
    
    def retrieve(
        self,
        query: str,
        allow_forgotten: bool = True,
    ) -> ReconstructionResult:
        """
        检索记忆
        
        返回重组后的记忆
        """
        return self.retrieval.retrieve(query, allow_forgotten)
    
    def retrieve_by_photo(
        self,
        photo_info: Dict[str, Any],
    ) -> ReconstructionResult:
        """
        通过照片信息检索记忆
        
        photo_info 格式：
        {
            "timestamp": 1714304000,  # 时间戳
            "location": "北京",        # 地点
            "faces": ["张三", "李四"], # 识别到的人脸
            "labels": ["烤鸭", "餐厅"], # 图像标签
        }
        """
        ctx = self.retrieval.parse_photo_info(photo_info)
        
        # 先尝试核心层
        core_results = self.core.retrieve(
            ctx.to_tags(),
            min_weight=0.2,
            limit=10,
        )
        
        if core_results:
            chunks = [c for c, _ in core_results]
            assembled = self._assemble_chunks(chunks, ctx)
            review_result, confidence = self._review_chunks(chunks, assembled, ctx)
            
            return ReconstructionResult(
                success=True,
                chunks=chunks,
                assembled_content=assembled,
                review_result=review_result,
                retrieval_path="core",
                confidence=confidence,
            )
        
        # 尝试伪遗忘层唤醒
        forgotten_results = self.forgotten.try_wake(ctx.to_tags())
        
        if forgotten_results:
            chunks = [c for c, _ in forgotten_results]
            assembled = self._assemble_chunks(chunks, ctx)
            review_result, confidence = self._review_chunks(chunks, assembled, ctx)
            
            return ReconstructionResult(
                success=True,
                chunks=chunks,
                assembled_content=assembled,
                review_result=review_result,
                retrieval_path="forgotten",
                confidence=confidence,
            )
        
        return ReconstructionResult(
            success=False,
            chunks=[],
            assembled_content="",
            review_result=ReviewResult.REJECTED,
            retrieval_path="none",
            confidence=0.0,
        )
    
    def _assemble_chunks(self, chunks, ctx):
        """组装碎片"""
        if not chunks:
            return ""
        if len(chunks) == 1:
            return chunks[0].content
        
        sorted_chunks = sorted(chunks, key=lambda c: c.importance, reverse=True)
        parts = [c.content for c in sorted_chunks if c.content]
        assembled = "；".join(parts)
        
        if ctx.time_relative:
            assembled = f"关于{ctx.time_relative}的记忆：{assembled}"
        
        return assembled
    
    def _review_chunks(self, chunks, assembled, ctx):
        """审阅碎片"""
        import math
        confidence = 0.0
        
        if len(chunks) == 1:
            confidence += 0.3
        elif len(chunks) <= 3:
            confidence += 0.4
        else:
            confidence += 0.5
        
        avg_importance = sum(c.importance for c in chunks) / len(chunks)
        confidence += avg_importance * 0.3
        
        total_recalls = sum(c.successful_recall_count for c in chunks)
        confidence += min(0.2, total_recalls * 0.05)
        
        confidence = max(0.0, min(1.0, confidence))
        
        if confidence >= 0.5:
            return ReviewResult.APPROVED, confidence
        return ReviewResult.QUESTIONABLE, confidence
    
    def feedback(
        self,
        query: str,
        accepted: bool,
        corrected_content: Optional[str] = None,
    ):
        """用户反馈"""
        self.retrieval.feedback(query, accepted, corrected_content)
        
        # 同步给人格适应层
        if accepted:
            self.persona.on_active_recall_continue()
        else:
            self.persona.on_active_recall_ignore()
    
    def on_active_recall_explicit_positive(self):
        """用户对主动提及表示惊喜"""
        return self.persona.on_active_recall_explicit_positive()
    
    def on_active_recall_explicit_negative(self):
        """用户对主动提及表示厌烦"""
        return self.persona.on_active_recall_explicit_negative()
    
    def should_trigger_active_recall(self) -> bool:
        """是否应该主动提及旧记忆"""
        return self.persona.should_trigger_active_recall()
    
    def get_persona_summary(self) -> Dict[str, Any]:
        """获取人格适应层摘要"""
        return self.persona.get_profile_summary()
    
    # ============ 维护 ============
    
    def maintain(self):
        """
        维护任务：
        1. 检查需要降级的记忆
        2. 降级到伪遗忘层
        3. 清理伪遗忘层
        4. 衰减长期未使用的记忆
        """
        now = time.time()
        
        # 核心层降级检查
        to_degrade = self.core.check_degrade()
        if to_degrade:
            degraded = self.core.degrade_chunks(to_degrade)
            for chunk in degraded:
                self.forgotten.archive(chunk)
        
        # 伪遗忘层清理
        self.forgotten.cleanup()
        
        # 核心层衰减未使用的记忆
        self.core.decay_all_unused()
        
        self.last_maintenance = now
    
    def auto_maintain_if_needed(self):
        """如果到了维护时间，自动维护"""
        if time.time() - self.last_maintenance > self.maintenance_interval:
            self.maintain()
    
    # ============ 持久化 ============
    
    def save(self):
        """保存所有数据"""
        import os
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.core.save(f"{self.data_dir}/core.json")
        self.forgotten.save(f"{self.data_dir}/forgotten.json")
        
        # 保存人格适应层
        import json
        persona_path = f"{self.data_dir}/persona.json"
        with open(persona_path, 'w', encoding='utf-8') as f:
            json.dump(self.persona.export_profile(), f, ensure_ascii=False, indent=2)
    
    def load(self) -> bool:
        """加载数据"""
        import os
        import json
        
        core_path = f"{self.data_dir}/core.json"
        forgotten_path = f"{self.data_dir}/forgotten.json"
        persona_path = f"{self.data_dir}/persona.json"
        
        core_loaded = self.core.load(core_path) if os.path.exists(core_path) else False
        forgotten_loaded = self.forgotten.load(forgotten_path) if os.path.exists(forgotten_path) else False
        
        # 加载人格适应层
        persona_loaded = False
        if os.path.exists(persona_path):
            try:
                with open(persona_path, 'r', encoding='utf-8') as f:
                    persona_data = json.load(f)
                self.persona = PersonaLayer.from_profile(persona_data)
                persona_loaded = True
            except Exception:
                pass
        
        return core_loaded or forgotten_loaded or persona_loaded
    
    # ============ 统计 ============
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计"""
        f_stats = self.forgotten.get_stats()
        
        return {
            "core_chunks": len(self.core),
            "forgotten_chunks": len(self.forgotten),
            "forgotten_stats": {
                "total": f_stats.total_chunks,
                "oldest_days": f_stats.oldest_age_days,
                "avg_weight": f_stats.avg_weight,
                "types": f_stats.chunk_types,
            },
            "retrieval_stats": self.retrieval.get_stats(),
            "persona_summary": self.persona.get_profile_summary(),
        }
    
    def get_recent_memories(self, limit: int = 10) -> List[Dict]:
        """获取最近的记忆"""
        all_chunks = list(self.core.chunks.values()) + list(self.forgotten.chunks.values())
        all_chunks.sort(key=lambda c: c.created_at, reverse=True)
        
        return [
            {
                "id": c.id,
                "content": c.content[:50],
                "layer": c.layer.value,
                "created_at": time.strftime("%Y-%m-%d %H:%M", time.localtime(c.created_at)),
                "importance": c.importance,
            }
            for c in all_chunks[:limit]
        ]
