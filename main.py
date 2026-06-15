"""
类人记忆系统 - 主入口

整合所有模块，提供统一的API
"""

from typing import Dict, List, Optional, Any
import json
import time

from memory_chunk import MemoryChunk, MemoryLayer
from memory_layer_core import MemoryLayerCore
from forgotten_layer import ForgottenLayer
from retrieval import MemoryRetrieval, QueryContext, ReconstructionResult, ReviewResult
from core.weight_system import MemoryType
from core.persona_layer import PersonaLayer, BehaviorType
from core.attention_system import AttentionOS, FocusWorkspace, Goal, ProcedureMemory
from core.agent_system import CognitiveAgent
from core.cognitive_state import ActionExpectation, CognitiveFrame, CognitiveState, ReflectionNote


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

        # 目标驱动注意力调度层
        self.attention = AttentionOS()
        self.cognitive_state = CognitiveState()
        
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
        self.cognitive_state.reinforce_from_feedback(accepted, corrected_content)
    
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

    # ============ 注意力调度 ============

    def start_goal(
        self,
        objective: str,
        constraints: Optional[List[str]] = None,
        open_loops: Optional[List[str]] = None,
        priority: float = 0.7,
    ) -> Goal:
        """开始一个会影响注意力选择的目标"""
        return self.attention.start_goal(objective, constraints, open_loops, priority)

    def update_goal(
        self,
        goal_id: str,
        status: Optional[str] = None,
        evidence: Optional[List[str]] = None,
        open_loops: Optional[List[str]] = None,
    ) -> Optional[Goal]:
        """更新目标状态、证据或未闭环事项"""
        return self.attention.update_goal(
            goal_id,
            status=status,
            evidence=evidence,
            open_loops=open_loops,
        )

    def add_procedure(
        self,
        title: str,
        steps: List[str],
        triggers: Optional[List[str]] = None,
        importance: float = 0.6,
        confidence: float = 0.6,
    ) -> ProcedureMemory:
        """添加一条程序记忆，也就是系统学会的一种做事方式"""
        return self.attention.add_procedure(
            title=title,
            steps=steps,
            triggers=triggers,
            importance=importance,
            confidence=confidence,
        )

    def record_procedure_use(self, procedure_id: str, success: Optional[bool] = None) -> bool:
        """记录程序记忆是否帮上忙，用于更新置信度"""
        return self.attention.record_procedure_use(procedure_id, success)

    def focus(
        self,
        query: str,
        memory_limit: int = 5,
        procedure_limit: int = 3,
        include_forgotten: bool = False,
    ) -> FocusWorkspace:
        """
        构建本轮注意力工作区。

        它不会直接生成回答，而是决定当前目标下哪些记忆和程序应该进入上下文。
        """
        chunks = list(self.core.chunks.values())
        if include_forgotten:
            chunks.extend(self.forgotten.chunks.values())
        workspace = self.attention.build_focus(
            query=query,
            memories=chunks,
            memory_limit=memory_limit,
            procedure_limit=procedure_limit,
        )
        self.attach_cognitive_context(workspace)
        return workspace

    def get_attention_summary(self) -> Dict[str, Any]:
        """获取注意力调度层摘要"""
        active_goal = self.attention.goal_stack.active()
        return {
            "active_goal": active_goal.to_dict() if active_goal else None,
            "goals": [goal.to_dict() for goal in self.attention.goal_stack.goals],
            "procedures": [procedure.to_dict() for procedure in self.attention.procedures],
            "workspace_history_count": len(self.attention.workspace_history),
        }

    # ============ Reflective cognition ============

    def observe_world(self, observation: Any):
        """Fold a new observation into the self/world model."""
        self.cognitive_state.observe(observation)

    def attach_cognitive_context(self, workspace: FocusWorkspace, tools: Optional[Any] = None) -> FocusWorkspace:
        """Attach current self/world state to a focus workspace."""
        frame = self.cognitive_state.build_frame(
            query=workspace.query,
            workspace=workspace,
            tools=tools,
        )
        context = frame.to_dict()
        workspace.cognitive_context = context
        if self.attention.workspace_history:
            latest = self.attention.workspace_history[-1]
            if latest.get("query") == workspace.query and latest.get("created_at") == workspace.created_at:
                latest["cognitive_context"] = context
        return workspace

    def build_cognitive_frame(
        self,
        query: str = "",
        workspace: Optional[FocusWorkspace] = None,
        tools: Optional[Any] = None,
    ) -> CognitiveFrame:
        """Build an inspectable cognitive frame for prompts or debugging."""
        return self.cognitive_state.build_frame(query=query, workspace=workspace, tools=tools)

    def predict_action(self, action: Any, tools: Optional[Any] = None) -> ActionExpectation:
        """Predict the likely outcome of an action before executing it."""
        return self.cognitive_state.predict_action(action, tools=tools)

    def reflect_episode(self, episode: Any) -> ReflectionNote:
        """Update self/world state from a completed agent episode."""
        return self.cognitive_state.reflect_episode(episode)

    def get_cognitive_summary(self) -> Dict[str, Any]:
        """Return current self-model, drives, world beliefs, and reflections."""
        return self.cognitive_state.get_summary()

    # ============ Agent 运行时 ============

    def create_agent(
        self,
        name: str = "cognitive-agent",
        auto_consolidate: bool = True,
    ) -> CognitiveAgent:
        """创建一个使用本记忆系统作为认知层的 Agent 运行时"""
        return CognitiveAgent(
            memory_system=self,
            name=name,
            auto_consolidate=auto_consolidate,
        )

    def create_openai_agent(
        self,
        name: str = "openai-compatible-agent",
        auto_consolidate: bool = True,
        env_file: Optional[str] = ".env",
        synthesize_tool_responses: bool = True,
        **api_overrides: Any,
    ) -> CognitiveAgent:
        """Create an agent backed by an OpenAI-compatible ChatCompletions API."""
        from core.llm_planner import LLMPlanner, LLMResponseSynthesizer

        agent = self.create_agent(name=name, auto_consolidate=auto_consolidate)
        planner = LLMPlanner.from_openai_compatible_env(
            env_file=env_file,
            **api_overrides,
        )
        agent.planner = planner
        if synthesize_tool_responses:
            agent.response_synthesizer = LLMResponseSynthesizer(planner.llm)
        return agent
    
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

        # 保存注意力调度层
        attention_path = f"{self.data_dir}/attention.json"
        with open(attention_path, 'w', encoding='utf-8') as f:
            json.dump(self.attention.to_dict(), f, ensure_ascii=False, indent=2)

        cognitive_path = f"{self.data_dir}/cognitive_state.json"
        with open(cognitive_path, 'w', encoding='utf-8') as f:
            json.dump(self.cognitive_state.to_dict(), f, ensure_ascii=False, indent=2)
    
    def load(self) -> bool:
        """加载数据"""
        import os
        import json
        
        core_path = f"{self.data_dir}/core.json"
        forgotten_path = f"{self.data_dir}/forgotten.json"
        persona_path = f"{self.data_dir}/persona.json"
        attention_path = f"{self.data_dir}/attention.json"
        cognitive_path = f"{self.data_dir}/cognitive_state.json"
        
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

        # 加载注意力调度层
        attention_loaded = False
        if os.path.exists(attention_path):
            try:
                with open(attention_path, 'r', encoding='utf-8') as f:
                    attention_data = json.load(f)
                self.attention = AttentionOS.from_dict(attention_data)
                attention_loaded = True
            except Exception:
                pass

        cognitive_loaded = False
        if os.path.exists(cognitive_path):
            try:
                with open(cognitive_path, 'r', encoding='utf-8') as f:
                    cognitive_data = json.load(f)
                self.cognitive_state = CognitiveState.from_dict(cognitive_data)
                cognitive_loaded = True
            except Exception:
                pass
        
        return core_loaded or forgotten_loaded or persona_loaded or attention_loaded or cognitive_loaded
    
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
            "attention_summary": self.get_attention_summary(),
            "cognitive_summary": self.get_cognitive_summary(),
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
