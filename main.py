"""
类人记忆系统 - 主入口

整合所有模块，提供统一的API

重构：支持可插拔存储后端（json / sqlite）
"""

from typing import Dict, List, Optional, Any
import json
import os
import time

from memory_chunk import MemoryChunk, MemoryLayer
from memory_layer_core import MemoryLayerCore
from forgotten_layer import ForgottenLayer
from retrieval import MemoryRetrieval, ReconstructionResult, ReviewResult
from core.weight_system import MemoryType
from core.persona_layer import PersonaLayer
from core.attention_system import AttentionOS, FocusWorkspace, Goal, ProcedureMemory
from core.agent_system import CognitiveAgent
from core.cognitive_state import ActionExpectation, CognitiveFrame, CognitiveState, ReflectionNote


class HumanLikeMemorySystem:
    """
    类人记忆系统

    使用示例：

    # 默认 JSON 后端（向后兼容）
    system = HumanLikeMemorySystem()

    # 使用 SQLite 后端
    system = HumanLikeMemorySystem(store_backend="sqlite")

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

        # 存储后端："json" 或 "sqlite"
        store_backend: str = "json",

        # 核心层参数
        core_decay_half_life: float = 7 * 24 * 3600,
        core_degrade_threshold: float = 0.15,

        # 伪遗忘层参数
        forgotten_cleanup_age_days: float = 365,

        # 检索参数
        retrieval_confidence_threshold: float = 0.5,

        # 混合检索（BM25 + Dense + RRF）。关闭则退回纯标签检索。
        enable_hybrid_retrieval: bool = True,

        # 安全与治理参数
        enable_pii_detection: bool = True,
        enable_audit_log: bool = True,
        audit_log_file: Optional[str] = None,

        # LLM 函数（可选）
        llm_fn: Optional[Any] = None,
    ):
        self.data_dir = data_dir
        self.store_backend = store_backend
        self.llm_fn = llm_fn

        # 创建存储后端
        core_store, forgotten_store = self._create_stores(store_backend, data_dir)

        # 初始化各层
        self.core = MemoryLayerCore(
            store=core_store,
            decay_half_life=core_decay_half_life,
            degrade_threshold=core_degrade_threshold,
        )

        self.forgotten = ForgottenLayer(
            store=forgotten_store,
            cleanup_age_days=forgotten_cleanup_age_days,
        )

        # 混合检索栈：把 BM25/Dense/RRF 真正接入生产检索路径。
        # 只索引核心层——伪遗忘层的记忆按设计必须通过线索唤醒进入，
        # 不应该出现在主动检索的候选里。
        self.query_planner = None
        if enable_hybrid_retrieval:
            from core.bm25_retriever import BM25Retriever
            from core.dense_retriever import DenseRetriever
            from core.query_planner import QueryPlanner
            self.query_planner = QueryPlanner(
                core_layer=self.core,
                forgotten_layer=None,
                bm25=BM25Retriever(),
                dense=DenseRetriever(),
            )

        self.retrieval = MemoryRetrieval(
            core_layer=self.core,
            forgotten_layer=self.forgotten,
            planner=self.query_planner,
            review_confidence_threshold=retrieval_confidence_threshold,
        )

        # 双时态事实取代：FACT/PREFERENCE 写入时的确定性决策表
        # （ADD / UPDATE / SUPERSEDE / NOOP，每个决策带具名规则可审计）
        from core.supersession import SupersessionEngine
        self.supersession = SupersessionEngine(self.core)

        # 人格适应层
        self.persona = PersonaLayer()

        # 目标驱动注意力调度层
        self.attention = AttentionOS()
        self.cognitive_state = CognitiveState()

        # 安全与治理层
        self.enable_pii_detection = enable_pii_detection
        self.enable_audit_log = enable_audit_log

        if enable_pii_detection:
            from core.pii_handler import PIIHandler
            self.pii_handler = PIIHandler()
        else:
            self.pii_handler = None

        if enable_audit_log:
            from core.audit_logger import AuditLogger
            log_file = audit_log_file or os.path.join(data_dir, "audit.log")
            self.audit_logger = AuditLogger(log_file=log_file)
        else:
            self.audit_logger = None

        # 定时任务
        self.last_maintenance = time.time()
        self.maintenance_interval = 6 * 3600  # 每6小时维护一次

    @staticmethod
    def _create_stores(backend: str, data_dir: str):
        """
        根据后端类型创建存储实例

        Args:
            backend: "json" 或 "sqlite"
            data_dir: 数据目录

        Returns:
            (core_store, forgotten_store) 元组
        """
        os.makedirs(data_dir, exist_ok=True)

        if backend == "sqlite":
            from core.sqlite_store import SqliteMemoryStore
            db_path = os.path.join(data_dir, "memory.db")
            core_store = SqliteMemoryStore(db_path, table_prefix="core_")
            forgotten_store = SqliteMemoryStore(db_path, table_prefix="forgotten_")
        else:
            # 默认 JSON 后端
            from core.json_store import JsonMemoryStore
            core_store = JsonMemoryStore(os.path.join(data_dir, "core.json"))
            forgotten_store = JsonMemoryStore(os.path.join(data_dir, "forgotten.json"))

        return core_store, forgotten_store

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

        # 新增：统一 schema 字段
        source: str = "user",  # user / system_extract / import / consolidation
        confidence: float = 0.8,
        valid_at: Optional[float] = None,
        invalid_at: Optional[float] = None,
        user_id: str = "default",
        session_id: Optional[str] = None,
        version: int = 1,
    ) -> str:
        """
        添加记忆

        返回记忆ID
        """
        # PII 检测与脱敏
        if self.pii_handler and self.pii_handler.has_pii(content):
            pii_types = list(self.pii_handler.get_pii_types(content))
            content = self.pii_handler.redact(content)

            # 记录 PII 检测审计
            if self.audit_logger:
                import hashlib
                text_hash = hashlib.md5(content.encode()).hexdigest()[:16]
                self.audit_logger.log_pii_detection(
                    text_hash=text_hash,
                    pii_types=pii_types,
                    action="redact",
                )

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

            # 新增字段
            source=source,
            confidence=confidence,
            valid_at=valid_at,
            invalid_at=invalid_at,
            user_id=user_id,
            session_id=session_id,
            version=version,
        )

        if target_layer == MemoryLayer.FORGOTTEN:
            self.forgotten.archive(chunk)
        else:
            # 写入决策表：FACT/PREFERENCE 可能命中 NOOP/UPDATE/SUPERSEDE，
            # 返回值非 None 时表示写入已被既有记忆吸收
            absorbed_id = self._apply_write_decision(chunk)
            if absorbed_id is not None:
                return absorbed_id
            self.core.add(chunk)
            if self.query_planner:
                self.query_planner.add_chunk(chunk)

        # 记录创建审计
        if self.audit_logger:
            self.audit_logger.log_memory_access(
                chunk_id=chunk.id,
                user_id=user_id,
                action="create",
                details={"source": source, "layer": target_layer.value if target_layer else "core"},
            )

        return chunk.id

    def _apply_write_decision(self, chunk: MemoryChunk) -> Optional[str]:
        """
        对待写入记忆执行取代决策表。

        返回值：
        - None: 按 ADD 正常写入（惊奇度已用于缩放重要性）
        - chunk_id: 写入被既有记忆吸收（NOOP 强化 / UPDATE 就地更新），
          调用方直接返回该 id

        SUPERSEDE 时旧记忆被标记失效并归档到伪遗忘层（"过时"是
        伪遗忘的正当理由），新记忆 parent_id 指向旧记忆，形成
        可追溯的双时态链条。
        """
        from core.supersession import surprise_scaled_importance

        now = time.time()
        decision = self.supersession.decide(chunk, now=now)

        if self.audit_logger:
            self.audit_logger.log_security_event(
                event_type="memory_write_decision",
                details=dict(decision.to_audit(), chunk_id=chunk.id),
                severity="info",
            )

        if decision.op == "noop":
            # 近重复：强化既有记忆，不新增
            self.core.access(decision.target_id)
            existing = self.core.get(decision.target_id)
            if existing:
                existing.successful_recall()
                self.core._store.put(existing)
            return decision.target_id

        if decision.op == "update":
            # 同一事实的更完整版本：就地更新，旧内容进 history
            existing = self.core.get(decision.target_id)
            if existing:
                history = existing.metadata.setdefault("history", [])
                history.append({
                    "content": existing.content,
                    "replaced_at": now,
                    "version": existing.version,
                })
                existing.content = chunk.content
                existing.keywords |= chunk.keywords
                existing.topics |= chunk.topics
                existing.persons |= chunk.persons
                existing.version += 1
                existing.updated_at = now
                existing.confidence = max(existing.confidence, chunk.confidence)
                # 重新索引（标签可能变化）
                self.core.add(existing)
                if self.query_planner:
                    self.query_planner.add_chunk(existing)
                return existing.id
            return None

        if decision.op == "supersede":
            # 新值取代旧值：旧记忆失效 -> 归档；新记忆携带链条
            old = self.core.get(decision.target_id)
            if old:
                old.invalid_at = now
                old.metadata["superseded_by"] = chunk.id
                old.review_note = "superseded"
                self.core.remove(old.id)
                if self.query_planner:
                    self.query_planner.remove_chunk(old.id)
                self.forgotten.archive(old)

                chunk.parent_id = old.id
                if chunk.valid_at is None:
                    chunk.valid_at = now
            return None  # 新记忆按正常路径写入

        # ADD：惊奇度门控编码——越出乎意料的信息越值得记住
        if decision.rule_id == "R4_new_fact":
            chunk.importance = surprise_scaled_importance(chunk.importance, decision.surprise)
            chunk.metadata["encoding_surprise"] = round(decision.surprise, 4)
        return None

    def add_raw_memory(
        self,
        text: str,
        user_id: str = "default",
        session_id: Optional[str] = None,
        importance: float = 0.5,
        confidence: float = 0.8,
        check_duplicate: bool = True,
    ) -> str:
        """
        添加原始文本记忆（自动抽取流水线）

        流程：raw_text -> extract_entities -> check_duplicate -> persist

        Args:
            text: 原始文本
            user_id: 用户ID
            session_id: 会话ID
            importance: 重要性（0-1）
            confidence: 置信度（0-1）
            check_duplicate: 是否检查重复

        Returns:
            记忆ID
        """
        from core.entity_extractor import EntityExtractor

        extractor = EntityExtractor()

        # 步骤0：PII 检测与脱敏
        if self.pii_handler and self.pii_handler.has_pii(text):
            pii_types = list(self.pii_handler.get_pii_types(text))
            text = self.pii_handler.redact(text)

            # 记录 PII 检测审计
            if self.audit_logger:
                import hashlib
                text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
                self.audit_logger.log_pii_detection(
                    text_hash=text_hash,
                    pii_types=pii_types,
                    action="redact",
                )

        # 步骤1：抽取实体（优先使用 LLM）
        if self.llm_fn:
            try:
                extracted = self._llm_extract(text)
                persons = set(extracted.get("persons", []))

                # location 可能是字符串或列表
                location_raw = extracted.get("location")
                if isinstance(location_raw, list):
                    location = location_raw[0] if location_raw else None
                else:
                    location = location_raw

                time_relative = extracted.get("time")
                if isinstance(time_relative, list):
                    time_relative = time_relative[0] if time_relative else None
                time_context = None
                time_absolute = None
                topics = set(extracted.get("topics", []))
                keywords = set(extracted.get("keywords", []))
                emotion_valence = float(extracted.get("emotion_valence", 0.0))
                emotion_intensity = float(extracted.get("emotion_intensity", 0.0))

                # 如果 LLM 返回了 importance，使用它
                if "importance" in extracted:
                    importance = float(extracted["importance"])

                print(f"[LLM] 抽取成功: persons={persons}, location={location}, topics={topics}")
            except Exception as e:
                print(f"[LLM] 抽取失败，回退到规则抽取: {e}")
                persons = extractor.extract_persons(text)
                location = extractor.extract_location(text)
                time_absolute, time_relative, time_context = extractor.extract_time(text)
                topics = extractor.extract_topics(text)
                keywords = extractor.extract_keywords(text)
                emotion_valence, emotion_intensity = extractor.extract_emotion(text)
        else:
            persons = extractor.extract_persons(text)
            location = extractor.extract_location(text)
            time_absolute, time_relative, time_context = extractor.extract_time(text)
            topics = extractor.extract_topics(text)
            keywords = extractor.extract_keywords(text)
            emotion_valence, emotion_intensity = extractor.extract_emotion(text)

        # 步骤2：检查重复（如果启用）
        if check_duplicate:
            existing = self._find_similar_memory(text)
            if existing:
                # 更新现有记忆的访问统计
                existing.access()
                existing.successful_recall()
                existing.version += 1
                existing.updated_at = time.time()
                self.core._store.put(existing)

                # 记录更新审计
                if self.audit_logger:
                    self.audit_logger.log_memory_access(
                        chunk_id=existing.id,
                        user_id=user_id,
                        action="update",
                        details={"reason": "duplicate_detected"},
                    )

                return existing.id

        # 步骤3：创建新记忆
        chunk = MemoryChunk(
            content=text,
            memory_type=MemoryType.INTERACTION,

            persons=persons,
            location=location,
            time_absolute=time_absolute,
            time_relative=time_relative,
            time_context=time_context,
            topics=topics,
            keywords=keywords,

            emotion_valence=emotion_valence,
            emotion_intensity=emotion_intensity,

            importance=importance,

            # 新增字段
            source="system_extract",
            confidence=confidence,
            user_id=user_id,
            session_id=session_id,
        )

        # 步骤4：存储
        self.core.add(chunk)
        if self.query_planner:
            self.query_planner.add_chunk(chunk)

        # 记录创建审计
        if self.audit_logger:
            self.audit_logger.log_memory_access(
                chunk_id=chunk.id,
                user_id=user_id,
                action="create",
                details={"source": "system_extract"},
            )

        return chunk.id

    def _find_similar_memory(self, text: str, threshold: float = 0.7):
        """
        查找相似记忆（简单实现：基于关键词重叠）

        Args:
            text: 要查找的文本
            threshold: 相似度阈值

        Returns:
            相似的 MemoryChunk 或 None
        """
        from core.entity_extractor import EntityExtractor

        extractor = EntityExtractor()
        new_keywords = extractor.extract_keywords(text)

        if not new_keywords:
            return None

        # 遍历核心层记忆（get_all 返回 Dict[str, MemoryChunk]）
        for chunk in self.core._store.get_all().values():
            existing_keywords = chunk.keywords
            if not existing_keywords:
                continue

            # 计算关键词重叠率
            overlap = len(new_keywords & existing_keywords)
            total = len(new_keywords | existing_keywords)
            similarity = overlap / total if total > 0 else 0

            if similarity >= threshold:
                return chunk

        return None

    def _llm_extract(self, text: str) -> dict:
        """
        使用 LLM 从文本中抽取结构化信息

        Args:
            text: 输入文本

        Returns:
            抽取结果字典
        """
        prompt = f"""请从以下文本中提取结构化信息，返回 JSON 格式。

要求：
1. persons: 涉及的人物列表（名字）
2. location: 地点（如果有）
3. time: 时间描述（如果有）
4. topics: 主题标签列表（2-5个）
5. keywords: 关键词列表（3-8个）
6. emotion_valence: 情绪效价（-1.0到1.0，负面到正面）
7. emotion_intensity: 情绪强度（0.0到1.0）
8. importance: 重要性（0.0到1.0）

文本：{text}

请直接返回 JSON，不要有其他内容："""

        # 传输错误必须向上传播：add_raw_memory 的 except 分支会回退到
        # 规则抽取。在这里吞掉错误并返回 {} 会让回退路径永远不触发，
        # 产出没有任何检索线索的记忆。
        result = self.llm_fn(prompt)
        print(f"[LLM] 原始返回: {result[:200]}...")

        # 尝试解析 JSON
        import re

        # 移除 markdown 代码块标记
        result = re.sub(r'```json\s*', '', result)
        result = re.sub(r'```\s*', '', result)

        # 尝试直接解析
        try:
            return json.loads(result.strip())
        except json.JSONDecodeError:
            pass

        # 提取 JSON 部分（支持嵌套）
        try:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

            # 尝试找到第一个 { 和最后一个 }
            start = result.find('{')
            end = result.rfind('}')
            if start != -1 and end != -1:
                return json.loads(result[start:end+1])
        except json.JSONDecodeError:
            pass

        # 完全无法解析同样按失败处理，触发调用方的规则抽取回退
        raise ValueError(f"LLM 返回无法解析为 JSON: {result[:100]}...")

    def retrieve(
        self,
        query: str,
        allow_forgotten: bool = True,
        user_id: str = "default",
    ) -> ReconstructionResult:
        """
        检索记忆

        返回重组后的记忆
        """
        start_time = time.time()
        result = self.retrieval.retrieve(query, allow_forgotten)
        latency = time.time() - start_time

        # 记录检索审计
        if self.audit_logger:
            import hashlib
            query_hash = hashlib.md5(query.encode()).hexdigest()[:16]
            self.audit_logger.log_retrieval(
                user_id=user_id,
                query_hash=query_hash,
                num_results=len(result.chunks),
                latency=latency,
            )

        return result
    
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
            # 锚点足够强的唤醒记忆提升回核心层（遗忘-唤醒闭环）。
            # 必须使用返回值：被提升条目已替换为核心层新对象
            chunks = self.retrieval.promote_woken(forgotten_results)
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

    def create_runtime(
        self,
        name: str = "cognitive-runtime-agent",
        auto_consolidate: bool = True,
        max_steps: int = 4,
    ):
        """Create a multi-step CognitiveRuntime with the default local planner."""
        from core.cognitive_runtime import CognitiveRuntime, CognitiveRuntimeConfig

        agent = self.create_agent(name=name, auto_consolidate=auto_consolidate)
        return CognitiveRuntime(agent, config=CognitiveRuntimeConfig(max_steps=max_steps))

    def create_openai_runtime(
        self,
        name: str = "openai-compatible-runtime-agent",
        auto_consolidate: bool = True,
        env_file: Optional[str] = ".env",
        max_steps: int = 4,
        **api_overrides: Any,
    ):
        """Create a multi-step CognitiveRuntime backed by an OpenAI-compatible API."""
        from core.cognitive_runtime import CognitiveRuntime, CognitiveRuntimeConfig, LLMRuntimeFinalizer

        agent = self.create_openai_agent(
            name=name,
            auto_consolidate=auto_consolidate,
            env_file=env_file,
            **api_overrides,
        )
        llm = getattr(agent.planner, "llm", None)
        finalizer = LLMRuntimeFinalizer(llm) if llm is not None else None
        return CognitiveRuntime(agent, config=CognitiveRuntimeConfig(max_steps=max_steps), finalizer=finalizer)
    
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
                # 降级即离开主动检索：同步从混合检索索引移除
                if self.query_planner:
                    self.query_planner.remove_chunk(chunk.id)
        
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
        os.makedirs(self.data_dir, exist_ok=True)

        # 保存核心层和伪遗忘层（委托给 store）
        self.core.save()
        self.forgotten.save()
        
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
        core_loaded = self.core.load()
        forgotten_loaded = self.forgotten.load()

        # 重建混合检索索引（BM25/Dense 索引只存在于内存）
        if self.query_planner and core_loaded:
            self.query_planner.index_chunks(self.core.chunks)

        # 加载人格适应层
        persona_path = f"{self.data_dir}/persona.json"
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
        attention_path = f"{self.data_dir}/attention.json"
        attention_loaded = False
        if os.path.exists(attention_path):
            try:
                with open(attention_path, 'r', encoding='utf-8') as f:
                    attention_data = json.load(f)
                self.attention = AttentionOS.from_dict(attention_data)
                attention_loaded = True
            except Exception:
                pass

        cognitive_path = f"{self.data_dir}/cognitive_state.json"
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
