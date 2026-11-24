"""Memory-R1 协议适配器 —— 将学习型记忆控制器嫁接到确定性认知内核。

源码级对齐 ACL 2026 `Memory-R1`（`yansikuan/memory-r1`）的核心双阶段机制，
同时用 `new-agent-memory` 的双时态链、ACT-R 激活与伪遗忘层补齐其原地覆写缺陷：

1. **Stage 1a · 原子事实抽取（Fact Extraction）**：
   从对话轮抽取自包含事实/偏好列表 `{"facts": [...]}`，替代脆弱的手写正则，
   天然处理同义改述、隐含状态与中英混杂表达。
2. **Stage 1b · 整数 ID 映射 + 四动作仲裁（Memory Manager）**：
   对每条新事实检索核心层 top-k 相关旧记忆，按 `Memory-R1` 防 UUID 幻觉设计
   将真实 `chunk.id` 临时映射为 `"0", "1", "2"...`，由模型（微调后的
   Qwen2.5-7B / LLaMA-3.1-8B vLLM 端点，或任意 OpenAI 兼容模型）输出：
   - `ADD`    -> 写入核心层（享受惊奇度编码与混合检索索引）
   - `UPDATE` -> **不原地覆写销毁历史**，而是走双时态取代：旧记忆置
                 `invalid_at` 并归档进 `ForgottenLayer`，新记忆 `parent_id`
                 指向旧记忆（现在时查新值，过去时仍能沿链追溯旧值！）
   - `DELETE` -> **不物理硬删**，置 `invalid_at` + `retracted_by="memory_r1"`
                 归档进伪遗忘层，全程可审计、可逆
   - `NONE`   -> 触发 `core.access()`（受 60s Pavlik 间隔效应去抖保护）
3. **Stage 2 · 回答前记忆蒸馏（Answer-Agent Memory Distillation）**：
   在把候选记忆喂给回答模型前，剔除检索带入的边缘噪声条目。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from core.entity_extractor import EntityExtractor
from core.llm_planner import LLMCallable, LLMError, OpenAICompatibleChatClient, OpenAICompatibleConfig
from core.weight_system import MemoryType
from memory_chunk import MemoryChunk


FACT_EXTRACTION_SYSTEM_PROMPT = """You are a Personal Information & Memory Organizer.
Extract self-contained, durable facts, preferences, plans, or workflows stated by the user.
Ignore greetings, transient chit-chat, questions, or anything the user explicitly asks NOT to remember.
Record facts in the SAME language as the user input.

Return ONLY valid JSON with this exact shape:
{"facts": ["fact 1", "fact 2"]}
If nothing durable should be stored, return:
{"facts": []}"""


MEMORY_R1_UPDATE_SYSTEM_PROMPT = """You are a Memory-R1 smart memory manager controlling an agent's long-term memory bank.
Compare the newly extracted facts with the existing memories and output structured operations:

- "ADD": New information not covered by any existing memory (assign a new integer id).
- "UPDATE": Consolidates, refines, or replaces an existing memory (MUST reuse that existing memory's exact integer id; merge complementary multi-session details or update changed values).
- "DELETE": Existing memory is explicitly contradicted or retracted without a replacement value (MUST reuse that existing memory's exact integer id).
- "NONE": Existing memory is unchanged or already captures the duplicate fact (keep its exact integer id).

Return ONLY valid JSON with this exact structure:
{
  "memory": [
    {
      "id": "0",
      "text": "updated or kept memory content",
      "event": "ADD | UPDATE | DELETE | NONE",
      "memory_type": "fact | preference | procedure | story",
      "old_memory": "previous text when event is UPDATE (optional)"
    }
  ]
}"""


MEMORY_DISTILL_SYSTEM_PROMPT = """You are the Memory-R1 distillation filter.
Given a user query and numbered candidate memories, select ONLY the integer IDs of memories that provide relevant evidence for answering the query.
Return ONLY valid JSON: {"selected_ids": ["0", "2"]}"""


def _extract_json_object(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object")
    return data


def _parse_memory_type(raw: Any) -> MemoryType:
    val = str(raw or "fact").strip().lower()
    mapping = {
        "fact": MemoryType.FACT,
        "preference": MemoryType.PREFERENCE,
        "procedure": MemoryType.PROCEDURE,
        "story": MemoryType.STORY,
        "idea": MemoryType.IDEA,
        "interaction": MemoryType.INTERACTION,
    }
    return mapping.get(val, MemoryType.FACT)


@dataclass
class MemoryR1Op:
    """单条经审计执行的 Memory-R1 记忆操作记录。"""

    event: str                       # ADD / UPDATE / DELETE / NONE
    text: str
    chunk_id: Optional[str] = None   # 写入或命中的真实 chunk id
    replaced_id: Optional[str] = None
    memory_type: str = "fact"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event,
            "text": self.text,
            "chunk_id": self.chunk_id,
            "replaced_id": self.replaced_id,
            "memory_type": self.memory_type,
        }


@dataclass
class MemoryR1TurnReport:
    """一轮对话经 Memory-R1 管理器处理后的完整可解释报告。"""

    extracted_facts: List[str] = field(default_factory=list)
    candidate_count: int = 0
    operations: List[MemoryR1Op] = field(default_factory=list)
    fallback_used: bool = False
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "extracted_facts": list(self.extracted_facts),
            "candidate_count": self.candidate_count,
            "operations": [op.to_dict() for op in self.operations],
            "fallback_used": self.fallback_used,
            "error": self.error,
        }


class MemoryR1Manager:
    """把 Memory-R1（RL 记忆控制器）协议对接到 HumanLikeMemorySystem。

    既可连接 vLLM 部署的 Memory-R1 微调权重（Qwen2.5-7B / LLaMA-3.1-8B），
    也可连接任意本地 Ollama / OpenAI 兼容模型；甚至支持为抽取器与仲裁器分别指定不同模型。
    """

    def __init__(
        self,
        memory_system: Any,
        manager_llm: LLMCallable,
        extractor_llm: Optional[LLMCallable] = None,
        candidate_top_k: int = 5,
    ):
        self.memory = memory_system
        self.manager_llm = manager_llm
        self.extractor_llm = extractor_llm or manager_llm
        self.candidate_top_k = max(1, int(candidate_top_k))
        self._entity_extractor = EntityExtractor()
        self.last_report: Optional[MemoryR1TurnReport] = None

    @classmethod
    def from_openai_compatible(
        cls,
        memory_system: Any,
        api_config: Optional[OpenAICompatibleConfig] = None,
        extractor_config: Optional[OpenAICompatibleConfig] = None,
        candidate_top_k: int = 5,
        **overrides: Any,
    ) -> "MemoryR1Manager":
        cfg = api_config or OpenAICompatibleConfig.from_env(**overrides)
        manager_client = OpenAICompatibleChatClient(cfg)
        extractor_client = (
            OpenAICompatibleChatClient(extractor_config)
            if extractor_config is not None
            else manager_client
        )
        return cls(
            memory_system=memory_system,
            manager_llm=manager_client,
            extractor_llm=extractor_client,
            candidate_top_k=candidate_top_k,
        )

    def _call(self, llm: LLMCallable, system_prompt: str, user_prompt: str) -> str:
        complete = getattr(llm, "complete", None)
        if callable(complete):
            try:
                return complete(user_prompt, system_prompt=system_prompt, emit_tokens=False)
            except TypeError:
                return complete(user_prompt, system_prompt=system_prompt)
        return llm(f"{system_prompt}\n\n{user_prompt}")

    def extract_facts(self, user_message: str, assistant_message: str = "") -> List[str]:
        """Stage 1a：从对话轮抽取值得长期保存的原子事实。"""
        text = (user_message or "").strip()
        if not text:
            return []
        convo = f"User: {text}"
        if assistant_message:
            convo += f"\nAssistant: {assistant_message.strip()}"

        raw = self._call(
            self.extractor_llm,
            FACT_EXTRACTION_SYSTEM_PROMPT,
            f"Conversation:\n{convo}",
        )
        data = _extract_json_object(raw)
        facts = data.get("facts") or []
        if not isinstance(facts, list):
            return []
        return [str(f).strip() for f in facts if isinstance(f, str) and str(f).strip()]

    def _collect_candidates(self, facts: Sequence[str]) -> List[MemoryChunk]:
        """为待仲裁的新事实搜集核心层候选旧记忆（去重保序）。"""
        seen: Dict[str, MemoryChunk] = {}
        for fact in facts:
            if hasattr(self.memory, "retrieve"):
                try:
                    res = self.memory.retrieve(fact, limit=self.candidate_top_k)
                    for ch in getattr(res, "chunks", []) or []:
                        if ch.id in self.memory.core.chunks and not ch.metadata.get("conversation_archive"):
                            seen.setdefault(ch.id, ch)
                except Exception:
                    pass
        # 若混合检索尚未命中（例如短库或纯改述），补充最近活跃的受管核心记忆兜底供模型比对
        if len(seen) < self.candidate_top_k:
            recent = sorted(
                (
                    c for c in self.memory.core.chunks.values()
                    if not c.metadata.get("conversation_archive")
                    and c.invalid_at is None
                ),
                key=lambda c: c.updated_at,
                reverse=True,
            )
            for ch in recent:
                if len(seen) >= self.candidate_top_k * 2:
                    break
                seen.setdefault(ch.id, ch)
        return list(seen.values())

    def ingest_turn(self, user_message: str, assistant_message: str = "") -> MemoryR1TurnReport:
        """执行完整 Memory-R1 提取 -> 整数映射仲裁 -> 双时态落库流水线。"""
        report = MemoryR1TurnReport()
        self.last_report = report

        facts = self.extract_facts(user_message, assistant_message)
        report.extracted_facts = facts
        if not facts:
            return report

        candidates = self._collect_candidates(facts)
        report.candidate_count = len(candidates)

        # Memory-R1 关键防幻觉技巧：把真实 UUID 映射为紧凑整数字符串 "0", "1", ...
        int_to_uuid: Dict[str, str] = {}
        old_memory_view: List[Dict[str, str]] = []
        for idx, chunk in enumerate(candidates):
            short_id = str(idx)
            int_to_uuid[short_id] = chunk.id
            old_memory_view.append({"id": short_id, "text": chunk.content})

        user_prompt = (
            "Current memory bank:\n"
            f"```json\n{json.dumps(old_memory_view, ensure_ascii=False, indent=2)}\n```\n\n"
            "Newly extracted facts:\n"
            f"```json\n{json.dumps(facts, ensure_ascii=False, indent=2)}\n```"
        )
        raw_decision = self._call(self.manager_llm, MEMORY_R1_UPDATE_SYSTEM_PROMPT, user_prompt)
        parsed = _extract_json_object(raw_decision)
        items = parsed.get("memory") or []
        if not isinstance(items, list):
            return report

        for entry in items:
            if not isinstance(entry, Mapping):
                continue
            event = str(entry.get("event") or "NONE").strip().upper()
            text = str(entry.get("text") or "").strip()
            short_id = str(entry.get("id") if entry.get("id") is not None else "").strip()
            mem_type = _parse_memory_type(entry.get("memory_type"))

            if event == "ADD":
                if not text:
                    continue
                new_id = self._apply_add(text, mem_type)
                report.operations.append(
                    MemoryR1Op(event="ADD", text=text, chunk_id=new_id, memory_type=mem_type.value)
                )
            elif event == "UPDATE":
                target_uuid = int_to_uuid.get(short_id)
                if not target_uuid or not text:
                    continue
                new_id = self._apply_bitemporal_update(target_uuid, text, mem_type)
                if new_id:
                    report.operations.append(
                        MemoryR1Op(
                            event="UPDATE",
                            text=text,
                            chunk_id=new_id,
                            replaced_id=target_uuid,
                            memory_type=mem_type.value,
                        )
                    )
            elif event == "DELETE":
                target_uuid = int_to_uuid.get(short_id)
                if not target_uuid:
                    continue
                if self._apply_soft_delete(target_uuid):
                    report.operations.append(
                        MemoryR1Op(event="DELETE", text=text, replaced_id=target_uuid)
                    )
            elif event in {"NONE", "NOOP"}:
                target_uuid = int_to_uuid.get(short_id)
                if target_uuid and self.memory.core.get(target_uuid) is not None:
                    # 复述强化走核心层 60s Pavlik 去抖访问
                    self.memory.core.access(target_uuid)
                    report.operations.append(
                        MemoryR1Op(event="NONE", text=text, chunk_id=target_uuid)
                    )

        return report

    def _apply_add(self, text: str, mem_type: MemoryType) -> Optional[str]:
        entities = self._entity_extractor.extract_all(text)
        return self.memory.add_memory(
            content=text,
            memory_type=mem_type,
            persons=list(entities["persons"]) or None,
            topics=list(entities["topics"]) or None,
            keywords=list(entities["keywords"]) or None,
            time_absolute=entities["time_absolute"],
            time_relative=entities["time_relative"],
            time_context=entities["time_context"],
            location=entities["location"],
            emotion_valence=entities["emotion_valence"],
            emotion_intensity=entities["emotion_intensity"],
            importance=0.55,
            confidence=0.85,
            source="memory_r1",
            metadata={"memr1_event": "ADD"},
        )

    def _apply_bitemporal_update(
        self,
        target_id: str,
        new_text: str,
        mem_type: MemoryType,
    ) -> Optional[str]:
        """双时态取代：保留 Memory-R1 跨表述语义合并能力，同时避免原地覆写抹除历史。"""
        old = self.memory.core.get(target_id)
        if old is None:
            return None

        now = time.time()
        entities = self._entity_extractor.extract_all(new_text)
        if self.memory.pii_handler and self.memory.pii_handler.has_pii(new_text):
            new_text = self.memory.pii_handler.redact(new_text)

        merged_persons = set(old.persons) | set(entities["persons"])
        merged_topics = set(old.topics) | set(entities["topics"])
        keywords = set(entities["keywords"])
        if self.memory.pii_handler:
            keywords = {k for k in keywords if not self.memory.pii_handler.has_pii(str(k))}

        new_chunk = MemoryChunk(
            content=new_text,
            memory_type=mem_type or old.memory_type,
            user_id=old.user_id,
            session_id=old.session_id,
            source="memory_r1",
            confidence=0.85,
            valid_at=now,
            parent_id=old.id,
            version=old.version + 1,
            time_absolute=entities["time_absolute"] or old.time_absolute,
            time_relative=entities["time_relative"],
            time_context=entities["time_context"],
            location=entities["location"] or old.location,
            persons=merged_persons,
            topics=merged_topics,
            keywords=keywords,
            emotion_valence=entities["emotion_valence"],
            emotion_intensity=entities["emotion_intensity"],
            importance=max(old.importance, 0.55),
            metadata={
                "memr1_event": "UPDATE",
                "supersedes": old.id,
                "write_decision": {
                    "op": "supersede",
                    "rule_id": "MemoryR1_RL_UPDATE",
                    "target_id": old.id,
                },
            },
        )

        # 旧事实置失效时间戳并归档至伪遗忘层（过去时查询仍可沿 parent_id 链追溯）
        old.invalid_at = now
        old.updated_at = now
        old.metadata["superseded_by"] = new_chunk.id
        self.memory.core.remove(old.id)
        self.memory.forgotten.archive(old)
        if getattr(self.memory, "query_planner", None):
            self.memory.query_planner.remove_chunk(old.id)

        new_id = self.memory.core.add(new_chunk)
        if getattr(self.memory, "query_planner", None):
            self.memory.query_planner.add_chunk(new_chunk)
        return new_id

    def _apply_soft_delete(self, target_id: str) -> bool:
        """可逆软删除：旧记忆归档至伪遗忘层并打上失效标记，而非物理销毁。"""
        old = self.memory.core.get(target_id)
        if old is None:
            return False
        now = time.time()
        old.invalid_at = now
        old.updated_at = now
        old.metadata["retracted_by"] = "memory_r1"
        self.memory.core.remove(old.id)
        self.memory.forgotten.archive(old)
        if getattr(self.memory, "query_planner", None):
            self.memory.query_planner.remove_chunk(old.id)
        return True

    def distill_chunks(self, query: str, chunks: Sequence[MemoryChunk]) -> List[MemoryChunk]:
        """Stage 2（Answer Agent 记忆蒸馏）：从候选记忆中筛选真正支撑回答的子集。"""
        chunk_list = list(chunks)
        if len(chunk_list) <= 1:
            return chunk_list

        numbered = [{"id": str(i), "text": c.content} for i, c in enumerate(chunk_list)]
        prompt = (
            f"Question: {query}\n\n"
            f"Candidate memories:\n{json.dumps(numbered, ensure_ascii=False, indent=2)}"
        )
        try:
            raw = self._call(self.manager_llm, MEMORY_DISTILL_SYSTEM_PROMPT, prompt)
            parsed = _extract_json_object(raw)
            raw_ids = parsed.get("selected_ids")
            if not isinstance(raw_ids, list):
                return chunk_list
            keep_indices = {str(x).strip() for x in raw_ids}
            selected = [ch for idx, ch in enumerate(chunk_list) if str(idx) in keep_indices]
            return selected or chunk_list
        except (LLMError, ValueError, TypeError, json.JSONDecodeError):
            return chunk_list
