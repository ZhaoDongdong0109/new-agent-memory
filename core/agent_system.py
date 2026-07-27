"""A minimal embodied agent loop built on memory and attention.

The loop is intentionally small and inspectable:

Observe -> Focus -> Decide -> Act -> Evaluate -> Remember -> Consolidate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
import json
import re
import time
import uuid

from core.attention_system import FocusItem, FocusWorkspace
from core.weight_system import MemoryType


def _now() -> float:
    return time.time()


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


# 明确的"不要写入记忆"表述；命中任意一条时视为用户否定了记忆写入
MEMORY_WRITE_NEGATIVE_MARKERS = [
    "do not remember",
    "don't remember",
    "do not save",
    "don't save",
    "do not store",
    "don't store",
    "do not record",
    "don't record",
    "不要记住",
    "不要保存",
    "不要存储",
    "不要记录",
    "别记住",
    "别保存",
    "别记录",
    "不保存",
    "不用保存",
]

# 明确的"请写入记忆"表述
MEMORY_WRITE_MARKERS = [
    "remember",
    "save this",
    "store this",
    "record this",
    "记住",
    "保存",
    "存储",
    "记录",
    "保存成记忆",
    "写入记忆",
]


def has_memory_write_intent(text: str) -> bool:
    """判断用户是否明确要求写入记忆；出现否定表述时返回 False。

    default_planner 和 LLMPlanner 共用这一判断，避免
    "please do NOT remember this secret" 这类否定请求被误存。
    """
    lowered = text.lower()
    if any(marker in lowered for marker in MEMORY_WRITE_NEGATIVE_MARKERS):
        return False
    return any(marker in lowered for marker in MEMORY_WRITE_MARKERS)


def tool_name_mentioned(tool_name: str, text: str) -> bool:
    """按词边界匹配工具名，避免 'search' 命中 'research' 这类子串误触发。

    边界只针对 ASCII 字母/数字/下划线，因此中文上下文里的
    "调用search工具" 仍然可以命中。
    """
    pattern = r"(?<![0-9a-z_])" + re.escape(tool_name.lower()) + r"(?![0-9a-z_])"
    return re.search(pattern, text.lower()) is not None


@dataclass
class Observation:
    """What the agent can perceive from its environment."""

    content: str
    source: str = "user"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "source": self.source,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        return cls(
            content=data.get("content", ""),
            source=data.get("source", "user"),
            metadata=dict(data.get("metadata", {})),
            timestamp=float(data.get("timestamp", _now())),
        )


@dataclass
class AgentAction:
    """An action selected by the agent."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "rationale": self.rationale,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentAction":
        return cls(
            name=data.get("name", "respond"),
            arguments=dict(data.get("arguments", {})),
            rationale=data.get("rationale", ""),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ActionResult:
    """The consequence of an action."""

    success: bool
    output: str = ""
    cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "cost": self.cost,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionResult":
        return cls(
            success=bool(data.get("success", False)),
            output=data.get("output", ""),
            cost=float(data.get("cost", 0.0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ExperienceEpisode:
    """A full action loop that can become future memory."""

    goal: str
    observation: Observation
    action: AgentAction
    result: ActionResult
    reward: float
    lesson: str = ""
    next_policy: str = ""
    focus_context: str = ""
    id: str = field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:10]}")
    created_at: float = field(default_factory=_now)
    # 0 表示尚未固化；非 0 表示已固化过，重复调用 consolidate 时跳过
    consolidated_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "observation": self.observation.to_dict(),
            "action": self.action.to_dict(),
            "result": self.result.to_dict(),
            "reward": self.reward,
            "lesson": self.lesson,
            "next_policy": self.next_policy,
            "focus_context": self.focus_context,
            "created_at": self.created_at,
            "consolidated_at": self.consolidated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperienceEpisode":
        return cls(
            id=data.get("id", f"exp_{uuid.uuid4().hex[:10]}"),
            goal=data.get("goal", ""),
            observation=Observation.from_dict(data.get("observation", {})),
            action=AgentAction.from_dict(data.get("action", {})),
            result=ActionResult.from_dict(data.get("result", {})),
            reward=float(data.get("reward", 0.0)),
            lesson=data.get("lesson", ""),
            next_policy=data.get("next_policy", ""),
            focus_context=data.get("focus_context", ""),
            created_at=float(data.get("created_at", _now())),
            consolidated_at=float(data.get("consolidated_at", 0.0)),
        )

    def to_memory_text(self) -> str:
        return (
            f"Goal: {self.goal}\n"
            f"Observation: {self.observation.content}\n"
            f"Action: {self.action.name} {self.action.arguments}\n"
            f"Result: {'success' if self.result.success else 'failure'} - {self.result.output}\n"
            f"Lesson: {self.lesson}\n"
            f"Next policy: {self.next_policy}"
        )

    def to_compact_memory_text(self) -> str:
        """运行时回合的紧凑摘要，避免把整段 runtime 脚手架写进长期记忆。"""
        original_task = str(self.observation.metadata.get("original_task", "")).strip()
        outcome = " ".join(str(self.result.output).split())
        if len(outcome) > 240:
            outcome = "..." + outcome[-237:]
        return (
            f"Goal: {self.goal}\n"
            f"Original task: {original_task}\n"
            f"Action: {self.action.name}\n"
            f"Outcome: {'success' if self.result.success else 'failure'} - {outcome}\n"
            f"Lesson: {self.lesson}"
        )


class ExperienceLayer:
    """Stores embodied episodes and consolidates them into memory."""

    def __init__(self, episodes: Optional[List[ExperienceEpisode]] = None):
        self.episodes = episodes or []

    def add(self, episode: ExperienceEpisode) -> str:
        self.episodes.append(episode)
        return episode.id

    def recent(self, limit: int = 10) -> List[ExperienceEpisode]:
        return self.episodes[-limit:]

    def consolidate(self, memory_system: Any, limit: int = 5) -> List[str]:
        """
        Turn recent high-signal episodes into durable memory and procedures.

        The memory_system is duck-typed so this layer stays independent from main.py.
        """
        created: List[str] = []
        for episode in self.recent(limit):
            # 幂等：已固化过的回合直接跳过，避免重复写入记忆和程序
            if episode.consolidated_at:
                continue
            # 低信号回合（含失败）不写入长期记忆；_derive_lesson 永远非空，
            # 因此这里只按 reward 判断
            if episode.reward < 0.35:
                episode.consolidated_at = _now()
                continue

            if episode.observation.metadata.get("runtime"):
                # runtime 脚手架观测只固化紧凑摘要，不存原始多 KB 上下文
                content = episode.to_compact_memory_text()
            else:
                content = episode.to_memory_text()

            memory_id = memory_system.add_memory(
                content=content,
                memory_type=MemoryType.STORY,
                topics=["experience", "agent", episode.action.name],
                importance=_clamp(0.35 + episode.reward * 0.45),
                metadata={"experience_id": episode.id, "kind": "episode"},
            )
            created.append(memory_id)

            if episode.result.success and episode.next_policy:
                title = f"Learned policy: {episode.action.name}"
                steps = [episode.next_policy]
                existing = self._find_existing_procedure(memory_system, title, steps)
                if existing is not None:
                    # 已学过同样的策略：只记一次使用，不再新增重复程序
                    self._record_procedure_use(memory_system, existing)
                    created.append(existing.id)
                else:
                    procedure = memory_system.add_procedure(
                        title=title,
                        steps=steps,
                        triggers=[episode.action.name, "experience", "success"],
                        importance=_clamp(0.45 + episode.reward * 0.4),
                        confidence=_clamp(0.45 + episode.reward * 0.4),
                    )
                    created.append(procedure.id)

            episode.consolidated_at = _now()

        return created

    @staticmethod
    def _find_existing_procedure(memory_system: Any, title: str, steps: List[str]) -> Optional[Any]:
        """在注意力层里查找 (title, steps) 完全相同的程序记忆。"""
        attention = getattr(memory_system, "attention", None)
        for procedure in getattr(attention, "procedures", None) or []:
            if getattr(procedure, "title", None) == title and list(getattr(procedure, "steps", [])) == list(steps):
                return procedure
        return None

    @staticmethod
    def _record_procedure_use(memory_system: Any, procedure: Any):
        record = getattr(memory_system, "record_procedure_use", None)
        if callable(record):
            record(procedure.id, True)
        elif hasattr(procedure, "record_use"):
            procedure.record_use(True)

    def to_dict(self) -> Dict[str, Any]:
        return {"episodes": [episode.to_dict() for episode in self.episodes]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperienceLayer":
        return cls([ExperienceEpisode.from_dict(item) for item in data.get("episodes", [])])


ToolHandler = Callable[[Dict[str, Any]], ActionResult]


@dataclass
class AgentTool:
    """A callable digital body part."""

    name: str
    description: str
    handler: ToolHandler
    cost: float = 0.0
    # 参数 JSON Schema：没有它，模型只能猜参数名（代码里曾到处是
    # 多别名兜底）。有 schema 的工具在提示里自带参数说明。
    parameters: Optional[Dict[str, Any]] = None

    def run(self, arguments: Dict[str, Any]) -> ActionResult:
        result = self.handler(arguments)
        result.cost += self.cost
        return result


class ToolRegistry:
    """Registry of actions the agent is allowed to take."""

    def __init__(self):
        self.tools: Dict[str, AgentTool] = {}

    def register(self, tool: AgentTool):
        self.tools[tool.name] = tool

    def run(self, action: AgentAction) -> ActionResult:
        tool = self.tools.get(action.name)
        if not tool:
            return ActionResult(False, f"Unknown tool: {action.name}", metadata={"missing_tool": action.name})
        try:
            return tool.run(action.arguments)
        except Exception as exc:
            return ActionResult(False, f"{type(exc).__name__}: {exc}")

    def describe(self) -> List[Dict[str, Any]]:
        described = []
        for tool in self.tools.values():
            entry: Dict[str, Any] = {
                "name": tool.name,
                "description": tool.description,
                "cost": tool.cost,
            }
            if tool.parameters:
                entry["parameters"] = tool.parameters
            described.append(entry)
        return described


Planner = Callable[[Observation, FocusWorkspace, ToolRegistry], AgentAction]
Evaluator = Callable[[Observation, AgentAction, ActionResult], float]
SynthesizedResponse = Union[ActionResult, str, None]
ResponseSynthesizer = Callable[[Observation, FocusWorkspace, AgentAction, ActionResult], SynthesizedResponse]


def default_planner(observation: Observation, workspace: FocusWorkspace, tools: ToolRegistry) -> AgentAction:
    """A safe planner that responds unless a tool trigger is obvious."""
    runtime_mode = bool(observation.metadata.get("runtime"))
    if runtime_mode:
        # runtime 脚手架文本会提到所有工具名，只允许根据原始任务触发工具，
        # 回复时也只带原始任务内容，避免把整段脚手架当成答案回显
        intent = str(observation.metadata.get("original_task") or observation.content)
    else:
        intent = observation.content

    for tool_name in tools.tools:
        if runtime_mode and tool_name in {"respond", "finish"}:
            continue
        if not tool_name_mentioned(tool_name, intent):
            continue
        if tool_name == "remember" and not has_memory_write_intent(intent):
            # 用户明确否定写入记忆（例如 "do not remember this secret"）时不触发
            continue
        return AgentAction(
            name=tool_name,
            arguments={"input": intent, "workspace": workspace.to_prompt_context()},
            rationale=f"Observation mentioned tool '{tool_name}'.",
        )

    return AgentAction(
        name="respond",
        arguments={
            "message": intent,
            "context": workspace.to_prompt_context(),
        },
        rationale="No explicit tool trigger; produce a contextual response.",
    )


def default_evaluator(observation: Observation, action: AgentAction, result: ActionResult) -> float:
    """Reward successful low-cost actions, penalize failures."""
    if not result.success:
        return 0.0
    cost_penalty = min(0.3, result.cost)
    return _clamp(0.7 - cost_penalty)


class _ExtractionProbe:
    """MemoryExtractor 期望 episode 形状；对话轮自动编码只有 observation"""

    def __init__(self, observation: Observation):
        self.id = f"turn_{uuid.uuid4().hex[:8]}"
        self.observation = observation
        self.action = None
        self.result = None
        self.reward = 0.0
        self.lesson = ""
        self.next_policy = ""
        self.goal = ""


class CognitiveAgent:
    """A first digital body for memory-driven agents."""

    def __init__(
        self,
        memory_system: Any,
        name: str = "cognitive-agent",
        planner: Planner = default_planner,
        evaluator: Evaluator = default_evaluator,
        response_synthesizer: Optional[ResponseSynthesizer] = None,
        experience_layer: Optional[ExperienceLayer] = None,
        auto_consolidate: bool = True,
        conversation_turns: int = 16,
        auto_extract: bool = True,
    ):
        from core.conversation import ConversationBuffer

        self.memory = memory_system
        self.name = name
        self.planner = planner
        self.evaluator = evaluator
        self.response_synthesizer = response_synthesizer
        self.experience = experience_layer or ExperienceLayer()
        self.tools = ToolRegistry()
        self.auto_consolidate = auto_consolidate
        # 会话工作记忆：多轮对话的连续性（溢出轮归档进长期记忆）
        self.conversation = ConversationBuffer(max_turns=conversation_turns)
        # 自动编码：每轮对话后从用户话语规则抽取事实进长期记忆
        # （走 add_memory 决策表 -> 取代链在聊天里真实生效）
        self.auto_extract = auto_extract
        self._extractor = None
        self._register_default_tools()

    def observe(self, content: str, source: str = "user", metadata: Optional[Dict[str, Any]] = None) -> Observation:
        return Observation(content=content, source=source, metadata=metadata or {})

    def run_turn(
        self,
        observation: Union[Observation, str],
        consolidate: Optional[bool] = None,
    ) -> ExperienceEpisode:
        if isinstance(observation, str):
            observation = self.observe(observation)

        if hasattr(self.memory, "observe_world"):
            self.memory.observe_world(observation)

        # 多轮上下文：把既有对话历史（不含本轮）交给规划提示——
        # 代词指代、"上面那个"从此有处可循。直接赋值而不是
        # setdefault：调用方復用旧 Observation 时，陈旧的历史块
        # 不得压过真实缓冲。
        if self.conversation.turns:
            observation.metadata["conversation"] = self.conversation.render_context()
        self.conversation.add("user", observation.content)

        workspace = self.build_workspace(observation)
        action = self.select_action(observation, workspace)
        episode = self.execute_action(observation, workspace, action, consolidate=consolidate)

        reply = episode.result.output if episode.result else ""
        if reply:
            self.conversation.add("assistant", reply)
        self._archive_conversation_overflow()
        if self.auto_extract:
            self._auto_encode(observation)
        # 历史块只服务于本轮提示；留在 metadata 里会随 episode
        # 持久化，30 轮聊天累计重复存储 50KB+（对抗审查实测）
        observation.metadata.pop("conversation", None)
        return episode

    def build_workspace(self, observation: Observation) -> FocusWorkspace:
        workspace = self.memory.focus(observation.content, include_forgotten=True)
        self._merge_production_recall(workspace, observation)
        if hasattr(self.memory, "attach_cognitive_context"):
            self.memory.attach_cognitive_context(workspace, tools=self.tools)
        return workspace

    def _merge_production_recall(self, workspace: FocusWorkspace, observation: Observation) -> None:
        """把生产检索管线的结果并入注意力工作区

        此前 agent 召回只走 focus() 的词元重叠打分，完全绕过
        BM25/RRF/双时态/时间窗/容量截断，且不记录访问——ACT-R
        频率效应与间隔效应对 agent 路径失效。这里补上主通路：
        retrieve() 命中的记忆按生产规则被"想起"（access 计数、
        共激活、取代链路由全部生效），再并入工作区供提示使用。

        三道闸（对抗审查实证补上）：
        1. 查询用原始任务而不是 runtime 脚手架——多 KB 样板文本
           做查询会给无关记忆刷访问计数、连虚假 Hebbian 边
        2. QUESTIONABLE（词汇覆盖警告/已过时标注）结果不并入
           提示——低相关命中不该穿上"记忆"的外衣喂给模型
        3. 归档的对话片段不并入——自己说过的话被捞回又被想起，
           自激励回环会让片段免于自然衰减并挤占工作区席位
        """
        if not hasattr(self.memory, "retrieve"):
            return
        query = str(observation.metadata.get("original_task") or observation.content)
        try:
            result = self.memory.retrieve(query, limit=5)
        except TypeError:
            # 旧签名（无 limit）容错
            result = self.memory.retrieve(query)
        except Exception:
            return
        if not getattr(result, "success", False):
            return
        review = getattr(result, "review_result", None)
        questionable = getattr(review, "value", review) == "questionable"
        merged = 0
        if not questionable:
            seen = {item.id for item in workspace.memories}
            for chunk in result.chunks:
                if chunk.id in seen:
                    continue
                if chunk.metadata.get("conversation_archive"):
                    continue
                workspace.memories.append(FocusItem(
                    id=chunk.id,
                    item_type="memory",
                    content=chunk.content,
                    score=round(float(result.confidence), 4),
                    reason="hybrid-retrieval",
                ))
                seen.add(chunk.id)
                merged += 1
        # 插在审计队首：LLM 提示只展示 audit[:8]，追加在尾部的
        # 记录（含"可能不相关"警告）对模型永远不可见
        workspace.audit.insert(0, {
            "stage": "production_recall",
            "hits": len(result.chunks),
            "merged": merged,
            "questionable": questionable,
            "confidence": round(float(result.confidence), 4),
            "path": getattr(result, "retrieval_path", ""),
            "note": getattr(result, "review_note", ""),
        })

    @staticmethod
    def _anchored_clauses(text: str, persons) -> str:
        """按逗号/分号切分子句，只保留含人物锚点的部分"""
        if not persons:
            return ""
        clauses = [c.strip() for c in re.split(r"[，,；;]", text) if c.strip()]
        if len(clauses) <= 1:
            return ""
        kept = [c for c in clauses if any(p in c for p in persons)]
        if not kept or len(kept) == len(clauses):
            return ""
        return "，".join(kept)

    def _archive_conversation_overflow(self) -> None:
        """溢出的旧对话轮归档进长期记忆（INTERACTION，自然衰减）

        带来源标记（conversation_archive）：这些片段可被显式检索
        （"上周我们聊过什么"），但不会被 _merge_production_recall
        自动捞回喂给提示——否则自己说过的话变成"记忆"又被想起，
        回环强化会让片段免于衰减。归档失败的轮次放回缓冲头部，
        下一轮重试，而不是无声丢失。
        """
        overflow = self.conversation.pop_overflow()
        if not overflow or not hasattr(self.memory, "add_memory"):
            return
        failed = []
        for turn in overflow:
            speaker = "用户" if turn.role == "user" else "助手"
            try:
                self.memory.add_memory(
                    content=f"对话片段（{speaker}）：{turn.content}",
                    importance=0.2,
                    source="conversation_archive",
                    metadata={"conversation_archive": True, "role": turn.role},
                )
            except Exception:
                failed.append(turn)
        if failed:
            self.conversation.turns[0:0] = failed

    # 疑问/祈使/暂态标记：这些话语不是可长期成立的事实陈述
    _NON_DECLARATIVE_MARKERS = (
        "?", "？", "吗", "呢", "怎么", "什么", "为什么", "哪", "几点", "多少",
        "帮我", "请你", "请帮", "给我", "麻烦", "help me", "please ",
    )
    # 自动编码的单句长度上限：整段多子句原话入库会让不相关的
    # 填充语参与取代判定（"周末我们约了饭"曾把住址事实误取代成
    # 养猫事实——对抗审查端到端复现）
    _AUTO_ENCODE_MAX_CHARS = 60

    def _auto_encode(self, observation: Observation) -> None:
        """每轮对话后自动编码：用户话语中的事实进长期记忆

        规则抽取（无 LLM 依赖）。四道闸（全部来自对抗审查实证）：
        1. 尊重否定意图："不要记住…"的内容一个字也不进库
        2. 只编码陈述句：疑问/祈使/暂态话语不是事实
        3. 单句长度上限：多子句原话的填充语会污染取代判定
        4. 必须有人物锚点或明确偏好标记：地点/话题词典太宽，
           "帮我写个测试"不该成为永久 FACT
        写入走 add_memory 决策表——重复陈述被 NOOP 强化，改口走
        SUPERSEDE 取代链。provenance 落库（source=system_extract、
        置信度/重要性低于显式写入）。
        """
        if observation.source != "user" or not hasattr(self.memory, "add_memory"):
            return
        text = observation.content or ""
        lowered = text.lower()
        if any(m in lowered for m in MEMORY_WRITE_NEGATIVE_MARKERS):
            return  # 用户明确拒绝记录——自动编码不得绕过否定意图
        if len(text) > self._AUTO_ENCODE_MAX_CHARS:
            return
        if any(m in lowered for m in self._NON_DECLARATIVE_MARKERS):
            return
        if self._extractor is None:
            from core.memory_extractor import MemoryExtractor
            self._extractor = MemoryExtractor()
        probe = _ExtractionProbe(observation)
        try:
            result = self._extractor.extract(probe)
        except Exception:
            return
        preference_marker = any(m in text for m in ("喜欢", "偏好", "讨厌", "习惯"))
        for spec in result.specs:
            if spec.metadata.get("kind") != "fact":
                continue  # 经验/程序类交给回合固化，避免双写
            if not spec.persons and not preference_marker:
                continue  # 地点/话题词典太宽，单靠它们不足以断定"事实"
            # 只保留承载锚点的子句：整句入库时，无关填充语
            # （"周末我们约了饭"）会参与取代判定，曾把住址事实
            # 误取代成养猫事实（对抗审查端到端复现）
            content = self._anchored_clauses(spec.content, spec.persons) or spec.content
            try:
                self.memory.add_memory(
                    content=content,
                    memory_type=spec.memory_type,
                    persons=list(spec.persons),
                    topics=list(spec.topics),
                    keywords=list(spec.keywords),
                    time_absolute=spec.time_absolute,
                    time_relative=spec.time_relative,
                    location=spec.location,
                    # 自动编码的置信度/重要性必须低于显式写入
                    # （惊奇度缩放后仍不超过显式 remember 的水平）
                    importance=min(spec.importance, 0.4),
                    confidence=0.6,
                    source="system_extract",
                    metadata=dict(spec.metadata, auto_encoded=True),
                    emotion_valence=spec.emotion_valence,
                    emotion_intensity=spec.emotion_intensity,
                )
            except Exception:
                return

    def select_action(self, observation: Observation, workspace: FocusWorkspace) -> AgentAction:
        return self.planner(observation, workspace, self.tools)

    def execute_action(
        self,
        observation: Observation,
        workspace: FocusWorkspace,
        action: AgentAction,
        synthesize: bool = True,
        consolidate: Optional[bool] = None,
    ) -> ExperienceEpisode:
        prediction = None
        if hasattr(self.memory, "predict_action"):
            prediction = self.memory.predict_action(action, tools=self.tools)

        tool_result = self.tools.run(action)
        if prediction is not None:
            tool_result.metadata.setdefault("prediction", prediction.to_dict())

        result = self._synthesize_result(observation, workspace, action, tool_result) if synthesize else tool_result

        reward = self.evaluator(observation, action, result)
        lesson = self._derive_lesson(observation, action, result, reward)
        next_policy = self._derive_next_policy(action, result, reward)

        active_goal = workspace.active_goal.objective if workspace.active_goal else ""
        episode = ExperienceEpisode(
            goal=active_goal,
            observation=observation,
            action=action,
            result=result,
            reward=reward,
            lesson=lesson,
            next_policy=next_policy,
            focus_context=workspace.to_prompt_context(),
        )
        self.experience.add(episode)

        if hasattr(self.memory, "reflect_episode"):
            self.memory.reflect_episode(episode)

        should_consolidate = self.auto_consolidate if consolidate is None else consolidate
        if should_consolidate:
            self.experience.consolidate(self.memory, limit=1)

        return episode

    def add_tool(
        self,
        name: str,
        description: str,
        handler: ToolHandler,
        cost: float = 0.0,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self.tools.register(AgentTool(
            name=name, description=description, handler=handler,
            cost=cost, parameters=parameters,
        ))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "auto_consolidate": self.auto_consolidate,
            "experience": self.experience.to_dict(),
            "tools": self.tools.describe(),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        memory_system: Any,
        planner: Planner = default_planner,
        evaluator: Evaluator = default_evaluator,
    ) -> "CognitiveAgent":
        return cls(
            memory_system=memory_system,
            name=data.get("name", "cognitive-agent"),
            planner=planner,
            evaluator=evaluator,
            experience_layer=ExperienceLayer.from_dict(data.get("experience", {})),
            auto_consolidate=bool(data.get("auto_consolidate", True)),
        )

    def _synthesize_result(
        self,
        observation: Observation,
        workspace: FocusWorkspace,
        action: AgentAction,
        tool_result: ActionResult,
    ) -> ActionResult:
        """Optionally turn a raw tool result into a final user-facing response."""
        if action.name == "respond" or self.response_synthesizer is None:
            return tool_result

        try:
            synthesized = self.response_synthesizer(observation, workspace, action, tool_result)
        except Exception as exc:
            tool_result.metadata.setdefault("synthesis_error", f"{type(exc).__name__}: {exc}")
            return tool_result

        if synthesized is None:
            tool_result.metadata.setdefault("synthesis_empty", True)
            return tool_result
        if isinstance(synthesized, str):
            synthesized = ActionResult(tool_result.success, synthesized)
        if not isinstance(synthesized, ActionResult) or not synthesized.output:
            tool_result.metadata.setdefault("synthesis_empty", True)
            return tool_result

        synthesized.success = bool(synthesized.success and tool_result.success)
        synthesized.cost += tool_result.cost
        synthesized.metadata.setdefault("kind", "response_synthesis")
        synthesized.metadata.setdefault("synthesized_from_tool", action.name)
        synthesized.metadata.setdefault("tool_result", tool_result.to_dict())
        if "prediction" in tool_result.metadata:
            synthesized.metadata.setdefault("prediction", tool_result.metadata["prediction"])
        return synthesized

    def _register_default_tools(self):
        self.add_tool(
            "respond", "Return a context-aware text response.", self._respond_tool,
            parameters={
                "type": "object",
                "properties": {"message": {"type": "string", "description": "Final user-facing answer."}},
                "required": ["message"],
            },
        )
        self.add_tool(
            "remember",
            "Store memory only when the user explicitly asks to remember/save/record something.",
            self._remember_tool,
            parameters={
                "type": "object",
                "properties": {"content": {"type": "string", "description": "The information to store."}},
                "required": ["content"],
            },
        )
        self.add_tool(
            "introspect",
            "Read current self-model, drives, world beliefs, and open questions.",
            self._introspect_tool,
            parameters={"type": "object", "properties": {}},
        )

    def _respond_tool(self, arguments: Dict[str, Any]) -> ActionResult:
        message = (
            arguments.get("message")
            or arguments.get("response")
            or arguments.get("content")
            or arguments.get("text")
            or arguments.get("input", "")
        )
        context = arguments.get("context") or arguments.get("workspace", "")
        if message:
            output = str(message)
        elif context:
            output = str(context)
        else:
            output = ""
        return ActionResult(True, output)

    def _remember_tool(self, arguments: Dict[str, Any]) -> ActionResult:
        content = arguments.get("content") or arguments.get("input", "")
        if not content:
            return ActionResult(False, "No content to remember.")
        memory_id = self.memory.add_memory(
            content=content,
            memory_type=MemoryType.INTERACTION,
            topics=["explicit", "agent"],
            importance=0.55,
            metadata={"source": self.name, "tool": "remember"},
        )
        return ActionResult(True, f"Stored memory {memory_id}", metadata={"memory_id": memory_id})

    def _introspect_tool(self, arguments: Dict[str, Any]) -> ActionResult:
        if not hasattr(self.memory, "get_cognitive_summary"):
            return ActionResult(False, "This memory system has no cognitive state.")
        summary = self.memory.get_cognitive_summary()
        if arguments.get("format") == "json":
            output = json.dumps(summary, ensure_ascii=False, indent=2)
        else:
            output = self._format_cognitive_summary(summary)
        return ActionResult(True, output, metadata={"kind": "cognitive_summary"})

    def _format_cognitive_summary(self, summary: Dict[str, Any]) -> str:
        identity = summary.get("identity", {})
        drives = summary.get("drives", {})
        open_questions = summary.get("open_questions", [])
        tool_stats = summary.get("tool_stats", {})

        drive_lines = []
        for name, drive in sorted(drives.items(), key=lambda item: item[1].get("urgency", 0), reverse=True)[:5]:
            drive_lines.append(f"- {name}: value={drive.get('value', 0):.2f}, urgency={drive.get('urgency', 0):.2f}")

        question_lines = [f"- {question}" for question in open_questions[-5:]] or ["- none"]
        tool_lines = []
        for name, stats in sorted(tool_stats.items()):
            attempts = int(stats.get("attempts", 0))
            successes = int(stats.get("successes", 0))
            tool_lines.append(f"- {name}: {successes}/{attempts} successes")
        if not tool_lines:
            tool_lines.append("- none")

        return "\n".join(
            [
                f"identity: {identity.get('name', self.name)}",
                f"mission: {identity.get('mission', '')}",
                "drives:",
                *drive_lines,
                "open_questions:",
                *question_lines,
                "tool_stats:",
                *tool_lines,
            ]
        )

    def _derive_lesson(
        self,
        observation: Observation,
        action: AgentAction,
        result: ActionResult,
        reward: float,
    ) -> str:
        if result.success:
            return f"When seeing '{observation.source}' input like this, action '{action.name}' worked."
        return f"Action '{action.name}' failed: {result.output}"

    def _derive_next_policy(self, action: AgentAction, result: ActionResult, reward: float) -> str:
        if result.success and reward >= 0.5:
            return f"Use '{action.name}' again when the focus workspace and observation match this pattern."
        if not result.success:
            return f"Before using '{action.name}' again, check tool availability and arguments."
        return ""
