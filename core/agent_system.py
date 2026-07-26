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

from core.attention_system import FocusWorkspace
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
        return [
            {"name": tool.name, "description": tool.description, "cost": tool.cost}
            for tool in self.tools.values()
        ]


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
    ):
        self.memory = memory_system
        self.name = name
        self.planner = planner
        self.evaluator = evaluator
        self.response_synthesizer = response_synthesizer
        self.experience = experience_layer or ExperienceLayer()
        self.tools = ToolRegistry()
        self.auto_consolidate = auto_consolidate
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

        workspace = self.build_workspace(observation)
        action = self.select_action(observation, workspace)
        return self.execute_action(observation, workspace, action, consolidate=consolidate)

    def build_workspace(self, observation: Observation) -> FocusWorkspace:
        workspace = self.memory.focus(observation.content, include_forgotten=True)
        if hasattr(self.memory, "attach_cognitive_context"):
            self.memory.attach_cognitive_context(workspace, tools=self.tools)
        return workspace

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

    def add_tool(self, name: str, description: str, handler: ToolHandler, cost: float = 0.0):
        self.tools.register(AgentTool(name=name, description=description, handler=handler, cost=cost))

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
        self.add_tool("respond", "Return a context-aware text response.", self._respond_tool)
        self.add_tool("remember", "Store memory only when the user explicitly asks to remember/save/record something.", self._remember_tool)
        self.add_tool("introspect", "Read current self-model, drives, world beliefs, and open questions.", self._introspect_tool)

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
