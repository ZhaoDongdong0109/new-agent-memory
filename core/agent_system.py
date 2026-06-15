"""A minimal embodied agent loop built on memory and attention.

The loop is intentionally small and inspectable:

Observe -> Focus -> Decide -> Act -> Evaluate -> Remember -> Consolidate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import json
import time
import uuid

from core.attention_system import FocusWorkspace
from core.weight_system import MemoryType


def _now() -> float:
    return time.time()


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentAction":
        return cls(
            name=data.get("name", "respond"),
            arguments=dict(data.get("arguments", {})),
            rationale=data.get("rationale", ""),
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
            if episode.reward < 0.35 and not episode.lesson:
                continue

            memory_id = memory_system.add_memory(
                content=episode.to_memory_text(),
                memory_type=MemoryType.STORY,
                topics=["experience", "agent", episode.action.name],
                importance=_clamp(0.35 + episode.reward * 0.45),
                metadata={"experience_id": episode.id, "kind": "episode"},
            )
            created.append(memory_id)

            if episode.result.success and episode.next_policy:
                procedure = memory_system.add_procedure(
                    title=f"Learned policy: {episode.action.name}",
                    steps=[episode.next_policy],
                    triggers=[episode.action.name, "experience", "success"],
                    importance=_clamp(0.45 + episode.reward * 0.4),
                    confidence=_clamp(0.45 + episode.reward * 0.4),
                )
                created.append(procedure.id)

        return created

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
    lower = observation.content.lower()
    for tool_name in tools.tools:
        if tool_name.lower() in lower:
            return AgentAction(
                name=tool_name,
                arguments={"input": observation.content, "workspace": workspace.to_prompt_context()},
                rationale=f"Observation mentioned tool '{tool_name}'.",
            )

    return AgentAction(
        name="respond",
        arguments={
            "message": observation.content,
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

        workspace = self.memory.focus(observation.content, include_forgotten=True)
        if hasattr(self.memory, "attach_cognitive_context"):
            self.memory.attach_cognitive_context(workspace, tools=self.tools)

        action = self.planner(observation, workspace, self.tools)
        prediction = None
        if hasattr(self.memory, "predict_action"):
            prediction = self.memory.predict_action(action, tools=self.tools)

        tool_result = self.tools.run(action)
        if prediction is not None:
            tool_result.metadata.setdefault("prediction", prediction.to_dict())

        result = self._synthesize_result(observation, workspace, action, tool_result)

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
