"""Multi-step cognitive runtime for memory-driven agents.

The runtime is the first layer above a single agent turn:

Observe -> Set/refresh goal -> Focus -> Plan one action -> Act -> Reflect
        -> Re-focus with trace -> Continue or Finish
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
import json
import time
import uuid

from core.agent_system import ActionResult, AgentAction, CognitiveAgent, ExperienceEpisode, Observation


RuntimeFinalizer = Callable[["CognitiveRun"], Union[str, ActionResult, None]]


def _now() -> float:
    return time.time()


def _shorten(text: Any, limit: int = 500) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


@dataclass
class CognitiveRuntimeConfig:
    """Configuration for the multi-step runtime loop."""

    max_steps: int = 4
    auto_goal: bool = True
    consolidate: bool = True
    stop_on_failure: bool = False
    finalize_completed_runs: bool = False
    max_trace_chars: int = 7000


@dataclass
class CognitiveStep:
    """One executed step in a cognitive runtime run."""

    index: int
    observation: Observation
    action: AgentAction
    result: ActionResult
    reward: float
    prediction: Optional[Dict[str, Any]] = None
    episode_id: str = ""
    workspace_context: str = ""
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "observation": self.observation.to_dict(),
            "action": self.action.to_dict(),
            "result": self.result.to_dict(),
            "reward": self.reward,
            "prediction": self.prediction,
            "episode_id": self.episode_id,
            "workspace_context": self.workspace_context,
            "created_at": self.created_at,
        }


@dataclass
class CognitiveRun:
    """A complete multi-step runtime trace."""

    original_observation: Observation
    goal: str
    steps: List[CognitiveStep]
    result: ActionResult
    completed: bool
    stop_reason: str = ""
    id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex[:10]}")
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "original_observation": self.original_observation.to_dict(),
            "goal": self.goal,
            "steps": [step.to_dict() for step in self.steps],
            "result": self.result.to_dict(),
            "completed": self.completed,
            "stop_reason": self.stop_reason,
            "created_at": self.created_at,
        }


class CognitiveRuntime:
    """A small executive loop that can execute several agent actions per task."""

    def __init__(
        self,
        agent: CognitiveAgent,
        config: Optional[CognitiveRuntimeConfig] = None,
        finalizer: Optional[RuntimeFinalizer] = None,
    ):
        self.agent = agent
        self.config = config or CognitiveRuntimeConfig()
        self.finalizer = finalizer
        self._register_control_tools()

    def run(
        self,
        task: Union[str, Observation],
        source: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CognitiveRun:
        if isinstance(task, Observation):
            original = task
        else:
            original = self.agent.observe(task, source=source, metadata=metadata or {})

        if hasattr(self.agent.memory, "observe_world"):
            self.agent.memory.observe_world(original)

        goal = self._ensure_goal(original)
        steps: List[CognitiveStep] = []
        completed = False
        stop_reason = "max_steps"

        for index in range(1, max(1, self.config.max_steps) + 1):
            step_observation = self._build_step_observation(original, goal, steps, index)
            workspace = self.agent.build_workspace(step_observation)
            action = self.agent.select_action(step_observation, workspace)
            action = self._guard_runtime_action(original, steps, action, index)
            episode = self.agent.execute_action(
                step_observation,
                workspace,
                action,
                synthesize=False,
                consolidate=False,
            )
            step = self._step_from_episode(index, step_observation, workspace.to_prompt_context(), episode)
            steps.append(step)

            if action.name == "finish":
                completed = True
                stop_reason = "finish"
                break
            if self.config.stop_on_failure and not episode.result.success:
                stop_reason = "failure"
                break

        result = self._final_result(original, goal, steps, completed, stop_reason)
        run = CognitiveRun(
            original_observation=original,
            goal=goal,
            steps=steps,
            result=result,
            completed=completed,
            stop_reason=stop_reason,
        )

        if self.config.consolidate:
            self.agent.experience.consolidate(self.agent.memory, limit=len(steps))
        if completed:
            self._complete_goal(goal, result.output)
        return run

    def _register_control_tools(self):
        if "finish" in self.agent.tools.tools:
            return

        def finish(arguments: Dict[str, Any]) -> ActionResult:
            message = str(
                arguments.get("message")
                or arguments.get("final_answer")
                or arguments.get("answer")
                or arguments.get("summary")
                or arguments.get("input", "")
            )
            next_questions = arguments.get("next_questions") or arguments.get("questions") or []
            if isinstance(next_questions, str):
                next_questions = [next_questions]
            if next_questions:
                question_lines = "\n".join(f"{index}. {question}" for index, question in enumerate(next_questions, start=1))
                message = (message + "\n\nNext questions:\n" + question_lines).strip()
            return ActionResult(True, str(message), metadata={"kind": "runtime_finish"})

        self.agent.add_tool(
            "finish",
            "End the CognitiveRuntime run only when the task is satisfied; put the final answer in message.",
            finish,
        )

    def _ensure_goal(self, observation: Observation) -> str:
        objective = f"Complete user task: {_shorten(observation.content, 180)}"
        if not self.config.auto_goal or not hasattr(self.agent.memory, "start_goal"):
            active = getattr(getattr(self.agent.memory, "attention", None), "goal_stack", None)
            current = active.active() if active and hasattr(active, "active") else None
            return current.objective if current else objective

        goal = self.agent.memory.start_goal(
            objective,
            constraints=[
                "Use memory and tools as evidence, not as replacement for the current task.",
                "Prefer reversible, inspectable actions.",
                "Use finish only after the requested answer is complete.",
            ],
            open_loops=["Decide next action", "Check result", "Finish with a user-facing answer"],
            priority=0.82,
        )
        return goal.objective

    def _complete_goal(self, objective: str, evidence: str):
        attention = getattr(self.agent.memory, "attention", None)
        goal_stack = getattr(attention, "goal_stack", None)
        if goal_stack is None:
            return
        active = goal_stack.active()
        if active and active.objective == objective and hasattr(self.agent.memory, "update_goal"):
            self.agent.memory.update_goal(active.id, status="completed", evidence=[_shorten(evidence, 240)])

    def _build_step_observation(
        self,
        original: Observation,
        goal: str,
        steps: List[CognitiveStep],
        index: int,
    ) -> Observation:
        trace = self._trace_text(steps)
        content = f"""
CognitiveRuntime step {index}/{self.config.max_steps}

Original user task:
{original.content}

Runtime goal:
{goal}

Completed steps:
{trace or "(none yet)"}

Choose exactly one next action.
Runtime rules:
- The original user task is the objective. Prior memory and open questions are evidence only.
- Use respond for intermediate reasoning or partial answers that may be needed before another tool.
- Use introspect when the task asks for self-checking or when coherence/drive state matters.
- Use finish only when the final user-facing answer is complete.
- If finish is not possible, choose the smallest useful next action.
""".strip()
        if len(content) > self.config.max_trace_chars:
            content = content[: self.config.max_trace_chars] + "\n...[runtime context truncated]"
        return self.agent.observe(
            content,
            source="runtime",
            metadata={
                "runtime": True,
                "original_task": original.content,
                "step_index": index,
                "goal": goal,
            },
        )

    def _trace_text(self, steps: List[CognitiveStep]) -> str:
        lines: List[str] = []
        for step in steps:
            lines.append(
                f"{step.index}. action={step.action.name}; "
                f"success={step.result.success}; "
                f"output={_shorten(step.result.output, 700)}"
            )
        return "\n".join(lines)

    def _step_from_episode(
        self,
        index: int,
        observation: Observation,
        workspace_context: str,
        episode: ExperienceEpisode,
    ) -> CognitiveStep:
        prediction = episode.result.metadata.get("prediction") if isinstance(episode.result.metadata, dict) else None
        return CognitiveStep(
            index=index,
            observation=observation,
            action=episode.action,
            result=episode.result,
            reward=episode.reward,
            prediction=prediction,
            episode_id=episode.id,
            workspace_context=workspace_context,
        )

    def _guard_runtime_action(
        self,
        original: Observation,
        steps: List[CognitiveStep],
        action: AgentAction,
        index: int,
    ) -> AgentAction:
        missing_tools = self._missing_required_tools(original, steps)
        if not missing_tools and action.name == "respond" and index >= self.config.max_steps:
            return AgentAction(
                name="finish",
                arguments={"message": self._action_message(action)},
                rationale="CognitiveRuntime guard: final step converted response into finish.",
            )
        if not missing_tools:
            return action

        if action.name == "finish":
            if self._answer_should_precede_tool(original, missing_tools[0]) and not self._has_executed(steps, "respond"):
                return AgentAction(
                    name="respond",
                    arguments={"message": self._action_message(action)},
                    rationale=(
                        "CognitiveRuntime guard: preserve answer as an intermediate response "
                        f"before required tool '{missing_tools[0]}' runs."
                    ),
                )
            return AgentAction(
                name=missing_tools[0],
                arguments={},
                rationale=f"CognitiveRuntime guard: cannot finish before required tool '{missing_tools[0]}' runs.",
            )

        if action.name == "respond":
            message = self._action_message(action).strip().lower()
            if any(message == name or message.startswith(name) for name in missing_tools):
                return AgentAction(
                    name=missing_tools[0],
                    arguments={},
                    rationale=(
                        f"CognitiveRuntime guard: responding with '{message}' "
                        f"is not the same as running '{missing_tools[0]}'."
                    ),
                )

        return action

    def _action_message(self, action: AgentAction) -> str:
        arguments = action.arguments or {}
        message = (
            arguments.get("message")
            or arguments.get("final_answer")
            or arguments.get("answer")
            or arguments.get("summary")
            or arguments.get("content")
            or arguments.get("input", "")
        )
        next_questions = arguments.get("next_questions") or arguments.get("questions") or []
        if isinstance(next_questions, str):
            next_questions = [next_questions]
        if next_questions:
            question_lines = "\n".join(f"{index}. {question}" for index, question in enumerate(next_questions, start=1))
            return (str(message) + "\n\nNext questions:\n" + question_lines).strip()
        return str(message)

    def _has_executed(self, steps: List[CognitiveStep], action_name: str) -> bool:
        return any(step.action.name == action_name for step in steps)

    def _answer_should_precede_tool(self, original: Observation, tool_name: str) -> bool:
        task = original.content.lower()
        tool_pos = task.find(tool_name.lower())
        if tool_pos < 0:
            return False
        answer_positions = [
            task.find(marker)
            for marker in ["answer", "respond", "回答", "用一句话", "一句话"]
            if task.find(marker) >= 0
        ]
        return bool(answer_positions and min(answer_positions) < tool_pos)

    def _missing_required_tools(self, original: Observation, steps: List[CognitiveStep]) -> List[str]:
        required: List[str] = []
        task = original.content.lower()
        for tool_name in self.agent.tools.tools:
            if tool_name in {"respond", "finish", "remember"}:
                continue
            if tool_name.lower() in task:
                required.append(tool_name)

        executed = {step.action.name for step in steps}
        return [name for name in required if name not in executed]

    def _final_result(
        self,
        original: Observation,
        goal: str,
        steps: List[CognitiveStep],
        completed: bool,
        stop_reason: str,
    ) -> ActionResult:
        provisional = self._fallback_final_output(steps, completed, stop_reason)
        if completed and provisional and not self.config.finalize_completed_runs:
            return ActionResult(
                True,
                provisional,
                metadata={"kind": "runtime_finish_output", "stop_reason": stop_reason, "completed": True},
            )
        run = CognitiveRun(
            original_observation=original,
            goal=goal,
            steps=steps,
            result=ActionResult(completed, provisional),
            completed=completed,
            stop_reason=stop_reason,
        )
        if self.finalizer is not None:
            try:
                finalized = self.finalizer(run)
            except Exception as exc:
                return ActionResult(
                    bool(provisional),
                    provisional,
                    metadata={"finalizer_error": f"{type(exc).__name__}: {exc}", "stop_reason": stop_reason},
                )
            if isinstance(finalized, ActionResult) and finalized.output:
                finalized.metadata.setdefault("stop_reason", stop_reason)
                finalized.metadata.setdefault("completed", completed)
                return finalized
            if isinstance(finalized, str) and finalized.strip():
                return ActionResult(completed, finalized.strip(), metadata={"kind": "runtime_finalized", "stop_reason": stop_reason})

        return ActionResult(
            bool(provisional),
            provisional,
            metadata={"kind": "runtime_fallback_final", "stop_reason": stop_reason, "completed": completed},
        )

    def _fallback_final_output(self, steps: List[CognitiveStep], completed: bool, stop_reason: str) -> str:
        for step in reversed(steps):
            if step.action.name == "finish" and step.result.output:
                return step.result.output
        for step in reversed(steps):
            if step.action.name == "respond" and step.result.output:
                return step.result.output
        if not steps:
            return "No runtime steps were executed."
        prefix = "" if completed else f"Runtime stopped before finish ({stop_reason}). "
        return prefix + "Last tool output: " + _shorten(steps[-1].result.output, 1000)


class LLMRuntimeFinalizer:
    """Use an LLM to turn a runtime trace into a final answer."""

    def __init__(self, llm: Callable[..., str], max_prompt_chars: int = 12000):
        self.llm = llm
        self.max_prompt_chars = max_prompt_chars
        self.last_prompt = ""
        self.last_output = ""

    def __call__(self, run: CognitiveRun) -> Optional[ActionResult]:
        prompt = self.build_prompt(run)
        self.last_prompt = prompt
        complete = getattr(self.llm, "complete", None)
        if callable(complete):
            output = complete(
                prompt,
                system_prompt="You are the final response layer of a multi-step cognitive runtime.",
            )
        else:
            output = self.llm(prompt)
        self.last_output = output.strip()
        if not self.last_output:
            return None
        return ActionResult(
            run.completed,
            self.last_output,
            metadata={"kind": "llm_runtime_finalizer", "completed": run.completed, "stop_reason": run.stop_reason},
        )

    def build_prompt(self, run: CognitiveRun) -> str:
        steps = []
        for step in run.steps:
            steps.append(
                {
                    "index": step.index,
                    "action": step.action.to_dict(),
                    "success": step.result.success,
                    "output": _shorten(step.result.output, 1200),
                    "reward": step.reward,
                }
            )
        prompt = f"""
Original user task:
{run.original_observation.content}

Runtime goal:
{run.goal}

Completed:
{run.completed}

Stop reason:
{run.stop_reason}

Executed steps:
{json.dumps(steps, ensure_ascii=False, indent=2)}

Write the final user-facing answer.
Rules:
- Answer the original user task, not prior open questions from memory.
- If the runtime did not complete every requested step, state the missing part plainly and give the next smallest action.
- Do not return JSON.
- Keep the answer concise and useful.
""".strip()
        if len(prompt) <= self.max_prompt_chars:
            return prompt
        return prompt[: self.max_prompt_chars] + "\n...[runtime trace truncated]"
