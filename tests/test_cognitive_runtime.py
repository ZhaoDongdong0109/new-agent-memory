import pytest

from new_agent_memory import (
    ActionResult,
    AgentAction,
    CognitiveRuntime,
    CognitiveRuntimeConfig,
    HumanLikeMemorySystem,
)


def test_cognitive_runtime_executes_multiple_steps_and_finishes():
    memory = HumanLikeMemorySystem()
    agent = memory.create_agent(auto_consolidate=False)
    actions = iter(
        [
            AgentAction(name="respond", arguments={"message": "intermediate answer"}),
            AgentAction(name="introspect", arguments={}),
            AgentAction(name="finish", arguments={"message": "final answer"}),
        ]
    )

    def planner(observation, workspace, tools):
        return next(actions)

    agent.planner = planner
    runtime = CognitiveRuntime(agent, config=CognitiveRuntimeConfig(max_steps=4, consolidate=False))

    run = runtime.run("answer, introspect, then finish")

    assert run.completed is True
    assert run.stop_reason == "finish"
    assert run.result.output == "final answer"
    assert [step.action.name for step in run.steps] == ["respond", "introspect", "finish"]
    assert len(agent.experience.episodes) == 3
    assert "finish" in agent.tools.tools


def test_cognitive_runtime_falls_back_when_no_finish_is_chosen():
    memory = HumanLikeMemorySystem()
    agent = memory.create_agent(auto_consolidate=False)

    def planner(observation, workspace, tools):
        return AgentAction(name="introspect", arguments={})

    agent.planner = planner
    runtime = CognitiveRuntime(agent, config=CognitiveRuntimeConfig(max_steps=2, consolidate=False))

    run = runtime.run("keep checking yourself")

    assert run.completed is False
    assert run.stop_reason == "max_steps"
    assert len(run.steps) == 2
    assert "Runtime stopped before finish" in run.result.output


def test_cognitive_runtime_uses_finalizer_for_trace_answer():
    memory = HumanLikeMemorySystem()
    agent = memory.create_agent(auto_consolidate=False)
    actions = iter(
        [
            AgentAction(name="respond", arguments={"message": "first"}),
            AgentAction(name="finish", arguments={"message": "done"}),
        ]
    )

    def planner(observation, workspace, tools):
        return next(actions)

    def finalizer(run):
        assert run.completed is True
        assert [step.action.name for step in run.steps] == ["respond", "finish"]
        return "finalized from trace"

    agent.planner = planner
    runtime = CognitiveRuntime(
        agent,
        config=CognitiveRuntimeConfig(max_steps=3, consolidate=False, finalize_completed_runs=True),
        finalizer=finalizer,
    )

    run = runtime.run("do two steps")

    assert run.result.success is True
    assert run.result.output == "finalized from trace"
    assert run.result.metadata["kind"] == "runtime_finalized"


def test_cognitive_runtime_prevents_finish_before_required_tool_runs():
    memory = HumanLikeMemorySystem()
    agent = memory.create_agent(auto_consolidate=False)
    actions = iter(
        [
            AgentAction(name="respond", arguments={"message": "answer"}),
            AgentAction(name="finish", arguments={"message": "too early"}),
            AgentAction(name="finish", arguments={"message": "done"}),
        ]
    )

    def planner(observation, workspace, tools):
        return next(actions)

    agent.planner = planner
    runtime = CognitiveRuntime(agent, config=CognitiveRuntimeConfig(max_steps=3, consolidate=False))

    run = runtime.run("answer, then introspect, then finish")

    assert [step.action.name for step in run.steps] == ["respond", "introspect", "finish"]
    assert run.completed is True
    assert run.result.output == "done"


def test_cognitive_runtime_finish_includes_next_questions():
    memory = HumanLikeMemorySystem()
    agent = memory.create_agent(auto_consolidate=False)

    def planner(observation, workspace, tools):
        return AgentAction(
            name="finish",
            arguments={"answer": "final", "next_questions": ["q1", "q2"]},
        )

    agent.planner = planner
    runtime = CognitiveRuntime(agent, config=CognitiveRuntimeConfig(max_steps=1, consolidate=False))

    run = runtime.run("finish with questions")

    assert run.completed is True
    assert "final" in run.result.output
    assert "1. q1" in run.result.output
    assert "2. q2" in run.result.output


def test_cognitive_runtime_preserves_answer_before_required_tool_order():
    memory = HumanLikeMemorySystem()
    agent = memory.create_agent(auto_consolidate=False)
    actions = iter(
        [
            AgentAction(name="finish", arguments={"message": "first answer"}),
            AgentAction(name="respond", arguments={"message": "introspect {}"}),
            AgentAction(name="respond", arguments={"message": "final answer"}),
        ]
    )

    def planner(observation, workspace, tools):
        return next(actions)

    agent.planner = planner
    runtime = CognitiveRuntime(agent, config=CognitiveRuntimeConfig(max_steps=3, consolidate=False))

    run = runtime.run("answer first, then introspect, then finish")

    assert [step.action.name for step in run.steps] == ["respond", "introspect", "finish"]
    assert run.completed is True
    assert run.result.output == "final answer"


def test_cognitive_runtime_default_planner_does_not_echo_scaffold():
    """回归：无 LLM 的默认规划器不应把 runtime 脚手架当成答案回显。"""
    memory = HumanLikeMemorySystem()
    runtime = memory.create_runtime(max_steps=2)

    run = runtime.run("What is the capital of France?")

    assert "CognitiveRuntime step" not in run.result.output
    assert "What is the capital of France?" in run.result.output
    # 固化的记忆里也不应出现整段脚手架
    assert all(
        "Choose exactly one next action" not in chunk.content
        for chunk in memory.core.chunks.values()
    )


def test_cognitive_runtime_closes_goal_after_incomplete_run():
    """回归：未完成的运行不能留下永久 active 的目标。"""
    memory = HumanLikeMemorySystem()
    agent = memory.create_agent(auto_consolidate=False)

    def planner(observation, workspace, tools):
        return AgentAction(name="introspect", arguments={})

    agent.planner = planner
    runtime = CognitiveRuntime(agent, config=CognitiveRuntimeConfig(max_steps=2, consolidate=False))

    run = runtime.run("keep checking yourself")

    assert run.completed is False
    summary = memory.get_attention_summary()
    assert summary["active_goal"] is None
    goal = summary["goals"][-1]
    assert goal["status"] == "suspended"
    assert goal["evidence"] and "max_steps" in goal["evidence"][0]


def test_cognitive_runtime_marks_goal_completed_on_finish():
    memory = HumanLikeMemorySystem()
    agent = memory.create_agent(auto_consolidate=False)

    def planner(observation, workspace, tools):
        return AgentAction(name="finish", arguments={"message": "done now"})

    agent.planner = planner
    runtime = CognitiveRuntime(agent, config=CognitiveRuntimeConfig(max_steps=2, consolidate=False))

    run = runtime.run("finish immediately")

    assert run.completed is True
    summary = memory.get_attention_summary()
    assert summary["active_goal"] is None
    goal = summary["goals"][-1]
    assert goal["status"] == "completed"
    assert goal["evidence"] and "done now" in goal["evidence"][0]


def test_cognitive_runtime_closes_goal_when_planner_raises():
    memory = HumanLikeMemorySystem()
    agent = memory.create_agent(auto_consolidate=False)

    def planner(observation, workspace, tools):
        raise ValueError("boom")

    agent.planner = planner
    runtime = CognitiveRuntime(agent, config=CognitiveRuntimeConfig(max_steps=2, consolidate=False))

    with pytest.raises(ValueError):
        runtime.run("some task")

    summary = memory.get_attention_summary()
    assert summary["active_goal"] is None
    assert summary["goals"][-1]["status"] == "suspended"


def test_cognitive_runtime_incomplete_run_result_is_not_success():
    """回归：未完成的运行 _final_result 不应返回 success=True。"""
    memory = HumanLikeMemorySystem()
    agent = memory.create_agent(auto_consolidate=False)

    def planner(observation, workspace, tools):
        return AgentAction(name="introspect", arguments={})

    agent.planner = planner
    runtime = CognitiveRuntime(agent, config=CognitiveRuntimeConfig(max_steps=2, consolidate=False))

    run = runtime.run("keep checking yourself")

    assert run.completed is False
    assert run.result.success is False
    assert run.result.metadata["completed"] is False


def test_cognitive_runtime_does_not_force_tool_on_substring_match():
    """回归：'research' 不应把 'search' 工具标成必须执行。"""
    memory = HumanLikeMemorySystem()
    agent = memory.create_agent(auto_consolidate=False)
    agent.add_tool("search", "Search the web.", lambda arguments: ActionResult(True, "results"))
    actions = iter(
        [
            AgentAction(name="respond", arguments={"message": "direct answer"}),
            AgentAction(name="finish", arguments={"message": "final"}),
        ]
    )

    def planner(observation, workspace, tools):
        return next(actions)

    agent.planner = planner
    runtime = CognitiveRuntime(agent, config=CognitiveRuntimeConfig(max_steps=3, consolidate=False))

    run = runtime.run("Please research quantum computing and answer briefly")

    assert [step.action.name for step in run.steps] == ["respond", "finish"]
    assert run.completed is True
    assert run.result.output == "final"


def test_cognitive_runtime_carries_blocked_finish_draft_into_next_observation():
    """回归：守卫拦下 finish 时，草稿答案要带入下一步观测而不是丢弃。"""
    memory = HumanLikeMemorySystem()
    agent = memory.create_agent(auto_consolidate=False)
    seen = []
    actions = iter(
        [
            AgentAction(name="finish", arguments={"message": "draft answer 42"}),
            AgentAction(name="finish", arguments={"message": "verified final"}),
        ]
    )

    def planner(observation, workspace, tools):
        seen.append(observation.content)
        return next(actions)

    agent.planner = planner
    runtime = CognitiveRuntime(agent, config=CognitiveRuntimeConfig(max_steps=3, consolidate=False))

    run = runtime.run("use introspect, then finish")

    assert [step.action.name for step in run.steps] == ["introspect", "finish"]
    assert run.result.output == "verified final"
    assert "Draft final answer" in seen[1]
    assert "draft answer 42" in seen[1]
