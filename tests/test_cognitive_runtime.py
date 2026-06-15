from new_agent_memory import (
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
