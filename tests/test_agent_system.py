from core.agent_system import default_planner
from new_agent_memory import (
    ActionResult,
    AgentAction,
    CognitiveAgent,
    ExperienceEpisode,
    ExperienceLayer,
    HumanLikeMemorySystem,
    Observation,
)


def test_cognitive_agent_runs_full_loop_and_consolidates_experience():
    memory = HumanLikeMemorySystem()
    memory.start_goal(
        "帮助用户完成代码任务",
        constraints=["先观察", "再行动", "最后复盘"],
    )
    agent = memory.create_agent(name="hermes-seed")

    episode = agent.run_turn("请根据上下文回复我")

    assert episode.result.success is True
    assert episode.reward > 0
    assert episode.lesson
    assert len(agent.experience.episodes) == 1
    assert memory.get_memory_stats()["core_chunks"] >= 1
    assert memory.get_attention_summary()["procedures"]


def test_agent_can_use_custom_tool_and_record_failure():
    memory = HumanLikeMemorySystem()
    agent = CognitiveAgent(memory_system=memory, auto_consolidate=False)

    def planner(observation, workspace, tools):
        return AgentAction(name="broken_tool", arguments={"input": observation.content})

    agent.planner = planner

    episode = agent.run_turn("请调用 broken_tool")

    assert episode.result.success is False
    assert episode.reward == 0
    assert "failed" in episode.lesson
    assert "check tool availability" in episode.next_policy


def test_agent_custom_tool_success_updates_experience():
    memory = HumanLikeMemorySystem()
    agent = CognitiveAgent(memory_system=memory, auto_consolidate=False)

    def echo_tool(arguments):
        return ActionResult(True, f"echo: {arguments['input']}", cost=0.1)

    agent.add_tool("echo", "Echo the input.", echo_tool, cost=0.05)
    episode = agent.run_turn("echo hello")

    assert episode.action.name == "echo"
    assert episode.result.success is True
    assert "echo hello" in episode.result.output
    assert 0 < episode.reward < 0.7


def test_agent_synthesizes_non_respond_tool_result():
    memory = HumanLikeMemorySystem()
    agent = CognitiveAgent(memory_system=memory, auto_consolidate=False)

    def inspect_tool(arguments):
        return ActionResult(True, "raw internal state")

    def planner(observation, workspace, tools):
        return AgentAction(name="inspect", arguments={"target": "state"})

    def synthesizer(observation, workspace, action, result):
        return ActionResult(True, f"Final answer from {result.output}")

    agent.add_tool("inspect", "Inspect current state.", inspect_tool)
    agent.planner = planner
    agent.response_synthesizer = synthesizer

    episode = agent.run_turn("inspect and tell me the next step")

    assert episode.action.name == "inspect"
    assert episode.result.output == "Final answer from raw internal state"
    assert episode.result.metadata["synthesized_from_tool"] == "inspect"
    assert episode.result.metadata["tool_result"]["output"] == "raw internal state"


def test_agent_skips_synthesizer_for_respond_action():
    memory = HumanLikeMemorySystem()
    agent = CognitiveAgent(memory_system=memory, auto_consolidate=False)
    calls = []

    def planner(observation, workspace, tools):
        return AgentAction(name="respond", arguments={"message": "direct"})

    def synthesizer(observation, workspace, action, result):
        calls.append(action.name)
        return "should not run"

    agent.planner = planner
    agent.response_synthesizer = synthesizer

    episode = agent.run_turn("hello")

    assert episode.result.output == "direct"
    assert calls == []


def test_consolidate_reuses_identical_learned_procedure():
    """回归：同样的成功回合不应每次都新增一条重复程序记忆。"""
    memory = HumanLikeMemorySystem()
    agent = memory.create_agent(name="dedupe-agent")

    agent.run_turn("hello one")
    agent.run_turn("hello two")
    agent.run_turn("hello three")

    procedures = memory.get_attention_summary()["procedures"]
    respond_procs = [p for p in procedures if p["title"] == "Learned policy: respond"]
    assert len(respond_procs) == 1
    # 第 2、3 次固化只记 use，不再新增
    assert respond_procs[0]["use_count"] == 2


def test_consolidate_is_idempotent_and_serializes_stamp():
    """回归：同一 episode 固化两次不应产生重复记忆/程序。"""
    memory = HumanLikeMemorySystem()
    agent = CognitiveAgent(memory_system=memory, auto_consolidate=False)

    episode = agent.run_turn("hello")

    first = agent.experience.consolidate(memory, limit=5)
    assert first
    assert episode.consolidated_at > 0
    assert agent.experience.consolidate(memory, limit=5) == []

    restored = ExperienceEpisode.from_dict(episode.to_dict())
    assert restored.consolidated_at == episode.consolidated_at


def test_consolidate_gates_low_reward_and_compacts_runtime_episodes():
    """回归：reward=0 的失败回合不入长期记忆；runtime 回合只存紧凑摘要。"""
    memory = HumanLikeMemorySystem()
    layer = ExperienceLayer()

    failed = ExperienceEpisode(
        goal="g",
        observation=Observation(content="broken run"),
        action=AgentAction(name="broken_tool"),
        result=ActionResult(False, "Unknown tool: broken_tool"),
        reward=0.0,
        lesson="Action 'broken_tool' failed: Unknown tool: broken_tool",
    )
    scaffold = "CognitiveRuntime step 2/4\n" + ("scaffold-line " * 200)
    runtime_episode = ExperienceEpisode(
        goal="Complete user task: answer briefly",
        observation=Observation(
            content=scaffold,
            source="runtime",
            metadata={"runtime": True, "original_task": "answer briefly"},
        ),
        action=AgentAction(name="respond"),
        result=ActionResult(True, "the actual answer"),
        reward=0.7,
    )
    layer.add(failed)
    layer.add(runtime_episode)

    created = layer.consolidate(memory, limit=5)

    assert len(created) == 1
    chunk = memory.core.chunks[created[0]]
    assert "scaffold-line" not in chunk.content
    assert "answer briefly" in chunk.content
    assert "the actual answer" in chunk.content
    # 低信号回合也被打上戳，避免以后重复评估
    assert failed.consolidated_at > 0
    assert layer.consolidate(memory, limit=5) == []


def test_default_planner_skips_remember_on_negated_request():
    """回归：'do not remember this secret' 不应触发 remember 存下秘密。"""
    memory = HumanLikeMemorySystem()
    agent = CognitiveAgent(memory_system=memory, auto_consolidate=False)

    episode = agent.run_turn("please do NOT remember this secret: k-9987")

    assert episode.action.name == "respond"
    assert all("k-9987" not in chunk.content for chunk in memory.core.chunks.values())


def test_default_planner_still_remembers_on_explicit_request():
    memory = HumanLikeMemorySystem()
    agent = CognitiveAgent(memory_system=memory, auto_consolidate=False)

    episode = agent.run_turn("please remember this: I prefer green tea")

    assert episode.action.name == "remember"
    assert episode.result.success is True


def test_default_planner_ignores_tool_names_inside_runtime_scaffold():
    """回归：runtime 脚手架提到 respond/introspect 等工具名时不应被子串误触发，
    回复内容应是原始任务而不是整段脚手架。"""
    memory = HumanLikeMemorySystem()
    agent = CognitiveAgent(memory_system=memory, auto_consolidate=False)

    scaffold = (
        "CognitiveRuntime step 1/4\n"
        "Use respond for intermediate reasoning.\n"
        "Use introspect when coherence matters.\n"
        "Use finish only when done. remember the rules.\n"
    ) * 20
    observation = agent.observe(
        scaffold,
        source="runtime",
        metadata={"runtime": True, "original_task": "法国的首都是哪里？"},
    )
    workspace = memory.focus("runtime")

    action = default_planner(observation, workspace, agent.tools)

    assert action.name == "respond"
    assert action.arguments["message"] == "法国的首都是哪里？"
