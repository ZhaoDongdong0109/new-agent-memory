from new_agent_memory import ActionResult, AgentAction, CognitiveAgent, HumanLikeMemorySystem


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
