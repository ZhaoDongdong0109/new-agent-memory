from new_agent_memory import AgentAction, CognitiveAgent, HumanLikeMemorySystem


def test_focus_workspace_includes_cognitive_context():
    memory = HumanLikeMemorySystem()
    memory.start_goal(
        "Build a self-correcting agent",
        constraints=["prefer reversible actions"],
    )
    agent = memory.create_agent(auto_consolidate=False)
    observation = agent.observe(
        "How should I safely test the repo?",
        metadata={"topics": ["testing"], "location": "workspace"},
    )

    memory.observe_world(observation)
    workspace = memory.focus("safe testing")

    assert workspace.cognitive_context is not None
    assert workspace.cognitive_context["identity"]
    assert workspace.cognitive_context["open_questions"]
    assert "Cognitive State" in workspace.to_prompt_context()


def test_agent_can_introspect_and_update_tool_priors():
    memory = HumanLikeMemorySystem()
    agent = memory.create_agent(auto_consolidate=False)

    episode = agent.run_turn("please introspect current state")

    assert episode.action.name == "introspect"
    assert episode.result.success is True
    assert "drives" in episode.result.output
    assert episode.result.metadata["prediction"]["action_name"] == "introspect"

    summary = memory.get_cognitive_summary()
    assert summary["interaction_count"] == 1
    assert summary["tool_stats"]["introspect"]["successes"] == 1.0
    assert summary["recent_reflections"][0]["outcome"] == "success"


def test_failed_action_creates_reflection_and_open_question():
    memory = HumanLikeMemorySystem()
    agent = CognitiveAgent(memory_system=memory, auto_consolidate=False)

    def planner(observation, workspace, tools):
        return AgentAction(name="missing_sensor", arguments={})

    agent.planner = planner

    episode = agent.run_turn("check the outside temperature")

    assert episode.result.success is False

    summary = memory.get_cognitive_summary()
    assert summary["tool_stats"]["missing_sensor"]["failures"] == 1.0
    assert any("missing_sensor" in question for question in summary["open_questions"])
    assert summary["recent_reflections"][0]["outcome"] == "failure"


def test_cognitive_state_round_trips(tmp_path):
    data_dir = tmp_path / "memory_data"
    memory = HumanLikeMemorySystem(data_dir=str(data_dir))
    agent = memory.create_agent(auto_consolidate=False)
    agent.run_turn("please introspect current state")
    memory.save()

    loaded = HumanLikeMemorySystem(data_dir=str(data_dir))

    assert loaded.load() is True
    summary = loaded.get_cognitive_summary()
    assert summary["interaction_count"] == 1
    assert summary["tool_stats"]["introspect"]["attempts"] == 1.0

    workspace = loaded.focus("current state")
    assert workspace.cognitive_context is not None
