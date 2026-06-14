from new_agent_memory import ActionResult, AgentAction, CognitiveAgent, HumanLikeMemorySystem, LLMPlanner


def test_llm_planner_parses_json_action():
    memory = HumanLikeMemorySystem()
    agent = CognitiveAgent(memory_system=memory, auto_consolidate=False)

    def fake_llm(prompt):
        return '{"name": "respond", "arguments": {"message": "hello"}, "rationale": "answer directly"}'

    planner = LLMPlanner(fake_llm)
    observation = agent.observe("hello")
    workspace = memory.focus("hello")

    action = planner(observation, workspace, agent.tools)

    assert action.name == "respond"
    assert action.arguments["message"] == "hello"
    assert "Available tools" in planner.last_prompt


def test_llm_planner_parses_fenced_json_and_uses_tool():
    memory = HumanLikeMemorySystem()
    agent = CognitiveAgent(memory_system=memory, auto_consolidate=False)

    def inspect_tool(arguments):
        return ActionResult(True, f"inspected {arguments['target']}")

    agent.add_tool("inspect", "Inspect a target.", inspect_tool)

    def fake_llm(prompt):
        return """```json
{"name": "inspect", "arguments": {"target": "workspace"}, "rationale": "need inspection"}
```"""

    agent.planner = LLMPlanner(fake_llm)
    episode = agent.run_turn("please inspect")

    assert episode.action.name == "inspect"
    assert episode.result.success is True
    assert "workspace" in episode.result.output


def test_llm_planner_falls_back_for_invalid_json():
    memory = HumanLikeMemorySystem()
    agent = CognitiveAgent(memory_system=memory, auto_consolidate=False)
    planner = LLMPlanner(lambda prompt: "not json")

    action = planner(agent.observe("hello"), memory.focus("hello"), agent.tools)

    assert action.name == "respond"
    assert action.rationale == "LLMPlanner fallback"
    assert "Could not parse" in action.arguments["message"]


def test_llm_planner_falls_back_for_unknown_tool():
    memory = HumanLikeMemorySystem()
    agent = CognitiveAgent(memory_system=memory, auto_consolidate=False)
    planner = LLMPlanner(lambda prompt: '{"name": "delete_world", "arguments": {}}')

    action = planner(agent.observe("hello"), memory.focus("hello"), agent.tools)

    assert action.name == "respond"
    assert "unavailable tool" in action.arguments["message"]
