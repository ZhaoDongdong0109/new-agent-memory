from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading

from new_agent_memory import (
    ActionResult,
    AgentAction,
    CognitiveAgent,
    HumanLikeMemorySystem,
    LLMPlanner,
    LLMPlannerConfig,
    LLMResponseSynthesizer,
    OpenAICompatibleChatClient,
    OpenAICompatibleConfig,
)


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
    assert action.rationale == "LLMPlanner direct response fallback"
    assert action.arguments["message"] == "not json"


def test_llm_planner_repairs_invalid_json_once():
    memory = HumanLikeMemorySystem()
    agent = CognitiveAgent(memory_system=memory, auto_consolidate=False)
    outputs = iter(
        [
            '```json\n{"name": "respond", "arguments": {"message": "broken"',
            '{"name": "respond", "arguments": {"message": "repaired"}, "rationale": "retry"}',
        ]
    )
    planner = LLMPlanner(lambda prompt: next(outputs))

    action = planner(agent.observe("hello"), memory.focus("hello"), agent.tools)

    assert action.name == "respond"
    assert action.arguments["message"] == "repaired"
    assert "Previous invalid output" in planner.last_prompt
    assert "Original task context" in planner.last_prompt
    assert "Observation:" in planner.last_prompt


def test_llm_planner_uses_heuristic_for_memory_request_after_invalid_json():
    memory = HumanLikeMemorySystem()
    agent = CognitiveAgent(memory_system=memory, auto_consolidate=False)
    planner = LLMPlanner(lambda prompt: "")

    action = planner(agent.observe("请把这个结论保存成记忆：需要更强的目标生成"), memory.focus("保存"), agent.tools)

    assert action.name == "remember"
    assert "需要更强的目标生成" in action.arguments["content"]


def test_llm_planner_heuristic_respects_negative_memory_intent():
    memory = HumanLikeMemorySystem()
    agent = CognitiveAgent(memory_system=memory, auto_consolidate=False)
    planner = LLMPlanner(lambda prompt: "")

    action = planner(agent.observe("please introspect current state; do not save this"), memory.focus("introspect"), agent.tools)

    assert action.name == "introspect"


def test_llm_planner_skips_tool_heuristics_inside_runtime_context():
    memory = HumanLikeMemorySystem()
    agent = CognitiveAgent(memory_system=memory, auto_consolidate=False)
    outputs = iter(["", "runtime direct answer"])
    planner = LLMPlanner(lambda prompt: next(outputs), config=LLMPlannerConfig(json_repair_attempts=0))
    observation = agent.observe(
        "CognitiveRuntime trace mentions remember and introspect many times.",
        source="runtime",
        metadata={"runtime": True, "original_task": "answer first, then self-check"},
    )

    action = planner(observation, memory.focus("runtime"), agent.tools)

    assert action.name == "respond"
    assert action.arguments["message"] == "runtime direct answer"


def test_llm_planner_rejects_runtime_remember_when_original_task_did_not_ask_to_save():
    memory = HumanLikeMemorySystem()
    agent = CognitiveAgent(memory_system=memory, auto_consolidate=False)
    outputs = iter(
        [
            '{"name": "remember", "arguments": {"content": "wrong"}, "rationale": "trace mentioned memory"}',
            "runtime answer instead",
        ]
    )
    planner = LLMPlanner(lambda prompt: next(outputs), config=LLMPlannerConfig(json_repair_attempts=0))
    observation = agent.observe(
        "CognitiveRuntime trace mentions remember.",
        source="runtime",
        metadata={"runtime": True, "original_task": "answer the question; do not save"},
    )

    action = planner(observation, memory.focus("runtime"), agent.tools)

    assert action.name == "respond"
    assert action.arguments["message"] == "runtime answer instead"


def test_llm_planner_rejects_remember_without_explicit_write_intent():
    memory = HumanLikeMemorySystem()
    agent = CognitiveAgent(memory_system=memory, auto_consolidate=False)
    outputs = iter(
        [
            '{"name": "remember", "arguments": {"content": "wrong"}, "rationale": "topic mentions memory"}',
            "应该先回答自检问题，而不是保存记忆。",
        ]
    )
    planner = LLMPlanner(lambda prompt: next(outputs))

    action = planner(agent.observe("作为记忆智能体，你最弱的能力是什么？"), memory.focus("自检"), agent.tools)

    assert action.name == "respond"
    assert action.arguments["message"] == "应该先回答自检问题，而不是保存记忆。"
    assert action.rationale == "LLMPlanner direct response fallback"


def test_llm_planner_rejects_remember_when_user_negates_save_intent():
    memory = HumanLikeMemorySystem()
    agent = CognitiveAgent(memory_system=memory, auto_consolidate=False)
    outputs = iter(
        [
            '{"name": "remember", "arguments": {"content": "wrong"}, "rationale": "misread save"}',
            "我会直接回答，不会保存这条内容。",
        ]
    )
    planner = LLMPlanner(lambda prompt: next(outputs))

    action = planner(agent.observe("请回答，不要保存记忆：你的目标是什么？"), memory.focus("自检"), agent.tools)

    assert action.name == "respond"
    assert action.arguments["message"] == "我会直接回答，不会保存这条内容。"


def test_llm_planner_direct_response_fallback_for_plain_chat():
    memory = HumanLikeMemorySystem()
    agent = CognitiveAgent(memory_system=memory, auto_consolidate=False)
    outputs = iter(["", "", "你好，我是一个记忆驱动的智能体。"])
    planner = LLMPlanner(lambda prompt: next(outputs))

    action = planner(agent.observe("你好"), memory.focus("你好"), agent.tools)

    assert action.name == "respond"
    assert action.arguments["message"] == "你好，我是一个记忆驱动的智能体。"
    assert action.rationale == "LLMPlanner direct response fallback"
    assert "answer the user directly" in planner.last_prompt


def test_llm_planner_falls_back_for_unknown_tool():
    memory = HumanLikeMemorySystem()
    agent = CognitiveAgent(memory_system=memory, auto_consolidate=False)
    planner = LLMPlanner(lambda prompt: '{"name": "delete_world", "arguments": {}}')

    action = planner(agent.observe("hello"), memory.focus("hello"), agent.tools)

    assert action.name == "respond"
    assert "unavailable tool" in action.arguments["message"]


def test_llm_planner_honors_explicit_tool_request_over_respond():
    memory = HumanLikeMemorySystem()
    agent = CognitiveAgent(memory_system=memory, auto_consolidate=False)
    planner = LLMPlanner(lambda prompt: '{"name": "respond", "arguments": {"message": "old context"}}')

    action = planner(agent.observe("please call introspect and then answer"), memory.focus("old context"), agent.tools)

    assert action.name == "introspect"
    assert action.rationale == "LLMPlanner guard: user explicitly requested tool 'introspect'."


def test_llm_response_synthesizer_answers_from_tool_result():
    memory = HumanLikeMemorySystem()
    agent = CognitiveAgent(memory_system=memory, auto_consolidate=False)

    def fake_llm(prompt):
        assert "Original user message:" in prompt
        assert "Tool result:" in prompt
        assert "raw cognitive summary" in prompt
        return "Next smallest experiment: review one dialogue, extract a hypothesis, then test recall."

    synthesizer = LLMResponseSynthesizer(fake_llm)
    observation = agent.observe("introspect and propose the next experiment")
    workspace = memory.focus("introspect")
    action = AgentAction(name="introspect", rationale="need self state")
    result = ActionResult(True, "raw cognitive summary")

    synthesized = synthesizer(observation, workspace, action, result)

    assert synthesized is not None
    assert synthesized.success is True
    assert "Next smallest experiment" in synthesized.output
    assert synthesized.metadata["kind"] == "llm_response_synthesis"
    assert "Do not return JSON" in synthesizer.last_prompt


def test_llm_response_synthesizer_rewrites_incomplete_output():
    memory = HumanLikeMemorySystem()
    agent = CognitiveAgent(memory_system=memory, auto_consolidate=False)
    outputs = iter(["Next experiment: start a goal (", "Next experiment: start one active goal and test recall."])

    synthesizer = LLMResponseSynthesizer(lambda prompt: next(outputs))
    observation = agent.observe("introspect and propose the next experiment")
    workspace = memory.focus("introspect")
    action = AgentAction(name="introspect")
    result = ActionResult(True, "curiosity is low")

    synthesized = synthesizer(observation, workspace, action, result)

    assert synthesized.output == "Next experiment: start one active goal and test recall."
    assert synthesized.metadata["rewritten_incomplete_output"] is True
    assert "Rewrite it as one complete" in synthesizer.last_prompt
    assert "Original user message is the task" in synthesizer.last_prompt


def test_llm_response_synthesizer_rewrites_empty_output():
    memory = HumanLikeMemorySystem()
    agent = CognitiveAgent(memory_system=memory, auto_consolidate=False)
    outputs = iter(["", "Complete answer after empty synthesis."])

    synthesizer = LLMResponseSynthesizer(lambda prompt: next(outputs))
    observation = agent.observe("introspect and answer")
    workspace = memory.focus("introspect")
    action = AgentAction(name="introspect")
    result = ActionResult(True, "state summary")

    synthesized = synthesizer(observation, workspace, action, result)

    assert synthesized.output == "Complete answer after empty synthesis."
    assert synthesized.metadata["rewritten_empty_output"] is True


def test_llm_response_synthesizer_omits_prior_open_questions_unless_requested():
    memory = HumanLikeMemorySystem()
    agent = CognitiveAgent(memory_system=memory, auto_consolidate=False)
    prompts = []

    def fake_llm(prompt):
        prompts.append(prompt)
        return "current task answer"

    synthesizer = LLMResponseSynthesizer(fake_llm)
    observation = agent.observe("run an experiment and evaluate coherence")
    workspace = memory.focus("experiment")
    workspace.cognitive_context = {"open_questions": ["old workspace question"]}
    action = AgentAction(name="introspect")
    result = ActionResult(
        True,
        "identity: agent\nopen_questions:\n- old question that should not steer the answer\ntool_stats:\n- introspect: 1/1 successes",
    )

    synthesizer(observation, workspace, action, result)

    assert "old question that should not steer the answer" not in prompts[0]
    assert "old workspace question" not in prompts[0]
    assert "prior open questions are not the current task" in prompts[0]


def test_openai_compatible_config_loads_dotenv(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=local-key",
                "OPENAI_BASE_URL=http://localhost:1234/v1",
                "OPENAI_MODEL=hermes-test",
                "OPENAI_TEMPERATURE=0.25",
                "OPENAI_MAX_TOKENS=256",
            ]
        ),
        encoding="utf-8",
    )

    config = OpenAICompatibleConfig.from_env(env_file=str(env_file), environ={})

    assert config.api_key == "local-key"
    assert config.base_url == "http://localhost:1234/v1"
    assert config.model == "hermes-test"
    assert config.temperature == 0.25
    assert config.max_tokens == 256
    assert config.chat_completions_url == "http://localhost:1234/v1/chat/completions"


def test_openai_compatible_client_posts_chat_completion():
    state = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            state["path"] = self.path
            state["authorization"] = self.headers.get("Authorization")
            state["payload"] = json.loads(self.rfile.read(length).decode("utf-8"))

            response = {
                "choices": [
                    {
                        "message": {
                            "content": '{"name": "respond", "arguments": {"message": "ok"}, "rationale": "test"}'
                        }
                    }
                ]
            }
            body = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        config = OpenAICompatibleConfig(
            api_key="test-key",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="fake-chat-model",
            timeout=5,
        )
        client = OpenAICompatibleChatClient(config)

        output = client("choose an action")
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert json.loads(output)["name"] == "respond"
    assert state["path"] == "/v1/chat/completions"
    assert state["authorization"] == "Bearer test-key"
    assert state["payload"]["model"] == "fake-chat-model"
    assert state["payload"]["messages"][1]["content"] == "choose an action"


def test_create_openai_agent_runs_against_compatible_endpoint():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            self.rfile.read(length)
            response = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "name": "respond",
                                    "arguments": {"message": "agent online"},
                                    "rationale": "single-turn smoke test",
                                }
                            )
                        }
                    }
                ]
            }
            body = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        memory = HumanLikeMemorySystem()
        agent = memory.create_openai_agent(
            env_file=None,
            api_key="test",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="fake-chat-model",
            auto_consolidate=False,
        )
        episode = agent.run_turn("hello")
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert episode.action.name == "respond"
    assert episode.result.success is True
    assert "agent online" in episode.result.output
