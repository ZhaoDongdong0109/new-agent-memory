from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading

from new_agent_memory import (
    ActionResult,
    AgentAction,
    CognitiveAgent,
    HumanLikeMemorySystem,
    LLMPlanner,
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
    assert action.rationale == "LLMPlanner fallback"
    assert "Could not parse" in action.arguments["message"]


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


def test_llm_planner_falls_back_for_unknown_tool():
    memory = HumanLikeMemorySystem()
    agent = CognitiveAgent(memory_system=memory, auto_consolidate=False)
    planner = LLMPlanner(lambda prompt: '{"name": "delete_world", "arguments": {}}')

    action = planner(agent.observe("hello"), memory.focus("hello"), agent.tools)

    assert action.name == "respond"
    assert "unavailable tool" in action.arguments["message"]


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
