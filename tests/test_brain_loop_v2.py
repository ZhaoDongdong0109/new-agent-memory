"""大脑主循环 v2 集成测试：

1. 原生 OpenAI `tools` / `tool_calls` 多消息协议（含安全守卫与 400 不支持自动回退）
2. SSE 流式输出（文本增量流式回调 + 分片 `tool_calls` 拼接）
3. CognitiveRuntime 多步循环接入会话工作记忆（多轮连续性 + 溢出归档 + 自动编码取代）
4. 第一人称画像事实与正式中文全名的自动编码与双时态取代链
5. 会话缓冲跨进程持久化（`conversation.json` 保存与恢复）
6. 回合固化 STORY 记忆不再因共享常量主题在睡眠巩固时退化为单一巨簇
7. CLI `--stream` / `--native-tools` / `--mcp-config` 端到端闭环
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import sys
import threading

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKE_MCP_SERVER = os.path.join(REPO_ROOT, "tests", "fixtures", "fake_mcp_server.py")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.agent_system import ActionResult, AgentAction  # noqa: E402
from core.cognitive_runtime import CognitiveRuntime, CognitiveRuntimeConfig  # noqa: E402
from core.llm_planner import LLMPlanner, OpenAICompatibleChatClient, OpenAICompatibleConfig  # noqa: E402
from core.weight_system import MemoryType  # noqa: E402
from main import HumanLikeMemorySystem  # noqa: E402
from new_agent_memory.cli import main as cli_main  # noqa: E402


def _make_system(tmp_path, **kwargs):
    kwargs.setdefault("enable_pii_detection", False)
    kwargs.setdefault("enable_audit_log", False)
    return HumanLikeMemorySystem(data_dir=str(tmp_path / "mem"), **kwargs)


def _start_http_server(handler_cls):
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


# ============ 1. 原生 tool-calling 多消息协议 ============


def test_native_tool_calling_sends_multi_turn_messages_and_runs_tool(tmp_path):
    captured_requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            captured_requests.append(payload)

            if len(captured_requests) == 1:
                # 第 1 轮：原生直接自然语言回复（不调工具）
                resp = {"choices": [{"message": {"role": "assistant", "content": "收到，我们在聊崇礼滑雪。"}}]}
            elif len(captured_requests) == 2:
                # 第 2 轮规划：原生 tool_calls 调用自定义工具
                resp = {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_weather_1",
                                        "type": "function",
                                        "function": {
                                            "name": "query_snow",
                                            "arguments": json.dumps({"resort": "崇礼"}, ensure_ascii=False),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            else:
                # 第 2 轮工具执行后的响应合成
                resp = {"choices": [{"message": {"role": "assistant", "content": "崇礼目前积雪厚度 80cm。"}}]}

            body = json.dumps(resp, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    server, thread = _start_http_server(Handler)
    try:
        system = _make_system(tmp_path)
        agent = system.create_openai_agent(
            env_file=None,
            api_key="k",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="fake-native",
            native_tools=True,
            auto_consolidate=False,
        )
        agent.auto_extract = False
        agent.add_tool(
            "query_snow",
            "查询滑雪场积雪厚度",
            lambda args: ActionResult(True, f"{args['resort']}: 80cm"),
            parameters={
                "type": "object",
                "properties": {"resort": {"type": "string"}},
                "required": ["resort"],
            },
        )

        ep1 = agent.run_turn("我们聊聊崇礼滑雪场")
        assert ep1.action.name == "respond"
        assert "崇礼" in ep1.result.output

        ep2 = agent.run_turn("帮我查一下那里的积雪")
        assert ep2.action.name == "query_snow"
        assert ep2.action.arguments == {"resort": "崇礼"}
        assert ep2.action.metadata.get("protocol") == "native_tools"
        assert "80cm" in ep2.result.output
    finally:
        server.shutdown()
        thread.join(timeout=5)

    second_req = captured_requests[1]
    assert "tools" in second_req
    tool_names = {t["function"]["name"] for t in second_req["tools"]}
    assert "query_snow" in tool_names
    roles = [m["role"] for m in second_req["messages"]]
    # system + 第1轮 user + 第1轮 assistant + 第2轮 user
    assert roles == ["system", "user", "assistant", "user"]


def test_native_tool_calling_enforces_negative_memory_guard_and_400_fallback(tmp_path):
    mode = {"reject_tools": False}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length).decode("utf-8"))

            if mode["reject_tools"] and "tools" in payload:
                err = b'{"error": {"message": "tools parameter is not supported by this model"}}'
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                self.wfile.write(err)
                return

            if "tools" in payload:
                # 模型错误尝试调用 remember，守卫应拦截
                resp = {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "c1",
                                        "type": "function",
                                        "function": {
                                            "name": "remember",
                                            "arguments": '{"content": "secret-token"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            else:
                resp = {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": '{"name": "respond", "arguments": {"message": "已安全直接回复"}}',
                            }
                        }
                    ]
                }

            body = json.dumps(resp, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    server, thread = _start_http_server(Handler)
    try:
        system = _make_system(tmp_path)
        planner = LLMPlanner.from_openai_compatible(
            OpenAICompatibleConfig(
                api_key="k",
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                model="m",
                native_tools=True,
            )
        )
        agent = system.create_agent(auto_consolidate=False)
        agent.planner = planner

        # 否定保存意图下，原生 remember 调用仍被守卫拦截
        ep = agent.run_turn("请不要保存这条内容：secret-token")
        assert ep.action.name == "respond"
        assert not system.core.chunks

        # 网关不支持 tools 返回 400 时，透明降级到提示 JSON 模式
        mode["reject_tools"] = True
        ep2 = agent.run_turn("普通问候")
        assert ep2.action.name == "respond"
        assert "已安全直接回复" in ep2.result.output
    finally:
        server.shutdown()
        thread.join(timeout=5)


# ============ 2. SSE 流式输出（文本流 + 分片 tool_calls） ============


def test_sse_streaming_accumulates_text_and_chunked_tool_calls():
    step = {"n": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            self.rfile.read(length)
            step["n"] += 1

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            if step["n"] == 1:
                chunks = [
                    {"choices": [{"delta": {"role": "assistant", "content": "流式"}}]},
                    {"choices": [{"delta": {"content": "输出"}}]},
                    {"choices": [{"delta": {"content": "成功！"}, "finish_reason": "stop"}]},
                ]
            else:
                chunks = [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_stream_1",
                                            "type": "function",
                                            "function": {"name": "calc_", "arguments": '{"x": '},
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {"name": "sum", "arguments": "123}"},
                                        }
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ]
                    },
                ]

            for ch in chunks:
                line = f"data: {json.dumps(ch, ensure_ascii=False)}\n\n"
                self.wfile.write(line.encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def log_message(self, format, *args):
            return

    server, thread = _start_http_server(Handler)
    tokens: list[str] = []
    try:
        cfg = OpenAICompatibleConfig(
            api_key="k",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="stream-model",
            stream=True,
            native_tools=True,
        )
        client = OpenAICompatibleChatClient(cfg, on_token=tokens.append)

        text = client.complete("测试流式")
        assert text == "流式输出成功！"
        assert tokens == ["流式", "输出", "成功！"]

        msg = client.complete_messages(
            [{"role": "user", "content": "算一下"}],
            tools=[{"type": "function", "function": {"name": "calc_sum", "parameters": {"type": "object"}}}],
        )
        tc = msg["tool_calls"][0]
        assert tc["id"] == "call_stream_1"
        assert tc["function"]["name"] == "calc_sum"
        assert json.loads(tc["function"]["arguments"]) == {"x": 123}
    finally:
        server.shutdown()
        thread.join(timeout=5)


# ============ 3. CognitiveRuntime 接入会话工作记忆与自动编码 ============


def test_cognitive_runtime_maintains_conversation_and_auto_encodes_supersession(tmp_path):
    system = _make_system(tmp_path)
    agent = system.create_agent(auto_consolidate=False)
    agent.conversation.max_turns = 2

    seen_conversations = []

    def planner(observation, workspace, tools):
        seen_conversations.append(observation.metadata.get("conversation", ""))
        return AgentAction(name="finish", arguments={"message": "已记录"})

    agent.planner = planner
    runtime = CognitiveRuntime(agent, config=CognitiveRuntimeConfig(max_steps=2, consolidate=False))

    run1 = runtime.run("小李现在住在里斯本")
    assert run1.completed is True
    v1_facts = [
        c for c in system.core.chunks.values()
        if c.memory_type == MemoryType.FACT and "里斯本" in c.content
    ]
    assert v1_facts

    run2 = runtime.run("小李现在住在柏林")
    assert run2.completed is True
    # 第 2 轮 runtime 步骤应当看到第 1 轮对话历史
    assert "里斯本" in seen_conversations[1]
    # 事实改口在 runtime 路径同样触发双时态取代：旧 FACT 归档至伪遗忘层，当前有效 FACT 为柏林
    assert system.core.get(v1_facts[0].id) is None
    assert system.forgotten.get(v1_facts[0].id).metadata.get("superseded_by")
    res = system.retrieve("小李现在住在哪")
    fact_contents = " ".join(
        c.content for c in res.chunks if not c.metadata.get("conversation_archive")
    )
    assert "柏林" in fact_contents and "里斯本" not in fact_contents
    # 溢出轮次已归档进长期记忆，且步骤 episode 不残留历史块膨胀
    assert any(c.metadata.get("conversation_archive") for c in system.core.chunks.values())
    assert all("conversation" not in ep.observation.metadata for ep in agent.experience.episodes)


# ============ 4. 第一人称与正式姓名自动编码 ============


def test_first_person_and_fullname_facts_auto_encode_and_supersede(tmp_path):
    system = _make_system(tmp_path)
    agent = system.create_agent(auto_consolidate=False)

    # 第一人称住址 + 填充语剥离 + 改口取代
    agent.run_turn("我现在住在上海，周末准备去爬山")
    agent.run_turn("我现在住在北京")
    res = system.retrieve("我现在住在哪")
    contents = " ".join(c.content for c in res.chunks)
    assert "北京" in contents and "上海" not in contents
    assert all("爬山" not in c.content for c in system.core.chunks.values() if c.memory_type == MemoryType.FACT)

    # 暂态动作不应误存为永久 FACT
    before_facts = len([c for c in system.core.chunks.values() if c.memory_type == MemoryType.FACT])
    agent.run_turn("我在调试记忆系统的检索模块")
    after_facts = len([c for c in system.core.chunks.values() if c.memory_type == MemoryType.FACT])
    assert after_facts == before_facts

    # 正式中文全名主语抽取
    agent.run_turn("王明现在住在深圳")
    assert any("深圳" in c.content and "王明" in c.persons for c in system.core.chunks.values())


# ============ 5. 会话缓冲跨进程持久化 + 6. 睡眠聚类去退化 ============


def test_conversation_persists_across_save_and_load(tmp_path):
    sys1 = _make_system(tmp_path)
    agent1 = sys1.create_agent(auto_consolidate=False)
    agent1.auto_extract = False
    agent1.run_turn("我正在设计机器人路径规划实验")
    sys1.save()

    sys2 = _make_system(tmp_path)
    assert sys2.load() is True
    agent2 = sys2.create_agent(auto_consolidate=False)
    assert any("路径规划" in t.content for t in agent2.conversation.turns)


def test_episode_consolidation_does_not_collapse_unrelated_stories_in_sleep(tmp_path):
    system = _make_system(tmp_path)
    agent = system.create_agent(auto_consolidate=True)
    agent.auto_extract = False

    # 2 条美食话题 + 2 条电影话题（各自不足 3 条最小簇规模）
    agent.run_turn("今天去餐厅吃烤鸭")
    agent.run_turn("晚上做饭煲汤")
    agent.run_turn("周末看了一场科幻电影")
    agent.run_turn("昨晚重温老电影")

    report = system.sleep()
    assert report.gists_created == [], "不相关的 Agent 回合因常量标签被错误聚成同一要点"


# ============ 7. CLI 端到端：--stream + --native-tools + --mcp-config ============


def test_cli_ask_with_stream_native_tools_and_mcp_config(tmp_path, capsys):
    mcp_cfg = tmp_path / ".mcp.json"
    mcp_cfg.write_text(
        json.dumps({"mcpServers": {"demo": {"command": sys.executable, "args": [FAKE_MCP_SERVER]}}}),
        encoding="utf-8",
    )
    calls = {"n": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            calls["n"] += 1

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            if calls["n"] == 1:
                assert any(t["function"]["name"] == "mcp__demo__add_numbers" for t in payload.get("tools", []))
                chunk = {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_mcp_1",
                                        "type": "function",
                                        "function": {
                                            "name": "mcp__demo__add_numbers",
                                            "arguments": '{"a": 7, "b": 35}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            else:
                chunk = {
                    "choices": [
                        {"delta": {"content": "计算结果是 42。"}, "finish_reason": "stop"}
                    ]
                }
            self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def log_message(self, format, *args):
            return

    server, thread = _start_http_server(Handler)
    try:
        code = cli_main(
            [
                "ask",
                "--data-dir",
                str(tmp_path / "memdata"),
                "--env-file",
                "",
                "--api-key",
                "test",
                "--base-url",
                f"http://127.0.0.1:{server.server_port}/v1",
                "--model",
                "fake-model",
                "--native-tools",
                "--stream",
                "--mcp-config",
                str(mcp_cfg),
                "7 加 35 等于多少",
            ]
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)

    out = capsys.readouterr().out
    assert code == 0
    assert "计算结果是 42。" in out
    # 流式已实时打印过一次，不应重复打印第二遍
    assert out.count("计算结果是 42。") == 1
