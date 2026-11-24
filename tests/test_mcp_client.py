"""大脑主循环 v2：外部 MCP 客户端与工具挂载测试。"""

from __future__ import annotations

import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKE_SERVER = os.path.join(REPO_ROOT, "tests", "fixtures", "fake_mcp_server.py")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.agent_system import AgentAction  # noqa: E402
from core.mcp_client import (  # noqa: E402
    MCPError,
    MCPStdioClient,
    MCPToolHub,
    load_mcp_config,
    mcp_result_to_action_result,
    register_mcp_tools,
    sanitize_mcp_tool_name,
)
from main import HumanLikeMemorySystem  # noqa: E402


def _make_system(tmp_path):
    return HumanLikeMemorySystem(
        data_dir=str(tmp_path / "mem"),
        enable_pii_detection=False,
        enable_audit_log=False,
    )


def test_mcp_stdio_client_initialize_paginate_and_call():
    with MCPStdioClient(sys.executable, [FAKE_SERVER], name="demo", timeout=10.0) as client:
        assert client.is_running
        assert client.server_info.get("name") == "fake-mcp"

        tools = client.list_tools()
        names = [t["name"] for t in tools]
        # 验证 nextCursor 翻页合并了两页工具
        assert names == ["echo", "add_numbers", "slow_op"]

        res = client.call_tool("echo", {"text": "你好 MCP"})
        assert res["isError"] is False
        assert res["content"][0]["text"] == "ECHO:你好 MCP"

        err_res = client.call_tool("missing_tool", {})
        action_res = mcp_result_to_action_result(err_res, server="demo", tool="missing_tool")
        assert action_res.success is False
        assert "Unknown tool" in action_res.output

        with pytest.raises(MCPError, match="-32601"):
            client.request("nonexistent/method")

    assert not client.is_running


def test_mcp_stdio_client_timeout_raises_typed_error():
    with MCPStdioClient(sys.executable, [FAKE_SERVER], name="demo", timeout=0.25) as client:
        with pytest.raises(MCPError, match="timed out"):
            client.call_tool("slow_op", {"seconds": 1.5})


def test_register_mcp_tools_into_cognitive_agent(tmp_path):
    system = _make_system(tmp_path)
    agent = system.create_agent(auto_consolidate=False)
    agent.auto_extract = False

    # 预先注册一个同名工具以验证不会覆盖已有工具
    agent.add_tool("mcp__demo__slow_op", "Existing local tool", lambda args: None)

    with MCPStdioClient(sys.executable, [FAKE_SERVER], name="demo", timeout=10.0) as client:
        registered = register_mcp_tools(agent, client)
        assert "mcp__demo__echo" in registered
        assert "mcp__demo__add_numbers" in registered
        assert "mcp__demo__slow_op" not in registered

        described = {t["name"]: t for t in agent.tools.describe()}
        assert described["mcp__demo__add_numbers"]["parameters"]["required"] == ["a", "b"]
        assert described["mcp__demo__echo"]["description"].startswith("[MCP:demo]")

        result = agent.tools.run(
            AgentAction(name="mcp__demo__add_numbers", arguments={"a": 19, "b": 23})
        )
        assert result.success is True
        assert result.output == "42"
        assert result.metadata["kind"] == "mcp_tool"


def test_sanitize_mcp_tool_name_enforces_charset_and_64_char_limit():
    short = sanitize_mcp_tool_name("my.server/1", "read:file")
    assert short == "mcp__my_server_1__read_file"

    long_a = sanitize_mcp_tool_name("server", "a" * 80 + "_one")
    long_b = sanitize_mcp_tool_name("server", "a" * 80 + "_two")
    assert len(long_a) <= 64 and len(long_b) <= 64
    assert long_a != long_b


def test_mcp_tool_hub_loads_config_and_isolates_broken_server(tmp_path):
    cfg_path = tmp_path / ".mcp.json"
    cfg_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "good": {"command": sys.executable, "args": [FAKE_SERVER]},
                    "broken": {"command": "definitely_nonexistent_binary_xyz_999"},
                    "remote_skipped": {"url": "https://example.com/mcp"},
                    "disabled_srv": {"command": sys.executable, "disabled": True},
                }
            }
        ),
        encoding="utf-8",
    )

    parsed = load_mcp_config(str(cfg_path))
    assert set(parsed.keys()) == {"good", "broken"}

    system = _make_system(tmp_path)
    agent = system.create_agent(auto_consolidate=False)

    with MCPToolHub.from_config(str(cfg_path), timeout=10.0) as hub:
        tools = hub.attach(agent)
        assert "mcp__good__echo" in tools
        assert "broken" in hub.errors
        res = agent.tools.run(AgentAction(name="mcp__good__echo", arguments={"text": "hub-ok"}))
        assert res.success is True
        assert res.output == "ECHO:hub-ok"
