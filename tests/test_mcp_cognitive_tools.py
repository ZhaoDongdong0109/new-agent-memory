"""MCP 认知工具面测试

memory_explain / memory_history / memory_sleep / memory_focus /
memory_feedback / memory_maintain 六个新工具，加上升级后的
memory_search（返回 id / 路径 / 置信度 / 弃答理由）。

直接实例化 MemoryMCPServer 调 _call_tool（stdio 协议正确性由
tests/test_mcp_server.py 的子进程测试覆盖）。
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.weight_system import MemoryType


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_DATA_DIR", str(tmp_path))
    # 禁用本地 LLM 探测，走规则抽取
    import mcp_server as mod
    monkeypatch.setattr(mod, "create_llm_fn", lambda: None)
    return mod.MemoryMCPServer()


def _call(server, name, arguments=None):
    return server._call_tool({"name": name, "arguments": arguments or {}})


def _text(result):
    assert "content" in result, result
    return result["content"][0]["text"]


def test_tools_listed(server):
    names = {t["name"] for t in server._list_tools()["tools"]}
    assert {
        "memory_search", "memory_add", "memory_stats",
        "memory_explain", "memory_history", "memory_sleep",
        "memory_focus", "memory_feedback", "memory_maintain",
    } <= names


def test_search_returns_ids(server):
    server.memory.add_memory(content="和老王确认了项目验收标准", persons=["老王"], topics=["项目"])
    text = _text(_call(server, "memory_search", {"query": "老王的项目"}))
    assert "mem_" in text, "search 结果必须携带 memory id 供溯源"
    assert "path=" in text and "confidence=" in text


def test_explain_full_breakdown(server):
    mid = server.memory.add_memory(
        content="小李喜欢埃塞俄比亚浅烘手冲",
        memory_type=MemoryType.PREFERENCE,
        persons=["小李"], topics=["偏好"], importance=0.8,
    )
    result = _call(server, "memory_explain", {"memory_id": mid})
    assert not result.get("isError")
    report = json.loads(_text(result))
    assert report["id"] == mid
    assert report["layer"] == "core"
    assert 0 < report["actr_activation"]["retention_P"] <= 1
    assert report["weight_factors"]["final_weight"] > 0
    assert report["actr_activation"]["decay_d_for_type"] == pytest.approx(0.44)


def test_explain_unknown_id(server):
    result = _call(server, "memory_explain", {"memory_id": "mem_nope"})
    assert result.get("isError")


def test_history_shows_supersession_chain(server):
    def add_city(city):
        return server.memory.add_memory(
            content=f"小李现在住在{city}",
            memory_type=MemoryType.FACT,
            persons=["小李"], topics=["居住"], keywords=["住", city],
        )

    id_v1 = add_city("里斯本")
    add_city("柏林")
    id_v3 = add_city("奥斯陆")

    # 从链上任意一环都能看到完整历史（这里用最旧的一环）
    text = _text(_call(server, "memory_history", {"memory_id": id_v1}))
    assert "CURRENT" in text and "奥斯陆" in text
    assert "superseded" in text and "里斯本" in text and "柏林" in text

    # 从链头也一样
    text_head = _text(_call(server, "memory_history", {"memory_id": id_v3}))
    assert "里斯本" in text_head


def test_sleep_consolidates_and_reports(server):
    for detail in ["聊了预算", "定了时间表", "评审了方案", "确认了验收"]:
        server.memory.add_memory(
            content=f"和老王开项目会{detail}",
            persons=["老王"], topics=["工作", "项目"], importance=0.6,
        )
    text = _text(_call(server, "memory_sleep"))
    assert "gists=1" in text
    assert "gist " in text and "4 episodes" in text


def test_focus_returns_workspace(server):
    server.memory.start_goal("修复测试失败并安全推送 PR", constraints=["先跑 pytest"])
    server.memory.add_memory(
        content="上次 maintain 失败是因为缺少 decay_all_unused",
        memory_type=MemoryType.FACT, topics=["pytest", "bugfix"], importance=0.7,
    )
    text = _text(_call(server, "memory_focus", {"query": "pytest 失败后怎么继续"}))
    assert "修复测试失败" in text  # 目标进入工作区


def test_feedback_adjusts_weights(server):
    mid = server.memory.add_memory(content="东京塔高度是 333 米", topics=["事实"], keywords=["东京塔"])
    before = server.memory.core.get(mid).recall_bias
    _call(server, "memory_feedback", {"query": "东京塔", "accepted": False})
    after = server.memory.core.get(mid).recall_bias
    assert after < before, "纠错反馈未降低 recall_bias"


def test_maintain_reports_counts(server):
    server.memory.add_memory(content="一条普通记忆", topics=["测试"])
    text = _text(_call(server, "memory_maintain"))
    assert "core:" in text and "forgotten:" in text


def test_add_with_fact_type_gets_supersession(server):
    """MCP memory_add 指定 fact 类型后进入双时态取代管理"""
    _call(server, "memory_add", {"content": "服务当前端口是 8420", "memory_type": "fact", "topics": ["配置"]})
    _call(server, "memory_add", {"content": "服务当前端口是 9000", "memory_type": "fact", "topics": ["配置"]})

    text = _text(_call(server, "memory_search", {"query": "现在的端口是多少"}))
    assert "9000" in text
    assert "8420" not in text, "被取代的旧端口冒充了现状"
