"""测试多 Agent 协作

测试 Claude Code 和 Codex 能否互相聊天、纠错、分配任务。

使用方式：
    # 启动 hub 服务器
    python -m hub.cli start

    # 运行测试
    python tests/test_multi_agent.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import threading
import urllib.request
from typing import Any, Dict, List

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.claude_adapter import ClaudeAdapter
from adapters.codex_adapter import CodexAdapter


# 测试配置
HUB_URL = "http://localhost:8420"
TEST_CHANNEL = "test_multi_agent"


def _api_get(path: str) -> Dict[str, Any]:
    """GET 请求"""
    url = HUB_URL + path
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _api_post(path: str, body: dict) -> Dict[str, Any]:
    """POST 请求"""
    url = HUB_URL + path
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_basic_messaging():
    """测试基本消息收发"""
    print("=" * 60)
    print("测试 1: 基本消息收发")
    print("=" * 60)

    # 创建两个 agent
    claude = ClaudeAdapter("TestClaude", hub_url=HUB_URL)
    codex = CodexAdapter("TestCodex", hub_url=HUB_URL)

    # 收集收到的消息
    claude_received = []
    codex_received = []

    # 重写 on_message 方法
    claude.on_message = lambda msg: claude_received.append(msg)
    codex.on_message = lambda msg: codex_received.append(msg)

    # 连接
    claude.connect()
    codex.connect()

    # 等待连接建立
    time.sleep(1)

    # Claude 发送消息
    claude.send("你好 Codex，我是 Claude", channel=TEST_CHANNEL)
    time.sleep(0.5)

    # Codex 发送消息
    codex.send("你好 Claude，我是 Codex", channel=TEST_CHANNEL)
    time.sleep(0.5)

    # 验证消息
    assert len(claude_received) >= 1, f"Claude 应该收到至少 1 条消息，实际收到 {len(claude_received)}"
    assert len(codex_received) >= 1, f"Codex 应该收到至少 1 条消息，实际收到 {len(codex_received)}"

    print(f"✅ Claude 收到 {len(claude_received)} 条消息")
    print(f"✅ Codex 收到 {len(codex_received)} 条消息")

    # 断开连接
    claude.disconnect()
    codex.disconnect()

    print()


def test_mention():
    """测试 @提及"""
    print("=" * 60)
    print("测试 2: @提及")
    print("=" * 60)

    # 创建两个 agent
    claude = ClaudeAdapter("TestClaude2", hub_url=HUB_URL)
    codex = CodexAdapter("TestCodex2", hub_url=HUB_URL)

    # 收集收到的消息
    claude_received = []
    codex_received = []

    # 重写 on_message 方法
    claude.on_message = lambda msg: claude_received.append(msg)
    codex.on_message = lambda msg: codex_received.append(msg)

    # 连接
    claude.connect()
    codex.connect()

    # 等待连接建立
    time.sleep(1)

    # Claude @Codex
    claude.send("@TestCodex2 请帮我看看这段代码", channel=TEST_CHANNEL)
    time.sleep(0.5)

    # 验证 Codex 收到消息
    assert len(codex_received) >= 1, f"Codex 应该收到至少 1 条消息，实际收到 {len(codex_received)}"

    # 检查是否被 @提及
    last_msg = codex_received[-1]
    assert "@TestCodex2" in last_msg["content"], f"消息应该包含 @提及，实际内容: {last_msg['content']}"

    print(f"✅ Codex 收到 @提及消息")

    # 断开连接
    claude.disconnect()
    codex.disconnect()

    print()


def test_task():
    """测试任务分配"""
    print("=" * 60)
    print("测试 3: 任务分配")
    print("=" * 60)

    # 创建两个 agent
    claude = ClaudeAdapter("TestClaude3", hub_url=HUB_URL)
    codex = CodexAdapter("TestCodex3", hub_url=HUB_URL)

    # 收集收到的任务
    codex_tasks = []

    # 重写 on_task 方法
    codex.on_task = lambda task: codex_tasks.append(task)

    # 连接
    claude.connect()
    codex.connect()

    # 等待连接建立
    time.sleep(1)

    # Claude 分配任务给 Codex
    claude.send_task(
        task="请重构 main.py 的第 42 行",
        assignee=codex.agent_id,
        channel=TEST_CHANNEL,
        priority="high",
        context={"file": "main.py", "line": 42}
    )
    time.sleep(0.5)

    # 验证 Codex 收到任务
    assert len(codex_tasks) >= 1, f"Codex 应该收到至少 1 个任务，实际收到 {len(codex_tasks)}"

    last_task = codex_tasks[-1]
    assert "重构" in last_task["content"], f"任务内容应该包含 '重构'，实际内容: {last_task['content']}"

    print(f"✅ Codex 收到任务")

    # 断开连接
    claude.disconnect()
    codex.disconnect()

    print()


def test_correction():
    """测试纠错"""
    print("=" * 60)
    print("测试 4: 纠错")
    print("=" * 60)

    # 创建两个 agent
    claude = ClaudeAdapter("TestClaude4", hub_url=HUB_URL)
    codex = CodexAdapter("TestCodex4", hub_url=HUB_URL)

    # 收到的纠错
    codex_corrections = []

    # 重写 on_correction 方法
    codex.on_correction = lambda msg: codex_corrections.append(msg)

    # 连接
    claude.connect()
    codex.connect()

    # 等待连接建立
    time.sleep(1)

    # Claude 指出 Codex 的错误
    claude.send_correction(
        target_message_id="msg_123",
        target_agent=codex.agent_id,
        issue="这里有除零错误",
        suggestion="检查分母是否为零",
        code_fix="if denominator != 0:\n    result = numerator / denominator",
        channel=TEST_CHANNEL
    )
    time.sleep(0.5)

    # 验证 Codex 收到纠错
    assert len(codex_corrections) >= 1, f"Codex 应该收到至少 1 条纠错，实际收到 {len(codex_corrections)}"

    last_correction = codex_corrections[-1]
    assert "除零" in last_correction["content"], f"纠错内容应该包含 '除零'，实际内容: {last_correction['content']}"

    print(f"✅ Codex 收到纠错")

    # 断开连接
    claude.disconnect()
    codex.disconnect()

    print()


def main():
    """运行所有测试"""
    print("🚀 开始测试多 Agent 协作")
    print()

    try:
        test_basic_messaging()
        test_mention()
        test_task()
        test_correction()

        print("=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        return 1
    except Exception as e:
        print(f"❌ 测试错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
