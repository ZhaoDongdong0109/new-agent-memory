"""统一 CLI 入口

支持启动各种 agent 适配器，接入 Hub。

使用方式：
    # 启动 Claude Code 适配器
    python -m adapters.cli claude --name "Claude"

    # 启动 Codex 适配器
    python -m adapters.cli codex --name "Codex"

    # 启动人类客户端
    python -m adapters.cli human --name "User"

    # 查看 hub 状态
    python -m adapters.cli status
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from typing import List


def _api_get(hub_url: str, path: str) -> dict:
    """GET 请求"""
    url = hub_url.rstrip("/") + path
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_claude(args: argparse.Namespace) -> int:
    """启动 Claude Code 适配器"""
    from adapters.claude_adapter import ClaudeAdapter
    agent = ClaudeAdapter(name=args.name, hub_url=args.hub)
    agent.connect()
    try:
        import time
        while agent._running:
            time.sleep(1)
    except KeyboardInterrupt:
        agent.disconnect()
    return 0


def run_codex(args: argparse.Namespace) -> int:
    """启动 Codex 适配器"""
    from adapters.codex_adapter import CodexAdapter
    agent = CodexAdapter(name=args.name, hub_url=args.hub)
    agent.connect()
    try:
        import time
        while agent._running:
            time.sleep(1)
    except KeyboardInterrupt:
        agent.disconnect()
    return 0


def run_human(args: argparse.Namespace) -> int:
    """启动人类客户端"""
    from adapters.claude_adapter import ClaudeAdapter
    # 人类客户端使用 Claude 适配器
    agent = ClaudeAdapter(name=args.name, hub_url=args.hub)
    agent.connect()
    try:
        import time
        while agent._running:
            time.sleep(1)
    except KeyboardInterrupt:
        agent.disconnect()
    return 0


def run_status(args: argparse.Namespace) -> int:
    """查看 hub 状态"""
    hub_url = (args.hub or "http://localhost:8420").rstrip("/")
    try:
        # 获取统计信息
        stats = _api_get(hub_url, "/api/stats")
        agents = stats.get("agents", {})
        print("Hub 状态:")
        print(f"  Agent:   {agents.get('online', 0)} 在线 / {agents.get('total', 0)} 总计")
        print(f"  频道:    {stats.get('channels', 0)}")
        print(f"  消息:    {stats.get('messages', 0)}")
        print(f"  运行时间: {int(stats.get('uptime_seconds', 0) / 60)} 分钟")
        print()

        # 获取 agent 列表
        agent_list = _api_get(hub_url, "/api/agents")
        agents = agent_list.get("agents", [])
        if agents:
            print("Agent 列表:")
            for agent in agents:
                status = "🟢" if agent["status"] == "online" else "⚪"
                print(f"  {status} {agent['name']} ({agent['agent_type']}) - {agent['status']}")
        else:
            print("没有注册的 agent")

        return 0

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        prog="adapters-cli",
        description="Multi-agent communication hub adapters.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Claude Code 适配器
    claude = subparsers.add_parser("claude", help="Start Claude Code adapter.")
    claude.add_argument("--name", default="Claude", help="Agent name.")
    claude.add_argument("--hub", default=None, help="Hub URL.")
    claude.set_defaults(func=run_claude)

    # Codex 适配器
    codex = subparsers.add_parser("codex", help="Start Codex adapter.")
    codex.add_argument("--name", default="Codex", help="Agent name.")
    codex.add_argument("--hub", default=None, help="Hub URL.")
    codex.set_defaults(func=run_codex)

    # 人类客户端
    human = subparsers.add_parser("human", help="Start human client.")
    human.add_argument("--name", default="User", help="Your name.")
    human.add_argument("--hub", default=None, help="Hub URL.")
    human.set_defaults(func=run_human)

    # Hub 状态
    status = subparsers.add_parser("status", help="Show hub status.")
    status.add_argument("--hub", default=None, help="Hub URL.")
    status.set_defaults(func=run_status)

    return parser


def main(argv: List[str] = None) -> int:
    """CLI 入口"""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
