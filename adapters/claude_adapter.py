"""Claude Code 适配器

让 Claude Code 能够接入 Hub，与其他 agent 交流。

功能：
- 自动连接到 hub
- 实时接收消息
- 支持 @提及回复
- 支持任务接收和执行
- 支持纠错消息

使用方式：
    # 作为独立进程运行
    python -m adapters.claude_adapter --name "Claude"

    # 或者在代码中使用
    from adapters.claude_adapter import ClaudeAdapter
    agent = ClaudeAdapter("Claude")
    agent.connect()
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from adapters.base_adapter import BaseAdapter

try:
    from hub_data.queue_protocol import QueueProtocol
except Exception:  # pragma: no cover - optional local file protocol
    QueueProtocol = None


class ClaudeAdapter(BaseAdapter):
    """Claude Code 适配器"""

    def __init__(self, name: str = "Claude", hub_url: str = None):
        """
        初始化 Claude Code 适配器

        Args:
            name: agent 名称
            hub_url: hub 服务器地址
        """
        super().__init__(name, agent_type="claude", hub_url=hub_url)
        self.hub_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hub_data")
        self.messages_dir = os.path.join(self.hub_data_dir, "messages")
        self.agent_file_name = name.lower()
        self.inbox_file = os.path.join(self.messages_dir, f"{self.agent_file_name}_inbox.json")
        self.outbox_file = os.path.join(self.messages_dir, f"{self.agent_file_name}_outbox.json")
        self.queue = QueueProtocol(self.agent_file_name, base_dir=self.messages_dir) if QueueProtocol else None
        self.outbox_queue = QueueProtocol(f"{self.agent_file_name}_outbox", base_dir=self.messages_dir) if QueueProtocol else None
        self._input_thread: Optional[threading.Thread] = None

    def connect(self):
        """连接到 hub"""
        super().connect()
        self._start_input_listener()
        self._print_help()

    def _print_help(self):
        """打印帮助信息"""
        print(f"[{self.name}] 命令：")
        print(f"  :help          - 显示帮助")
        print(f"  :channels      - 列出频道")
        print(f"  :agents        - 列出 agent")
        print(f"  :switch <id>   - 切换频道")
        print(f"  :quit          - 退出")
        print(f"  @<name> <msg>  - @提及某个 agent")
        print(f"  其他内容       - 发送到当前频道")
        print()

    def _start_input_listener(self):
        """启动输入监听线程"""
        def _input_loop():
            current_channel = "general"
            while self._running:
                try:
                    line = input()
                    if not line:
                        continue

                    # 处理命令
                    if line.startswith(":"):
                        self._handle_command(line, current_channel)
                        continue

                    # 处理 @提及
                    if line.startswith("@"):
                        self._handle_mention(line, current_channel)
                        continue

                    # 普通消息
                    self.send(line, channel=current_channel)

                except EOFError:
                    break
                except KeyboardInterrupt:
                    break

        self._input_thread = threading.Thread(target=_input_loop, daemon=True)
        self._input_thread.start()

    def _handle_command(self, command: str, current_channel: str):
        """处理命令"""
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == ":help":
            self._print_help()

        elif cmd == ":channels":
            channels = self.list_channels()
            print(f"[{self.name}] 频道列表：")
            for ch in channels:
                print(f"  - {ch['id']}: {ch['name']}")

        elif cmd == ":agents":
            agents = self.list_agents()
            print(f"[{self.name}] Agent 列表：")
            for agent in agents:
                status = "🟢" if agent["status"] == "online" else "⚪"
                print(f"  {status} {agent['name']} ({agent['agent_type']})")

        elif cmd == ":switch" and len(parts) > 1:
            new_channel = parts[1]
            print(f"[{self.name}] 切换到频道: {new_channel}")
            # 这里可以实现频道切换逻辑

        elif cmd == ":quit":
            self.disconnect()
            sys.exit(0)

        else:
            print(f"[{self.name}] 未知命令: {command}")

    def _handle_mention(self, line: str, current_channel: str):
        """处理 @提及"""
        # 解析 @name message
        parts = line.split(" ", 1)
        if len(parts) < 2:
            print(f"[{self.name}] 格式: @<name> <message>")
            return

        mention = parts[0][1:]  # 去掉 @
        message = parts[1]

        # 查找目标 agent
        agents = self.list_agents()
        target_agent = None
        for agent in agents:
            if agent["name"].lower() == mention.lower():
                target_agent = agent
                break

        if not target_agent:
            print(f"[{self.name}] 找不到 agent: {mention}")
            return

        # 发送消息
        self.send(f"@{mention} {message}", channel=current_channel)

    def on_message(self, msg: Dict[str, Any]):
        """收到消息时的处理"""
        sender = msg.get("sender_name", "unknown")
        content = msg.get("content", "")
        channel = msg.get("channel_id", "general")
        message_type = msg.get("message_type", "text")

        # 打印消息
        timestamp = datetime.fromtimestamp(msg.get("created_at", 0)).strftime("%H:%M:%S")
        print(f"[{timestamp}] [{channel}] {sender}: {content}")

        # 写入 inbox 文件（供其他程序读取）
        self._write_inbox(msg)

        # 检查是否 @自己
        if self._is_mentioned(content):
            print(f"[{self.name}] 被 @{sender} 提及！")
            # 这里可以自动回复或等待用户输入

    def on_task(self, task: Dict[str, Any]):
        """收到任务时的处理"""
        sender = task.get("sender_name", "unknown")
        content = task.get("content", "")
        priority = task.get("metadata", {}).get("priority", "normal")

        print(f"[{self.name}] 收到任务 (优先级: {priority}):")
        print(f"  来自: {sender}")
        print(f"  任务: {content}")

        # 写入任务文件
        self._write_inbox(task)
        self._write_task(task)

    def on_correction(self, correction: Dict[str, Any]):
        """收到纠错时的处理"""
        sender = correction.get("sender_name", "unknown")
        content = correction.get("content", "")

        print(f"[{self.name}] 收到纠错:")
        print(f"  来自: {sender}")
        print(f"  内容: {content}")
        self._write_inbox(correction)

    def _is_mentioned(self, content: str) -> bool:
        """检查是否被 @提及"""
        content_lower = content.lower()
        return (f"@{self.name.lower()}" in content_lower or
                f"@{self.name.lower().replace(' ', '')}" in content_lower)

    def _write_inbox(self, msg: Dict[str, Any]):
        """写入 inbox 文件"""
        try:
            os.makedirs(self.messages_dir, exist_ok=True)
            with open(self.inbox_file, "w", encoding="utf-8") as f:
                json.dump(msg, f, ensure_ascii=False, indent=2)
            if self.queue:
                self.queue.write_message(msg)
        except Exception:
            pass

    def _write_task(self, task: Dict[str, Any]):
        """写入任务文件"""
        try:
            task_file = "/tmp/claude_hub_task.json"
            with open(task_file, "w", encoding="utf-8") as f:
                json.dump(task, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def check_outbox(self) -> Optional[str]:
        """检查 outbox 文件（供外部程序写入回复）"""
        if self.outbox_queue:
            try:
                queued = self.outbox_queue.read_messages()
                if queued:
                    msg = queued[0]
                    self.outbox_queue.mark_processed(msg["id"])
                    return msg.get("content")
            except Exception:
                pass

        if not os.path.exists(self.outbox_file):
            return None
        try:
            with open(self.outbox_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            os.remove(self.outbox_file)
            return data.get("content")
        except Exception:
            return None


def main(argv: List[str] = None) -> int:
    """CLI 入口"""
    parser = argparse.ArgumentParser(
        prog="claude-adapter",
        description="Claude Code adapter for Communication Hub.",
    )
    parser.add_argument("--name", default="Claude", help="Agent name.")
    parser.add_argument("--hub", default=None, help="Hub URL.")

    args = parser.parse_args(argv)

    agent = ClaudeAdapter(name=args.name, hub_url=args.hub)
    agent.connect()

    try:
        # 保持运行
        while agent._running:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        agent.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
