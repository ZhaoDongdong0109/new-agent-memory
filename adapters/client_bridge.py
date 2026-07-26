"""客户端 Agent 桥接器

把任何命令行程序（stdin/stdout）桥接到 Communication Hub。
适合千问、Codex 等客户端形式的 agent。

原理：
1. 启动子进程（agent 的 CLI）
2. 轮询 Hub 获取新消息，写入子进程的 stdin
3. 读取子进程的 stdout，发回 Hub

用法：
  python adapters/client_bridge.py --name "千问" --cmd "qwen-cli chat"
  python adapters/client_bridge.py --name "Codex" --cmd "codex-cli"

环境变量：
  HUB_URL — Hub 地址 (默认 http://localhost:8420)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
import urllib.request
from typing import Any, Dict, List, Optional


def _hub_url() -> str:
    return os.environ.get("HUB_URL", "http://localhost:8420").rstrip("/")


class ClientBridge:
    """Bridges a CLI-based agent to the Communication Hub."""

    def __init__(
        self,
        name: str,
        cmd: str,
        hub_url: str = "http://localhost:8420",
        agent_type: str = "generic",
        color: str = "#6366F1",
        emoji: str = "🤖",
        channel: str = "general",
        poll_interval: float = 2.0,
    ):
        self.name = name
        self.cmd = cmd
        self.hub_url = hub_url.rstrip("/")
        self.agent_type = agent_type
        self.color = color
        self.emoji = emoji
        self.channel = channel
        self.poll_interval = poll_interval
        self.agent_id: Optional[str] = None
        self._seen_ids: set = set()
        self._process: Optional[subprocess.Popen] = None
        self._stdout_lines: List[str] = []
        self._stdout_lock = threading.Lock()

    # -- Hub API --------------------------------------------------------------

    def hub_register(self) -> str:
        data = self._hub_post("/api/agents/register", {
            "name": self.name,
            "agent_type": self.agent_type,
            "color": self.color,
            "avatar_emoji": self.emoji,
        })
        self.agent_id = data["id"]
        return self.agent_id

    def hub_send(self, content: str):
        if not self.agent_id:
            self.hub_register()
        self._hub_post(f"/api/channels/{self.channel}/messages", {
            "sender_id": self.agent_id,
            "content": content,
            "message_type": "text",
        })

    def hub_poll(self):
        data = self._hub_get(f"/api/channels/{self.channel}/messages?limit=50")
        messages = data.get("messages", [])
        new_msgs = [m for m in messages if m["id"] not in self._seen_ids]
        for m in new_msgs:
            self._seen_ids.add(m["id"])
        return new_msgs

    def hub_heartbeat(self, status: str = "online"):
        if self.agent_id:
            self._hub_post(f"/api/agents/{self.agent_id}/heartbeat", {"status": status})

    # -- Process management ---------------------------------------------------

    def start_process(self):
        """Start the child process."""
        print(f"[{self.name}] Starting: {self.cmd}")
        self._process = subprocess.Popen(
            self.cmd,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # Background thread to read stdout
        reader = threading.Thread(target=self._read_stdout, daemon=True)
        reader.start()

    def _read_stdout(self):
        """Read lines from the child process stdout."""
        if not self._process or not self._process.stdout:
            return
        for line in self._process.stdout:
            line = line.strip()
            if line:
                with self._stdout_lock:
                    self._stdout_lines.append(line)
        # Process ended
        with self._stdout_lock:
            self._stdout_lines.append(f"[{self.name}] Process exited")

    def send_to_process(self, text: str):
        """Write a line to the child process stdin."""
        if self._process and self._process.stdin:
            try:
                self._process.stdin.write(text + "\n")
                self._process.stdin.flush()
            except Exception as exc:
                print(f"[{self.name}] stdin write error: {exc}")

    def drain_stdout(self) -> List[str]:
        """Get and clear buffered stdout lines."""
        with self._stdout_lock:
            lines = list(self._stdout_lines)
            self._stdout_lines.clear()
        return lines

    # -- Main loop ------------------------------------------------------------

    def run(self):
        """Main event loop."""
        self.hub_register()
        self.hub_heartbeat("online")
        self.start_process()

        print(f"[{self.name}] Connected. Channel: #{self.channel}")
        print(f"[{self.name}] Watching for hub messages and forwarding to process...")

        try:
            while True:
                # Hub → Process
                try:
                    new_msgs = self.hub_poll()
                    for msg in new_msgs:
                        if msg.get("sender_id") == self.agent_id:
                            continue
                        sender = msg.get("sender_name", "unknown")
                        content = msg.get("content", "")
                        print(f"[Hub → {self.name}] {sender}: {content}")
                        self.send_to_process(f"[{sender}]: {content}")
                except Exception as exc:
                    print(f"[{self.name}] Hub poll error: {exc}")

                # Process → Hub
                lines = self.drain_stdout()
                for line in lines:
                    if line.startswith("[") and "]:" in line:
                        # Already formatted as "[sender]: content", skip echoes
                        continue
                    print(f"[{self.name} → Hub] {line}")
                    self.hub_send(line)

                # Check if process is still alive
                if self._process and self._process.poll() is not None:
                    print(f"[{self.name}] Process exited with code {self._process.returncode}")
                    # Drain remaining output
                    lines = self.drain_stdout()
                    for line in lines:
                        self.hub_send(line)
                    break

                self.hub_heartbeat("online")
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            print(f"\n[{self.name}] Shutting down...")
        finally:
            self.hub_heartbeat("offline")
            if self._process:
                self._process.terminate()

    # -- HTTP helpers ---------------------------------------------------------

    def _hub_get(self, path: str) -> Dict[str, Any]:
        url = self.hub_url + path
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _hub_post(self, path: str, body: dict) -> Dict[str, Any]:
        url = self.hub_url + path
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bridge a CLI agent to the Hub")
    parser.add_argument("--name", required=True, help="Agent display name")
    parser.add_argument("--cmd", required=True, help="Command to run the agent CLI")
    parser.add_argument("--hub-url", default=_hub_url(), help="Hub URL")
    parser.add_argument("--type", default="generic", help="Agent type")
    parser.add_argument("--color", default="#6366F1", help="Hex color")
    parser.add_argument("--emoji", default="🤖", help="Avatar emoji")
    parser.add_argument("--channel", default="general", help="Channel")
    parser.add_argument("--interval", type=float, default=2.0, help="Poll interval")
    args = parser.parse_args(argv)

    bridge = ClientBridge(
        name=args.name,
        cmd=args.cmd,
        hub_url=args.hub_url,
        agent_type=args.type,
        color=args.color,
        emoji=args.emoji,
        channel=args.channel,
        poll_interval=args.interval,
    )
    bridge.run()


if __name__ == "__main__":
    main()
