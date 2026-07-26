"""Hermes ↔ Hub 双向连接器

Hermes 有 HTTP API，这个连接器：
1. 轮询 Hub 获取新消息
2. 将新消息转发给 Hermes 的 API
3. 将 Hermes 的回复发回 Hub

用法：
  python adapters/hermes_connector.py --hermes-url http://localhost:HERMES_PORT

环境变量：
  HERMES_URL    — Hermes API 地址 (默认 http://localhost:8080)
  HUB_URL       — Hub 地址 (默认 http://localhost:8420)
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from typing import Any, Dict, Optional


def _hub_url() -> str:
    return os.environ.get("HUB_URL", "http://localhost:8420").rstrip("/")


def _hermes_url() -> str:
    return os.environ.get("HERMES_URL", "http://localhost:8080").rstrip("/")


class HermesConnector:
    """Bidirectional bridge between Hermes API and the Communication Hub."""

    def __init__(
        self,
        hermes_url: str = "http://localhost:8080",
        hub_url: str = "http://localhost:8420",
        agent_name: str = "Hermes",
        poll_interval: float = 2.0,
        channel: str = "general",
    ):
        self.hermes_url = hermes_url.rstrip("/")
        self.hub_url = hub_url.rstrip("/")
        self.agent_name = agent_name
        self.poll_interval = poll_interval
        self.channel = channel
        self.agent_id: Optional[str] = None
        self._seen_ids: set = set()

    # -- Hub API --------------------------------------------------------------

    def hub_register(self) -> str:
        data = self._hub_post("/api/agents/register", {
            "name": self.agent_name,
            "agent_type": "hermes",
            "color": "#10B981",
            "avatar_emoji": "🧠",
        })
        self.agent_id = data["id"]
        print(f"[Hermes Connector] Registered with hub: {self.agent_id}")
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

    # -- Hermes API -----------------------------------------------------------

    def hermes_send(self, message: str) -> str:
        """Send a message to Hermes and get a response.

        Adjust this method to match Hermes's actual API format.
        The default assumes a simple /chat or /v1/chat/completions style API.
        """
        # Try OpenAI-compatible format first
        try:
            return self._hermes_openai_format(message)
        except Exception:
            pass

        # Try simple /chat endpoint
        try:
            return self._hermes_simple_chat(message)
        except Exception:
            pass

        # Fallback: just return the raw response
        try:
            data = self._hermes_post("/api/generate", {"prompt": message})
            return data.get("response") or data.get("text") or data.get("content") or str(data)
        except Exception as exc:
            return f"[Hermes Connector] Error: {exc}"

    def _hermes_openai_format(self, message: str) -> str:
        body = {
            "model": "hermes",
            "messages": [{"role": "user", "content": message}],
            "max_tokens": 1024,
        }
        data = self._hermes_post("/v1/chat/completions", body)
        return data["choices"][0]["message"]["content"]

    def _hermes_simple_chat(self, message: str) -> str:
        data = self._hermes_post("/chat", {"message": message})
        return data.get("response") or data.get("reply") or data.get("content") or str(data)

    # -- Main loop ------------------------------------------------------------

    def run(self):
        """Main event loop: poll hub → forward to Hermes → send reply back."""
        self.hub_register()
        self.hub_heartbeat("online")
        print(f"[Hermes Connector] Connected. Channel: #{self.channel}")
        print(f"[Hermes Connector] Hermes URL: {self.hermes_url}")
        print(f"[Hermes Connector] Polling every {self.poll_interval}s...")

        try:
            while True:
                try:
                    new_msgs = self.hub_poll()
                    for msg in new_msgs:
                        # Skip messages from self
                        if msg.get("sender_id") == self.agent_id:
                            continue

                        sender = msg.get("sender_name", "unknown")
                        content = msg.get("content", "")
                        print(f"[Hub → Hermes] {sender}: {content}")

                        # Forward to Hermes
                        try:
                            reply = self.hermes_send(f"来自 {sender} 的消息: {content}")
                            print(f"[Hermes → Hub] {reply}")
                            self.hub_send(reply)
                        except Exception as exc:
                            print(f"[Hermes Connector] Hermes error: {exc}")

                    self.hub_heartbeat("online")
                except Exception as exc:
                    print(f"[Hermes Connector] Poll error: {exc}")

                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            print("\n[Hermes Connector] Shutting down...")
            self.hub_heartbeat("offline")

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

    def _hermes_post(self, path: str, body: dict) -> Dict[str, Any]:
        url = self.hermes_url + path
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Hermes ↔ Hub connector")
    parser.add_argument("--hermes-url", default=_hermes_url(), help="Hermes API URL")
    parser.add_argument("--hub-url", default=_hub_url(), help="Hub URL")
    parser.add_argument("--channel", default="general", help="Channel to watch")
    parser.add_argument("--interval", type=float, default=2.0, help="Poll interval (seconds)")
    args = parser.parse_args(argv)

    connector = HermesConnector(
        hermes_url=args.hermes_url,
        hub_url=args.hub_url,
        channel=args.channel,
        poll_interval=args.interval,
    )
    connector.run()


if __name__ == "__main__":
    main()
