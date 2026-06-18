"""HTTP polling adapter for connecting agents to the Communication Hub.

Agents use this adapter to register, send messages, and poll for new
messages.  Uses only Python stdlib (urllib.request).
"""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Any, Callable, Dict, List, Optional


class PollingAdapter:
    """Connects an agent to the hub via HTTP polling."""

    def __init__(
        self,
        hub_url: str = "http://localhost:8420",
        agent_name: str = "agent",
        agent_type: str = "generic",
        color: str = "#6366F1",
        avatar_emoji: str = "🤖",
        poll_interval: float = 2.0,
        on_message: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.hub_url = hub_url.rstrip("/")
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.color = color
        self.avatar_emoji = avatar_emoji
        self.poll_interval = poll_interval
        self.on_message = on_message
        self.agent_id: Optional[str] = None
        self._running = False
        self._last_seen_ids: set = set()

    # -- Registration ---------------------------------------------------------

    def register(self) -> str:
        """Register with the hub and return the agent ID."""
        data = self._post("/api/agents/register", {
            "name": self.agent_name,
            "agent_type": self.agent_type,
            "color": self.color,
            "avatar_emoji": self.avatar_emoji,
        })
        self.agent_id = data["id"]
        return self.agent_id

    def heartbeat(self, status: str = "online"):
        """Send a heartbeat to the hub."""
        if not self.agent_id:
            return
        self._post(f"/api/agents/{self.agent_id}/heartbeat", {"status": status})

    # -- Messaging ------------------------------------------------------------

    def send(self, channel_id: str, content: str, message_type: str = "text") -> Dict[str, Any]:
        """Send a message to a channel."""
        if not self.agent_id:
            self.register()
        return self._post(f"/api/channels/{channel_id}/messages", {
            "sender_id": self.agent_id,
            "content": content,
            "message_type": message_type,
        })

    def get_messages(self, channel_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent messages from a channel."""
        data = self._get(f"/api/channels/{channel_id}/messages?limit={limit}")
        return data.get("messages", [])

    def list_channels(self) -> List[Dict[str, Any]]:
        """List all channels."""
        data = self._get("/api/channels")
        return data.get("channels", [])

    def list_agents(self) -> List[Dict[str, Any]]:
        """List all agents."""
        data = self._get("/api/agents")
        return data.get("agents", [])

    # -- Polling loop ---------------------------------------------------------

    def poll(self, channel_id: str) -> List[Dict[str, Any]]:
        """Poll for new messages since last check."""
        messages = self.get_messages(channel_id)
        new_msgs = [m for m in messages if m["id"] not in self._last_seen_ids]
        for m in new_msgs:
            self._last_seen_ids.add(m["id"])
        return new_msgs

    def start(self, channel_id: str = "general"):
        """Start the polling loop (blocking)."""
        if not self.agent_id:
            self.register()
        self._running = True
        self.heartbeat("online")
        print(f"[{self.agent_name}] Connected to hub (id={self.agent_id})")

        try:
            while self._running:
                try:
                    new_msgs = self.poll(channel_id)
                    for msg in new_msgs:
                        if self.on_message and msg.get("sender_id") != self.agent_id:
                            self.on_message(msg)
                    self.heartbeat("online")
                except Exception as exc:
                    print(f"[{self.agent_name}] Poll error: {exc}")
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        """Stop the polling loop."""
        self._running = False
        if self.agent_id:
            try:
                self.heartbeat("offline")
            except Exception:
                pass

    # -- HTTP helpers ---------------------------------------------------------

    def _get(self, path: str) -> Dict[str, Any]:
        url = self.hub_url + path
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post(self, path: str, body: dict) -> Dict[str, Any]:
        url = self.hub_url + path
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
