"""Data models for the Agent Communication Hub.

Follows the same @dataclass + to_dict/from_dict pattern used throughout
the new-agent-memory project (see memory_chunk.py, agent_system.py).
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _now() -> float:
    return time.time()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclass
class Agent:
    """A registered participant in the hub."""

    id: str = field(default_factory=lambda: _new_id("agent"))
    name: str = ""
    agent_type: str = "generic"  # claude-code, hermes, qianwen, codex, human, generic
    status: str = "offline"  # online, offline, idle, busy
    color: str = "#6366F1"
    avatar_emoji: str = "🤖"
    registered_at: float = field(default_factory=_now)
    last_seen: float = field(default_factory=_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "agent_type": self.agent_type,
            "status": self.status,
            "color": self.color,
            "avatar_emoji": self.avatar_emoji,
            "registered_at": self.registered_at,
            "last_seen": self.last_seen,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Agent:
        return cls(
            id=data.get("id", _new_id("agent")),
            name=data.get("name", ""),
            agent_type=data.get("agent_type", "generic"),
            status=data.get("status", "offline"),
            color=data.get("color", "#6366F1"),
            avatar_emoji=data.get("avatar_emoji", "🤖"),
            registered_at=float(data.get("registered_at", _now())),
            last_seen=float(data.get("last_seen", _now())),
            metadata=dict(data.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# Channel
# ---------------------------------------------------------------------------

@dataclass
class Channel:
    """A named conversation channel."""

    id: str = "general"
    name: str = "general"
    description: str = ""
    created_at: float = field(default_factory=_now)
    created_by: str = "system"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Channel:
        return cls(
            id=data.get("id", "general"),
            name=data.get("name", "general"),
            description=data.get("description", ""),
            created_at=float(data.get("created_at", _now())),
            created_by=data.get("created_by", "system"),
        )


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """One message in a channel."""

    id: str = field(default_factory=lambda: _new_id("msg"))
    channel_id: str = "general"
    sender_id: str = ""
    sender_name: str = ""
    content: str = ""
    message_type: str = "text"  # text, task, task_result, error, correction, collaboration
    reply_to: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "channel_id": self.channel_id,
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "content": self.content,
            "message_type": self.message_type,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }
        if self.reply_to:
            d["reply_to"] = self.reply_to
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Message:
        return cls(
            id=data.get("id", _new_id("msg")),
            channel_id=data.get("channel_id", "general"),
            sender_id=data.get("sender_id", ""),
            sender_name=data.get("sender_name", ""),
            content=data.get("content", ""),
            message_type=data.get("message_type", "text"),
            reply_to=data.get("reply_to"),
            metadata=dict(data.get("metadata", {})),
            created_at=float(data.get("created_at", _now())),
        )


# ---------------------------------------------------------------------------
# HubStore — in-memory store with JSON file persistence
# ---------------------------------------------------------------------------

class HubStore:
    """Thread-safe in-memory store with JSON persistence.

    Mirrors the save/load pattern of HumanLikeMemorySystem in main.py.
    """

    def __init__(self, data_dir: str = "./hub_data"):
        self.data_dir = data_dir
        self._lock = threading.Lock()
        self.agents: Dict[str, Agent] = {}
        self.channels: Dict[str, Channel] = {}
        self.messages: List[Message] = []
        self._ensure_default_channel()

    def _ensure_default_channel(self):
        if "general" not in self.channels:
            self.channels["general"] = Channel(
                id="general",
                name="general",
                description="General discussion",
                created_by="system",
            )

    # -- Agents ---------------------------------------------------------------

    def register_agent(self, agent: Agent) -> Agent:
        with self._lock:
            self.agents[agent.id] = agent
            return agent

    def update_agent_status(self, agent_id: str, status: str) -> bool:
        with self._lock:
            agent = self.agents.get(agent_id)
            if not agent:
                return False
            agent.status = status
            agent.last_seen = _now()
            return True

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        with self._lock:
            return self.agents.get(agent_id)

    def list_agents(self) -> List[Agent]:
        with self._lock:
            return list(self.agents.values())

    # -- Channels -------------------------------------------------------------

    def create_channel(self, channel: Channel) -> Channel:
        with self._lock:
            self.channels[channel.id] = channel
            return channel

    def get_channel(self, channel_id: str) -> Optional[Channel]:
        with self._lock:
            return self.channels.get(channel_id)

    def list_channels(self) -> List[Channel]:
        with self._lock:
            return list(self.channels.values())

    # -- Messages -------------------------------------------------------------

    def add_message(self, message: Message) -> Message:
        with self._lock:
            self.messages.append(message)
            return message

    def get_messages(
        self,
        channel_id: str,
        limit: int = 50,
        before: Optional[float] = None,
    ) -> List[Message]:
        with self._lock:
            filtered = [m for m in self.messages if m.channel_id == channel_id]
            if before is not None:
                filtered = [m for m in filtered if m.created_at < before]
            # Return most recent first, then reverse for chronological display
            filtered.sort(key=lambda m: m.created_at, reverse=True)
            page = filtered[:limit]
            page.reverse()
            return page

    def count_messages(self, channel_id: Optional[str] = None) -> int:
        with self._lock:
            if channel_id:
                return sum(1 for m in self.messages if m.channel_id == channel_id)
            return len(self.messages)

    def search_messages(self, query: str, limit: int = 20) -> List[Message]:
        with self._lock:
            q = query.lower()
            hits = [m for m in self.messages if q in m.content.lower()]
            hits.sort(key=lambda m: m.created_at, reverse=True)
            return hits[:limit]

    # -- Persistence ----------------------------------------------------------

    def save(self):
        with self._lock:
            os.makedirs(self.data_dir, exist_ok=True)
            self._write_json("agents.json", [a.to_dict() for a in self.agents.values()])
            self._write_json("channels.json", [c.to_dict() for c in self.channels.values()])
            self._write_json("messages.json", [m.to_dict() for m in self.messages])

    def load(self) -> bool:
        with self._lock:
            agents_path = os.path.join(self.data_dir, "agents.json")
            if not os.path.exists(agents_path):
                return False
            try:
                agent_list = self._read_json("agents.json")
                self.agents = {d["id"]: Agent.from_dict(d) for d in agent_list}

                channel_list = self._read_json("channels.json")
                self.channels = {d["id"]: Channel.from_dict(d) for d in channel_list}

                msg_list = self._read_json("messages.json")
                self.messages = [Message.from_dict(d) for d in msg_list]
            except Exception:
                return False
            self._ensure_default_channel()
            return True

    def _write_json(self, filename: str, data: Any):
        path = os.path.join(self.data_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _read_json(self, filename: str) -> Any:
        path = os.path.join(self.data_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
