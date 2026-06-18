"""Tests for the Agent Communication Hub."""

from __future__ import annotations

import json
import queue
import time

import pytest

from hub.models import Agent, Channel, Message, HubStore
from hub.sse import SSEBroadcaster


# ---------------------------------------------------------------------------
# HubStore tests
# ---------------------------------------------------------------------------

class TestHubStore:
    def test_register_agent(self, tmp_path):
        store = HubStore(data_dir=str(tmp_path))
        agent = Agent(name="Test", agent_type="generic")
        store.register_agent(agent)
        assert store.get_agent(agent.id) is not None
        assert store.get_agent(agent.id).name == "Test"

    def test_list_agents(self, tmp_path):
        store = HubStore(data_dir=str(tmp_path))
        store.register_agent(Agent(name="A"))
        store.register_agent(Agent(name="B"))
        assert len(store.list_agents()) == 2

    def test_update_agent_status(self, tmp_path):
        store = HubStore(data_dir=str(tmp_path))
        agent = Agent(name="Test")
        store.register_agent(agent)
        store.update_agent_status(agent.id, "busy")
        assert store.get_agent(agent.id).status == "busy"

    def test_update_nonexistent_agent(self, tmp_path):
        store = HubStore(data_dir=str(tmp_path))
        assert store.update_agent_status("nope", "online") is False

    def test_default_channel_exists(self, tmp_path):
        store = HubStore(data_dir=str(tmp_path))
        channels = store.list_channels()
        assert any(c.id == "general" for c in channels)

    def test_create_channel(self, tmp_path):
        store = HubStore(data_dir=str(tmp_path))
        ch = Channel(id="tasks", name="tasks", description="Task coordination")
        store.create_channel(ch)
        assert store.get_channel("tasks") is not None
        assert len(store.list_channels()) == 2  # general + tasks

    def test_send_and_get_messages(self, tmp_path):
        store = HubStore(data_dir=str(tmp_path))
        agent = Agent(name="Test", id="agent_test")
        store.register_agent(agent)
        for i in range(3):
            store.add_message(Message(
                channel_id="general",
                sender_id="agent_test",
                sender_name="Test",
                content=f"msg {i}",
            ))
        msgs = store.get_messages("general")
        assert len(msgs) == 3
        assert msgs[0].content == "msg 0"
        assert msgs[2].content == "msg 2"

    def test_message_pagination(self, tmp_path):
        store = HubStore(data_dir=str(tmp_path))
        for i in range(20):
            store.add_message(Message(channel_id="general", content=f"m{i}"))
        page = store.get_messages("general", limit=5)
        assert len(page) == 5
        assert page[0].content == "m15"  # Most recent 5, reversed

    def test_message_before_timestamp(self, tmp_path):
        store = HubStore(data_dir=str(tmp_path))
        store.add_message(Message(channel_id="general", content="old", created_at=100.0))
        store.add_message(Message(channel_id="general", content="new", created_at=200.0))
        msgs = store.get_messages("general", before=150.0)
        assert len(msgs) == 1
        assert msgs[0].content == "old"

    def test_search_messages(self, tmp_path):
        store = HubStore(data_dir=str(tmp_path))
        store.add_message(Message(channel_id="general", content="Hello world"))
        store.add_message(Message(channel_id="general", content="Python is great"))
        store.add_message(Message(channel_id="general", content="Hello Python"))
        hits = store.search_messages("python")
        assert len(hits) == 2

    def test_count_messages(self, tmp_path):
        store = HubStore(data_dir=str(tmp_path))
        for i in range(5):
            store.add_message(Message(channel_id="general", content=f"m{i}"))
        store.add_message(Message(channel_id="other", content="x"))
        assert store.count_messages() == 6
        assert store.count_messages("general") == 5

    def test_save_and_load_round_trip(self, tmp_path):
        store = HubStore(data_dir=str(tmp_path))
        agent = Agent(name="Saved", id="agent_saved")
        store.register_agent(agent)
        store.create_channel(Channel(id="tasks", name="tasks"))
        store.add_message(Message(channel_id="general", sender_name="Saved", content="hello"))
        store.save()

        store2 = HubStore(data_dir=str(tmp_path))
        assert store2.load() is True
        assert store2.get_agent("agent_saved").name == "Saved"
        assert store2.get_channel("tasks") is not None
        msgs = store2.get_messages("general")
        assert len(msgs) == 1
        assert msgs[0].content == "hello"

    def test_load_nonexistent(self, tmp_path):
        store = HubStore(data_dir=str(tmp_path / "nope"))
        assert store.load() is False


# ---------------------------------------------------------------------------
# SSEBroadcaster tests
# ---------------------------------------------------------------------------

class TestSSEBroadcaster:
    def test_broadcast_reaches_registered_clients(self):
        sse = SSEBroadcaster()
        q1 = sse.register()
        q2 = sse.register()
        sse.broadcast("message", {"content": "hello"})
        assert not q1.empty()
        assert not q2.empty()
        data1 = q1.get_nowait()
        assert "event: message" in data1
        assert "hello" in data1

    def test_unregister_stops_delivery(self):
        sse = SSEBroadcaster()
        q1 = sse.register()
        sse.unregister(q1)
        sse.broadcast("message", {"content": "hello"})
        assert q1.empty()

    def test_client_count(self):
        sse = SSEBroadcaster()
        assert sse.client_count == 0
        q1 = sse.register()
        assert sse.client_count == 1
        q2 = sse.register()
        assert sse.client_count == 2
        sse.unregister(q1)
        assert sse.client_count == 1

    def test_format_sse_message(self):
        formatted = SSEBroadcaster._format("test", {"key": "val"})
        assert formatted.startswith("event: test\ndata: ")
        assert formatted.endswith("\n\n")
        parsed = json.loads(formatted.split("data: ", 1)[1].strip())
        assert parsed["key"] == "val"


# ---------------------------------------------------------------------------
# Agent / Channel / Message data model tests
# ---------------------------------------------------------------------------

class TestModels:
    def test_agent_round_trip(self):
        a = Agent(name="Claude", agent_type="claude-code", color="#7C3AED")
        d = a.to_dict()
        a2 = Agent.from_dict(d)
        assert a2.name == "Claude"
        assert a2.agent_type == "claude-code"
        assert a2.color == "#7C3AED"

    def test_channel_round_trip(self):
        c = Channel(id="tasks", name="tasks", description="desc")
        d = c.to_dict()
        c2 = Channel.from_dict(d)
        assert c2.id == "tasks"
        assert c2.description == "desc"

    def test_message_round_trip(self):
        m = Message(
            channel_id="general",
            sender_id="agent_1",
            sender_name="Bot",
            content="hello",
            reply_to="msg_prev",
        )
        d = m.to_dict()
        assert d["reply_to"] == "msg_prev"
        m2 = Message.from_dict(d)
        assert m2.content == "hello"
        assert m2.reply_to == "msg_prev"

    def test_message_no_reply_to(self):
        m = Message(content="hi")
        d = m.to_dict()
        assert "reply_to" not in d
