from __future__ import annotations

from hub_data.auto_claude import is_low_value_ack, should_respond


def test_auto_claude_ignores_automation_ack():
    msg = {
        "id": "msg_ack",
        "sender_name": "Claude",
        "content": "收到 Codex！我们继续协作。",
        "metadata": {"automation_id": "auto_claude"},
    }

    assert is_low_value_ack(msg) is True
    assert should_respond(msg, {"responded_ids": []}) is False


def test_auto_claude_responds_to_codex_task_like_message():
    msg = {
        "id": "msg_task",
        "sender_name": "Codex",
        "content": "Please verify the queue protocol next.",
        "message_type": "text",
        "metadata": {},
    }

    assert should_respond(msg, {"responded_ids": []}) is True


def test_auto_claude_does_not_reply_twice():
    msg = {
        "id": "msg_done",
        "sender_name": "Codex",
        "content": "Please verify the queue protocol next.",
        "message_type": "text",
        "metadata": {},
    }

    assert should_respond(msg, {"responded_ids": ["msg_done"]}) is False
