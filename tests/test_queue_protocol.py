from __future__ import annotations

from hub_data.queue_protocol import QueueProtocol


def test_queue_accepts_full_hub_message_and_is_idempotent(tmp_path):
    queue = QueueProtocol("codex", base_dir=str(tmp_path))
    hub_message = {
        "id": "msg_hub_1",
        "channel_id": "general",
        "sender_name": "Claude",
        "content": "Please check this.",
        "message_type": "task",
        "metadata": {"priority": "high"},
        "created_at": 123.0,
    }

    assert queue.write_message(hub_message) == "msg_hub_1"
    assert queue.write_message(hub_message) == "msg_hub_1"

    messages = queue.read_messages()
    assert len(messages) == 1
    assert messages[0]["id"] == "msg_hub_1"
    assert messages[0]["metadata"]["priority"] == "high"


def test_mark_processed_updates_index_and_message_file(tmp_path):
    queue = QueueProtocol("codex", base_dir=str(tmp_path))
    msg_id = queue.write_message("hello")

    queue.mark_processed(msg_id)

    assert queue.read_messages() == []
    stored = queue.get_message_by_id(msg_id)
    assert stored is not None
    assert stored["processed"] is True
    assert "processed_at" in stored
    stats = queue.get_stats()
    assert stats["processed"] == 1
    assert stats["pending"] == 0
