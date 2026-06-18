"""Optional bridge between the Communication Hub and HumanLikeMemorySystem.

When enabled, every message posted to the hub is also stored as a memory
chunk so agents can later recall past conversations through the memory
system's retrieval pipeline.
"""

from __future__ import annotations

from typing import Any, Optional

from hub.models import Message


class MemoryBridge:
    """Stores hub messages into HumanLikeMemorySystem."""

    def __init__(self, memory_system: Any):
        self.memory = memory_system

    def on_message(self, message: Message):
        """Called after a message is stored in the hub."""
        if message.message_type == "system":
            return  # Don't store system events as memories

        self.memory.add_memory(
            content=f"[{message.sender_name}] {message.content}",
            memory_type="interaction",
            persons=[message.sender_name],
            topics=["chat", message.channel_id],
            importance=0.4,
            metadata={
                "hub_message_id": message.id,
                "channel": message.channel_id,
                "source": "hub",
            },
        )

    def search(self, query: str) -> str:
        """Retrieve relevant memories for a query."""
        result = self.memory.retrieve(query, allow_forgotten=True)
        return result.assembled_content if result else ""

    def save(self):
        """Persist the memory system."""
        self.memory.save()
