"""Server-Sent Events (SSE) broadcaster for real-time push.

SSE is chosen over WebSocket because it works with Python's stdlib
http.server without protocol upgrade complexity.  The browser natively
supports SSE via the EventSource API.
"""

from __future__ import annotations

import json
import queue
import threading
from typing import Any, List


class SSEBroadcaster:
    """Manages SSE client connections and broadcasts events.

    Each connected browser tab registers a queue.  When an event is
    broadcast, it is placed into every client's queue.  The HTTP handler
    reads from the queue and writes SSE-formatted data to the response.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._clients: List[queue.Queue] = []

    def register(self) -> queue.Queue:
        """Register a new SSE client and return its event queue."""
        q: queue.Queue = queue.Queue(maxsize=256)
        with self._lock:
            self._clients.append(q)
        return q

    def unregister(self, q: queue.Queue):
        """Remove an SSE client."""
        with self._lock:
            try:
                self._clients.remove(q)
            except ValueError:
                pass

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    def broadcast(self, event_type: str, data: Any):
        """Send an event to all connected clients.

        Args:
            event_type: SSE event name (e.g. "message", "agent_status").
            data: JSON-serializable payload.
        """
        payload = self._format(event_type, data)
        with self._lock:
            dead: List[queue.Queue] = []
            for q in self._clients:
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._clients.remove(q)

    @staticmethod
    def _format(event_type: str, data: Any) -> str:
        """Format a value as an SSE message."""
        body = json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else data
        return f"event: {event_type}\ndata: {body}\n\n"
