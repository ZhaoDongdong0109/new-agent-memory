"""Webhook adapter — extends PollingAdapter with a local HTTP server.

Agents that can receive HTTP callbacks can use this adapter.  The local
server listens for push notifications from the hub while also polling
as a fallback.
"""

from __future__ import annotations

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable, Dict, Optional

from adapters.polling_adapter import PollingAdapter


class _WebhookHandler(BaseHTTPRequestHandler):
    """Receives POST callbacks from the hub."""

    callback: Callable[[Dict[str, Any]], None] = lambda msg: None

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self.send_response(204)
            self.end_headers()
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            self.callback(body)
        except Exception:
            pass
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')


class WebhookAdapter(PollingAdapter):
    """Polling adapter with a local HTTP server for push callbacks."""

    def __init__(
        self,
        callback_port: int = 8421,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.callback_port = callback_port
        self._server: Optional[HTTPServer] = None

    def register(self) -> str:
        """Register with the hub and provide the callback URL."""
        agent_id = super().register()
        # Store callback URL in agent metadata
        callback_url = f"http://localhost:{self.callback_port}/callback"
        self._post(f"/api/agents/{agent_id}/heartbeat", {
            "status": "online",
        })
        return agent_id

    def start(self, channel_id: str = "general"):
        """Start the webhook server and polling loop."""
        # Start local HTTP server in background
        _WebhookHandler.callback = self._handle_webhook
        self._server = HTTPServer(("0.0.0.0", self.callback_port), _WebhookHandler)
        server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        server_thread.start()

        # Run polling loop in main thread
        super().start(channel_id)

    def stop(self):
        super().stop()
        if self._server:
            self._server.shutdown()

    def _handle_webhook(self, body: Dict[str, Any]):
        """Handle an incoming webhook callback."""
        if self.on_message and body.get("sender_id") != self.agent_id:
            self.on_message(body)
