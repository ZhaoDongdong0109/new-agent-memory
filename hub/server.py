"""HTTP server for the Agent Communication Hub.

Uses http.server.HTTPServer with ThreadingMixIn for concurrent connections.
All state lives in HubStore (thread-safe).  Real-time push uses SSE.
"""

from __future__ import annotations

import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Any, Optional
from urllib.parse import urlparse, parse_qs

from hub.models import Agent, Channel, Message, HubStore
from hub.sse import SSEBroadcaster
from hub.memory_bridge import MemoryBridge


_start_time: float = time.time()


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class HubHandler(BaseHTTPRequestHandler):
    """Request handler for the Communication Hub."""

    store: HubStore = None  # type: ignore[assignment]
    sse: SSEBroadcaster = None  # type: ignore[assignment]
    memory_bridge: Optional[MemoryBridge] = None
    web_ui_html: str = ""

    # Silence per-request logging
    def log_message(self, format, *args):
        pass

    # -- Routing --------------------------------------------------------------

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        qs = parse_qs(parsed.query)

        routes = {
            "/": self._serve_ui,
            "/ui": self._serve_ui,
            "/api/agents": self._get_agents,
            "/api/channels": self._get_channels,
            "/api/events": self._sse_stream,
            "/api/stats": self._get_stats,
            "/api/messages/search": self._search_messages,
        }

        if path in routes:
            routes[path](qs)
            return

        # /api/agents/{id}
        if path.startswith("/api/agents/") and path.count("/") == 3:
            agent_id = path.split("/")[3]
            self._get_agent(agent_id)
            return

        # /api/channels/{id}/messages
        if path.startswith("/api/channels/") and path.endswith("/messages"):
            parts = path.split("/")
            if len(parts) == 5:
                channel_id = parts[3]
                self._get_messages(channel_id, qs)
                return

        self._json_response(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        body = self._read_body()

        # /api/agents/register
        if path == "/api/agents/register":
            self._register_agent(body)
            return

        # /api/agents/{id}/heartbeat
        if path.startswith("/api/agents/") and path.endswith("/heartbeat"):
            parts = path.split("/")
            if len(parts) == 5:
                agent_id = parts[3]
                self._heartbeat(agent_id, body)
                return

        # /api/channels
        if path == "/api/channels":
            self._create_channel(body)
            return

        # /api/channels/{id}/messages
        if path.startswith("/api/channels/") and path.endswith("/messages"):
            parts = path.split("/")
            if len(parts) == 5:
                channel_id = parts[3]
                self._send_message(channel_id, body)
                return

        self._json_response(404, {"error": "not found"})

    # -- Agent endpoints ------------------------------------------------------

    def _register_agent(self, body: dict):
        agent = Agent(
            name=body.get("name", "anonymous"),
            agent_type=body.get("agent_type", "generic"),
            color=body.get("color", "#6366F1"),
            avatar_emoji=body.get("avatar_emoji", "🤖"),
            metadata=body.get("metadata", {}),
        )
        self.store.register_agent(agent)
        self.sse.broadcast("agent_status", {
            "id": agent.id,
            "name": agent.name,
            "status": "online",
            "avatar_emoji": agent.avatar_emoji,
            "color": agent.color,
        })
        self._json_response(200, agent.to_dict())

    def _heartbeat(self, agent_id: str, body: dict):
        status = body.get("status", "online")
        ok = self.store.update_agent_status(agent_id, status)
        if not ok:
            self._json_response(404, {"error": "agent not found"})
            return
        agent = self.store.get_agent(agent_id)
        self.sse.broadcast("agent_status", {
            "id": agent_id,
            "name": agent.name if agent else "",
            "status": status,
        })
        self._json_response(200, {"ok": True, "last_seen": agent.last_seen if agent else 0})

    def _get_agents(self, qs: dict):
        agents = self.store.list_agents()
        self._json_response(200, {"agents": [a.to_dict() for a in agents]})

    def _get_agent(self, agent_id: str):
        agent = self.store.get_agent(agent_id)
        if not agent:
            self._json_response(404, {"error": "agent not found"})
            return
        self._json_response(200, agent.to_dict())

    # -- Channel endpoints ----------------------------------------------------

    def _create_channel(self, body: dict):
        name = body.get("name", "").strip()
        if not name:
            self._json_response(400, {"error": "channel name required"})
            return
        channel_id = body.get("id", name.lower().replace(" ", "-"))
        sender_id = body.get("created_by", "system")
        channel = Channel(
            id=channel_id,
            name=name,
            description=body.get("description", ""),
            created_by=sender_id,
        )
        self.store.create_channel(channel)
        self.sse.broadcast("channel_created", channel.to_dict())
        self._json_response(200, channel.to_dict())

    def _get_channels(self, qs: dict):
        channels = self.store.list_channels()
        self._json_response(200, {"channels": [c.to_dict() for c in channels]})

    # -- Message endpoints ----------------------------------------------------

    def _send_message(self, channel_id: str, body: dict):
        channel = self.store.get_channel(channel_id)
        if not channel:
            self._json_response(404, {"error": "channel not found"})
            return

        sender_id = body.get("sender_id", "")
        sender_name = body.get("sender_name", "anonymous")
        # Look up agent name if sender_id is provided
        if sender_id:
            agent = self.store.get_agent(sender_id)
            if agent:
                sender_name = agent.name

        message = Message(
            channel_id=channel_id,
            sender_id=sender_id,
            sender_name=sender_name,
            content=body.get("content", ""),
            message_type=body.get("message_type", "text"),
            reply_to=body.get("reply_to"),
            metadata=body.get("metadata", {}),
        )
        self.store.add_message(message)

        # Push via SSE
        self.sse.broadcast("message", message.to_dict())

        # Optional memory bridge
        if self.memory_bridge:
            try:
                self.memory_bridge.on_message(message)
            except Exception:
                pass  # Don't let bridge errors break messaging

        self._json_response(200, message.to_dict())

    def _get_messages(self, channel_id: str, qs: dict):
        channel = self.store.get_channel(channel_id)
        if not channel:
            self._json_response(404, {"error": "channel not found"})
            return

        limit = int(qs.get("limit", ["50"])[0])
        before = float(qs["before"][0]) if "before" in qs else None
        messages = self.store.get_messages(channel_id, limit=limit, before=before)
        has_more = len(messages) == limit
        self._json_response(200, {
            "messages": [m.to_dict() for m in messages],
            "has_more": has_more,
        })

    def _search_messages(self, qs: dict):
        query = qs.get("q", [""])[0]
        limit = int(qs.get("limit", ["20"])[0])
        if not query:
            self._json_response(400, {"error": "q parameter required"})
            return
        messages = self.store.search_messages(query, limit=limit)
        self._json_response(200, {
            "messages": [m.to_dict() for m in messages],
            "total": len(messages),
        })

    # -- SSE ------------------------------------------------------------------

    def _sse_stream(self, qs: dict):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        client_queue = self.sse.register()
        try:
            # Send initial keepalive
            self.wfile.write(b": keepalive\n\n")
            self.wfile.flush()

            while True:
                try:
                    data = client_queue.get(timeout=30)
                    self.wfile.write(data.encode("utf-8"))
                    self.wfile.flush()
                except Exception:
                    # Timeout — send keepalive comment to detect disconnect
                    try:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                    except Exception:
                        break  # Client disconnected
        except Exception:
            pass
        finally:
            self.sse.unregister(client_queue)

    # -- Stats ----------------------------------------------------------------

    def _get_stats(self, qs: dict):
        agents = self.store.list_agents()
        online = sum(1 for a in agents if a.status == "online")
        self._json_response(200, {
            "agents": {"total": len(agents), "online": online},
            "channels": len(self.store.list_channels()),
            "messages": self.store.count_messages(),
            "uptime_seconds": round(time.time() - _start_time, 1),
        })

    # -- Web UI ---------------------------------------------------------------

    def _serve_ui(self, qs: dict):
        html = self.__class__.web_ui_html
        if not html:
            self._json_response(503, {"error": "web UI not loaded"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html.encode("utf-8"))))
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    # -- Helpers --------------------------------------------------------------

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        try:
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def _json_response(self, status: int, data: Any):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


def start_hub(
    host: str = "0.0.0.0",
    port: int = 8420,
    data_dir: str = "./hub_data",
    memory_bridge: Optional[MemoryBridge] = None,
):
    """Start the Communication Hub server."""
    from hub.web_ui import generate_web_ui_html

    store = HubStore(data_dir=data_dir)
    store.load()  # Restore previous state if available

    sse = SSEBroadcaster()

    # Wire up class-level references
    HubHandler.store = store
    HubHandler.sse = sse
    HubHandler.memory_bridge = memory_bridge
    HubHandler.web_ui_html = generate_web_ui_html()

    server = ThreadingHTTPServer((host, port), HubHandler)

    print(f"Agent Communication Hub started")
    print(f"  Web UI:  http://localhost:{port}/ui")
    print(f"  API:     http://localhost:{port}/api")
    print(f"  SSE:     http://localhost:{port}/api/events")
    if memory_bridge:
        print(f"  Memory:  integration enabled")
    print()

    # Save on shutdown
    import atexit
    atexit.register(lambda: _shutdown(store, memory_bridge))

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _shutdown(store: HubStore, bridge: Optional[MemoryBridge]):
    store.save()
    if bridge:
        bridge.save()
