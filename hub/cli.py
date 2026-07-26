"""Command line entry point for the Agent Communication Hub."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-hub",
        description="Multi-agent communication hub.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # hub start
    start = subparsers.add_parser("start", help="Start the hub server.")
    start.add_argument("--host", default="0.0.0.0", help="Bind address.")
    start.add_argument("--port", type=int, default=8420, help="Bind port.")
    start.add_argument("--data-dir", default="./hub_data", help="Persistence directory.")
    start.add_argument("--memory-integration", action="store_true",
                       help="Enable HumanLikeMemorySystem bridge.")
    start.add_argument("--memory-data-dir", default="./memory_data",
                       help="Memory system data directory.")
    start.set_defaults(func=run_start)

    # hub status
    status = subparsers.add_parser("status", help="Show hub status.")
    status.add_argument("--hub", default="http://localhost:8420", help="Hub URL.")
    status.set_defaults(func=run_status)

    # hub send
    send = subparsers.add_parser("send", help="Send a message.")
    send.add_argument("--hub", default="http://localhost:8420", help="Hub URL.")
    send.add_argument("--channel", default="general", help="Channel ID.")
    send.add_argument("--sender", required=True, help="Sender agent name.")
    send.add_argument("--message", required=True, help="Message content.")
    send.set_defaults(func=run_send)

    return parser


def run_start(args: argparse.Namespace) -> int:
    from hub.server import start_hub

    bridge = None
    if args.memory_integration:
        try:
            from main import HumanLikeMemorySystem
            memory = HumanLikeMemorySystem(data_dir=args.memory_data_dir)
            memory.load()
            from hub.memory_bridge import MemoryBridge
            bridge = MemoryBridge(memory)
            print(f"Memory bridge enabled (data: {args.memory_data_dir})")
        except Exception as exc:
            print(f"Warning: could not enable memory bridge: {exc}")

    start_hub(
        host=args.host,
        port=args.port,
        data_dir=args.data_dir,
        memory_bridge=bridge,
    )
    return 0


def run_status(args: argparse.Namespace) -> int:
    import urllib.request
    url = args.hub.rstrip("/") + "/api/stats"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        agents = data.get("agents", {})
        print("Hub status:")
        print(f"  Agents:   {agents.get('online', 0)} online / {agents.get('total', 0)} total")
        print(f"  Channels: {data.get('channels', 0)}")
        print(f"  Messages: {data.get('messages', 0)}")
        print(f"  Uptime:   {int(data.get('uptime_seconds', 0) / 60)}m")
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def run_send(args: argparse.Namespace) -> int:
    import urllib.request
    url = args.hub.rstrip("/") + "/api/channels/" + args.channel + "/messages"
    body = json.dumps({
        "sender_name": args.sender,
        "content": args.message,
        "message_type": "text",
    }).encode("utf-8")
    try:
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        print(f"Sent: {data.get('id', '?')}")
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def main(argv: Any = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
