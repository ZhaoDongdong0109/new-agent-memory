"""CLI bridge — stdin/stdout to HTTP adapter for the Communication Hub.

The simplest way for a command-line agent to connect to the hub.
Reads lines from stdin, sends them as messages, polls for incoming
messages and prints them to stdout.
"""

from __future__ import annotations

import argparse
import sys
import threading
from typing import Any

from adapters.polling_adapter import PollingAdapter


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cli-bridge",
        description="Connect a CLI agent to the Communication Hub.",
    )
    parser.add_argument("--name", required=True, help="Agent name.")
    parser.add_argument("--type", default="generic", help="Agent type.")
    parser.add_argument("--color", default="#6366F1", help="Hex color.")
    parser.add_argument("--emoji", default="🤖", help="Avatar emoji.")
    parser.add_argument("--hub", default="http://localhost:8420", help="Hub URL.")
    parser.add_argument("--channel", default="general", help="Default channel.")
    args = parser.parse_args(argv)

    adapter = PollingAdapter(
        hub_url=args.hub,
        agent_name=args.name,
        agent_type=args.type,
        color=args.color,
        avatar_emoji=args.emoji,
        on_message=lambda msg: print(f"\n[{msg['sender_name']}]: {msg['content']}\n> ", end="", flush=True),
    )
    adapter.register()
    adapter.heartbeat("online")
    print(f"[{args.name}] Connected to hub (id={adapter.agent_id})")
    print(f"[{args.name}] Channel: #{args.channel}")
    print(f"[{args.name}] Type messages and press Enter. Commands: :quit, :switch <channel>, :channels, :agents")
    print()

    # Background polling thread
    def poll_loop():
        while adapter._running:
            try:
                new_msgs = adapter.poll(args.channel)
                for msg in new_msgs:
                    if msg.get("sender_id") != adapter.agent_id:
                        print(f"\n[{msg['sender_name']}]: {msg['content']}")
            except Exception:
                pass
            import time
            time.sleep(adapter.poll_interval)

    adapter._running = True
    poller = threading.Thread(target=poll_loop, daemon=True)
    poller.start()

    # Main input loop
    try:
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not line:
                continue

            cmd = line.lower()
            if cmd in {":q", ":quit", ":exit"}:
                break
            if cmd == ":channels":
                for ch in adapter.list_channels():
                    print(f"  #{ch['id']} — {ch.get('description', '')}")
                continue
            if cmd == ":agents":
                for a in adapter.list_agents():
                    status = a.get("status", "?")
                    print(f"  {a['avatar_emoji']} {a['name']} ({status})")
                continue
            if cmd.startswith(":switch "):
                new_ch = line[8:].strip()
                if new_ch:
                    args.channel = new_ch
                    print(f"Switched to #{new_ch}")
                continue

            adapter.send(args.channel, line)
    finally:
        adapter.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
