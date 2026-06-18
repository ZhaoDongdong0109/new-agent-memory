"""Claude Code ↔ Hub 实时桥接器

持续轮询 Hub，当有新消息时：
1. 收集未读消息
2. 构造上下文
3. 通过文件写入通知 Claude Code 有新消息

Claude Code 可以通过读取这个文件来获取待回复的消息。

用法：
  python adapters/claude_code_bridge.py

环境变量：
  HUB_URL — Hub 地址 (默认 http://localhost:8420)
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional

# Bypass proxy for localhost
_proxy_handler = urllib.request.ProxyHandler({})
_opener = urllib.request.build_opener(_proxy_handler)

HUB_URL = os.environ.get("HUB_URL", "http://localhost:8420").rstrip("/")
AGENT_ID = "agent_059026b8ab"  # Claude Code 的 agent ID
INBOX_FILE = "/tmp/claude_hub_inbox.json"
OUTBOX_FILE = "/tmp/claude_hub_outbox.json"
POLL_INTERVAL = 3  # 秒


def _api_get(path: str) -> Dict[str, Any]:
    url = HUB_URL + path
    with _opener.open(url, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _api_post(path: str, body: dict) -> Dict[str, Any]:
    url = HUB_URL + path
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with _opener.open(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def heartbeat():
    _api_post(f"/api/agents/{AGENT_ID}/heartbeat", {"status": "online"})


def poll_messages(channel: str = "general", seen_ids: set = None) -> List[Dict]:
    if seen_ids is None:
        seen_ids = set()
    data = _api_get(f"/api/channels/{channel}/messages?limit=50")
    messages = data.get("messages", [])
    new_msgs = [m for m in messages if m["id"] not in seen_ids and m.get("sender_id") != AGENT_ID]
    for m in new_msgs:
        seen_ids.add(m["id"])
    return new_msgs


def send_message(content: str, channel: str = "general"):
    _api_post(f"/api/channels/{channel}/messages", {
        "sender_id": AGENT_ID,
        "content": content,
        "message_type": "text",
    })


def write_inbox(messages: List[Dict]):
    """Write new messages to inbox file for Claude Code to read."""
    with open(INBOX_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


def check_outbox() -> Optional[str]:
    """Check if Claude Code has written a reply to send."""
    if not os.path.exists(OUTBOX_FILE):
        return None
    try:
        with open(OUTBOX_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        os.remove(OUTBOX_FILE)
        return data.get("content")
    except Exception:
        return None


def run():
    seen_ids: set = set()

    # Load existing message IDs to avoid replay
    try:
        data = _api_get("/api/channels/general/messages?limit=50")
        for m in data.get("messages", []):
            seen_ids.add(m["id"])
    except Exception:
        pass

    print(f"[Claude Code Bridge] Watching hub at {HUB_URL}")
    print(f"[Claude Code Bridge] Inbox: {INBOX_FILE}")
    print(f"[Claude Code Bridge] Outbox: {OUTBOX_FILE}")
    print(f"[Claude Code Bridge] Polling every {POLL_INTERVAL}s...")
    print()

    try:
        while True:
            try:
                heartbeat()

                # Check for new messages
                new_msgs = poll_messages("general", seen_ids)
                if new_msgs:
                    print(f"[Claude Code Bridge] {len(new_msgs)} new message(s)")
                    write_inbox(new_msgs)
                    for m in new_msgs:
                        print(f"  [{m['sender_name']}]: {m['content']}")

                # Check if Claude Code wants to send something
                reply = check_outbox()
                if reply:
                    print(f"[Claude Code Bridge] Sending reply: {reply[:50]}...")
                    send_message(reply)

            except Exception as exc:
                print(f"[Claude Code Bridge] Error: {exc}")

            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("\n[Claude Code Bridge] Stopped")


if __name__ == "__main__":
    run()
