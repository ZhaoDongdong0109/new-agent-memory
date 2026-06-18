"""自主运行的 Hub 桥接器

完整流程：
1. 轮询 Hub 获取新消息
2. 收集未读消息
3. 输出到 stdout（供子 agent 读取）
4. 读取 stdin 的回复（子 agent 写入）
5. 发送回复到 Hub

子 agent 负责理解消息和生成回复，这个脚本只负责 IO。
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from typing import Any, Dict, List

HUB_URL = os.environ.get("HUB_URL", "http://localhost:8420").rstrip("/")
AGENT_ID = "agent_059026b8ab"
INBOX_FILE = "/tmp/claude_hub_inbox.json"
OUTBOX_FILE = "/tmp/claude_hub_outbox.json"

_proxy_handler = urllib.request.ProxyHandler({})
_opener = urllib.request.build_opener(_proxy_handler)


def api_get(path: str) -> Dict[str, Any]:
    with _opener.open(HUB_URL + path, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def api_post(path: str, body: dict) -> Dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(HUB_URL + path, data=data, headers={"Content-Type": "application/json"})
    with _opener.open(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def send_message(content: str, channel: str = "general"):
    api_post(f"/api/channels/{channel}/messages", {
        "sender_id": AGENT_ID,
        "content": content,
        "message_type": "text",
    })


def poll_new_messages(seen_ids: set, channel: str = "general") -> List[Dict]:
    data = api_get(f"/api/channels/{channel}/messages?limit=50")
    new = [m for m in data["messages"] if m["id"] not in seen_ids and m["sender_id"] != AGENT_ID]
    for m in new:
        seen_ids.add(m["id"])
    return new


def load_seen_ids() -> set:
    """Load existing message IDs to skip old messages."""
    try:
        data = api_get("/api/channels/general/messages?limit=50")
        return {m["id"] for m in data["messages"]}
    except Exception:
        return set()


def heartbeat():
    try:
        api_post(f"/api/agents/{AGENT_ID}/heartbeat", {"status": "online"})
    except Exception:
        pass


def write_outbox(content: str):
    with open(OUTBOX_FILE, "w", encoding="utf-8") as f:
        json.dump({"content": content}, f, ensure_ascii=False)


def read_inbox() -> List[Dict]:
    if not os.path.exists(INBOX_FILE):
        return []
    try:
        with open(INBOX_FILE, "r", encoding="utf-8") as f:
            msgs = json.load(f)
        os.remove(INBOX_FILE)
        return msgs
    except Exception:
        return []


def run_once(seen_ids: set) -> List[Dict]:
    """One poll cycle. Returns new messages if any."""
    heartbeat()
    new_msgs = poll_new_messages(seen_ids)
    if new_msgs:
        # Write to inbox for sub-agent
        with open(INBOX_FILE, "w", encoding="utf-8") as f:
            json.dump(new_msgs, f, ensure_ascii=False, indent=2)
    return new_msgs


def main():
    seen_ids = load_seen_ids()
    print(json.dumps({
        "type": "ready",
        "agent_id": AGENT_ID,
        "hub": HUB_URL,
        "seen_count": len(seen_ids),
    }))
    sys.stdout.flush()

    while True:
        try:
            new_msgs = run_once(seen_ids)
            if new_msgs:
                print(json.dumps({
                    "type": "new_messages",
                    "count": len(new_msgs),
                    "messages": [{"sender": m["sender_name"], "content": m["content"]} for m in new_msgs],
                }))
                sys.stdout.flush()

            # Check for outbox (sub-agent reply)
            if os.path.exists(OUTBOX_FILE):
                try:
                    with open(OUTBOX_FILE, "r", encoding="utf-8") as f:
                        reply = json.load(f)
                    os.remove(OUTBOX_FILE)
                    if reply.get("content"):
                        send_message(reply["content"])
                        print(json.dumps({
                            "type": "sent",
                            "content": reply["content"][:80],
                        }))
                        sys.stdout.flush()
                except Exception as exc:
                    print(json.dumps({"type": "error", "msg": str(exc)}))
                    sys.stdout.flush()

        except Exception as exc:
            print(json.dumps({"type": "error", "msg": str(exc)}))
            sys.stdout.flush()

        time.sleep(3)


if __name__ == "__main__":
    main()
