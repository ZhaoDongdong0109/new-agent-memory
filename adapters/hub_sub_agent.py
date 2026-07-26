"""Hub 子 Agent 工作逻辑

职责：
1. 日常聊天：回复 Hub 上的普通消息，简短友好
2. 任务识别：如果消息像任务指令，写入 /tmp/claude_hub_task.json
3. 继续监听：处理完一条消息后，等 15 秒再检查新消息

任务识别规则：
- 包含动词：帮我、执行、做、写、改、查、分析、重构、部署
- 明确 @Claude Code

用法：
  python adapters/hub_sub_agent.py

环境变量：
  HUB_URL — Hub 地址 (默认 http://localhost:8420)
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional

# Bypass proxy for localhost
_proxy_handler = urllib.request.ProxyHandler({})
_opener = urllib.request.build_opener(_proxy_handler)

HUB_URL = os.environ.get("HUB_URL", "http://localhost:8420").rstrip("/")
AGENT_ID = "agent_059026b8ab"  # Claude Code 的 agent ID
TASK_FILE = "/tmp/claude_hub_task.json"
POLL_INTERVAL = 15  # 秒 - 处理完消息后等待时间

# 任务识别关键词
TASK_KEYWORDS = [
    "帮我", "执行", "做", "写", "改", "查", "分析", "重构", "部署",
    "请", "麻烦", "能不能", "可以", "需要", "想要", "希望",
    "实现", "修改", "优化", "修复", "调试", "测试", "运行",
    "创建", "删除", "更新", "配置", "安装", "启动", "停止",
]

# @Claude Code 模式
AT_CLAUDE_CODE = ["@claude code", "@claudecode", "@claude-code"]


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
    """发送心跳到 Hub"""
    try:
        _api_post(f"/api/agents/{AGENT_ID}/heartbeat", {"status": "online"})
    except Exception:
        pass


def poll_messages(channel: str = "general", seen_ids: set = None) -> List[Dict]:
    """轮询 Hub 获取新消息"""
    if seen_ids is None:
        seen_ids = set()
    data = _api_get(f"/api/channels/{channel}/messages?limit=50")
    messages = data.get("messages", [])
    new_msgs = [m for m in messages if m["id"] not in seen_ids and m.get("sender_id") != AGENT_ID]
    for m in new_msgs:
        seen_ids.add(m["id"])
    return new_msgs


def send_message(content: str, channel: str = "general"):
    """发送消息到 Hub"""
    _api_post(f"/api/channels/{channel}/messages", {
        "sender_id": AGENT_ID,
        "content": content,
        "message_type": "text",
    })


def is_task_message(content: str) -> bool:
    """判断消息是否为任务指令"""
    content_lower = content.lower()

    # 检查是否 @Claude Code
    for pattern in AT_CLAUDE_CODE:
        if pattern in content_lower:
            return True

    # 检查是否包含任务关键词
    for keyword in TASK_KEYWORDS:
        if keyword in content:
            return True

    return False


def write_task_file(task: str, sender: str):
    """将任务写入 /tmp/claude_hub_task.json"""
    task_data = {
        "task": task,
        "sender": sender,
        "timestamp": datetime.now().isoformat(),
    }
    with open(TASK_FILE, "w", encoding="utf-8") as f:
        json.dump(task_data, f, ensure_ascii=False, indent=2)
    print(f"[Hub Sub-Agent] Task written to {TASK_FILE}")


def generate_chat_reply(content: str, sender: str) -> str:
    """生成简短友好的聊天回复"""
    replies = [
        "收到！有什么我可以帮忙的吗？",
        "嗯嗯，了解了~",
        "好的，我在听！",
        "收到！有什么需要随时说~",
        "嗯，明白了！",
        "好的，随时找我~",
        "收到！有什么可以帮你的吗？",
        "嗯嗯，我在呢~",
    ]
    return random.choice(replies)


def process_message(msg: Dict) -> Optional[str]:
    """处理单条消息，返回要发送的回复"""
    content = msg.get("content", "")
    sender = msg.get("sender_name", "unknown")

    if is_task_message(content):
        # 识别为任务
        write_task_file(content, sender)
        return "收到，我来处理这个任务 🫡"
    else:
        # 普通聊天
        return generate_chat_reply(content, sender)


def main():
    """主循环"""
    seen_ids: set = set()

    # 加载现有消息 ID 避免重复处理
    try:
        data = _api_get("/api/channels/general/messages?limit=50")
        for m in data.get("messages", []):
            seen_ids.add(m["id"])
    except Exception:
        pass

    print("[Hub Sub-Agent] 启动")
    print(f"[Hub Sub-Agent] Hub 地址: {HUB_URL}")
    print(f"[Hub Sub-Agent] Agent ID: {AGENT_ID}")
    print(f"[Hub Sub-Agent] 任务文件: {TASK_FILE}")
    print(f"[Hub Sub-Agent] 轮询间隔: {POLL_INTERVAL}秒")
    print()

    try:
        while True:
            try:
                heartbeat()

                # 检查新消息
                new_msgs = poll_messages("general", seen_ids)
                if new_msgs:
                    print(f"[Hub Sub-Agent] 收到 {len(new_msgs)} 条新消息")

                    for msg in new_msgs:
                        content = msg.get("content", "")
                        sender = msg.get("sender_name", "unknown")
                        print(f"  [{sender}]: {content[:80]}...")

                        # 处理消息并获取回复
                        reply = process_message(msg)
                        if reply:
                            send_message(reply)
                            print(f"  [回复]: {reply}")

                # 等待指定时间再检查
                time.sleep(POLL_INTERVAL)

            except Exception as exc:
                print(f"[Hub Sub-Agent] 错误: {exc}")
                time.sleep(5)

    except KeyboardInterrupt:
        print("\n[Hub Sub-Agent] 停止")


if __name__ == "__main__":
    main()
