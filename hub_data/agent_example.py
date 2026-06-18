#!/usr/bin/env python3
"""Agent 自动回复示例

使用方式：
    # 直接运行
    python3 hub_data/agent_example.py

    # 或者修改 AGENT_NAME 后运行
    AGENT_NAME=codex python3 hub_data/agent_example.py
"""

import json
import os
import sys
import time

# 配置
AGENT_NAME = os.environ.get("AGENT_NAME", "claude")
HUB_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "messages")
INBOX_FILE = os.path.join(HUB_DATA_DIR, f"{AGENT_NAME}_inbox.json")
OUTBOX_FILE = os.path.join(HUB_DATA_DIR, f"{AGENT_NAME}_outbox.json")


def read_inbox():
    """读取消息"""
    if not os.path.exists(INBOX_FILE):
        return None
    try:
        with open(INBOX_FILE, encoding="utf-8") as f:
            msg = json.load(f)
        os.remove(INBOX_FILE)
        return msg
    except Exception as e:
        print(f"读取 inbox 失败: {e}")
        return None


def send_reply(content):
    """发送回复"""
    try:
        with open(OUTBOX_FILE, "w", encoding="utf-8") as f:
            json.dump({"content": content}, f, ensure_ascii=False)
        print(f"已回复: {content}")
    except Exception as e:
        print(f"发送回复失败: {e}")


def process_message(msg):
    """处理消息"""
    sender = msg.get("sender_name", "unknown")
    content = msg.get("content", "")
    message_type = msg.get("message_type", "text")
    metadata = msg.get("metadata", {})

    print(f"收到来自 {sender} 的消息: {content}")

    # 根据消息类型处理
    if message_type == "task":
        # 任务消息
        priority = metadata.get("priority", "normal")
        print(f"  任务优先级: {priority}")
        send_reply(f"收到任务！我来处理: {content}")

    elif message_type == "correction":
        # 纠错消息
        issue = metadata.get("issue", "")
        suggestion = metadata.get("suggestion", "")
        print(f"  问题: {issue}")
        print(f"  建议: {suggestion}")
        send_reply(f"收到纠错！我来修复: {issue}")

    elif f"@{AGENT_NAME}" in content.lower():
        # @提及
        send_reply(f"收到！我是 {AGENT_NAME}，我来看看...")

    else:
        # 普通消息
        send_reply(f"嗯嗯，了解了~")


def main():
    """主循环"""
    print(f"启动 {AGENT_NAME} 自动回复机器人...")
    print(f"监听: {INBOX_FILE}")
    print(f"回复: {OUTBOX_FILE}")
    print()

    # 确保目录存在
    os.makedirs(HUB_DATA_DIR, exist_ok=True)

    try:
        while True:
            msg = read_inbox()
            if msg:
                process_message(msg)
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n{AGENT_NAME} 自动回复机器人已停止")


if __name__ == "__main__":
    main()
