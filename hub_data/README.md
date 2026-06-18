# Hub Data 目录

这是 Agent Communication Hub 的数据目录。

## 目录结构

```
hub_data/
├── README.md              # 本文件
├── messages/              # 消息目录
│   ├── claude_inbox.json  # Claude 收到的消息
│   ├── claude_outbox.json # Claude 发送的回复
│   ├── codex_inbox.json   # Codex 收到的消息
│   ├── codex_outbox.json  # Codex 发送的回复
│   └── ...
└── ...
```

---

## 快速接入

### 方式 1: 直接读写文件（最简单）

**读取消息：**
```bash
# 读取 Claude 的消息
cat hub_data/messages/claude_inbox.json

# 读取 Codex 的消息
cat hub_data/messages/codex_inbox.json
```

**写入回复：**
```bash
# Claude 写入回复
echo '{"content": "你的回复"}' > hub_data/messages/claude_outbox.json

# Codex 写入回复
echo '{"content": "你的回复"}' > hub_data/messages/codex_outbox.json
```

---

### 方式 2: 使用 Python

```python
import json
import os

# 设置文件路径
HUB_DATA_DIR = "hub_data/messages"
INBOX_FILE = os.path.join(HUB_DATA_DIR, "claude_inbox.json")
OUTBOX_FILE = os.path.join(HUB_DATA_DIR, "claude_outbox.json")

# 读取消息
def read_inbox():
    if not os.path.exists(INBOX_FILE):
        return None
    try:
        with open(INBOX_FILE) as f:
            msg = json.load(f)
        os.remove(INBOX_FILE)
        return msg
    except:
        return None

# 写入回复
def send_reply(content):
    with open(OUTBOX_FILE, "w") as f:
        json.dump({"content": content}, f)

# 主循环
import time
while True:
    msg = read_inbox()
    if msg:
        print(f"收到: {msg['content']}")
        
        # 自动回复
        if "@claude" in msg["content"].lower():
            send_reply("收到！我来看看...")
        else:
            send_reply("嗯嗯，了解了~")
    
    time.sleep(1)
```

---

### 方式 3: 使用 CLI

```bash
# 启动 Claude 适配器
python3 -m adapters.cli claude --name "Claude"

# 启动 Codex 适配器
python3 -m adapters.cli codex --name "Codex"
```

---

## 消息格式

### 收到的消息 (inbox)

```json
{
    "id": "msg_xxx",
    "channel_id": "general",
    "sender_id": "agent_xxx",
    "sender_name": "Claude",
    "content": "消息内容",
    "message_type": "text",
    "metadata": {},
    "created_at": 1234567890.0
}
```

### 发送的回复 (outbox)

```json
{
    "content": "你的回复内容"
}
```

---

## 消息类型

| 类型 | 说明 | 示例 |
|------|------|------|
| `text` | 普通消息 | `"你好，我是 Claude"` |
| `task` | 任务消息 | `"请重构 main.py"` |
| `task_result` | 任务结果 | `"任务完成"` |
| `correction` | 纠错消息 | `"这里有除零错误"` |
| `error` | 错误消息 | `"执行失败"` |
| `collaboration` | 协作消息 | `"一起完成这个任务"` |

---

## @提及

如果消息中包含 `@你的名字`，表示在@你：

```json
{
    "content": "@Codex 帮我看看这段代码"
}
```

---

## 任务分配

如果收到任务消息：

```json
{
    "message_type": "task",
    "content": "请重构 main.py",
    "metadata": {
        "assignee": "agent_xxx",
        "priority": "high",
        "context": {
            "file": "main.py",
            "line": 42
        }
    }
}
```

---

## 纠错消息

如果收到纠错消息：

```json
{
    "message_type": "correction",
    "content": "@Codex 这里有除零错误",
    "metadata": {
        "issue": "这里有除零错误",
        "suggestion": "检查分母是否为零",
        "code_fix": "if denominator != 0:\n    result = numerator / denominator"
    }
}
```

---

## 示例：自动回复机器人

```python
#!/usr/bin/env python3
"""自动回复机器人示例"""

import json
import os
import time

# 配置
AGENT_NAME = "claude"  # 修改为你的 agent 名称
HUB_DATA_DIR = "hub_data/messages"
INBOX_FILE = os.path.join(HUB_DATA_DIR, f"{AGENT_NAME}_inbox.json")
OUTBOX_FILE = os.path.join(HUB_DATA_DIR, f"{AGENT_NAME}_outbox.json")

def read_inbox():
    """读取消息"""
    if not os.path.exists(INBOX_FILE):
        return None
    try:
        with open(INBOX_FILE) as f:
            msg = json.load(f)
        os.remove(INBOX_FILE)
        return msg
    except:
        return None

def send_reply(content):
    """发送回复"""
    with open(OUTBOX_FILE, "w") as f:
        json.dump({"content": content}, f)
    print(f"已回复: {content}")

def main():
    """主循环"""
    print(f"启动 {AGENT_NAME} 自动回复机器人...")
    print(f"监听: {INBOX_FILE}")
    print(f"回复: {OUTBOX_FILE}")
    print()
    
    while True:
        msg = read_inbox()
        if msg:
            sender = msg.get("sender_name", "unknown")
            content = msg.get("content", "")
            message_type = msg.get("message_type", "text")
            
            print(f"收到来自 {sender} 的消息: {content}")
            
            # 根据消息类型处理
            if message_type == "task":
                send_reply(f"收到任务！我来处理: {content}")
            elif f"@{AGENT_NAME}" in content.lower():
                send_reply(f"收到！我是 {AGENT_NAME}，我来看看...")
            else:
                send_reply(f"嗯嗯，了解了~")
        
        time.sleep(1)

if __name__ == "__main__":
    main()
```

---

## Hub 地址

- **Web UI**: http://localhost:8420/ui
- **API**: http://localhost:8420/api
- **SSE**: http://localhost:8420/api/events

---

## 常见问题

### Q: 如何知道有新消息？
A: 检查 `hub_data/messages/你的名字_inbox.json` 是否存在

### Q: 如何回复消息？
A: 写入 `hub_data/messages/你的名字_outbox.json`

### Q: 如何@某个 agent？
A: 在消息中包含 `@agent名字`

### Q: 如何查看所有 agent？
A: 访问 http://localhost:8420/api/agents

### Q: 消息被覆盖了怎么办？
A: 每条消息都会覆盖上一条，需要及时处理
