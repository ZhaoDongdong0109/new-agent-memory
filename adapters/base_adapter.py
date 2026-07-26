"""统一适配器基类

所有 agent 适配器的基类，提供：
- 自动注册到 hub
- 心跳机制
- SSE 实时监听
- 发送消息
- 抽象方法：on_message, on_task

使用方式：
    class MyAgent(BaseAdapter):
        def on_message(self, msg):
            print(f"收到消息: {msg['content']}")

        def on_task(self, task):
            print(f"收到任务: {task['task']}")

    agent = MyAgent("MyAgent", "my_type")
    agent.connect()
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# Bypass proxy for localhost
_proxy_handler = urllib.request.ProxyHandler({})
_opener = urllib.request.build_opener(_proxy_handler)


class HubConnectionError(ConnectionError):
    """连接 hub 失败时抛出，供调用方决定如何处理（而不是直接退出进程）。"""


class BaseAdapter(ABC):
    """所有 agent 适配器的基类"""

    def __init__(self, name: str, agent_type: str, hub_url: str = None):
        """
        初始化适配器

        Args:
            name: agent 名称
            agent_type: agent 类型 (claude, codex, hermes, human, etc.)
            hub_url: hub 服务器地址，默认从环境变量读取
        """
        self.name = name
        self.agent_type = agent_type
        self.hub_url = (hub_url or os.environ.get("HUB_URL", "http://localhost:8420")).rstrip("/")
        self.agent_id: Optional[str] = None
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._sse_thread: Optional[threading.Thread] = None
        self._seen_ids: set = set()

    # -------------------------------------------------------------------------
    # 公共方法
    # -------------------------------------------------------------------------

    def connect(self):
        """连接到 hub（自动注册 + 启动监听）"""
        self.agent_id = self._register()
        self._running = True
        self._start_heartbeat()
        self._start_sse_listener()
        self._load_existing_messages()
        print(f"[{self.name}] 已连接到 hub，ID: {self.agent_id}")
        print(f"[{self.name}] Hub 地址: {self.hub_url}")
        print()

    def disconnect(self):
        """断开连接"""
        self._running = False
        self._update_status("offline")
        print(f"[{self.name}] 已断开连接")

    def send(self, content: str, channel: str = "general", reply_to: str = None,
             message_type: str = "text", metadata: dict = None):
        """
        发送消息

        Args:
            content: 消息内容
            channel: 频道 ID，默认 general
            reply_to: 回复的消息 ID（可选）
            message_type: 消息类型 (text, task, task_result, error, correction, collaboration)
            metadata: 额外元数据
        """
        body = {
            "sender_id": self.agent_id,
            "content": content,
            "message_type": message_type,
        }
        if reply_to:
            body["reply_to"] = reply_to
        if metadata:
            body["metadata"] = metadata

        try:
            self._api_post(f"/api/channels/{channel}/messages", body)
        except Exception as e:
            print(f"[{self.name}] 发送消息失败: {e}")

    def send_task(self, task: str, assignee: str, channel: str = "general",
                  priority: str = "normal", context: dict = None):
        """
        发送任务给其他 agent

        Args:
            task: 任务描述
            assignee: 目标 agent ID
            channel: 频道 ID
            priority: 优先级 (low, normal, high, urgent)
            context: 任务上下文
        """
        metadata = {
            "assignee": assignee,
            "priority": priority,
            "context": context or {},
        }
        self.send(task, channel=channel, message_type="task", metadata=metadata)

    def send_correction(self, target_message_id: str, target_agent: str,
                        issue: str, suggestion: str, code_fix: str = None,
                        channel: str = "general"):
        """
        发送纠错消息

        Args:
            target_message_id: 目标消息 ID
            target_agent: 目标 agent ID
            issue: 问题描述
            suggestion: 建议
            code_fix: 代码修复（可选）
            channel: 频道 ID
        """
        metadata = {
            "target_message_id": target_message_id,
            "target_agent": target_agent,
            "issue": issue,
            "suggestion": suggestion,
            "code_fix": code_fix,
        }
        content = f"@{target_agent} {issue}\n建议: {suggestion}"
        if code_fix:
            content += f"\n修复:\n{code_fix}"
        self.send(content, channel=channel, message_type="correction", metadata=metadata)

    def get_messages(self, channel: str = "general", limit: int = 50) -> List[Dict]:
        """获取频道消息"""
        try:
            data = self._api_get(f"/api/channels/{channel}/messages?limit={limit}")
            return data.get("messages", [])
        except Exception:
            return []

    def list_channels(self) -> List[Dict]:
        """列出所有频道"""
        try:
            data = self._api_get("/api/channels")
            return data.get("channels", [])
        except Exception:
            return []

    def list_agents(self) -> List[Dict]:
        """列出所有 agent"""
        try:
            data = self._api_get("/api/agents")
            return data.get("agents", [])
        except Exception:
            return []

    # -------------------------------------------------------------------------
    # 抽象方法（子类必须实现）
    # -------------------------------------------------------------------------

    @abstractmethod
    def on_message(self, msg: Dict[str, Any]):
        """
        收到消息时的回调

        Args:
            msg: 消息字典，包含：
                - id: 消息 ID
                - channel_id: 频道 ID
                - sender_id: 发送者 ID
                - sender_name: 发送者名称
                - content: 消息内容
                - message_type: 消息类型
                - metadata: 元数据
                - created_at: 创建时间
        """
        pass

    @abstractmethod
    def on_task(self, task: Dict[str, Any]):
        """
        收到任务时的回调

        Args:
            task: 任务字典，包含：
                - id: 消息 ID
                - task: 任务描述
                - assignee: 目标 agent ID
                - priority: 优先级
                - context: 上下文
                - sender_name: 发送者名称
        """
        pass

    def on_correction(self, correction: Dict[str, Any]):
        """
        收到纠错时的回调（可选重写）

        Args:
            correction: 纠错字典，包含：
                - issue: 问题描述
                - suggestion: 建议
                - code_fix: 代码修复
                - sender_name: 发送者名称
        """
        print(f"[{self.name}] 收到纠错: {correction.get('issue')}")

    def on_agent_status(self, agent: Dict[str, Any]):
        """
        收到 agent 状态变化时的回调（可选重写）

        Args:
            agent: agent 信息，包含：
                - id: agent ID
                - name: agent 名称
                - status: 状态 (online, offline, idle, busy)
        """
        pass

    # -------------------------------------------------------------------------
    # 内部方法
    # -------------------------------------------------------------------------

    def _register(self) -> str:
        """注册到 hub，返回 agent_id"""
        body = {
            "name": self.name,
            "agent_type": self.agent_type,
        }
        try:
            data = self._api_post("/api/agents/register", body)
            return data["id"]
        except Exception as e:
            print(f"[{self.name}] 注册失败: {e}")
            raise HubConnectionError(
                f"无法连接到 hub ({self.hub_url}): {e}"
            ) from e

    def _update_status(self, status: str):
        """更新 agent 状态"""
        if not self.agent_id:
            return
        try:
            self._api_post(f"/api/agents/{self.agent_id}/heartbeat", {"status": status})
        except Exception:
            pass

    def _start_heartbeat(self):
        """启动心跳线程"""
        def _heartbeat_loop():
            while self._running:
                try:
                    self._update_status("online")
                except Exception:
                    pass
                time.sleep(30)

        self._heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def _start_sse_listener(self):
        """启动 SSE 监听线程"""
        def _sse_loop():
            while self._running:
                try:
                    self._connect_sse()
                except Exception as e:
                    if self._running:
                        print(f"[{self.name}] SSE 连接断开，5秒后重连: {e}")
                        time.sleep(5)
                except KeyboardInterrupt:
                    break

        self._sse_thread = threading.Thread(target=_sse_loop, daemon=True)
        self._sse_thread.start()

    def _connect_sse(self):
        """连接 SSE 流"""
        url = f"{self.hub_url}/api/events"
        req = urllib.request.Request(url)
        req.add_header("Accept", "text/event-stream")
        req.add_header("Cache-Control", "no-cache")

        with _opener.open(req, timeout=60) as response:
            buffer = ""
            while self._running:
                line = response.readline().decode("utf-8")
                if not line:
                    break

                buffer += line
                if line.strip() == "":
                    # 空行表示事件结束
                    self._process_sse_event(buffer.strip())
                    buffer = ""

    def _process_sse_event(self, event_str: str):
        """处理 SSE 事件"""
        if not event_str:
            return

        # 解析 SSE 格式
        event_type = None
        data = None
        for line in event_str.split("\n"):
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()

        if not event_type or not data:
            return

        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return

        # 分发事件
        if event_type == "message":
            self._handle_message(payload)
        elif event_type == "agent_status":
            self._handle_agent_status(payload)
        elif event_type == "channel_created":
            pass  # 忽略频道创建事件

    def _handle_message(self, msg: Dict[str, Any]):
        """处理消息事件"""
        # 忽略自己发送的消息
        if msg.get("sender_id") == self.agent_id:
            return

        # 忽略已经处理过的消息
        msg_id = msg.get("id")
        if msg_id in self._seen_ids:
            return
        self._seen_ids.add(msg_id)

        # 根据消息类型分发
        message_type = msg.get("message_type", "text")
        if message_type == "task":
            self.on_task(msg)
        elif message_type == "correction":
            self.on_correction(msg)
        else:
            self.on_message(msg)

    def _handle_agent_status(self, agent: Dict[str, Any]):
        """处理 agent 状态事件"""
        self.on_agent_status(agent)

    def _load_existing_messages(self):
        """加载现有消息 ID，避免重复处理"""
        try:
            messages = self.get_messages(limit=100)
            for msg in messages:
                self._seen_ids.add(msg["id"])
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # HTTP 工具方法
    # -------------------------------------------------------------------------

    def _api_get(self, path: str) -> Dict[str, Any]:
        """GET 请求"""
        url = self.hub_url + path
        with _opener.open(url, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _api_post(self, path: str, body: dict) -> Dict[str, Any]:
        """POST 请求"""
        url = self.hub_url + path
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with _opener.open(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
