#!/usr/bin/env python3
"""队列式文件协议

解决单文件覆盖风险，实现更安全的消息传递。

使用方式：
    from hub_data.queue_protocol import QueueProtocol

    # 创建队列
    queue = QueueProtocol("codex")

    # 写入消息
    queue.write_message("你好")

    # 读取消息
    messages = queue.read_messages()

    # 标记消息已处理
    queue.mark_processed("msg_xxx")
"""

import json
import os
import re
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional


class QueueProtocol:
    """队列式文件协议"""

    def __init__(self, agent_name: str, base_dir: str = None):
        """
        初始化队列协议

        Args:
            agent_name: agent 名称 (codex, claude, etc.)
            base_dir: 基础目录，默认为 hub_data/messages
        """
        self.agent_name = agent_name
        self.base_dir = base_dir or os.path.join(os.path.dirname(__file__), "messages")
        self.queue_dir = os.path.join(self.base_dir, "queue", agent_name)
        self.processed_file = os.path.join(self.queue_dir, "processed.json")

        # 确保目录存在
        os.makedirs(self.queue_dir, exist_ok=True)

        # 加载已处理的消息 ID
        self.processed_ids = self._load_processed_ids()

    def write_message(self, content: Any, message_type: str = "text",
                      metadata: dict = None) -> str:
        """
        写入消息到队列

        Args:
            content: 消息内容
            message_type: 消息类型
            metadata: 元数据

        Returns:
            消息 ID
        """
        message = self._normalize_message(content, message_type, metadata)
        msg_id = message["id"]

        # Idempotent writes let adapters enqueue the same hub message safely
        # from polling and SSE paths without duplicating queue entries.
        if self.get_message_by_id(msg_id) is not None:
            return msg_id

        # 写入队列文件
        filename = f"{self._safe_filename_part(msg_id)}_{int(time.time())}.json"
        filepath = os.path.join(self.queue_dir, filename)
        self._write_json_atomic(filepath, message)

        print(f"[QueueProtocol] 写入消息: {msg_id}")
        return msg_id

    def read_messages(self, include_processed: bool = False) -> List[Dict]:
        """
        读取队列中的消息

        Args:
            include_processed: 是否包含已处理的消息

        Returns:
            消息列表
        """
        messages = []

        # 遍历队列目录
        for filename in os.listdir(self.queue_dir):
            if not filename.endswith(".json") or filename == "processed.json":
                continue

            filepath = os.path.join(self.queue_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    msg = json.load(f)

                # 检查是否已处理
                msg_id = msg.get("id")
                is_processed = msg_id in self.processed_ids or msg.get("processed") is True
                if not include_processed and is_processed:
                    continue

                messages.append(msg)

            except Exception as e:
                print(f"[QueueProtocol] 读取消息失败: {filename} - {e}")

        # 按时间排序
        messages.sort(key=lambda m: m.get("timestamp", ""))
        return messages

    def mark_processed(self, msg_id: str):
        """
        标记消息已处理

        Args:
            msg_id: 消息 ID
        """
        if msg_id not in self.processed_ids:
            self.processed_ids.add(msg_id)
            self._save_processed_ids()

        msg_file = self._find_message_file(msg_id)
        if msg_file:
            try:
                with open(msg_file, "r", encoding="utf-8") as f:
                    msg = json.load(f)
                msg["processed"] = True
                msg["processed_at"] = datetime.now().isoformat()
                self._write_json_atomic(msg_file, msg)
            except Exception as e:
                print(f"[QueueProtocol] 更新消息状态失败: {msg_id} - {e}")

        print(f"[QueueProtocol] 标记已处理: {msg_id}")

    def get_message_by_id(self, msg_id: str) -> Optional[Dict]:
        """
        根据 ID 获取消息

        Args:
            msg_id: 消息 ID

        Returns:
            消息字典，如果不存在返回 None
        """
        for filename in os.listdir(self.queue_dir):
            if not filename.endswith(".json") or filename == "processed.json":
                continue

            filepath = os.path.join(self.queue_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    msg = json.load(f)
                if msg.get("id") == msg_id:
                    return msg
            except Exception:
                continue

        return None

    def clear_processed(self):
        """清理已处理的消息文件"""
        for filename in os.listdir(self.queue_dir):
            if not filename.endswith(".json") or filename == "processed.json":
                continue

            filepath = os.path.join(self.queue_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    msg = json.load(f)

                msg_id = msg.get("id")
                if msg_id in self.processed_ids or msg.get("processed") is True:
                    os.remove(filepath)
                    print(f"[QueueProtocol] 清理已处理消息: {msg_id}")

            except Exception as e:
                print(f"[QueueProtocol] 清理失败: {filename} - {e}")

    def get_stats(self) -> Dict:
        """
        获取队列统计信息

        Returns:
            统计信息字典
        """
        total = 0
        processed = 0
        pending = 0

        for filename in os.listdir(self.queue_dir):
            if not filename.endswith(".json") or filename == "processed.json":
                continue

            filepath = os.path.join(self.queue_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    msg = json.load(f)

                msg_id = msg.get("id")
                total += 1

                if msg_id in self.processed_ids or msg.get("processed") is True:
                    processed += 1
                else:
                    pending += 1

            except Exception:
                continue

        return {
            "total": total,
            "processed": processed,
            "pending": pending,
            "queue_dir": self.queue_dir,
        }

    def _load_processed_ids(self) -> set:
        """加载已处理的消息 ID"""
        if not os.path.exists(self.processed_file):
            return set()

        try:
            with open(self.processed_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return set(data.get("processed_ids", []))
        except Exception:
            return set()

    def _save_processed_ids(self):
        """保存已处理的消息 ID"""
        data = {
            "processed_ids": sorted(self.processed_ids),
            "updated_at": datetime.now().isoformat(),
        }

        with open(self.processed_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _normalize_message(self, content: Any, message_type: str, metadata: dict = None) -> Dict:
        if isinstance(content, dict):
            message = dict(content)
            message["id"] = message.get("id") or f"msg_{uuid.uuid4().hex[:10]}"
            message["content"] = message.get("content", "")
            message["message_type"] = message.get("message_type", message_type)
            if metadata:
                merged_metadata = dict(message.get("metadata") or {})
                merged_metadata.update(metadata)
                message["metadata"] = merged_metadata
            else:
                message["metadata"] = message.get("metadata") or {}
            message["timestamp"] = message.get("timestamp") or self._timestamp_from_message(message)
            message["processed"] = bool(message.get("processed", False))
            return message

        return {
            "id": f"msg_{uuid.uuid4().hex[:10]}",
            "content": str(content),
            "message_type": message_type,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "processed": False,
        }

    def _timestamp_from_message(self, message: Dict) -> str:
        created_at = message.get("created_at")
        if isinstance(created_at, (int, float)):
            return datetime.fromtimestamp(created_at).isoformat()
        return datetime.now().isoformat()

    def _find_message_file(self, msg_id: str) -> Optional[str]:
        for filename in os.listdir(self.queue_dir):
            if not filename.endswith(".json") or filename == "processed.json":
                continue
            filepath = os.path.join(self.queue_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    msg = json.load(f)
                if msg.get("id") == msg_id:
                    return filepath
            except Exception:
                continue
        return None

    def _write_json_atomic(self, filepath: str, data: Dict):
        tmp_path = f"{filepath}.{uuid.uuid4().hex}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, filepath)

    def _safe_filename_part(self, value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def main():
    """测试队列协议"""
    print("测试队列式文件协议...")
    print()

    # 创建 Codex 队列
    codex_queue = QueueProtocol("codex")

    # 写入测试消息
    msg_id1 = codex_queue.write_message("你好 Codex！")
    msg_id2 = codex_queue.write_message("请帮我检查文件。", message_type="task")
    msg_id3 = codex_queue.write_message("这里有错误。", message_type="correction")

    print()

    # 读取消息
    messages = codex_queue.read_messages()
    print(f"队列中有 {len(messages)} 条消息：")
    for msg in messages:
        print(f"  - {msg['id']}: {msg['content'][:30]}...")

    print()

    # 标记已处理
    codex_queue.mark_processed(msg_id1)
    codex_queue.mark_processed(msg_id2)

    # 再次读取
    messages = codex_queue.read_messages()
    print(f"处理后剩余 {len(messages)} 条消息：")
    for msg in messages:
        print(f"  - {msg['id']}: {msg['content'][:30]}...")

    print()

    # 统计信息
    stats = codex_queue.get_stats()
    print(f"队列统计：")
    print(f"  总计: {stats['total']}")
    print(f"  已处理: {stats['processed']}")
    print(f"  待处理: {stats['pending']}")
    print(f"  队列目录: {stats['queue_dir']}")

    print()

    # 清理已处理消息
    codex_queue.clear_processed()

    print()
    print("测试完成！")


if __name__ == "__main__":
    main()
