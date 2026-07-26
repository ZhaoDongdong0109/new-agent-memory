"""
安全审计日志

记录安全相关事件：
- 记忆访问
- 数据删除
- PII 检测
- 安全事件
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional


class AuditLogger:
    """
    安全审计日志

    记录安全相关事件，用于审计和追踪。
    """

    def __init__(
        self,
        log_file: str = "audit.log",
        json_format: bool = True,
    ):
        """
        Args:
            log_file: 日志文件路径
            json_format: 是否使用 JSON 格式
        """
        self.log_file = log_file
        self.json_format = json_format
        self.logger = self._setup_logger()

    def log_memory_access(
        self,
        chunk_id: str,
        user_id: str,
        action: str,
        details: Optional[Dict] = None,
    ):
        """
        记录记忆访问

        Args:
            chunk_id: 记忆 ID
            user_id: 用户 ID
            action: 操作类型 (create, read, update, delete)
            details: 额外详情
        """
        event = {
            "event_type": "memory_access",
            "chunk_id": chunk_id,
            "user_id": user_id,
            "action": action,
            "timestamp": time.time(),
        }
        if details:
            event["details"] = details

        self.logger.info(json.dumps(event, ensure_ascii=False))

    def log_data_deletion(
        self,
        user_id: str,
        deleted: Dict,
        reason: str = "user_request",
    ):
        """
        记录数据删除

        Args:
            user_id: 用户 ID
            deleted: 删除统计
            reason: 删除原因
        """
        event = {
            "event_type": "data_deletion",
            "user_id": user_id,
            "deleted": deleted,
            "reason": reason,
            "timestamp": time.time(),
        }

        self.logger.warning(json.dumps(event, ensure_ascii=False))

    def log_pii_detection(
        self,
        text_hash: str,
        pii_types: List[str],
        action: str = "redact",
    ):
        """
        记录 PII 检测

        Args:
            text_hash: 文本哈希（不记录原文）
            pii_types: 检测到的 PII 类型
            action: 执行的操作 (redact, anonymize, block)
        """
        event = {
            "event_type": "pii_detection",
            "text_hash": text_hash,
            "pii_types": pii_types,
            "action": action,
            "timestamp": time.time(),
        }

        self.logger.info(json.dumps(event, ensure_ascii=False))

    def log_security_event(
        self,
        event_type: str,
        details: Dict,
        severity: str = "info",
    ):
        """
        记录安全事件

        Args:
            event_type: 事件类型
            details: 事件详情
            severity: 严重程度 (info, warning, error, critical)
        """
        event = {
            "event_type": f"security_{event_type}",
            "severity": severity,
            "details": details,
            "timestamp": time.time(),
        }

        log_func = {
            "info": self.logger.info,
            "warning": self.logger.warning,
            "error": self.logger.error,
            "critical": self.logger.critical,
        }.get(severity, self.logger.info)

        log_func(json.dumps(event, ensure_ascii=False))

    def log_retrieval(
        self,
        user_id: str,
        query_hash: str,
        num_results: int,
        latency: float,
    ):
        """
        记录检索操作

        Args:
            user_id: 用户 ID
            query_hash: 查询哈希
            num_results: 结果数量
            latency: 延迟（秒）
        """
        event = {
            "event_type": "retrieval",
            "user_id": user_id,
            "query_hash": query_hash,
            "num_results": num_results,
            "latency": latency,
            "timestamp": time.time(),
        }

        self.logger.info(json.dumps(event, ensure_ascii=False))

    def log_consolidation(
        self,
        episode_id: str,
        num_memories: int,
        extractor_type: str,
    ):
        """
        记录巩固操作

        Args:
            episode_id: Episode ID
            num_memories: 生成的记忆数量
            extractor_type: 抽取器类型
        """
        event = {
            "event_type": "consolidation",
            "episode_id": episode_id,
            "num_memories": num_memories,
            "extractor_type": extractor_type,
            "timestamp": time.time(),
        }

        self.logger.info(json.dumps(event, ensure_ascii=False))

    def _setup_logger(self) -> logging.Logger:
        """设置日志器

        每个日志文件使用独立的 logger 名称，避免多个 AuditLogger 实例
        （指向不同文件）共享同一个 handler 而写错文件。
        """
        # 以日志文件的绝对路径区分 logger，保证不同文件之间彼此隔离。
        target = os.path.abspath(self.log_file)
        logger = logging.getLogger(f"audit.{target}")
        logger.setLevel(logging.INFO)
        # 审计日志独立落盘，不向 root logger 传播（避免重复输出）。
        logger.propagate = False

        # 避免为同一个文件重复添加 handler。
        for existing in logger.handlers:
            if isinstance(existing, logging.FileHandler) and \
                    os.path.abspath(getattr(existing, "baseFilename", "")) == target:
                return logger

        # 文件 handler
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        handler = logging.FileHandler(target, encoding="utf-8")

        if self.json_format:
            # JSON 格式不需要 formatter
            pass
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)

        logger.addHandler(handler)

        return logger

    def get_recent_events(
        self,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """
        获取最近的审计事件

        Args:
            event_type: 事件类型过滤
            limit: 返回数量限制

        Returns:
            事件列表
        """
        events = []

        if not os.path.exists(self.log_file):
            return events

        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        event = json.loads(line)
                        if event_type and event.get("event_type") != event_type:
                            continue
                        events.append(event)
                    except json.JSONDecodeError:
                        continue

            # 返回最近的事件
            return events[-limit:]

        except Exception as e:
            print(f"[AuditLogger] 读取日志失败: {e}")
            return []
