"""
可插拔存储后端接口

设计原则：
- MemoryStore: 记忆碎片的 CRUD + 持久化
- KeyValueStore: 通用 KV 存储（用于 persona/attention/cognitive_state）
- 所有后端必须实现相同的接口，支持热切换
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Set


class MemoryStore(ABC):
    """
    记忆碎片存储接口

    职责：
    - 存储和检索 MemoryChunk 对象
    - 持久化到后端（JSON/SQLite/Postgres 等）
    - 支持按 ID 随机访问和全量遍历
    """

    @abstractmethod
    def put(self, chunk) -> None:
        """
        存储一个记忆碎片

        Args:
            chunk: MemoryChunk 对象
        """
        ...

    @abstractmethod
    def get(self, chunk_id: str):
        """
        按 ID 获取记忆碎片

        Args:
            chunk_id: 碎片 ID

        Returns:
            MemoryChunk 对象，不存在返回 None
        """
        ...

    @abstractmethod
    def delete(self, chunk_id: str) -> bool:
        """
        删除记忆碎片

        Args:
            chunk_id: 碎片 ID

        Returns:
            是否成功删除
        """
        ...

    @abstractmethod
    def get_all(self) -> Dict[str, Any]:
        """
        获取所有记忆碎片

        Returns:
            Dict[chunk_id, MemoryChunk]
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """获取碎片总数"""
        ...

    @abstractmethod
    def save(self) -> None:
        """持久化到后端"""
        ...

    @abstractmethod
    def load(self) -> bool:
        """
        从后端加载

        Returns:
            是否成功加载（文件/表不存在返回 False）
        """
        ...


class IndexStore(ABC):
    """
    倒排索引存储接口

    职责：
    - 存储和检索倒排索引（key -> set of chunk_ids）
    - 支持内存或持久化两种模式
    """

    @abstractmethod
    def add(self, index_name: str, key: str, chunk_id: str) -> None:
        """添加索引条目"""
        ...

    @abstractmethod
    def remove(self, index_name: str, key: str, chunk_id: str) -> None:
        """删除索引条目"""
        ...

    @abstractmethod
    def get(self, index_name: str, key: str) -> Set[str]:
        """获取某个 key 的所有 chunk_ids"""
        ...

    @abstractmethod
    def remove_chunk(self, chunk_id: str) -> None:
        """删除某个 chunk_id 在所有索引中的条目"""
        ...

    @abstractmethod
    def clear(self) -> None:
        """清空所有索引"""
        ...

    @abstractmethod
    def save(self) -> None:
        """持久化索引"""
        ...

    @abstractmethod
    def load(self) -> bool:
        """加载索引"""
        ...


class KeyValueStore(ABC):
    """
    通用 KV 存储接口

    职责：
    - 存储和检索 JSON 可序列化的值
    - 支持 namespace 隔离（persona/attention/cognitive_state）
    """

    @abstractmethod
    def get(self, namespace: str, key: str) -> Optional[Any]:
        """获取值"""
        ...

    @abstractmethod
    def put(self, namespace: str, key: str, value: Any) -> None:
        """存储值"""
        ...

    @abstractmethod
    def delete(self, namespace: str, key: str) -> bool:
        """删除值"""
        ...

    @abstractmethod
    def save(self) -> None:
        """持久化"""
        ...

    @abstractmethod
    def load(self) -> bool:
        """加载"""
        ...
