"""
JSON 文件存储后端

保持与现有 core.json / forgotten.json 格式完全兼容。
内存中维护 dict，save 时写文件，load 时读文件。
"""

import json
import os
import time
from typing import Dict, Optional, Any

from core.store import MemoryStore
from memory_chunk import MemoryChunk


class JsonMemoryStore(MemoryStore):
    """
    JSON 文件存储后端

    特点：
    - 与现有 core.json / forgotten.json 格式 100% 兼容
    - 内存中维护 dict，读写性能好
    - save 时原子写入（先写临时文件再 rename）
    """

    def __init__(self, filepath: str):
        """
        Args:
            filepath: JSON 文件路径，如 ./memory_data/core.json
        """
        self.filepath = filepath
        self.chunks: Dict[str, MemoryChunk] = {}
        self._loaded = False

    def put(self, chunk: MemoryChunk) -> None:
        self.chunks[chunk.id] = chunk

    def get(self, chunk_id: str) -> Optional[MemoryChunk]:
        return self.chunks.get(chunk_id)

    def delete(self, chunk_id: str) -> bool:
        if chunk_id in self.chunks:
            del self.chunks[chunk_id]
            return True
        return False

    def get_all(self) -> Dict[str, MemoryChunk]:
        return dict(self.chunks)

    def count(self) -> int:
        return len(self.chunks)

    def save(self) -> None:
        """原子写入 JSON 文件"""
        data = {
            "chunks": {cid: chunk.to_dict() for cid, chunk in self.chunks.items()},
            "timestamp": time.time(),
        }

        # 原子写入：先写临时文件再 rename
        tmp_path = self.filepath + ".tmp"
        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.filepath)

    def load(self) -> bool:
        """从 JSON 文件加载"""
        if not os.path.exists(self.filepath):
            return False

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            chunks_data = data.get("chunks", {})
            self.chunks = {}
            for cid, chunk_dict in chunks_data.items():
                self.chunks[cid] = MemoryChunk.from_dict(chunk_dict)

            self._loaded = True
            return True
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"[JsonMemoryStore] 加载失败: {e}")
            return False


class JsonKeyValueStore:
    """
    JSON KV 存储后端

    用于 persona/attention/cognitive_state 等子系统。
    每个 namespace 对应一个 JSON 文件。
    """

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: 数据目录，如 ./memory_data
        """
        self.data_dir = data_dir
        self._cache: Dict[str, Any] = {}

    def get(self, namespace: str, key: str) -> Optional[Any]:
        data = self._cache.get(namespace, {})
        return data.get(key)

    def put(self, namespace: str, key: str, value: Any) -> None:
        if namespace not in self._cache:
            self._cache[namespace] = {}
        self._cache[namespace][key] = value

    def delete(self, namespace: str, key: str) -> bool:
        if namespace in self._cache and key in self._cache[namespace]:
            del self._cache[namespace][key]
            return True
        return False

    def save(self) -> None:
        """将所有 namespace 写入各自的 JSON 文件"""
        os.makedirs(self.data_dir, exist_ok=True)
        for namespace, data in self._cache.items():
            filepath = os.path.join(self.data_dir, f"{namespace}.json")
            tmp_path = filepath + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, filepath)

    def load(self) -> bool:
        """从 JSON 文件加载所有 namespace"""
        if not os.path.exists(self.data_dir):
            return False

        loaded = False
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                namespace = filename[:-5]
                filepath = os.path.join(self.data_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        self._cache[namespace] = json.load(f)
                    loaded = True
                except (json.JSONDecodeError, IOError):
                    pass
        return loaded
