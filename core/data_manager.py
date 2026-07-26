"""
数据管理器

提供用户数据删除、导出等功能。
"""

import json
import os
import time
from typing import Dict


class DataManager:
    """
    数据管理器

    提供用户数据删除、导出等功能。
    """

    def __init__(self, memory_system):
        """
        Args:
            memory_system: HumanLikeMemorySystem 实例
        """
        self.memory = memory_system

    def delete_all_user_data(self, user_id: str = "default") -> Dict:
        """
        删除用户所有数据

        Args:
            user_id: 用户 ID

        Returns:
            删除统计
        """
        deleted = {
            "core_memories": 0,
            "forgotten_memories": 0,
            "persona": False,
            "attention": False,
            "cognitive_state": False,
        }

        # 删除核心层记忆
        if hasattr(self.memory, 'core') and self.memory.core:
            chunks_to_delete = []
            for chunk_id, chunk in self.memory.core._store.get_all().items():
                if self._belongs_to_user(chunk, user_id):
                    chunks_to_delete.append(chunk_id)

            for chunk_id in chunks_to_delete:
                self.memory.core.remove(chunk_id)
                deleted["core_memories"] += 1

        # 删除伪遗忘层记忆
        if hasattr(self.memory, 'forgotten') and self.memory.forgotten:
            chunks_to_delete = []
            for chunk_id, chunk in self.memory.forgotten._store.get_all().items():
                if self._belongs_to_user(chunk, user_id):
                    chunks_to_delete.append(chunk_id)

            for chunk_id in chunks_to_delete:
                self.memory.forgotten.remove(chunk_id)
                deleted["forgotten_memories"] += 1

        # 重置 persona
        if hasattr(self.memory, 'persona'):
            from core.persona_layer import PersonaLayer
            self.memory.persona = PersonaLayer()
            deleted["persona"] = True

        # 重置 attention
        if hasattr(self.memory, 'attention'):
            from core.attention_system import AttentionOS
            self.memory.attention = AttentionOS()
            deleted["attention"] = True

        # 重置 cognitive state
        if hasattr(self.memory, 'cognitive_state'):
            from core.cognitive_state import CognitiveState
            self.memory.cognitive_state = CognitiveState()
            deleted["cognitive_state"] = True

        # 保存更改
        if hasattr(self.memory, 'save'):
            self.memory.save()

        return deleted

    def export_user_data(self, user_id: str = "default") -> Dict:
        """
        导出用户数据

        Args:
            user_id: 用户 ID

        Returns:
            用户数据字典
        """
        data = {
            "user_id": user_id,
            "export_time": time.time(),
            "core_memories": [],
            "forgotten_memories": [],
            "persona": None,
            "attention": None,
            "cognitive_state": None,
        }

        # 导出核心层记忆
        if hasattr(self.memory, 'core') and self.memory.core:
            for chunk_id, chunk in self.memory.core._store.get_all().items():
                if self._belongs_to_user(chunk, user_id):
                    data["core_memories"].append(chunk.to_dict())

        # 导出伪遗忘层记忆
        if hasattr(self.memory, 'forgotten') and self.memory.forgotten:
            for chunk_id, chunk in self.memory.forgotten._store.get_all().items():
                if self._belongs_to_user(chunk, user_id):
                    data["forgotten_memories"].append(chunk.to_dict())

        # 导出 persona
        if hasattr(self.memory, 'persona'):
            data["persona"] = self.memory.persona.export_profile()

        # 导出 attention
        if hasattr(self.memory, 'attention'):
            data["attention"] = self.memory.attention.to_dict()

        # 导出 cognitive state
        if hasattr(self.memory, 'cognitive_state'):
            data["cognitive_state"] = self.memory.cognitive_state.to_dict()

        return data

    def export_to_file(self, user_id: str, output_file: str) -> bool:
        """
        导出用户数据到文件

        Args:
            user_id: 用户 ID
            output_file: 输出文件路径

        Returns:
            是否成功
        """
        try:
            data = self.export_user_data(user_id)
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"[DataManager] 导出失败: {e}")
            return False

    def get_data_stats(self, user_id: str = "default") -> Dict:
        """
        获取用户数据统计

        Args:
            user_id: 用户 ID

        Returns:
            统计信息
        """
        stats = {
            "user_id": user_id,
            "core_memories": 0,
            "forgotten_memories": 0,
            "total_size_bytes": 0,
        }

        # 统计核心层记忆
        if hasattr(self.memory, 'core') and self.memory.core:
            for chunk_id, chunk in self.memory.core._store.get_all().items():
                if self._belongs_to_user(chunk, user_id):
                    stats["core_memories"] += 1
                    stats["total_size_bytes"] += len(chunk.content.encode("utf-8"))

        # 统计伪遗忘层记忆
        if hasattr(self.memory, 'forgotten') and self.memory.forgotten:
            for chunk_id, chunk in self.memory.forgotten._store.get_all().items():
                if self._belongs_to_user(chunk, user_id):
                    stats["forgotten_memories"] += 1
                    stats["total_size_bytes"] += len(chunk.content.encode("utf-8"))

        return stats

    def _belongs_to_user(self, chunk, user_id: str) -> bool:
        """
        检查记忆是否属于用户

        默认所有记忆都属于 "default" 用户。
        可以通过 metadata.user_id 字段区分用户。
        """
        if user_id == "default":
            # 默认用户拥有所有没有明确用户标记的记忆
            chunk_user = chunk.metadata.get("user_id", "default")
            return chunk_user == "default"
        else:
            # 非默认用户只拥有明确标记的记忆
            chunk_user = chunk.metadata.get("user_id", "default")
            return chunk_user == user_id
