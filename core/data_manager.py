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

    def delete_all_user_data(
        self,
        user_id: str = "default",
        force_reset_global: bool = False,
    ) -> Dict:
        """
        删除用户所有数据

        注意：persona / attention / cognitive_state 是全局共享状态，
        并不按用户隔离。为避免删除单个用户时连带清空其他用户仍在
        依赖的全局层，只有在删除后核心层与伪遗忘层不再残留任何记忆
        （即被删除者是唯一用户），或调用方显式传入
        force_reset_global=True 时，才会重置这三个全局层。

        Args:
            user_id: 用户 ID
            force_reset_global: 即使仍存在其他用户的记忆，也强制重置
                全局 persona / attention / cognitive 状态

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

            planner = getattr(self.memory, "query_planner", None)
            for chunk_id in chunks_to_delete:
                self.memory.core.remove(chunk_id)
                # 同步清理混合检索索引：残留 id 会占用其他用户
                # 检索结果的融合名额
                if planner:
                    planner.remove_chunk(chunk_id)
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

        # 全局层重置策略：仅当删除后不再残留任何记忆（被删除者是
        # 唯一用户），或调用方显式要求时才重置，见方法 docstring。
        reset_global = force_reset_global or not self._has_remaining_memories()

        # 重置 persona
        if reset_global and hasattr(self.memory, 'persona'):
            from core.persona_layer import PersonaLayer
            self.memory.persona = PersonaLayer()
            deleted["persona"] = True

        # 重置 attention
        if reset_global and hasattr(self.memory, 'attention'):
            from core.attention_system import AttentionOS
            self.memory.attention = AttentionOS()
            deleted["attention"] = True

        # 重置 cognitive state
        if reset_global and hasattr(self.memory, 'cognitive_state'):
            from core.cognitive_state import CognitiveState
            self.memory.cognitive_state = CognitiveState()
            deleted["cognitive_state"] = True

        # 保存更改
        if hasattr(self.memory, 'save'):
            self.memory.save()

        # 写入审计日志（防御式：memory 系统可能没有 audit_logger 或为 None）
        audit_logger = getattr(self.memory, 'audit_logger', None)
        if audit_logger is not None and hasattr(audit_logger, 'log_data_deletion'):
            try:
                audit_logger.log_data_deletion(user_id=user_id, deleted=deleted)
            except Exception as e:
                print(f"[DataManager] 审计记录失败: {e}")

        return deleted

    def _has_remaining_memories(self) -> bool:
        """检查核心层与伪遗忘层是否仍残留任何记忆"""
        for layer_name in ('core', 'forgotten'):
            layer = getattr(self.memory, layer_name, None)
            if layer and layer._store.get_all():
                return True
        return False

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

        优先读取 MemoryChunk 的一等字段 user_id（main.add_memory /
        add_raw_memory 写入的位置）；一等字段缺失或仍为缺省值
        "default" 时，回退到 metadata["user_id"]（兼容旧数据）；
        两者都没有明确标记时归属 "default" 用户。
        """
        chunk_user = getattr(chunk, "user_id", None)
        if not chunk_user or chunk_user == "default":
            # 一等字段未明确标记用户，回退到 metadata（兼容旧数据）
            metadata = getattr(chunk, "metadata", None) or {}
            chunk_user = metadata.get("user_id") or chunk_user or "default"
        return chunk_user == user_id
