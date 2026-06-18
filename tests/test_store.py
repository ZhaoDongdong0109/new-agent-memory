"""
存储后端测试

测试 JsonMemoryStore 和 SqliteMemoryStore 的功能和一致性。
"""

import json
import os
import tempfile
import time
import pytest

from memory_chunk import MemoryChunk, MemoryLayer, MemoryType
from core.json_store import JsonMemoryStore
from core.sqlite_store import SqliteMemoryStore


def make_test_chunk(chunk_id: str = "test_001", content: str = "测试记忆") -> MemoryChunk:
    """创建测试用的 MemoryChunk"""
    return MemoryChunk(
        id=chunk_id,
        content=content,
        summary="测试摘要",
        memory_type=MemoryType.INTERACTION,
        time_absolute="2026-06-18",
        time_relative="今天",
        time_context="中午",
        location="北京",
        persons={"张三", "李四"},
        topics={"测试", "开发"},
        keywords={"python", "memory"},
        emotion_valence=0.5,
        emotion_intensity=0.7,
        importance=0.8,
        layer=MemoryLayer.CORE,
        created_at=time.time(),
        updated_at=time.time(),
        last_accessed=time.time(),
        access_count=5,
        associations={"other_001": 0.3},
        metadata={"test": True},
    )


class TestJsonMemoryStore:
    """测试 JSON 存储后端"""

    def test_put_and_get(self, tmp_path):
        """测试存储和获取"""
        filepath = str(tmp_path / "test.json")
        store = JsonMemoryStore(filepath)

        chunk = make_test_chunk()
        store.put(chunk)

        retrieved = store.get("test_001")
        assert retrieved is not None
        assert retrieved.id == "test_001"
        assert retrieved.content == "测试记忆"

    def test_get_nonexistent(self, tmp_path):
        """测试获取不存在的记录"""
        filepath = str(tmp_path / "test.json")
        store = JsonMemoryStore(filepath)

        result = store.get("nonexistent")
        assert result is None

    def test_delete(self, tmp_path):
        """测试删除"""
        filepath = str(tmp_path / "test.json")
        store = JsonMemoryStore(filepath)

        chunk = make_test_chunk()
        store.put(chunk)
        assert store.count() == 1

        result = store.delete("test_001")
        assert result is True
        assert store.count() == 0
        assert store.get("test_001") is None

    def test_delete_nonexistent(self, tmp_path):
        """测试删除不存在的记录"""
        filepath = str(tmp_path / "test.json")
        store = JsonMemoryStore(filepath)

        result = store.delete("nonexistent")
        assert result is False

    def test_get_all(self, tmp_path):
        """测试获取所有记录"""
        filepath = str(tmp_path / "test.json")
        store = JsonMemoryStore(filepath)

        store.put(make_test_chunk("chunk_1", "记忆1"))
        store.put(make_test_chunk("chunk_2", "记忆2"))
        store.put(make_test_chunk("chunk_3", "记忆3"))

        all_chunks = store.get_all()
        assert len(all_chunks) == 3
        assert "chunk_1" in all_chunks
        assert "chunk_2" in all_chunks
        assert "chunk_3" in all_chunks

    def test_count(self, tmp_path):
        """测试计数"""
        filepath = str(tmp_path / "test.json")
        store = JsonMemoryStore(filepath)

        assert store.count() == 0

        store.put(make_test_chunk("chunk_1"))
        assert store.count() == 1

        store.put(make_test_chunk("chunk_2"))
        assert store.count() == 2

    def test_save_and_load(self, tmp_path):
        """测试持久化"""
        filepath = str(tmp_path / "test.json")
        store = JsonMemoryStore(filepath)

        store.put(make_test_chunk("chunk_1", "记忆1"))
        store.put(make_test_chunk("chunk_2", "记忆2"))
        store.save()

        # 创建新的 store 实例加载
        store2 = JsonMemoryStore(filepath)
        result = store2.load()
        assert result is True

        assert store2.count() == 2
        chunk = store2.get("chunk_1")
        assert chunk is not None
        assert chunk.content == "记忆1"

    def test_load_nonexistent(self, tmp_path):
        """测试加载不存在的文件"""
        filepath = str(tmp_path / "nonexistent.json")
        store = JsonMemoryStore(filepath)

        result = store.load()
        assert result is False

    def test_complex_fields(self, tmp_path):
        """测试复杂字段的序列化/反序列化"""
        filepath = str(tmp_path / "test.json")
        store = JsonMemoryStore(filepath)

        chunk = make_test_chunk()
        store.put(chunk)
        store.save()

        store2 = JsonMemoryStore(filepath)
        store2.load()

        retrieved = store2.get("test_001")
        assert retrieved.persons == {"张三", "李四"}
        assert retrieved.topics == {"测试", "开发"}
        assert retrieved.associations == {"other_001": 0.3}
        assert retrieved.metadata == {"test": True}


class TestSqliteMemoryStore:
    """测试 SQLite 存储后端"""

    def test_put_and_get(self, tmp_path):
        """测试存储和获取"""
        db_path = str(tmp_path / "test.db")
        store = SqliteMemoryStore(db_path)
        store.load()

        chunk = make_test_chunk()
        store.put(chunk)

        retrieved = store.get("test_001")
        assert retrieved is not None
        assert retrieved.id == "test_001"
        assert retrieved.content == "测试记忆"

    def test_get_nonexistent(self, tmp_path):
        """测试获取不存在的记录"""
        db_path = str(tmp_path / "test.db")
        store = SqliteMemoryStore(db_path)
        store.load()

        result = store.get("nonexistent")
        assert result is None

    def test_delete(self, tmp_path):
        """测试删除"""
        db_path = str(tmp_path / "test.db")
        store = SqliteMemoryStore(db_path)
        store.load()

        chunk = make_test_chunk()
        store.put(chunk)
        assert store.count() == 1

        result = store.delete("test_001")
        assert result is True
        assert store.count() == 0
        assert store.get("test_001") is None

    def test_delete_nonexistent(self, tmp_path):
        """测试删除不存在的记录"""
        db_path = str(tmp_path / "test.db")
        store = SqliteMemoryStore(db_path)
        store.load()

        result = store.delete("nonexistent")
        assert result is False

    def test_get_all(self, tmp_path):
        """测试获取所有记录"""
        db_path = str(tmp_path / "test.db")
        store = SqliteMemoryStore(db_path)
        store.load()

        store.put(make_test_chunk("chunk_1", "记忆1"))
        store.put(make_test_chunk("chunk_2", "记忆2"))
        store.put(make_test_chunk("chunk_3", "记忆3"))

        all_chunks = store.get_all()
        assert len(all_chunks) == 3
        assert "chunk_1" in all_chunks
        assert "chunk_2" in all_chunks
        assert "chunk_3" in all_chunks

    def test_count(self, tmp_path):
        """测试计数"""
        db_path = str(tmp_path / "test.db")
        store = SqliteMemoryStore(db_path)
        store.load()

        assert store.count() == 0

        store.put(make_test_chunk("chunk_1"))
        assert store.count() == 1

        store.put(make_test_chunk("chunk_2"))
        assert store.count() == 2

    def test_persistence(self, tmp_path):
        """测试持久化（SQLite 自动持久化）"""
        db_path = str(tmp_path / "test.db")
        store = SqliteMemoryStore(db_path)
        store.load()

        store.put(make_test_chunk("chunk_1", "记忆1"))
        store.put(make_test_chunk("chunk_2", "记忆2"))

        # 创建新的 store 实例加载
        store2 = SqliteMemoryStore(db_path)
        store2.load()

        assert store2.count() == 2
        chunk = store2.get("chunk_1")
        assert chunk is not None
        assert chunk.content == "记忆1"

    def test_complex_fields(self, tmp_path):
        """测试复杂字段的序列化/反序列化"""
        db_path = str(tmp_path / "test.db")
        store = SqliteMemoryStore(db_path)
        store.load()

        chunk = make_test_chunk()
        store.put(chunk)

        # 重新加载
        store2 = SqliteMemoryStore(db_path)
        store2.load()

        retrieved = store2.get("test_001")
        assert retrieved.persons == {"张三", "李四"}
        assert retrieved.topics == {"测试", "开发"}
        assert retrieved.associations == {"other_001": 0.3}
        assert retrieved.metadata == {"test": True}

    def test_table_prefix(self, tmp_path):
        """测试表前缀（用于区分 core/forgotten）"""
        db_path = str(tmp_path / "test.db")

        core_store = SqliteMemoryStore(db_path, table_prefix="core_")
        forgotten_store = SqliteMemoryStore(db_path, table_prefix="forgotten_")
        core_store.load()
        forgotten_store.load()

        core_chunk = make_test_chunk("core_001", "核心记忆")
        forgotten_chunk = make_test_chunk("forgotten_001", "遗忘记忆")
        forgotten_chunk.layer = MemoryLayer.FORGOTTEN

        core_store.put(core_chunk)
        forgotten_store.put(forgotten_chunk)

        # 验证两个表独立
        assert core_store.count() == 1
        assert forgotten_store.count() == 1
        assert core_store.get("core_001") is not None
        assert forgotten_store.get("forgotten_001") is not None
        assert core_store.get("forgotten_001") is None
        assert forgotten_store.get("core_001") is None

    def test_index_operations(self, tmp_path):
        """测试索引操作"""
        db_path = str(tmp_path / "test.db")
        store = SqliteMemoryStore(db_path)
        store.load()

        # 添加带索引的数据
        chunk1 = make_test_chunk("chunk_1")
        chunk1.topics = {"python", "memory"}
        chunk1.location = "北京"
        chunk1.persons = {"张三"}

        chunk2 = make_test_chunk("chunk_2")
        chunk2.topics = {"python", "test"}
        chunk2.location = "上海"
        chunk2.persons = {"李四"}

        store.put(chunk1)
        store.put(chunk2)

        # 测试索引查询
        beijing_chunks = store.get_index("location_idx", "北京")
        assert "chunk_1" in beijing_chunks
        assert "chunk_2" not in beijing_chunks

        python_chunks = store.get_index("topic_idx", "python")
        assert "chunk_1" in python_chunks
        assert "chunk_2" in python_chunks

    def test_stats(self, tmp_path):
        """测试统计功能"""
        db_path = str(tmp_path / "test.db")
        store = SqliteMemoryStore(db_path)
        store.load()

        store.set_stat("total_recall_success", 42)
        store.set_stat("total_recall_fail", 3)

        assert store.get_stat("total_recall_success") == 42
        assert store.get_stat("total_recall_fail") == 3

    def test_rebuild_indexes(self, tmp_path):
        """测试重建索引"""
        db_path = str(tmp_path / "test.db")
        store = SqliteMemoryStore(db_path)
        store.load()

        chunk = make_test_chunk()
        chunk.topics = {"python", "memory"}
        chunk.location = "北京"
        chunk.persons = {"张三"}
        store.put(chunk)

        # 重建索引
        store.rebuild_indexes()

        # 验证索引仍然有效
        beijing_chunks = store.get_index("location_idx", "北京")
        assert "test_001" in beijing_chunks


class TestStoreConsistency:
    """测试 JSON 和 SQLite 后端的一致性"""

    def test_round_trip_consistency(self, tmp_path):
        """测试两种后端存储相同数据后结果一致"""
        # 创建测试数据
        chunks = [
            make_test_chunk("chunk_1", "记忆1"),
            make_test_chunk("chunk_2", "记忆2"),
            make_test_chunk("chunk_3", "记忆3"),
        ]

        # JSON 后端
        json_path = str(tmp_path / "test.json")
        json_store = JsonMemoryStore(json_path)
        for chunk in chunks:
            json_store.put(chunk)
        json_store.save()

        # SQLite 后端
        db_path = str(tmp_path / "test.db")
        sqlite_store = SqliteMemoryStore(db_path)
        sqlite_store.load()
        for chunk in chunks:
            sqlite_store.put(chunk)

        # 比较结果
        json_store2 = JsonMemoryStore(json_path)
        json_store2.load()

        sqlite_store2 = SqliteMemoryStore(db_path)
        sqlite_store2.load()

        assert json_store2.count() == sqlite_store2.count()

        for chunk_id in ["chunk_1", "chunk_2", "chunk_3"]:
            json_chunk = json_store2.get(chunk_id)
            sqlite_chunk = sqlite_store2.get(chunk_id)

            assert json_chunk is not None
            assert sqlite_chunk is not None
            assert json_chunk.content == sqlite_chunk.content
            assert json_chunk.importance == sqlite_chunk.importance
            assert json_chunk.persons == sqlite_chunk.persons
            assert json_chunk.topics == sqlite_chunk.topics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
