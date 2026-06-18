"""
SQLite 存储后端

特点：
- 使用 Python 内置 sqlite3，无外部依赖
- 支持倒排索引持久化
- 事务安全，支持并发读
- 适合中等规模数据（百万级记忆碎片）
"""

import json
import os
import sqlite3
import time
from typing import Dict, List, Optional, Set, Any

from core.store import MemoryStore, IndexStore
from memory_chunk import MemoryChunk, MemoryLayer, MemoryType


class SqliteMemoryStore(MemoryStore):
    """
    SQLite 记忆碎片存储后端

    表结构：
    - memory_chunks: 主表，存储所有 MemoryChunk 字段
    - time_idx / topic_idx / location_idx / person_idx: 倒排索引
    - stats: 统计信息
    """

    def __init__(self, db_path: str, table_prefix: str = ""):
        """
        Args:
            db_path: SQLite 数据库文件路径
            table_prefix: 表名前缀，用于区分 core/forgotten（如 "core_"）
        """
        self.db_path = db_path
        self.table_prefix = table_prefix
        self.conn: Optional[sqlite3.Connection] = None
        self._ensure_connection()

    def _ensure_connection(self):
        """确保数据库连接存在"""
        if self.conn is None:
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")

    def _table(self, name: str) -> str:
        """带前缀的表名"""
        return f"{self.table_prefix}{name}"

    def _ensure_schema(self):
        """创建表结构"""
        t = self._table

        # 主表
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {t('memory_chunks')} (
                id TEXT PRIMARY KEY,
                layer TEXT NOT NULL,
                content TEXT NOT NULL,
                summary TEXT DEFAULT '',
                memory_type TEXT DEFAULT 'interaction',
                tags TEXT DEFAULT '{{}}',
                time_absolute TEXT,
                time_relative TEXT,
                time_context TEXT,
                location TEXT,
                location_detail TEXT,
                persons TEXT DEFAULT '[]',
                person_count INTEGER DEFAULT 0,
                topics TEXT DEFAULT '[]',
                keywords TEXT DEFAULT '[]',
                emotion_valence REAL DEFAULT 0.0,
                emotion_intensity REAL DEFAULT 0.0,
                emotion_tags TEXT DEFAULT '[]',
                connection_value REAL DEFAULT 0.0,
                importance REAL DEFAULT 0.5,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                last_accessed REAL,
                access_count INTEGER DEFAULT 0,
                successful_recall_count INTEGER DEFAULT 0,
                associations TEXT DEFAULT '{{}}',
                review_status TEXT DEFAULT 'pending',
                review_note TEXT,
                reconstruction_count INTEGER DEFAULT 0,
                parent_id TEXT,
                metadata TEXT DEFAULT '{{}}'
            )
        """)

        # 倒排索引表
        for idx_name in ['time_idx', 'topic_idx', 'location_idx', 'person_idx']:
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {t(idx_name)} (
                    key TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    PRIMARY KEY (key, chunk_id)
                )
            """)

        # 统计表
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {t('stats')} (
                key TEXT PRIMARY KEY,
                value REAL NOT NULL
            )
        """)

        self.conn.commit()

    def put(self, chunk: MemoryChunk) -> None:
        """存储一个记忆碎片"""
        self._ensure_connection()

        t = self._table
        d = chunk.to_dict()

        # 插入/更新主表
        self.conn.execute(f"""
            INSERT OR REPLACE INTO {t('memory_chunks')} (
                id, layer, content, summary, memory_type, tags,
                time_absolute, time_relative, time_context,
                location, location_detail,
                persons, person_count, topics, keywords,
                emotion_valence, emotion_intensity, emotion_tags,
                connection_value, importance,
                created_at, updated_at, last_accessed,
                access_count, successful_recall_count,
                associations, review_status, review_note,
                reconstruction_count, parent_id, metadata
            ) VALUES (
                :id, :layer, :content, :summary, :memory_type, :tags,
                :time_absolute, :time_relative, :time_context,
                :location, :location_detail,
                :persons, :person_count, :topics, :keywords,
                :emotion_valence, :emotion_intensity, :emotion_tags,
                :connection_value, :importance,
                :created_at, :updated_at, :last_accessed,
                :access_count, :successful_recall_count,
                :associations, :review_status, :review_note,
                :reconstruction_count, :parent_id, :metadata
            )
        """, {
            "id": d["id"],
            "layer": d["layer"],
            "content": d["content"],
            "summary": d.get("summary", ""),
            "memory_type": d.get("memory_type", "interaction"),
            "tags": json.dumps(d.get("tags", {}), ensure_ascii=False),
            "time_absolute": d.get("time_absolute"),
            "time_relative": d.get("time_relative"),
            "time_context": d.get("time_context"),
            "location": d.get("location"),
            "location_detail": d.get("location_detail"),
            "persons": json.dumps(list(d.get("persons", [])), ensure_ascii=False),
            "person_count": d.get("person_count", 0),
            "topics": json.dumps(list(d.get("topics", [])), ensure_ascii=False),
            "keywords": json.dumps(list(d.get("keywords", [])), ensure_ascii=False),
            "emotion_valence": d.get("emotion_valence", 0.0),
            "emotion_intensity": d.get("emotion_intensity", 0.0),
            "emotion_tags": json.dumps(list(d.get("emotion_tags", [])), ensure_ascii=False),
            "connection_value": d.get("connection_value", 0.0),
            "importance": d.get("importance", 0.5),
            "created_at": d["created_at"],
            "updated_at": d["updated_at"],
            "last_accessed": d.get("last_accessed"),
            "access_count": d.get("access_count", 0),
            "successful_recall_count": d.get("successful_recall_count", 0),
            "associations": json.dumps(d.get("associations", {}), ensure_ascii=False),
            "review_status": d.get("review_status", "pending"),
            "review_note": d.get("review_note"),
            "reconstruction_count": d.get("reconstruction_count", 0),
            "parent_id": d.get("parent_id"),
            "metadata": json.dumps(d.get("metadata", {}), ensure_ascii=False),
        })

        # 更新倒排索引
        self._update_indexes(chunk)
        self.conn.commit()

    def get(self, chunk_id: str) -> Optional[MemoryChunk]:
        """按 ID 获取记忆碎片"""
        self._ensure_connection()

        t = self._table
        cursor = self.conn.execute(
            f"SELECT * FROM {t('memory_chunks')} WHERE id = ?",
            (chunk_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_chunk(row)

    def delete(self, chunk_id: str) -> bool:
        """删除记忆碎片"""
        self._ensure_connection()

        t = self._table
        # 先删除索引
        for idx_name in ['time_idx', 'topic_idx', 'location_idx', 'person_idx']:
            self.conn.execute(
                f"DELETE FROM {t(idx_name)} WHERE chunk_id = ?",
                (chunk_id,)
            )
        # 删除主记录
        cursor = self.conn.execute(
            f"DELETE FROM {t('memory_chunks')} WHERE id = ?",
            (chunk_id,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def get_all(self) -> Dict[str, MemoryChunk]:
        """获取所有记忆碎片"""
        self._ensure_connection()

        t = self._table
        cursor = self.conn.execute(f"SELECT * FROM {t('memory_chunks')}")
        result = {}
        for row in cursor:
            chunk = self._row_to_chunk(row)
            result[chunk.id] = chunk
        return result

    def count(self) -> int:
        """获取碎片总数"""
        self._ensure_connection()

        t = self._table
        cursor = self.conn.execute(f"SELECT COUNT(*) FROM {t('memory_chunks')}")
        return cursor.fetchone()[0]

    def save(self) -> None:
        """提交事务（SQLite 自动持久化）"""
        if self.conn:
            self.conn.commit()

    def load(self) -> bool:
        """确保表结构存在"""
        try:
            self._ensure_connection()
            self._ensure_schema()
            return True
        except sqlite3.Error as e:
            print(f"[SqliteMemoryStore] 加载失败: {e}")
            return False

    def close(self):
        """关闭连接"""
        if self.conn:
            self.conn.close()
            self.conn = None

    # ============ 索引操作 ============

    def _update_indexes(self, chunk: MemoryChunk):
        """更新倒排索引"""
        t = self._table
        cid = chunk.id

        # 先删除旧索引
        for idx_name in ['time_idx', 'topic_idx', 'location_idx', 'person_idx']:
            self.conn.execute(
                f"DELETE FROM {t(idx_name)} WHERE chunk_id = ?",
                (cid,)
            )

        # 时间索引
        if chunk.time_absolute:
            key = chunk.time_absolute[:7]  # 年-月
            self.conn.execute(
                f"INSERT OR IGNORE INTO {t('time_idx')} (key, chunk_id) VALUES (?, ?)",
                (key, cid)
            )

        # 主题索引
        for topic in chunk.topics:
            self.conn.execute(
                f"INSERT OR IGNORE INTO {t('topic_idx')} (key, chunk_id) VALUES (?, ?)",
                (topic, cid)
            )

        # 地点索引
        if chunk.location:
            self.conn.execute(
                f"INSERT OR IGNORE INTO {t('location_idx')} (key, chunk_id) VALUES (?, ?)",
                (chunk.location, cid)
            )

        # 人物索引
        for person in chunk.persons:
            self.conn.execute(
                f"INSERT OR IGNORE INTO {t('person_idx')} (key, chunk_id) VALUES (?, ?)",
                (person, cid)
            )

    def get_index(self, index_name: str, key: str) -> Set[str]:
        """获取某个索引 key 的所有 chunk_ids"""
        self._ensure_connection()

        t = self._table
        cursor = self.conn.execute(
            f"SELECT chunk_id FROM {t(index_name + '_idx')} WHERE key = ?",
            (key,)
        )
        return {row[0] for row in cursor}

    def rebuild_indexes(self):
        """重建所有倒排索引"""
        self._ensure_connection()

        t = self._table
        # 清空索引
        for idx_name in ['time_idx', 'topic_idx', 'location_idx', 'person_idx']:
            self.conn.execute(f"DELETE FROM {t(idx_name)}")

        # 遍历所有 chunk 重建索引
        cursor = self.conn.execute(f"SELECT * FROM {t('memory_chunks')}")
        for row in cursor:
            chunk = self._row_to_chunk(row)
            self._update_indexes(chunk)

        self.conn.commit()

    # ============ 统计操作 ============

    def get_stat(self, key: str) -> float:
        """获取统计值"""
        self._ensure_connection()

        t = self._table
        cursor = self.conn.execute(
            f"SELECT value FROM {t('stats')} WHERE key = ?",
            (key,)
        )
        row = cursor.fetchone()
        return row[0] if row else 0.0

    def set_stat(self, key: str, value: float):
        """设置统计值"""
        self._ensure_connection()

        t = self._table
        self.conn.execute(
            f"INSERT OR REPLACE INTO {t('stats')} (key, value) VALUES (?, ?)",
            (key, value)
        )
        self.conn.commit()

    # ============ 内部方法 ============

    def _row_to_chunk(self, row: sqlite3.Row) -> MemoryChunk:
        """将数据库行转换为 MemoryChunk"""
        d = dict(row)

        # 解析 JSON 字段
        d["tags"] = json.loads(d.get("tags") or "{}")
        d["persons"] = set(json.loads(d.get("persons") or "[]"))
        d["topics"] = set(json.loads(d.get("topics") or "[]"))
        d["keywords"] = set(json.loads(d.get("keywords") or "[]"))
        d["emotion_tags"] = set(json.loads(d.get("emotion_tags") or "[]"))
        d["associations"] = json.loads(d.get("associations") or "{}")
        d["metadata"] = json.loads(d.get("metadata") or "{}")

        # 转换枚举
        d["memory_type"] = MemoryType(d.get("memory_type", "interaction"))
        d["layer"] = MemoryLayer(d.get("layer", "core"))

        return MemoryChunk.from_dict(d)
