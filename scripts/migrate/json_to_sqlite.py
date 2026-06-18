#!/usr/bin/env python3
"""
JSON → SQLite 迁移脚本

将现有的 core.json 和 forgotten.json 迁移到 SQLite 数据库。

使用方式：
    python scripts/migrate/json_to_sqlite.py [--data-dir ./memory_data]

迁移内容：
    1. core.json → memory.db (core_memory_chunks 表)
    2. forgotten.json → memory.db (forgotten_memory_chunks 表)
    3. 验证数据一致性
    4. 可选：备份原 JSON 文件
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# 添加项目根目录到 path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from memory_chunk import MemoryChunk
from core.json_store import JsonMemoryStore
from core.sqlite_store import SqliteMemoryStore


def backup_json_files(data_dir: str) -> str:
    """备份 JSON 文件"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(data_dir, f"backup_{timestamp}")
    os.makedirs(backup_dir, exist_ok=True)

    for filename in ["core.json", "forgotten.json"]:
        src = os.path.join(data_dir, filename)
        if os.path.exists(src):
            dst = os.path.join(backup_dir, filename)
            shutil.copy2(src, dst)
            print(f"  备份: {filename} → {backup_dir}/{filename}")

    return backup_dir


def migrate_layer(
    json_store: JsonMemoryStore,
    sqlite_store: SqliteMemoryStore,
    layer_name: str,
) -> int:
    """
    迁移一个层的数据

    Args:
        json_store: JSON 存储后端
        sqlite_store: SQLite 存储后端
        layer_name: 层名称（用于日志）

    Returns:
        迁移的记录数
    """
    print(f"\n迁移 {layer_name}...")

    # 从 JSON 加载
    if not json_store.load():
        print(f"  {layer_name}: JSON 文件不存在或为空，跳过")
        return 0

    chunks = json_store.get_all()
    print(f"  {layer_name}: 从 JSON 加载了 {len(chunks)} 条记忆")

    # 写入 SQLite
    count = 0
    for chunk_id, chunk in chunks.items():
        sqlite_store.put(chunk)
        count += 1

    # 验证
    sqlite_count = sqlite_store.count()
    if sqlite_count != count:
        print(f"  ⚠️ {layer_name}: 迁移后记录数不一致！JSON={count}, SQLite={sqlite_count}")
        return 0

    print(f"  ✅ {layer_name}: 成功迁移 {count} 条记忆")
    return count


def verify_migration(
    json_store: JsonMemoryStore,
    sqlite_store: SqliteMemoryStore,
    layer_name: str,
) -> bool:
    """
    验证迁移结果

    Args:
        json_store: JSON 存储后端
        sqlite_store: SQLite 存储后端
        layer_name: 层名称

    Returns:
        是否验证通过
    """
    print(f"\n验证 {layer_name}...")

    json_chunks = json_store.get_all()
    sqlite_chunks = sqlite_store.get_all()

    # 检查数量
    if len(json_chunks) != len(sqlite_chunks):
        print(f"  ❌ {layer_name}: 记录数不一致 JSON={len(json_chunks)} SQLite={len(sqlite_chunks)}")
        return False

    # 检查每条记录
    mismatches = 0
    for chunk_id, json_chunk in json_chunks.items():
        sqlite_chunk = sqlite_chunks.get(chunk_id)
        if sqlite_chunk is None:
            print(f"  ❌ {layer_name}: SQLite 中缺少记录 {chunk_id}")
            mismatches += 1
            continue

        # 比较关键字段
        json_dict = json_chunk.to_dict()
        sqlite_dict = sqlite_chunk.to_dict()

        for key in ["content", "summary", "memory_type", "importance", "layer"]:
            if json_dict.get(key) != sqlite_dict.get(key):
                print(f"  ❌ {layer_name}: 记录 {chunk_id} 字段 {key} 不一致")
                print(f"     JSON:   {json_dict.get(key)}")
                print(f"     SQLite: {sqlite_dict.get(key)}")
                mismatches += 1

    if mismatches > 0:
        print(f"  ❌ {layer_name}: 发现 {mismatches} 处不一致")
        return False

    print(f"  ✅ {layer_name}: 验证通过，{len(json_chunks)} 条记录完全一致")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="将 JSON 记忆数据迁移到 SQLite"
    )
    parser.add_argument(
        "--data-dir",
        default="./memory_data",
        help="数据目录路径（默认: ./memory_data）"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="不备份 JSON 文件"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="只验证，不执行迁移"
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    print(f"数据目录: {data_dir}")

    # 检查 JSON 文件是否存在
    core_json = os.path.join(data_dir, "core.json")
    forgotten_json = os.path.join(data_dir, "forgotten.json")

    if not os.path.exists(core_json) and not os.path.exists(forgotten_json):
        print("没有找到 JSON 文件，无需迁移")
        return 0

    # 备份
    if not args.no_backup and not args.verify_only:
        print("\n备份 JSON 文件...")
        backup_dir = backup_json_files(data_dir)
        print(f"备份完成: {backup_dir}")

    # 创建存储实例
    json_core = JsonMemoryStore(core_json)
    json_forgotten = JsonMemoryStore(forgotten_json)

    db_path = os.path.join(data_dir, "memory.db")
    sqlite_core = SqliteMemoryStore(db_path, table_prefix="core_")
    sqlite_forgotten = SqliteMemoryStore(db_path, table_prefix="forgotten_")

    # 确保 SQLite 表结构存在
    sqlite_core.load()
    sqlite_forgotten.load()

    if args.verify_only:
        # 只验证
        print("\n=== 验证模式 ===")
        json_core.load()
        json_forgotten.load()

        core_ok = verify_migration(json_core, sqlite_core, "核心层")
        forgotten_ok = verify_migration(json_forgotten, sqlite_forgotten, "伪遗忘层")

        if core_ok and forgotten_ok:
            print("\n✅ 验证通过！")
            return 0
        else:
            print("\n❌ 验证失败！")
            return 1
    else:
        # 执行迁移
        print("\n=== 开始迁移 ===")

        core_count = migrate_layer(json_core, sqlite_core, "核心层")
        forgotten_count = migrate_layer(json_forgotten, sqlite_forgotten, "伪遗忘层")

        print(f"\n迁移完成！")
        print(f"  核心层: {core_count} 条")
        print(f"  伪遗忘层: {forgotten_count} 条")
        print(f"  数据库: {db_path}")

        # 自动验证
        print("\n=== 自动验证 ===")
        json_core.load()
        json_forgotten.load()

        core_ok = verify_migration(json_core, sqlite_core, "核心层")
        forgotten_ok = verify_migration(json_forgotten, sqlite_forgotten, "伪遗忘层")

        if core_ok and forgotten_ok:
            print("\n✅ 迁移并验证成功！")
            print(f"\n下一步：")
            print(f"  1. 测试 SQLite 后端: python -c \"from main import HumanLikeMemorySystem; s = HumanLikeMemorySystem(store_backend='sqlite'); s.load(); print('OK')\"")
            print(f"  2. 确认无误后，可删除 JSON 备份: rm -rf {backup_dir}")
            return 0
        else:
            print("\n❌ 迁移验证失败！请检查数据")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
