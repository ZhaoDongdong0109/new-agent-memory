"""
DataManager 用户数据隔离与审计回归测试

覆盖的缺陷：
1. _belongs_to_user 只读 metadata["user_id"]，而 main.add_memory 写入的是
   MemoryChunk 的一等字段 user_id —— 导致导出/删除跨用户泄露或失效。
2. delete_all_user_data 不写审计日志，且删除单个用户时会连带重置
   全局 persona / attention / cognitive 状态。
"""

from core.data_manager import DataManager
from memory_chunk import MemoryChunk, MemoryLayer
from new_agent_memory import HumanLikeMemorySystem


def make_system(tmp_path):
    """创建隔离在临时目录中的记忆系统（关闭 PII 与审计以简化测试）"""
    return HumanLikeMemorySystem(
        data_dir=str(tmp_path / "memory_data"),
        enable_pii_detection=False,
        enable_audit_log=False,
    )


def add_three_users(system):
    """为 alice / bob / default 各添加一条核心记忆，返回 chunk id 映射"""
    return {
        "alice": system.add_memory(
            content="alice 的私密记忆", topics=["secret"], user_id="alice"
        ),
        "bob": system.add_memory(
            content="bob 的私密记忆", topics=["secret"], user_id="bob"
        ),
        "default": system.add_memory(content="默认用户的记忆", topics=["public"]),
    }


class RecordingAuditLogger:
    """记录 log_data_deletion 调用的假审计器"""

    def __init__(self):
        self.deletions = []

    def log_data_deletion(self, user_id, deleted, reason="user_request"):
        self.deletions.append({"user_id": user_id, "deleted": deleted, "reason": reason})


# ============ Bug 1：用户归属判断（导出/删除隔离） ============


def test_export_returns_own_memories_only(tmp_path):
    system = make_system(tmp_path)
    add_three_users(system)
    manager = DataManager(system)

    alice_data = manager.export_user_data("alice")
    assert len(alice_data["core_memories"]) == 1
    assert alice_data["core_memories"][0]["user_id"] == "alice"

    bob_data = manager.export_user_data("bob")
    assert len(bob_data["core_memories"]) == 1
    assert bob_data["core_memories"][0]["user_id"] == "bob"


def test_export_default_does_not_leak_other_users(tmp_path):
    system = make_system(tmp_path)
    add_three_users(system)
    manager = DataManager(system)

    default_data = manager.export_user_data("default")
    assert len(default_data["core_memories"]) == 1
    assert default_data["core_memories"][0]["user_id"] == "default"


def test_export_covers_forgotten_layer(tmp_path):
    system = make_system(tmp_path)
    chunk = MemoryChunk(
        content="alice 被遗忘的记忆", user_id="alice", layer=MemoryLayer.FORGOTTEN
    )
    system.forgotten.archive(chunk)
    manager = DataManager(system)

    alice_data = manager.export_user_data("alice")
    assert len(alice_data["forgotten_memories"]) == 1
    assert manager.export_user_data("default")["forgotten_memories"] == []


def test_delete_removes_only_target_user(tmp_path):
    system = make_system(tmp_path)
    ids = add_three_users(system)
    manager = DataManager(system)

    deleted = manager.delete_all_user_data("alice")
    assert deleted["core_memories"] == 1

    remaining = system.core._store.get_all()
    assert ids["alice"] not in remaining
    assert ids["bob"] in remaining
    assert ids["default"] in remaining


def test_delete_default_does_not_delete_other_users(tmp_path):
    system = make_system(tmp_path)
    ids = add_three_users(system)
    manager = DataManager(system)

    deleted = manager.delete_all_user_data("default")
    assert deleted["core_memories"] == 1

    remaining = system.core._store.get_all()
    assert ids["default"] not in remaining
    assert ids["alice"] in remaining
    assert ids["bob"] in remaining


def test_get_data_stats_isolated_per_user(tmp_path):
    system = make_system(tmp_path)
    add_three_users(system)
    manager = DataManager(system)

    assert manager.get_data_stats("alice")["core_memories"] == 1
    assert manager.get_data_stats("default")["core_memories"] == 1


def test_belongs_to_user_metadata_fallback(tmp_path):
    """一等字段缺省时回退到 metadata["user_id"]（兼容旧数据）"""
    manager = DataManager(make_system(tmp_path))
    legacy_chunk = MemoryChunk(content="旧数据", metadata={"user_id": "carol"})

    assert manager._belongs_to_user(legacy_chunk, "carol") is True
    assert manager._belongs_to_user(legacy_chunk, "default") is False

    plain_chunk = MemoryChunk(content="无标记")
    assert manager._belongs_to_user(plain_chunk, "default") is True
    assert manager._belongs_to_user(plain_chunk, "carol") is False


# ============ Bug 2：审计日志与全局状态重置 ============


def test_delete_logs_data_deletion_audit(tmp_path):
    system = make_system(tmp_path)
    add_three_users(system)
    recorder = RecordingAuditLogger()
    system.audit_logger = recorder
    manager = DataManager(system)

    deleted = manager.delete_all_user_data("alice")

    assert len(recorder.deletions) == 1
    assert recorder.deletions[0]["user_id"] == "alice"
    assert recorder.deletions[0]["deleted"] == deleted


def test_delete_tolerates_missing_audit_logger(tmp_path):
    """memory 系统没有 audit_logger 属性时不应抛异常"""

    class BareMemorySystem:
        pass

    manager = DataManager(BareMemorySystem())
    deleted = manager.delete_all_user_data("alice")
    assert deleted["core_memories"] == 0


def test_global_state_preserved_when_other_users_remain(tmp_path):
    system = make_system(tmp_path)
    add_three_users(system)
    manager = DataManager(system)
    persona_before = system.persona
    attention_before = system.attention
    cognitive_before = system.cognitive_state

    deleted = manager.delete_all_user_data("alice")

    assert deleted["persona"] is False
    assert deleted["attention"] is False
    assert deleted["cognitive_state"] is False
    assert system.persona is persona_before
    assert system.attention is attention_before
    assert system.cognitive_state is cognitive_before


def test_global_state_reset_when_last_user_deleted(tmp_path):
    system = make_system(tmp_path)
    system.add_memory(content="alice 独占系统", user_id="alice")
    manager = DataManager(system)
    persona_before = system.persona

    deleted = manager.delete_all_user_data("alice")

    assert deleted["persona"] is True
    assert deleted["attention"] is True
    assert deleted["cognitive_state"] is True
    assert system.persona is not persona_before


def test_force_reset_global_flag(tmp_path):
    system = make_system(tmp_path)
    add_three_users(system)
    manager = DataManager(system)
    persona_before = system.persona

    deleted = manager.delete_all_user_data("alice", force_reset_global=True)

    assert deleted["persona"] is True
    assert system.persona is not persona_before
