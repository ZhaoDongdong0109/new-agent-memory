"""
安全功能测试

测试 PII 检测、脱敏、数据删除、审计日志。
"""

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, '.')

from core.pii_handler import PIIHandler
from core.data_manager import DataManager
from core.audit_logger import AuditLogger


def test_pii_handler():
    """测试 PII 处理器"""
    print("=== 测试 PII 处理器 ===")

    handler = PIIHandler()

    # 测试 PII 检测
    text1 = "我的手机号是13812345678，邮箱是test@example.com"
    pii_list = handler.detect(text1)
    print(f"检测 PII: {text1}")
    print(f"  结果: {pii_list}")
    assert len(pii_list) >= 2
    assert any(pii["type"] == "phone" for pii in pii_list)
    assert any(pii["type"] == "email" for pii in pii_list)

    # 测试脱敏
    redacted = handler.redact(text1)
    print(f"脱敏结果: {redacted}")
    assert "13812345678" not in redacted
    assert "test@example.com" not in redacted
    assert "[REDACTED]" in redacted

    # 测试匿名化
    anonymized = handler.anonymize_text(text1)
    print(f"匿名化结果: {anonymized}")
    assert "138****5678" in anonymized
    assert "t***@example.com" in anonymized

    # 测试 has_pii
    assert handler.has_pii(text1) == True
    assert handler.has_pii("今天天气很好") == False

    # 测试 get_pii_types
    pii_types = handler.get_pii_types(text1)
    print(f"PII 类型: {pii_types}")
    assert "phone" in pii_types
    assert "email" in pii_types

    # 测试身份证号
    text2 = "我的身份证号是110101199001011234"
    pii_list = handler.detect(text2)
    print(f"身份证检测: {text2}")
    print(f"  结果: {pii_list}")
    assert len(pii_list) >= 1
    assert any(pii["type"] == "id_card" for pii in pii_list)

    print("✅ PII 处理器测试通过\n")


def test_data_manager():
    """测试数据管理器"""
    print("=== 测试数据管理器 ===")

    # 创建模拟的记忆系统
    class MockChunk:
        def __init__(self, chunk_id, content, user_id="default"):
            self.id = chunk_id
            self.content = content
            self.metadata = {"user_id": user_id}

        def to_dict(self):
            return {"id": self.id, "content": self.content, "metadata": self.metadata}

    class MockStore:
        def __init__(self):
            self._data = {}

        def get_all(self):
            return self._data

        def delete(self, chunk_id):
            if chunk_id in self._data:
                del self._data[chunk_id]
                return True
            return False

        def put(self, chunk):
            self._data[chunk.id] = chunk

    class MockCore:
        def __init__(self):
            self._store = MockStore()

        def remove(self, chunk_id):
            return self._store.delete(chunk_id)

    class MockForgotten:
        def __init__(self):
            self._store = MockStore()

        def remove(self, chunk_id):
            return self._store.delete(chunk_id)

    class MockPersona:
        def export_profile(self):
            return {"user_id": "default"}

    class MockAttention:
        def to_dict(self):
            return {"goals": []}

    class MockCognitiveState:
        def to_dict(self):
            return {"drives": {}}

    class MockMemorySystem:
        def __init__(self):
            self.core = MockCore()
            self.forgotten = MockForgotten()
            self.persona = MockPersona()
            self.attention = MockAttention()
            self.cognitive_state = MockCognitiveState()

        def save(self):
            pass

    # 创建测试数据
    mock_memory = MockMemorySystem()
    mock_memory.core._store._data = {
        "mem_001": MockChunk("mem_001", "记忆1", "default"),
        "mem_002": MockChunk("mem_002", "记忆2", "default"),
        "mem_003": MockChunk("mem_003", "记忆3", "user1"),
    }
    mock_memory.forgotten._store._data = {
        "mem_004": MockChunk("mem_004", "记忆4", "default"),
    }

    # 创建数据管理器
    manager = DataManager(mock_memory)

    # 测试数据统计
    stats = manager.get_data_stats("default")
    print(f"数据统计: {stats}")
    assert stats["core_memories"] == 2
    assert stats["forgotten_memories"] == 1

    # 测试导出
    data = manager.export_user_data("default")
    print(f"导出数据: core={len(data['core_memories'])}, forgotten={len(data['forgotten_memories'])}")
    assert len(data["core_memories"]) == 2
    assert len(data["forgotten_memories"]) == 1

    # 测试删除
    deleted = manager.delete_all_user_data("default")
    print(f"删除结果: {deleted}")
    assert deleted["core_memories"] == 2
    assert deleted["forgotten_memories"] == 1

    # 验证删除后的统计
    stats = manager.get_data_stats("default")
    print(f"删除后统计: {stats}")
    assert stats["core_memories"] == 0
    assert stats["forgotten_memories"] == 0

    print("✅ 数据管理器测试通过\n")


def test_audit_logger():
    """测试审计日志"""
    print("=== 测试审计日志 ===")

    with tempfile.TemporaryDirectory() as tmp_dir:
        log_file = os.path.join(tmp_dir, "audit.log")

        # 创建审计日志器
        logger = AuditLogger(log_file=log_file)

        # 记录记忆访问
        logger.log_memory_access(
            chunk_id="mem_001",
            user_id="user1",
            action="create",
            details={"content_length": 100},
        )

        # 记录数据删除
        logger.log_data_deletion(
            user_id="user1",
            deleted={"core_memories": 5, "forgotten_memories": 2},
            reason="user_request",
        )

        # 记录 PII 检测
        logger.log_pii_detection(
            text_hash="abc123",
            pii_types=["phone", "email"],
            action="redact",
        )

        # 记录安全事件
        logger.log_security_event(
            event_type="unauthorized_access",
            details={"ip": "192.168.1.1", "path": "/api/data"},
            severity="warning",
        )

        # 验证日志文件
        assert os.path.exists(log_file)

        # 读取日志
        events = logger.get_recent_events(limit=10)
        print(f"日志事件数: {len(events)}")
        assert len(events) == 4

        # 检查事件类型
        event_types = [e.get("event_type") for e in events]
        print(f"事件类型: {event_types}")
        assert "memory_access" in event_types
        assert "data_deletion" in event_types
        assert "pii_detection" in event_types
        assert "security_unauthorized_access" in event_types

        # 按类型过滤
        pii_events = logger.get_recent_events(event_type="pii_detection")
        print(f"PII 事件数: {len(pii_events)}")
        assert len(pii_events) == 1

    print("✅ 审计日志测试通过\n")


def test_benchmark_metrics():
    """测试 benchmark 指标"""
    print("=== 测试 benchmark 指标 ===")

    from scripts.bench.metrics import (
        recall_at_k,
        precision_at_k,
        mrr,
        ndcg_at_k,
        f1_at_k,
        mean_latency,
        p50_latency,
        p95_latency,
        p99_latency,
    )

    # 测试 Recall@K
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    expected = {"doc1", "doc3", "doc5"}

    r5 = recall_at_k(retrieved, expected, 5)
    print(f"Recall@5: {r5}")
    assert r5 == 1.0

    r3 = recall_at_k(retrieved, expected, 3)
    print(f"Recall@3: {r3}")
    assert r3 == 2 / 3

    # 测试 Precision@K
    p5 = precision_at_k(retrieved, expected, 5)
    print(f"Precision@5: {p5}")
    assert p5 == 3 / 5

    # 测试 MRR
    m = mrr(retrieved, expected)
    print(f"MRR: {m}")
    assert m == 1.0

    # 测试 NDCG@K
    n5 = ndcg_at_k(retrieved, expected, 5)
    print(f"NDCG@5: {n5}")
    assert n5 > 0

    # 测试 F1@K
    f5 = f1_at_k(retrieved, expected, 5)
    print(f"F1@5: {f5}")
    assert f5 > 0

    # 测试延迟统计
    latencies = [0.01, 0.02, 0.03, 0.04, 0.05]
    print(f"Mean: {mean_latency(latencies):.3f}")
    print(f"P50: {p50_latency(latencies):.3f}")
    print(f"P95: {p95_latency(latencies):.3f}")
    print(f"P99: {p99_latency(latencies):.3f}")

    print("✅ Benchmark 指标测试通过\n")


if __name__ == "__main__":
    test_pii_handler()
    test_data_manager()
    test_audit_logger()
    test_benchmark_metrics()
    print("🎉 所有测试通过！")
