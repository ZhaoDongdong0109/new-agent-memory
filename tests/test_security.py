"""
安全功能测试

测试 PII 检测、脱敏、数据删除、审计日志。
"""

import os
import sys
import tempfile

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
    assert handler.has_pii(text1)
    assert not handler.has_pii("今天天气很好")

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


def test_pii_overlapping_matches_preserve_adjacent_text():
    """回归测试：重叠匹配不应损坏相邻文本

    此前 redact/anonymize 基于原文偏移逐个替换重叠区间，
    后续替换使用失效偏移，导致相邻文本被删除。
    """
    print("=== 测试重叠 PII 匹配 ===")

    handler = PIIHandler()
    text = "身份证110101199001011234的用户"

    redacted = handler.redact(text)
    print(f"脱敏结果: {redacted}")
    assert redacted == "身份证[REDACTED]的用户"
    assert redacted.endswith("的用户")  # 相邻文本必须保留
    assert "110101199001011234" not in redacted

    anonymized = handler.anonymize_text(text)
    print(f"匿名化结果: {anonymized}")
    assert anonymized == "身份证1101**********1234的用户"
    assert anonymized.endswith("的用户")

    # 相邻（首尾相接）的 PII 也不应损坏文本
    text2 = "13812345678test@example.com"
    redacted2 = handler.redact(text2)
    print(f"相邻 PII 脱敏: {redacted2}")
    assert "13812345678" not in redacted2
    assert "test@example.com" not in redacted2

    # 手机号嵌入邮箱本地部分（部分重叠）
    text3 = "联系a13812345678@example.com结束"
    redacted3 = handler.redact(text3)
    print(f"部分重叠脱敏: {redacted3}")
    assert redacted3 == "联系[REDACTED]结束"
    anonymized3 = handler.anonymize_text(text3)
    print(f"部分重叠匿名化: {anonymized3}")
    assert anonymized3 == "联系a***@example.com结束"

    print("✅ 重叠 PII 匹配测试通过\n")


def test_pii_numeric_boundaries_no_false_positives():
    """回归测试：非 PII 数字串不应被误报

    此前无边界保护的数字模式会命中订单号、版本号、
    以及更长数字串内嵌的"手机号"片段。
    """
    print("=== 测试数字边界误报 ===")

    handler = PIIHandler()

    # 订单号（16 位但 Luhn 校验失败）不应被识别为银行卡
    text1 = "订单号 2026072612345678"
    print(f"订单号: {handler.redact(text1)}")
    assert handler.redact(text1) == text1
    assert not handler.has_pii(text1)

    # 版本号（超过 4 段的点分数字串）不应被识别为 IP
    text2 = "版本 10.2.3.4.5"
    print(f"版本号: {handler.redact(text2)}")
    assert handler.redact(text2) == text2
    assert not handler.has_pii(text2)

    # 八位组超出 0-255 的点分数字串不是 IP
    text3 = "代码999.999.999.999测试"
    assert handler.redact(text3) == text3

    # 嵌在更长数字串中的"手机号"片段不应被识别
    text4 = "工单99138123456780号"
    print(f"工单号: {handler.redact(text4)}")
    assert handler.redact(text4) == text4
    assert not handler.has_pii(text4)

    print("✅ 数字边界误报测试通过\n")


def test_pii_real_values_still_detected():
    """回归测试：加了边界与校验后，真实 PII 仍能被识别"""
    print("=== 测试真实 PII 仍被识别 ===")

    handler = PIIHandler()

    # Luhn 合法的银行卡号
    text1 = "卡号4111111111111111请脱敏"
    types1 = handler.get_pii_types(text1)
    print(f"银行卡: {types1}")
    assert "bank_card" in types1
    assert "4111111111111111" not in handler.redact(text1)

    # 真实手机号
    assert "phone" in handler.get_pii_types("联系电话13812345678")

    # 合法 IP 地址
    text2 = "服务器地址192.168.1.100"
    assert "ip_address" in handler.get_pii_types(text2)
    assert "192.168.1.100" not in handler.redact(text2)

    # 身份证号
    assert "id_card" in handler.get_pii_types("身份证110101199001011234")

    print("✅ 真实 PII 识别测试通过\n")


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
    test_pii_overlapping_matches_preserve_adjacent_text()
    test_pii_numeric_boundaries_no_false_positives()
    test_pii_real_values_still_detected()
    test_data_manager()
    test_audit_logger()
    test_benchmark_metrics()
    print("🎉 所有测试通过！")
