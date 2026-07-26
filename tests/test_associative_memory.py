"""联想记忆测试：Hebbian 共激活 + 扩散激活 + 联想唤醒

验证三个类人回忆性质：
1. 一起被检索到的记忆会互相连线（fire together, wire together）
2. 检索命中 A 时，与 A 强关联的 B 会被联想想起（即使 B 不匹配查询）
3. 强关联可以把归档在伪遗忘层的记忆联想唤醒并提升回核心层

以及按类型分层的衰减半衰期（故事比交互细节持久）。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import HumanLikeMemorySystem
from memory_chunk import MemoryLayer
from core.weight_system import MemoryType, halflife_multiplier

DAY = 24 * 3600


def _make_system(tmp_path, **kwargs):
    kwargs.setdefault("enable_pii_detection", False)
    kwargs.setdefault("enable_audit_log", False)
    return HumanLikeMemorySystem(data_dir=str(tmp_path), **kwargs)


# ============ 共激活 ============

def test_coretrieved_memories_wire_together(tmp_path):
    """同一次检索命中的记忆之间应建立 Hebbian 关联"""
    system = _make_system(tmp_path)
    id_a = system.add_memory(content="上周和团队讨论了记忆系统的架构设计", keywords=["记忆系统"])
    id_b = system.add_memory(content="记忆系统的检索模块需要支持扩散激活", keywords=["记忆系统"])

    result = system.retrieve("记忆系统")
    assert result.success
    hit_ids = {c.id for c in result.chunks}
    assert {id_a, id_b} <= hit_ids

    chunk_a = system.core.get(id_a)
    assert chunk_a.associations.get(id_b, 0.0) > 0.0, "共同检索未建立关联"
    chunk_b = system.core.get(id_b)
    assert chunk_b.associations.get(id_a, 0.0) > 0.0, "关联应是双向的"


def test_repeated_coactivation_strengthens_edge(tmp_path):
    """重复共激活使关联边权单调增强（有上限）"""
    system = _make_system(tmp_path)
    id_a = system.add_memory(content="调试了向量检索的召回问题", keywords=["检索"])
    id_b = system.add_memory(content="检索模块的分数融合需要归一化", keywords=["检索"])

    weights = []
    for _ in range(3):
        system.retrieve("检索")
        weights.append(system.core.get(id_a).associations.get(id_b, 0.0))

    assert weights[0] < weights[1] < weights[2] <= 1.0


# ============ 扩散激活联想回忆 ============

def test_spreading_activation_recalls_associate(tmp_path):
    """查询只命中 A，但与 A 强关联的 B 应被联想想起"""
    system = _make_system(tmp_path, enable_hybrid_retrieval=False)
    id_a = system.add_memory(
        content="和老王聊了滑雪计划", topics=["旅行"], persons=["老王"],
    )
    id_b = system.add_memory(
        content="崇礼的雪季是十一月到三月", topics=["天气"],
    )
    # 建立强关联（模拟长期共同使用后的边权）
    system.core.strengthen_association(id_a, id_b, strength=0.8)

    # 查询只匹配 A 的主题/人物，B（topics=天气）不匹配
    result = system.retrieve("和老王一起去的旅行")
    assert result.success
    hit_ids = {c.id for c in result.chunks}
    assert id_a in hit_ids
    assert id_b in hit_ids, "强关联记忆未被扩散激活想起"

    # 审计轨迹可解释
    trace = dict(system.retrieval.last_activation_trace)
    assert id_b in trace and trace[id_b] > 0


def test_weak_association_is_not_recalled(tmp_path):
    """弱关联（低于激活阈值）不应被联想想起——联想是有选择性的"""
    system = _make_system(tmp_path, enable_hybrid_retrieval=False)
    id_a = system.add_memory(content="和老王聊了滑雪计划", topics=["旅行"], persons=["老王"])
    id_b = system.add_memory(content="昨天的午饭是牛肉面", topics=["美食"])
    system.core.strengthen_association(id_a, id_b, strength=0.1)  # 1跳激活 = 0.05 < 0.15

    result = system.retrieve("和老王一起去的旅行")
    assert result.success
    assert id_b not in {c.id for c in result.chunks}


# ============ 联想唤醒归档记忆 ============

def test_association_wakes_and_promotes_forgotten_memory(tmp_path):
    """强关联可以把伪遗忘层的记忆联想唤醒并提升回核心层"""
    system = _make_system(tmp_path, enable_hybrid_retrieval=False)
    id_a = system.add_memory(content="和老王聊了滑雪计划", topics=["旅行"], persons=["老王"])
    id_b = system.add_memory(content="三年前在崇礼摔断过雪板", topics=["天气"])
    system.core.strengthen_association(id_a, id_b, strength=0.8)

    # 手动把 B 归档（模拟它早已被遗忘）
    chunk_b = system.core.remove(id_b)
    system.forgotten.archive(chunk_b)
    assert system.core.get(id_b) is None

    # 检索命中 A：1 跳激活 = 1.0 * 0.8 * 0.5 = 0.4 >= 0.25 -> 唤醒并提升
    result = system.retrieve("和老王一起去的旅行")
    assert result.success
    assert id_b in {c.id for c in result.chunks}, "归档记忆未被联想唤醒"

    promoted = system.core.get(id_b)
    assert promoted is not None, "强激活的归档记忆应被提升回核心层"
    assert promoted.layer == MemoryLayer.CORE
    assert system.forgotten.get(id_b) is None
    assert system.retrieval.total_assoc_wakes >= 1


# ============ 按类型分层的衰减 ============

def test_halflife_multiplier_ordering():
    """半衰期倍率保持记忆科学排序：故事 > 想法 > 偏好 > 事实 > 交互"""
    m = halflife_multiplier
    assert m(MemoryType.STORY) > m(MemoryType.IDEA) > m(MemoryType.PREFERENCE) \
        > m(MemoryType.FACT) > m(MemoryType.INTERACTION) == 1.0
    # 字符串形式与未知类型的容错
    assert m("story") == m(MemoryType.STORY)
    assert m("unknown-type") == 1.0


def test_story_decays_slower_than_interaction(tmp_path):
    """同龄的故事记忆权重应高于交互记忆（程序性/叙事性记忆更持久）"""
    system = _make_system(tmp_path)
    id_story = system.add_memory(
        content="大学时和室友骑行了川藏线", memory_type=MemoryType.STORY,
    )
    id_chat = system.add_memory(
        content="随口问了一句今天天气", memory_type=MemoryType.INTERACTION,
    )
    for cid in (id_story, id_chat):
        chunk = system.core.get(cid)
        chunk.created_at -= 30 * DAY
        chunk.last_accessed -= 30 * DAY
        system.core._store.put(chunk)
        system.core._invalidate_weight(cid)

    w_story = system.core.calc_weight(system.core.get(id_story)).final
    w_chat = system.core.calc_weight(system.core.get(id_chat)).final
    assert w_story > w_chat, (
        f"30 天后 STORY({w_story:.3f}) 应比 INTERACTION({w_chat:.3f}) 持久"
    )
