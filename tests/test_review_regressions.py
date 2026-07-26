"""对抗审查发现的缺陷回归测试

覆盖第二轮对抗审查确认的问题：
1. [critical] 混合检索让遗忘唤醒路径饿死（核心命中时 try_wake 永不运行）
2. [high] 关联密度构成不可衰减的权重下限（共激活会让记忆永远无法遗忘）
3. [high] 元数据腿"有锚点但索引零命中"时全库泛滥进 RRF
4. [high] 删除用户后残留 id 占用其他用户的检索融合名额
5. [medium] 扩散激活重复传播（同一节点多路径激活后向外传播多次）
6. [medium] allow_forgotten=False 不再约束联想唤醒
7. [medium] 提升后结果列表持有过期对象（SQLite 上 layer 标记错误）
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import HumanLikeMemorySystem
from memory_chunk import MemoryLayer

DAY = 24 * 3600


def _make_system(tmp_path, **kwargs):
    kwargs.setdefault("enable_pii_detection", False)
    kwargs.setdefault("enable_audit_log", False)
    return HumanLikeMemorySystem(data_dir=str(tmp_path), **kwargs)


def _age_chunk(system, chunk_id, days):
    chunk = system.core.get(chunk_id)
    chunk.created_at -= days * DAY
    chunk.last_accessed -= days * DAY
    system.core._store.put(chunk)
    system.core._invalidate_weight(chunk_id)


# ============ 1. 唤醒不能被核心命中饿死 ============

def test_wake_reaches_through_core_hits(tmp_path):
    """核心层有命中时，强线索仍应唤醒归档记忆（both 路径）"""
    system = _make_system(tmp_path)

    # 一条会被查询命中的核心记忆
    system.add_memory(content="和老王讨论了下个月的旅行预算", persons=["老王"], topics=["旅行"])

    # 一条归档记忆，与查询线索强匹配
    archived_id = system.add_memory(
        content="五年前和老王一起去哈尔滨看冰雕", persons=["老王"], topics=["旅行"],
        importance=0.5,
    )
    chunk = system.core.remove(archived_id)
    system.forgotten.archive(chunk)

    result = system.retrieve("和老王一起去的旅行")
    assert result.success
    assert result.retrieval_path == "both", (
        f"核心命中时唤醒路径不可达（path={result.retrieval_path}）——生命周期饿死"
    )
    assert archived_id in [c.id for c in result.chunks]
    # 强锚点唤醒应已提升回核心层
    assert system.core.get(archived_id) is not None
    assert system.forgotten.get(archived_id) is None


# ============ 2. 关联不能构成永久权重下限 ============

def test_heavily_associated_memory_can_still_be_forgotten(tmp_path):
    """积累了大量 Hebbian 关联的记忆，长期不用仍必须能降级"""
    system = _make_system(tmp_path)
    chunk_id = system.add_memory(content="一条被频繁共同检索过的记忆", importance=0.5)

    chunk = system.core.get(chunk_id)
    # 模拟长期共激活积累的关联（10 条强边）
    for i in range(10):
        chunk.associations[f"fake_{i}"] = 1.0
    system.core._store.put(chunk)

    _age_chunk(system, chunk_id, 365)
    weight = system.core.calc_weight(system.core.get(chunk_id)).final
    assert weight < system.core.degrade_threshold, (
        f"一年未用、仅靠关联密度支撑的记忆权重 {weight:.3f} 仍高于降级阈值——"
        "关联构成了永久权重下限"
    )


# ============ 3. 元数据腿：有锚点但零命中时必须返回空 ============

def test_metadata_leg_no_flood_on_anchor_miss(tmp_path):
    """查询带主题锚点但索引零命中时，元数据腿不得把全库当命中"""
    system = _make_system(tmp_path)
    for i in range(5):
        system.add_memory(content=f"无关记忆 {i}", topics=["工作"])

    leg = system.query_planner._metadata_filter({"topics": {"不存在的主题"}})
    assert leg == [], "锚点未命中时元数据腿泛滥：全库被灌入 RRF 最大权重腿"


# ============ 4. 用户删除必须同步清理混合索引 ============

def test_user_deletion_purges_hybrid_index(tmp_path):
    """删除 alice 后：她的内容不可检索，bob 的检索不受残留 id 干扰"""
    from core.data_manager import DataManager

    system = _make_system(tmp_path)
    for i in range(6):
        system.add_memory(content=f"alice 的私有笔记 {i} 关于量子计算", user_id="alice")
    bob_id = system.add_memory(content="bob 研究的是量子计算的纠错码", user_id="bob")

    dm = DataManager(system)
    deleted = dm.delete_all_user_data("alice")
    assert deleted["core_memories"] == 6

    result = system.retrieve("量子计算")
    assert result.success, "残留 id 占用融合名额，把 bob 的真实结果挤掉了"
    ids = [c.id for c in result.chunks]
    assert bob_id in ids
    assert all(system.core.get(cid) is not None for cid in ids)


# ============ 5. 扩散激活不得重复传播 ============

def test_no_double_propagation(tmp_path):
    """同一节点被多路径激活后，只向外传播一次（激活值可精确预测）"""
    system = _make_system(tmp_path, enable_hybrid_retrieval=False)
    id_a1 = system.add_memory(content="种子甲", topics=["旅行"], persons=["老王"])
    id_a2 = system.add_memory(content="种子乙", topics=["旅行"], persons=["老王"])
    id_b = system.add_memory(content="中间节点", topics=["天气"])
    id_c = system.add_memory(content="末端节点", topics=["天气"])

    # 两条路径汇入 B，B 再连 C
    system.core.strengthen_association(id_a1, id_b, strength=0.5)
    system.core.strengthen_association(id_a2, id_b, strength=0.5)
    system.core.strengthen_association(id_b, id_c, strength=0.9)
    # strengthen 是饱和递增的，读出实际边权用于精确断言
    w_a1b = system.core.get(id_a1).associations[id_b]
    w_bc = system.core.get(id_b).associations[id_c]

    result = system.retrieve("和老王一起去的旅行")
    assert result.success
    trace = dict(system.retrieval.last_activation_trace)

    # B 的激活 = 两路贡献之和；C = B 单次传播
    expected_b = min(1.0, w_a1b * 0.5 * 2)
    expected_c = expected_b * w_bc * 0.5
    assert trace[id_b] == pytest.approx(expected_b, abs=1e-6)
    assert trace.get(id_c, 0.0) == pytest.approx(expected_c, abs=1e-6), (
        "C 的激活高于单次传播的预测值——B 重复传播了"
    )


# ============ 6. allow_forgotten=False 约束联想唤醒 ============

def test_allow_forgotten_false_blocks_associative_wake(tmp_path):
    """反馈路径（allow_forgotten=False）不得触碰伪遗忘层"""
    system = _make_system(tmp_path, enable_hybrid_retrieval=False)
    id_a = system.add_memory(content="核心记忆", topics=["旅行"], persons=["老王"])
    id_b = system.add_memory(content="已归档的关联记忆", topics=["天气"])
    system.core.strengthen_association(id_a, id_b, strength=0.8)
    chunk_b = system.core.remove(id_b)
    system.forgotten.archive(chunk_b)

    result = system.retrieve("和老王一起去的旅行", allow_forgotten=False)
    assert result.success
    assert id_b not in [c.id for c in result.chunks]
    assert system.forgotten.get(id_b) is not None, "allow_forgotten=False 时归档记忆被动了"
    assert system.core.get(id_b) is None


# ============ 7. SQLite 上提升后的对象必须是新鲜的 ============

def test_promoted_chunks_fresh_on_sqlite(tmp_path):
    """SQLite 后端：结果里被提升的 chunk 的 layer 必须已是 CORE"""
    system = _make_system(tmp_path, store_backend="sqlite")
    archived_id = system.add_memory(
        content="和老王一起去哈尔滨看冰雕", persons=["老王"], topics=["旅行"],
    )
    chunk = system.core.remove(archived_id)
    system.forgotten.archive(chunk)

    result = system.retrieve("和老王一起去的旅行")
    assert result.success
    woken = [c for c in result.chunks if c.id == archived_id]
    assert woken, "归档记忆未被唤醒"
    assert woken[0].layer == MemoryLayer.CORE, (
        "结果里的对象是提升前的过期副本（SQLite 返回副本语义）"
    )
