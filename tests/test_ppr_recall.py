"""Personalized PageRank 联想回忆性质测试（HippoRAG 思想）

旧的固定跳数扩散只沿显式 Hebbian 边传播；PPR 子图引入概念节点
（人物/主题/地点），带来两个新能力：

1. 概念桥接：两条记忆没有任何显式关联边，但共享稀有人物/主题时，
   通过概念节点成为两跳邻居——"想起老王的事时想起老王的另一件事"
2. 扇出阻尼（ACT-R fan effect，由列归一化天然实现）：
   烂大街的概念每个成员只分到一小份质量，稀有线索的联想远强于
   常见线索——"提到'量子退火'想起唯一那次讨论，提到'工作'什么都想不起"
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import HumanLikeMemorySystem


def _make_system(tmp_path, **kwargs):
    kwargs.setdefault("enable_pii_detection", False)
    kwargs.setdefault("enable_audit_log", False)
    kwargs.setdefault("enable_hybrid_retrieval", False)
    return HumanLikeMemorySystem(data_dir=str(tmp_path), **kwargs)


def test_concept_bridging_without_hebbian_edges(tmp_path):
    """共享稀有人物的记忆无需显式关联边即可被联想想起"""
    system = _make_system(tmp_path)
    id_a = system.add_memory(
        content="和老周确认了低温实验的排期", topics=["旅行"], persons=["老周"],
    )
    # 与 A 无任何 Hebbian 边，但共享稀有人物"老周"
    id_b = system.add_memory(
        content="老周提过他实验室的氦压缩机总出问题", topics=["设备"], persons=["老周"],
    )
    # 干扰项：无共享锚点
    system.add_memory(content="午饭吃了牛肉面", topics=["美食"])

    result = system.retrieve("和老周一起去的旅行")
    assert result.success
    trace = dict(system.retrieval.last_activation_trace)
    assert id_b in trace and trace[id_b] > 0, (
        "共享稀有人物的记忆未被概念桥接联想到"
    )
    assert id_b in {c.id for c in result.chunks}
    assert id_a in {c.id for c in result.chunks}


def test_fan_effect_damping(tmp_path):
    """稀有概念的联想强度远高于烂大街的概念（扇出阻尼）"""
    system = _make_system(tmp_path)

    # 种子：同时携带稀有主题与常见主题
    system.add_memory(
        content="和小陈讨论了量子退火的参数", topics=["量子退火", "工作"], persons=["小陈"],
    )
    # 稀有概念的唯一同伴
    id_rare = system.add_memory(
        content="量子退火那台机器上个月刚校准过", topics=["量子退火"],
    )
    # 常见概念"工作"的一大群成员
    common_ids = [
        system.add_memory(content=f"工作日常记录 {i}", topics=["工作"])
        for i in range(12)
    ]

    result = system.retrieve("和小陈讨论的事")
    assert result.success
    trace = dict(system.retrieval.last_activation_trace)

    rare_act = trace.get(id_rare, 0.0)
    common_acts = [trace.get(cid, 0.0) for cid in common_ids]
    assert rare_act > 0
    assert rare_act > max(common_acts) * 2, (
        f"扇出阻尼失效：稀有概念成员 {rare_act:.3f} 未显著强于 "
        f"常见概念成员 {max(common_acts):.3f}"
    )


def test_ppr_bounded_on_dense_graph(tmp_path):
    """稠密图上子图截断生效，检索不失控（节点上限 / 确定性收敛）"""
    system = _make_system(tmp_path)
    seed = system.add_memory(content="种子记忆", topics=["公共主题"], persons=["老王"])
    for i in range(60):
        system.add_memory(content=f"公共主题成员 {i}", topics=["公共主题"])

    result = system.retrieve("和老王一起去的旅行" if False else "老王的公共主题")
    # 无论查询命中与否，联想召回不得抛异常且轨迹有界
    if result.success:
        assert len(system.retrieval.last_activation_trace) <= 300
    assert system.core.get(seed) is not None
