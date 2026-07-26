"""双时态事实取代 + 时间感知检索 + 可审计弃答测试

验收标准来自 2025-2026 调研：知识更新是所有已发表记忆系统最弱的
能力（MemoryAgentBench 上 SOTA 接近随机）。本套测试即
knowledge-update probe 的确定性版本：

1. 版本化事实链（里斯本 -> 柏林 -> 奥斯陆）：现在时查询只返回当前值，
   过去时查询沿取代链带回历史值
2. 写入决策表：NOOP（近重复强化）/ UPDATE（扩展合并）/
   SUPERSEDE（换值取代）/ ADD（新事实 + 惊奇度缩放）
3. 时间窗口：显式时间表达过滤候选；窗口外查询可审计弃答
4. 惊奇度门控：越出乎意料的信息编码时重要性越高
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import HumanLikeMemorySystem
from core.time_parser import parse_query_window, query_tense, chunk_time_range
from core.weight_system import MemoryType


def _make_system(tmp_path, **kwargs):
    kwargs.setdefault("enable_pii_detection", False)
    kwargs.setdefault("enable_audit_log", False)
    return HumanLikeMemorySystem(data_dir=str(tmp_path), **kwargs)


def _add_fact(system, content, **kwargs):
    kwargs.setdefault("memory_type", MemoryType.FACT)
    return system.add_memory(content=content, **kwargs)


# ============ 事实取代链（旗舰场景）============

def test_supersession_chain_lisbon_berlin_oslo(tmp_path):
    """版本化事实链：现在时查现值，过去时回溯历史"""
    system = _make_system(tmp_path)

    id_v1 = _add_fact(system, "小李现在住在里斯本",
                      persons=["小李"], topics=["居住"], keywords=["住", "里斯本"])
    id_v2 = _add_fact(system, "小李现在住在柏林",
                      persons=["小李"], topics=["居住"], keywords=["住", "柏林"])
    id_v3 = _add_fact(system, "小李现在住在奥斯陆",
                      persons=["小李"], topics=["居住"], keywords=["住", "奥斯陆"])

    # v1、v2 已被取代：失效并归档，链条完整
    assert system.core.get(id_v1) is None
    assert system.core.get(id_v2) is None
    old_v1 = system.forgotten.get(id_v1)
    old_v2 = system.forgotten.get(id_v2)
    assert old_v1 is not None and old_v1.invalid_at is not None
    assert old_v2 is not None and old_v2.invalid_at is not None
    assert old_v1.metadata["superseded_by"] == id_v2
    assert old_v2.metadata["superseded_by"] == id_v3

    current = system.core.get(id_v3)
    assert current is not None
    assert current.parent_id == id_v2
    assert system.forgotten.get(id_v2).parent_id == id_v1

    # 现在时查询：只返回当前值
    result = system.retrieve("小李现在住在哪")
    assert result.success
    ids = [c.id for c in result.chunks]
    assert id_v3 in ids
    assert id_v1 not in ids and id_v2 not in ids, "被取代的旧值冒充了现状"

    # 过去时查询：沿取代链带回历史值
    past = system.retrieve("小李以前住在哪")
    assert past.success
    past_ids = [c.id for c in past.chunks]
    assert id_v2 in past_ids, "取代链未追溯到柏林"
    assert id_v1 in past_ids, "取代链未追溯到里斯本"


def test_noop_reinforces_duplicate(tmp_path):
    """近重复写入不新增，而是强化既有记忆"""
    system = _make_system(tmp_path)
    id_first = _add_fact(system, "小李喜欢喝手冲咖啡",
                         persons=["小李"], topics=["偏好"], keywords=["咖啡"],
                         memory_type=MemoryType.PREFERENCE)
    id_second = _add_fact(system, "小李喜欢喝手冲咖啡",
                          persons=["小李"], topics=["偏好"], keywords=["咖啡"],
                          memory_type=MemoryType.PREFERENCE)

    assert id_second == id_first, "近重复应命中 NOOP 返回既有 id"
    assert len(system.core) == 1
    reinforced = system.core.get(id_first)
    assert reinforced.successful_recall_count >= 1


def test_update_merges_extension(tmp_path):
    """新内容完整覆盖旧内容并扩展 -> 就地更新，旧内容进 history"""
    system = _make_system(tmp_path)
    id_v1 = _add_fact(system, "小李喜欢咖啡",
                      persons=["小李"], topics=["偏好"], keywords=["咖啡"],
                      memory_type=MemoryType.PREFERENCE)
    id_v2 = _add_fact(system, "小李喜欢咖啡，尤其是埃塞俄比亚的浅烘手冲咖啡",
                      persons=["小李"], topics=["偏好"], keywords=["咖啡", "手冲"],
                      memory_type=MemoryType.PREFERENCE)

    assert id_v2 == id_v1, "扩展应命中 UPDATE 返回既有 id"
    updated = system.core.get(id_v1)
    assert "埃塞俄比亚" in updated.content
    assert updated.version == 2
    history = updated.metadata.get("history", [])
    assert history and "小李喜欢咖啡" in history[0]["content"]


def test_unrelated_fact_is_added(tmp_path):
    """不同事实槽正常 ADD，互不干扰"""
    system = _make_system(tmp_path)
    id_a = _add_fact(system, "小李住在柏林",
                     persons=["小李"], topics=["居住"], keywords=["住", "柏林"])
    id_b = _add_fact(system, "老王养了一只橘猫",
                     persons=["老王"], topics=["宠物"], keywords=["猫"])
    assert id_a != id_b
    assert system.core.get(id_a) is not None
    assert system.core.get(id_b) is not None


def test_interaction_memories_not_managed(tmp_path):
    """INTERACTION 类型不参与取代（事件记录没有"当前值"语义）"""
    system = _make_system(tmp_path)
    id_a = system.add_memory(content="今天和小李聊了搬家的事",
                             persons=["小李"], topics=["居住"])
    id_b = system.add_memory(content="今天又和小李聊了搬家的事",
                             persons=["小李"], topics=["居住"])
    assert id_a != id_b
    assert len(system.core) == 2


# ============ 惊奇度门控编码 ============

def test_surprising_fact_encoded_stronger(tmp_path):
    """同底价重要性下，意外信息比熟悉信息编码更强"""
    system = _make_system(tmp_path)
    _add_fact(system, "小李在银行工作，是一名风控分析师",
              persons=["小李"], topics=["工作"], keywords=["银行", "风控"],
              importance=0.5)
    # 相关但内容有重叠的事实（低惊奇）
    id_low = _add_fact(system, "小李在银行工作，最近在做风控模型",
                       persons=["小李"], topics=["工作"], keywords=["银行", "风控", "模型"],
                       importance=0.5)
    # 完全无关的新事实（高惊奇）
    id_high = _add_fact(system, "老王周末在学习驯鹰",
                        persons=["老王"], topics=["爱好"], keywords=["驯鹰"],
                        importance=0.5)

    high_chunk = system.core.get(id_high)
    assert high_chunk.metadata.get("encoding_surprise", 0) > 0.9
    assert high_chunk.importance > 0.5, "高惊奇事实应被增强编码"

    low_chunk = system.core.get(id_low)
    if low_chunk is not None and "encoding_surprise" in low_chunk.metadata:
        assert low_chunk.importance <= high_chunk.importance


# ============ 时间解析 ============

def test_parse_query_window_frozen_clock():
    """冻结时钟下时间表达解析完全确定"""
    frozen = time.mktime((2026, 7, 26, 12, 0, 0, 0, 0, -1))

    def now_fn():
        return frozen

    w = parse_query_window("昨天吃了什么", now_fn=now_fn)
    assert time.localtime(w[0]).tm_mday == 25

    w = parse_query_window("2024年3月的出差", now_fn=now_fn)
    assert time.localtime(w[0]).tm_year == 2024
    assert time.localtime(w[0]).tm_mon == 3

    w = parse_query_window("3年前的旅行", now_fn=now_fn)
    assert time.localtime(w[0]).tm_year == 2023

    w = parse_query_window("what happened last year", now_fn=now_fn)
    assert time.localtime(w[0]).tm_year == 2025

    assert parse_query_window("没有时间表达的查询", now_fn=now_fn) is None


def test_query_tense_detection():
    assert query_tense("小李现在住在哪") == "present"
    assert query_tense("小李以前住在哪") == "past"
    assert query_tense("where did he use to live") == "past"
    assert query_tense("普通查询") is None


def test_chunk_time_range_formats(tmp_path):
    system = _make_system(tmp_path)
    cid = system.add_memory(content="出差记录", time_absolute="2024-03-15")
    start, end = chunk_time_range(system.core.get(cid))
    assert time.localtime(start).tm_year == 2024
    assert time.localtime(start).tm_mon == 3
    assert time.localtime(start).tm_mday == 15
    assert end > start


# ============ 时间窗口检索与可审计弃答 ============

def test_time_window_filters_results(tmp_path):
    """显式时间表达过滤掉窗口外的记忆"""
    system = _make_system(tmp_path)
    id_2024 = system.add_memory(
        content="2024年3月在上海出差见了客户",
        time_absolute="2024-03-10", location="上海", topics=["出差"],
    )
    system.add_memory(
        content="2026年6月在上海出差参加发布会",
        time_absolute="2026-06-20", location="上海", topics=["出差"],
    )

    result = system.retrieve("2024年3月在上海出差做了什么")
    assert result.success
    ids = [c.id for c in result.chunks]
    assert ids == [id_2024], f"时间窗口未过滤：{ids}"


def test_abstain_when_window_empty(tmp_path):
    """窗口内没有记忆时弃答，并给出最接近的记忆时间"""
    system = _make_system(tmp_path)
    system.add_memory(
        content="2026年6月在上海出差参加发布会",
        time_absolute="2026-06-20", location="上海", topics=["出差"],
    )

    result = system.retrieve("2020年在上海出差做了什么")
    assert not result.success, "窗口外的内容冒充了答案"
    assert "2026-06" in result.review_note, (
        f"弃答理由应给出最接近记忆的时间：{result.review_note}"
    )
