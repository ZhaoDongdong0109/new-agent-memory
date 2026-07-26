"""遗忘-唤醒生命周期集成测试

验证系统的招牌能力真实存在：
    添加 -> 老化 -> maintain() 降级 -> 线索唤醒 -> 提升回核心层 -> 再次可检索

以及支撑这个闭环的各项核心修复：
- 权重下限低于降级阈值（静态因子被时间衰减门控）
- 权重平局时 heap 检索不崩溃
- adjust_after_recall 的反馈真实生效（recall_bias）
- SQLite 后端访问统计持久化
- 查询/写入主题词汇表打通（中英文互通）
- 倒排索引键快照（原地修改标签不会留下脏索引）
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import HumanLikeMemorySystem
from memory_chunk import MemoryChunk, MemoryLayer
from memory_layer_core import MemoryLayerCore
from core.json_store import JsonMemoryStore

DAY = 24 * 3600


def _make_system(tmp_path, **kwargs):
    kwargs.setdefault("enable_pii_detection", False)
    kwargs.setdefault("enable_audit_log", False)
    return HumanLikeMemorySystem(data_dir=str(tmp_path), **kwargs)


def _age_chunk(system, chunk_id, days):
    """把一条记忆的时间戳整体推回 days 天前"""
    chunk = system.core.get(chunk_id)
    chunk.created_at -= days * DAY
    chunk.last_accessed -= days * DAY
    chunk.updated_at -= days * DAY
    system.core._store.put(chunk)
    system.core._invalidate_weight(chunk_id)


# ============ 权重模型 ============

def test_weight_floor_is_below_degrade_threshold(tmp_path):
    """老化后的默认记忆权重必须能降到降级阈值以下（旧模型下限 0.195 > 0.15）"""
    system = _make_system(tmp_path)
    chunk_id = system.add_memory(content="一条普通的记忆", importance=0.5)

    fresh_weight = system.core.calc_weight(system.core.get(chunk_id)).final
    assert fresh_weight > system.core.degrade_threshold

    _age_chunk(system, chunk_id, 120)
    aged_weight = system.core.calc_weight(system.core.get(chunk_id)).final
    assert aged_weight < system.core.degrade_threshold, (
        f"120 天未使用的默认记忆权重 {aged_weight:.3f} 仍高于 "
        f"降级阈值 {system.core.degrade_threshold}，遗忘机制失效"
    )


def test_important_memory_decays_slower(tmp_path):
    """高重要性/强情绪记忆衰减更慢，但同样会衰减"""
    system = _make_system(tmp_path)
    plain = system.add_memory(content="普通记忆", importance=0.3)
    vital = system.add_memory(
        content="重要记忆", importance=1.0,
        emotion_valence=0.8, emotion_intensity=0.9,
    )
    for cid in (plain, vital):
        _age_chunk(system, cid, 30)

    w_plain = system.core.calc_weight(system.core.get(plain)).final
    w_vital = system.core.calc_weight(system.core.get(vital)).final
    assert w_vital > w_plain


# ============ 检索稳定性 ============

def test_retrieve_no_crash_on_weight_ties(tmp_path):
    """权重完全相同的多条记忆参与检索时不应 TypeError（heap 平局比较 chunk）"""
    core = MemoryLayerCore(store=JsonMemoryStore(str(tmp_path / "core.json")))
    now = time.time()
    for i in range(5):
        chunk = MemoryChunk(
            content=f"相同权重的记忆 {i}",
            topics={"测试"},
            created_at=now,
            last_accessed=now,
        )
        core.add(chunk)

    results = core.retrieve({"topics": {"测试"}}, limit=3)
    assert len(results) == 3

    top = core.get_top(limit=3)
    assert len(top) == 3


# ============ 回忆反馈 ============

def test_failed_recall_feedback_lowers_weight(tmp_path):
    """错误回忆的惩罚必须真实生效并持久化（旧代码计算后丢弃）"""
    system = _make_system(tmp_path)
    chunk_id = system.add_memory(content="可能是错误的记忆", importance=0.5)

    before = system.core.calc_weight(system.core.get(chunk_id)).final
    system.core.adjust_after_recall(chunk_id, success=False)
    after = system.core.calc_weight(system.core.get(chunk_id)).final

    assert after < before
    assert system.core.get(chunk_id).recall_bias < 0

    # 反复纠错把记忆推向降级阈值，且有下限
    for _ in range(10):
        system.core.adjust_after_recall(chunk_id, success=False)
    assert system.core.get(chunk_id).recall_bias >= -0.25


def test_successful_recall_feedback_raises_weight(tmp_path):
    system = _make_system(tmp_path)
    chunk_id = system.add_memory(content="被确认正确的记忆", importance=0.5)

    before = system.core.calc_weight(system.core.get(chunk_id)).final
    system.core.adjust_after_recall(chunk_id, success=True)
    after = system.core.calc_weight(system.core.get(chunk_id)).final

    assert after > before
    assert system.core.get(chunk_id).successful_recall_count == 1


# ============ SQLite 后端持久化 ============

def test_sqlite_access_stats_persist(tmp_path):
    """SQLite 后端 get() 返回副本，access() 必须写回否则统计静默丢失"""
    system = _make_system(tmp_path, store_backend="sqlite")
    chunk_id = system.add_memory(content="SQLite 后端的记忆", topics=["测试"])

    system.core.access(chunk_id)
    system.core.access(chunk_id)

    fresh = system.core.get(chunk_id)
    assert fresh.access_count == 2


def test_sqlite_recall_bias_persists(tmp_path):
    system = _make_system(tmp_path, store_backend="sqlite")
    chunk_id = system.add_memory(content="SQLite 反馈持久化", topics=["测试"])
    system.core.adjust_after_recall(chunk_id, success=False)
    assert system.core.get(chunk_id).recall_bias < 0


# ============ 词汇表打通 ============

def test_chinese_query_matches_extracted_memory(tmp_path):
    """自动抽取产出中文主题，自然语言查询必须能命中（旧代码两套词汇零交集）"""
    system = _make_system(tmp_path, enable_hybrid_retrieval=False)
    system.add_raw_memory("中午和客户吃了牛肉面，聊得很开心", check_duplicate=False)

    result = system.retrieve("我们最近吃了什么")
    assert result.success, "中文查询无法命中自动抽取的记忆——词汇表仍然割裂"
    assert result.retrieval_path == "core"


def test_english_topics_match_chinese_query(tmp_path):
    """README 风格的英文主题标签也要能被中文查询命中"""
    system = _make_system(tmp_path, enable_hybrid_retrieval=False)
    system.add_memory(
        content="和客户在北京餐厅吃了烤鸭",
        topics=["food", "business"],
        location="北京",
    )
    result = system.retrieve("在北京吃了什么")
    assert result.success


# ============ 混合检索 ============

def test_hybrid_retrieval_is_wired_in(tmp_path):
    """默认配置下 QueryPlanner 必须真实参与检索（旧代码从未实例化）"""
    system = _make_system(tmp_path)
    assert system.query_planner is not None
    assert system.retrieval.planner is system.query_planner

    system.add_memory(content="昨天下午调试了向量检索的召回问题", keywords=["向量检索", "召回"])
    result = system.retrieve("向量检索的召回")
    assert result.success
    assert result.retrieval_path == "hybrid"


def test_hybrid_index_rebuilt_on_load(tmp_path):
    system = _make_system(tmp_path)
    system.add_memory(content="持久化之后依然要能被混合检索找到", keywords=["持久化"])
    system.save()

    reloaded = _make_system(tmp_path)
    assert reloaded.load()
    result = reloaded.retrieve("持久化")
    assert result.success


# ============ 索引一致性 ============

def test_index_snapshot_prevents_stale_keys(tmp_path):
    """原地修改标签后重新 add()，旧索引键必须被精确清除"""
    core = MemoryLayerCore(store=JsonMemoryStore(str(tmp_path / "core.json")))
    chunk = MemoryChunk(content="标签会变化的记忆", topics={"旧主题"})
    core.add(chunk)

    # JSON 后端按引用返回：原地修改后再 add，是最容易产生索引漂移的路径
    chunk.topics.clear()
    chunk.topics.add("新主题")
    core.add(chunk)

    stale = core.topic_index.get("旧主题", set())
    assert chunk.id not in stale, "旧主题索引键未被清除，索引已漂移"
    assert chunk.id in core.topic_index.get("新主题", set())


# ============ 照片检索 ============

def test_photo_timestamp_maps_to_daypart(tmp_path):
    """照片时间戳必须换算成语义时段，时钟时间任何写入路径都不会存储"""
    system = _make_system(tmp_path)
    noon = time.mktime((2026, 7, 20, 12, 30, 0, 0, 0, -1))
    ctx = system.retrieval.parse_photo_info({"timestamp": noon, "location": "北京"})
    assert ctx.time_context == "中午"


# ============ 完整生命周期（招牌能力） ============

def test_full_forget_wake_promote_cycle(tmp_path):
    """添加 -> 老化 -> 降级 -> 线索唤醒 -> 提升 -> 再次常规可检索"""
    system = _make_system(tmp_path)

    chunk_id = system.add_memory(
        content="和老王一起去哈尔滨看了冰雕，晚上吃了锅包肉",
        persons=["老王"],
        topics=["旅行"],
        importance=0.5,
    )

    # 1. 老化 120 天后维护：记忆应降级到伪遗忘层
    _age_chunk(system, chunk_id, 120)
    system.maintain()

    assert system.core.get(chunk_id) is None, "老化记忆未被降级"
    archived = system.forgotten.get(chunk_id)
    assert archived is not None
    assert archived.layer == MemoryLayer.FORGOTTEN

    # 2. 无线索查询不应唤醒（伪遗忘层不参与主动检索）
    blind = system.retrieve("最近工作进展怎么样")
    assert chunk_id not in [c.id for c in blind.chunks]

    # 3. 带强线索（人物 + 主题）的查询唤醒并提升
    result = system.retrieve("和老王一起去的那次旅行")
    assert result.success, "线索唤醒失败"
    assert result.retrieval_path == "forgotten"
    assert chunk_id in [c.id for c in result.chunks]

    # 4. 记忆已回到核心层，伪遗忘层不再持有
    promoted = system.core.get(chunk_id)
    assert promoted is not None, "唤醒的记忆未被提升回核心层"
    assert promoted.layer == MemoryLayer.CORE
    assert system.forgotten.get(chunk_id) is None
    assert promoted.successful_recall_count >= 1
    assert promoted.recall_bias > 0  # 再巩固奖励

    # 5. 提升后的记忆重新参与常规检索（含混合索引）
    again = system.retrieve("和老王一起去的那次旅行")
    assert again.success
    assert again.retrieval_path in ("hybrid", "core")
    assert chunk_id in [c.id for c in again.chunks]

    # 6. 统计口径记录了这次提升
    assert system.retrieval.total_promoted >= 1


def test_weak_cue_wakes_but_does_not_promote(tmp_path):
    """弱线索唤醒的记忆保持归档状态，但留下唤醒痕迹"""
    system = _make_system(tmp_path)
    chunk_id = system.add_memory(
        content="某次不重要的闲聊",
        topics=["旅行"],
        persons=["小李"],
        importance=0.0,  # 低重要性 -> 唤醒得分低
    )
    _age_chunk(system, chunk_id, 120)
    system.maintain()
    assert system.forgotten.get(chunk_id) is not None

    # persons+topics 匹配但 importance=0：得分 0.15+0.15+0 = 0.30
    # temp_weight = 0.50 < promote_threshold 0.55 -> 唤醒但不提升
    result = system.retrieve("和小李一起去的旅行")
    if result.success and result.retrieval_path == "forgotten":
        still_archived = system.forgotten.get(chunk_id)
        assert still_archived is not None, "弱唤醒不应提升"
        assert still_archived.successful_recall_count >= 1, "唤醒痕迹未持久化"
