"""ACT-R 基线激活模型性质测试

验证 B = ln(Σ t^-d) + 逻辑斯蒂保持率映射产生教科书级的认知效应：
- 幂律遗忘：保持率单调下降，且尾部比指数衰减厚
- 频率效应：使用次数多的记忆保持率更高
- 近因效应：同样次数下，最近用过的保持率更高
- 类型分层：STORY > FACT > INTERACTION 的持久性排序
- Petrov O(k) 近似：日志截断后激活与完整日志接近
- 生命周期兼容：新记忆可检索，旧记忆可降级
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.json_store import JsonMemoryStore
from core.weight_system import MemoryType
from memory_chunk import MemoryChunk
from memory_layer_core import MemoryLayerCore

DAY = 24 * 3600


def _core(tmp_path):
    return MemoryLayerCore(store=JsonMemoryStore(str(tmp_path / "core.json")))


def _aged_chunk(days_old, access_ages_days=(), memory_type=MemoryType.INTERACTION):
    """构造一条 days_old 天前创建、在指定天数前访问过的记忆"""
    now = time.time()
    created = now - days_old * DAY
    log = [created] + [now - d * DAY for d in sorted(access_ages_days, reverse=True)]
    last = max(log)
    return MemoryChunk(
        content="测试记忆",
        memory_type=memory_type,
        created_at=created,
        last_accessed=last,
        access_count=len(access_ages_days),
        access_log=log[-MemoryChunk.ACCESS_LOG_SIZE:],
    )


def test_power_law_forgetting_monotone(tmp_path):
    """保持率随年龄单调下降"""
    core = _core(tmp_path)
    ages = [0.01, 1, 7, 30, 120, 365]
    retentions = []
    for age in ages:
        wf = core.calc_weight(_aged_chunk(age))
        retentions.append(wf.retention)
        core.weight_cache.clear()
    assert all(a > b for a, b in zip(retentions, retentions[1:])), retentions
    # 幂律尾部：120 -> 365 天的相对跌幅远小于 1 -> 7 天（厚尾）
    early_drop = retentions[1] / max(retentions[2], 1e-9)
    late_drop = retentions[4] / max(retentions[5], 1e-9)
    assert late_drop < early_drop


def test_frequency_effect(tmp_path):
    """同样的最近访问时间，历史使用次数多者保持率更高"""
    core = _core(tmp_path)
    rare = _aged_chunk(60, access_ages_days=(30,))
    frequent = _aged_chunk(60, access_ages_days=(55, 50, 45, 40, 35, 30))
    w_rare = core.calc_weight(rare).retention
    core.weight_cache.clear()
    w_freq = core.calc_weight(frequent).retention
    assert w_freq > w_rare


def test_recency_effect(tmp_path):
    """同样的使用次数，最近用过者保持率更高"""
    core = _core(tmp_path)
    stale = _aged_chunk(60, access_ages_days=(40, 35, 30))
    recent = _aged_chunk(60, access_ages_days=(12, 6, 1))
    w_stale = core.calc_weight(stale).retention
    core.weight_cache.clear()
    w_recent = core.calc_weight(recent).retention
    assert w_recent > w_stale


def test_type_stratified_persistence(tmp_path):
    """同龄未使用记忆：故事 > 事实 > 交互 的保持率排序"""
    core = _core(tmp_path)
    retentions = {}
    for mtype in (MemoryType.STORY, MemoryType.FACT, MemoryType.INTERACTION):
        wf = core.calc_weight(_aged_chunk(90, memory_type=mtype))
        retentions[mtype] = wf.retention
        core.weight_cache.clear()
    assert retentions[MemoryType.STORY] > retentions[MemoryType.FACT]
    assert retentions[MemoryType.FACT] > retentions[MemoryType.INTERACTION]


def test_petrov_approximation_close_to_exact(tmp_path):
    """访问次数超过日志容量后，尾部近似与完整日志的激活接近"""
    core = _core(tmp_path)
    now = time.time()
    created = now - 100 * DAY
    # 20 次访问均匀分布在过去 100 天
    all_events = [created] + [now - d * DAY for d in range(95, 0, -5)]

    exact = MemoryChunk(
        content="完整日志", created_at=created, last_accessed=max(all_events),
        access_count=len(all_events) - 1, access_log=list(all_events),
    )
    # 手动放宽日志上限做精确参照
    exact_b = None
    try:
        MemoryChunk.ACCESS_LOG_SIZE = 64
        exact_b = core._base_level_activation(exact, now)
    finally:
        MemoryChunk.ACCESS_LOG_SIZE = 8

    truncated = MemoryChunk(
        content="截断日志", created_at=created, last_accessed=max(all_events),
        access_count=len(all_events) - 1,
        access_log=all_events[-MemoryChunk.ACCESS_LOG_SIZE:],
    )
    approx_b = core._base_level_activation(truncated, now)

    assert abs(approx_b - exact_b) < 0.5, (
        f"Petrov 近似偏差过大: exact={exact_b:.3f} approx={approx_b:.3f}"
    )


def test_lifecycle_calibration(tmp_path):
    """校准锚点：新记忆可检索（>0.2），120 天未用可降级（<0.15）"""
    core = _core(tmp_path)
    fresh = core.calc_weight(_aged_chunk(0.001)).final
    assert fresh > 0.2, f"新记忆权重 {fresh:.3f} 低于检索门槛"
    core.weight_cache.clear()
    aged = core.calc_weight(_aged_chunk(120)).final
    assert aged < core.degrade_threshold, f"旧记忆权重 {aged:.3f} 无法降级"


def test_reawakened_memory_recovers(tmp_path):
    """久未使用的记忆被重新访问后，保持率恢复（再巩固）"""
    core = _core(tmp_path)
    chunk = _aged_chunk(120)
    core.add(chunk)
    before = core.calc_weight(chunk).retention
    core.access(chunk.id)
    after = core.calc_weight(core.get(chunk.id)).retention
    assert after > before + 0.3, f"重新访问后保持率未恢复: {before:.3f} -> {after:.3f}"
