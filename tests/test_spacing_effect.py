"""Pavlik & Anderson (2005) 间隔效应测试

朴素 BLA 的盲区：每次使用贡献相同痕迹，突击复习 5 次与分散复习
5 次编码一样强——与百年间隔效应实证相悖。扩展后每次使用事件的
衰减速率取决于复习瞬间的激活：

    d_j = base_d + c * e^(m_j)      c = 0.277（论文拟合值）

刚用过就再用（激活高）-> 该次痕迹衰减快，边际收益小；
快忘了才复习（激活近阈值）-> 痕迹接近基线衰减，最耐久。
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import HumanLikeMemorySystem
from memory_chunk import MemoryChunk
from core.weight_system import (
    PAVLIK_DECAY_CEIL,
    actr_decay,
    pavlik_event_decay,
)

DAY = 24 * 3600.0


def _make_system(tmp_path, **kwargs):
    kwargs.setdefault("enable_pii_detection", False)
    kwargs.setdefault("enable_audit_log", False)
    return HumanLikeMemorySystem(data_dir=str(tmp_path), **kwargs)


# ============ 单元：事件衰减公式 ============

def test_pavlik_event_decay_properties():
    base = 0.5
    # 激活极低（首次编码前 = -inf 的近似）：回到类型基线
    assert abs(pavlik_event_decay(-50.0, base) - base) < 1e-6
    # 激活越高衰减越快（单调）
    d_low = pavlik_event_decay(-6.0, base)
    d_mid = pavlik_event_decay(-3.0, base)
    d_high = pavlik_event_decay(-1.0, base)
    assert base < d_low < d_mid < d_high
    # 上限保证幂律积分收敛
    assert pavlik_event_decay(10.0, base) == PAVLIK_DECAY_CEIL


# ============ 集成：分散复习比突击复习耐久 ============

def test_spaced_practice_beats_massed(tmp_path, monkeypatch):
    """同样 4 次复习：分散（每 12 天）显著强于突击（前 8 分钟内）"""
    t0 = 1_700_000_000.0
    clock = {"now": t0}
    monkeypatch.setattr(time, "time", lambda: clock["now"])

    system = _make_system(tmp_path)
    id_massed = system.add_memory(content="突击复习的临时知识点")
    id_spaced = system.add_memory(content="分散复习的长期知识点")
    id_single = system.add_memory(content="只编码一次的对照知识点")

    # 突击：编码后 2/4/6/8 分钟连续复习（间隔 > 60s 去抖窗口）
    for minutes in (2, 4, 6, 8):
        clock["now"] = t0 + minutes * 60
        system.core.access(id_massed)

    # 分散：每 12 天复习一次
    for day in (12, 24, 36, 48):
        clock["now"] = t0 + day * DAY
        system.core.access(id_spaced)

    # 复习时激活高 -> 记录的事件衰减高于基线；快忘了才复习 -> 接近基线
    base = actr_decay("interaction")
    massed_decays = [d for d in system.core.get(id_massed).access_decays if d]
    spaced_decays = [d for d in system.core.get(id_spaced).access_decays if d]
    assert all(d > base + 0.01 for d in massed_decays), massed_decays
    assert all(d < base + 0.005 for d in spaced_decays), spaced_decays

    # 60 天后的激活对比
    test_t = t0 + 60 * DAY
    clock["now"] = test_t
    B = {
        name: system.core._base_level_activation(system.core.get(cid), test_t)
        for name, cid in (
            ("massed", id_massed), ("spaced", id_spaced), ("single", id_single),
        )
    }
    assert B["spaced"] > B["massed"] + 0.5, B
    # 复习仍然有收益（突击也强于单次编码），但边际收益打折
    assert B["massed"] > B["single"], B
    gain_massed = B["massed"] - B["single"]
    gain_spaced = B["spaced"] - B["single"]
    assert gain_massed < 0.7 * gain_spaced, B


# ============ 兼容：老数据没有 access_decays ============

def test_legacy_chunk_without_decays_still_computes(tmp_path):
    """access_log 长于 access_decays（老数据/直接调用）时回退基线 d"""
    system = _make_system(tmp_path)
    cid = system.add_memory(content="迁移前的老记忆")
    chunk = system.core.get(cid)
    now = time.time()
    chunk.access_log = [now - 30 * DAY, now - 20 * DAY, now - 10 * DAY]
    chunk.access_decays = []  # 老数据没有逐事件衰减
    chunk.access_count = 3
    chunk.last_accessed = now - 10 * DAY
    chunk.created_at = now - 30 * DAY
    B = system.core._base_level_activation(chunk, now)
    assert B == B  # 不 NaN 不抛异常
    # 序列化往返保留字段
    restored = MemoryChunk.from_dict(chunk.to_dict())
    assert restored.access_decays == chunk.access_decays


def test_access_records_event_decay(tmp_path):
    """核心层访问路径记录本次事件的衰减速率"""
    system = _make_system(tmp_path)
    cid = system.add_memory(content="被访问的记忆", keywords=["访问"])
    chunk = system.core.get(cid)
    # 编码事件本身没有逐事件衰减记录（等价于基线）
    assert chunk.access_decays == []

    chunk.last_accessed -= 120  # 跳过去抖窗口
    system.core._store.put(chunk)
    system.core.access(cid)
    decays = system.core.get(cid).access_decays
    assert len(decays) == 1 and decays[0] is not None
    assert decays[0] >= actr_decay(chunk.memory_type)


# ============ 对抗审查发现的三个缺陷的回归 ============

def test_sqlite_backend_persists_event_decays(tmp_path):
    """SQLite 后端（生产 MCP 路径）必须持久化逐事件衰减

    对抗审查实测：schema/迁移/INSERT/行解析四处全部漏掉
    access_decays，put->get 一个来回衰减即丢失，整个间隔效应
    在生产后端上曾是静默 no-op。
    """
    system = _make_system(tmp_path, store_backend="sqlite")
    cid = system.add_memory(content="走 SQLite 的记忆")
    chunk = system.core.get(cid)
    chunk.last_accessed -= 120
    system.core._store.put(chunk)
    system.core.access(cid)

    # 同进程往返
    reloaded = system.core.get(cid)
    assert len(reloaded.access_decays) == 1
    assert reloaded.access_decays[0] is not None

    # None 值与滑动累计的往返（JSON null <-> None 无损）
    chunk = system.core.get(cid)
    chunk.access_decays = [None, 0.62]
    chunk.evicted_decay_sum = 3.3
    chunk.evicted_decay_count = 6
    system.core._store.put(chunk)
    back = system.core.get(cid)
    assert back.access_decays == [None, 0.62]
    assert back.evicted_decay_sum == 3.3 and back.evicted_decay_count == 6


def test_tail_integral_keeps_spacing_penalty(tmp_path, monkeypatch):
    """密集复习把事件推入 Petrov 尾部后，间隔惩罚不得丢失

    对抗审查实测（修复前）：20 次突击访问后 91% 的激活和来自
    按基线 d 记账的尾部，突击与分散的激活差距被抹平。
    """
    t0 = 1_700_000_000.0
    clock = {"now": t0}
    monkeypatch.setattr(time, "time", lambda: clock["now"])

    system = _make_system(tmp_path)
    id_crammed = system.add_memory(content="突击灌入的知识点甲")
    id_spaced = system.add_memory(content="分散复习的知识点乙")

    # 突击：61 秒一次，共 20 次（全部越过去抖，前 13 个事件被裁入尾部）
    for i in range(1, 21):
        clock["now"] = t0 + i * 61
        system.core.access(id_crammed)
    # 分散：每天一次，共 20 次
    for i in range(1, 21):
        clock["now"] = t0 + i * DAY
        system.core.access(id_spaced)

    # 突击痕迹被裁剪后滑动累计仍记着高衰减
    crammed = system.core.get(id_crammed)
    assert crammed.evicted_decay_count >= 10
    assert crammed.evicted_decay_sum / crammed.evicted_decay_count > 0.52

    test_t = t0 + 27 * DAY  # 分散复习结束一周后
    B_crammed = system.core._base_level_activation(system.core.get(id_crammed), test_t)
    B_spaced = system.core._base_level_activation(system.core.get(id_spaced), test_t)
    assert B_spaced > B_crammed + 0.5, (B_spaced, B_crammed)


def test_duplicate_ingestion_records_pavlik_decay(tmp_path):
    """去重路径的重复灌入按当时激活记衰减，不再享受基线耐久"""
    system = _make_system(tmp_path)
    first_id = system.add_raw_memory("发布流程要先跑测试再触发工作流")
    chunk = system.core.get(first_id)
    chunk.last_accessed -= 120  # 越过去抖
    system.core._store.put(chunk)

    second_id = system.add_raw_memory("发布流程要先跑测试再触发工作流")
    assert second_id == first_id, "重复灌入应命中去重"
    decays = system.core.get(first_id).access_decays
    assert decays and decays[-1] is not None, "去重路径未记录事件衰减"
    from core.weight_system import actr_decay
    assert decays[-1] > actr_decay(system.core.get(first_id).memory_type)
