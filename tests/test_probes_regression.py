"""确定性探针回归门

小规模快速版（CI 每次运行），断言各探针的下限分数。
机制改动导致检索/知识更新质量退化时，这里最先报警。

完整规模：python scripts/bench/tot_lite.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.bench.tot_lite import (
    capacity_probe,
    knowledge_update_probe,
    temporal_probe,
)


def test_temporal_probe_gate(tmp_path):
    r = temporal_probe(str(tmp_path), seed=7, num_facts=10, num_distractors=25)
    assert r["recall@5"] >= 0.9, f"时序检索召回退化: {r}"
    assert r["window_precision"] >= 0.9, f"时间窗口精确率退化: {r}"


def test_knowledge_update_probe_gate(tmp_path):
    r = knowledge_update_probe(str(tmp_path), seed=7, num_chains=5)
    assert r["cr_acc@1"] >= 0.8, f"当前值命中率退化: {r}"
    assert r["stale_rate"] <= 0.1, f"被取代旧值混入现在时结果: {r}"
    assert r["history_recall"] >= 0.9, f"取代链历史召回退化: {r}"


def test_capacity_probe_gate(tmp_path):
    # 阈值留有余量：chunk id 是 uuid，平局排序存在跨进程波动
    # （实测 5 次：recall@10 ∈ [0.87, 0.93]，mrr ∈ [0.77, 0.82]）。
    # 回归门是下限报警，不是精确值断言。
    r = capacity_probe(str(tmp_path), seed=7, store_size=300, num_gold=15)
    assert r["recall@10"] >= 0.8, f"容量召回退化: {r}"
    assert r["mrr"] >= 0.6, f"容量排序质量退化: {r}"
