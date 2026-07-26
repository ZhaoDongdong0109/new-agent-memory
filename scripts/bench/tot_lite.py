"""ToT-lite：确定性时序记忆评测生成器与探针

设计原则（来自 2025-2026 评测调研的教训）：
- 无 LLM 评审：答案由构造已知，评分是集合运算，不是模型打分
- 种子驱动：random.Random(seed)，同种子同数据同结论，可进 CI
- 无污染：数据是合成的，不可能出现在任何模型的训练集里
- 事实的时间用固定历史年份（2023-2025），运行时时钟不影响窗口判定

三个探针：

1. temporal_probe   时序检索：黄金事实混入干扰项，按"YYYY年M月+人物"
                    查询，度量证据 Recall@k 与窗口精确率
2. knowledge_update_probe
                    知识更新（全行业最弱能力）：v1->v2->v3 取代链，
                    现在时查询度量 CR-Acc@1（当前值命中率）与
                    stale-rate（被取代旧值混入率），过去时查询度量
                    历史链召回
3. capacity_probe   容量门：N 条干扰项下黄金事实的 Recall@10 / MRR /
                    p50/p95 延迟——机制改进的性能回归门
"""

from __future__ import annotations

import os
import random
import sys
import time
from typing import Dict, List

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from main import HumanLikeMemorySystem  # noqa: E402
from core.weight_system import MemoryType  # noqa: E402
from scripts.bench.metrics import (  # noqa: E402
    mrr,
    p50_latency,
    p95_latency,
    recall_at_k,
)

PERSONS = ["老王", "小李", "张三", "陈晨", "阿明", "赵姐", "小周", "何叔"]
LOCATIONS = ["北京", "上海", "杭州", "成都", "深圳", "西安", "苏州", "青岛"]
ACTIVITIES = ["参加行业峰会", "考察新工厂", "谈成一笔合作", "做技术分享", "验收项目", "参加婚礼"]
CITIES = ["里斯本", "柏林", "奥斯陆", "巴黎", "马德里", "维也纳", "布拉格", "赫尔辛基"]
DISTRACTOR_SNIPPETS = [
    "读完了一本关于分布式系统的书",
    "修好了阳台上的椅子",
    "试了新的咖啡豆感觉一般",
    "跑步五公里刷新了记录",
    "给绿萝换了个大盆",
    "学会了一道新菜",
]


def _fresh_system(data_dir: str) -> HumanLikeMemorySystem:
    return HumanLikeMemorySystem(
        data_dir=data_dir,
        enable_pii_detection=False,
        enable_audit_log=False,
    )


# ============ 探针 1：时序检索 ============

def temporal_probe(data_dir: str, seed: int = 42, num_facts: int = 24,
                   num_distractors: int = 60, top_k: int = 5) -> Dict:
    """
    黄金事实：{date}，{person} 在 {location} {activity}，time_absolute 精确。
    每个事实的 (年, 月, 人物) 组合唯一，答案由构造已知。
    查询："{Y}年{M}月{person}做了什么"
    """
    rng = random.Random(seed)
    system = _fresh_system(data_dir)

    # 生成黄金事实（唯一的 年-月-人物 槽）
    slots = set()
    gold = []
    while len(gold) < num_facts:
        year = rng.choice([2023, 2024, 2025])
        month = rng.randint(1, 12)
        person = rng.choice(PERSONS)
        if (year, month, person) in slots:
            continue
        slots.add((year, month, person))
        day = rng.randint(1, 28)
        location = rng.choice(LOCATIONS)
        activity = rng.choice(ACTIVITIES)
        gold.append({
            "year": year, "month": month, "person": person,
            "content": f"{year}年{month}月{day}日，{person}在{location}{activity}",
            "time_absolute": f"{year:04d}-{month:02d}-{day:02d}",
        })

    # 打乱插入：黄金事实 + 干扰项
    inserts = list(gold)
    for i in range(num_distractors):
        year = rng.choice([2023, 2024, 2025])
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        inserts.append({
            "content": f"{year}年{month}月{day}日，{rng.choice(DISTRACTOR_SNIPPETS)}",
            "time_absolute": f"{year:04d}-{month:02d}-{day:02d}",
        })
    rng.shuffle(inserts)

    for item in inserts:
        cid = system.add_memory(
            content=item["content"],
            time_absolute=item["time_absolute"],
        )
        item["id"] = cid

    # 查询与评分
    recalls, mrrs, window_hits, latencies = [], [], [], []
    from core.time_parser import chunk_time_range, parse_query_window
    for fact in gold:
        query = f"{fact['year']}年{fact['month']}月{fact['person']}做了什么"
        t0 = time.time()
        result = system.retrieve(query)
        latencies.append(time.time() - t0)

        retrieved = [c.id for c in result.chunks]
        expected = {fact["id"]}
        recalls.append(recall_at_k(retrieved, expected, top_k))
        mrrs.append(mrr(retrieved, expected))

        # 窗口精确率：返回的所有记忆必须落在查询的时间窗口内
        window = parse_query_window(query)
        if result.chunks and window:
            in_window = sum(
                1 for c in result.chunks
                if chunk_time_range(c)[0] <= window[1] and chunk_time_range(c)[1] >= window[0]
            )
            window_hits.append(in_window / len(result.chunks))

    return {
        "name": "temporal_probe (ToT-lite)",
        "seed": seed,
        "facts": num_facts,
        "distractors": num_distractors,
        f"recall@{top_k}": sum(recalls) / len(recalls),
        "mrr": sum(mrrs) / len(mrrs),
        "window_precision": sum(window_hits) / len(window_hits) if window_hits else 0.0,
        "p50_ms": p50_latency(latencies) * 1000,
        "p95_ms": p95_latency(latencies) * 1000,
    }


# ============ 探针 2：知识更新 ============

def knowledge_update_probe(data_dir: str, seed: int = 42, num_chains: int = 8) -> Dict:
    """
    每条链：同一人物的居住地 v1 -> v2 -> v3（写入时应触发 SUPERSEDE）。
    现在时查询：CR-Acc@1 = top-1 是否当前值；stale-rate = 结果中
    被取代旧值的占比。过去时查询：历史值召回率。
    """
    rng = random.Random(seed)
    system = _fresh_system(data_dir)

    chains = []
    persons = rng.sample(PERSONS, min(num_chains, len(PERSONS)))
    for person in persons:
        cities = rng.sample(CITIES, 3)
        ids = []
        for city in cities:
            cid = system.add_memory(
                content=f"{person}现在住在{city}",
                memory_type=MemoryType.FACT,
                persons=[person],
                topics=["居住"],
                keywords=["住", city],
            )
            ids.append(cid)
        chains.append({"person": person, "cities": cities, "ids": ids})

    cr_hits, stale_counts, result_counts, history_recalls = [], [], [], []
    for chain in chains:
        current_id = chain["ids"][-1]
        superseded = set(chain["ids"][:-1])

        # 现在时：top-1 应是当前值，被取代旧值不应出现
        result = system.retrieve(f"{chain['person']}现在住在哪")
        retrieved = [c.id for c in result.chunks]
        cr_hits.append(1.0 if retrieved and retrieved[0] == current_id else 0.0)
        stale_counts.append(sum(1 for cid in retrieved if cid in superseded))
        result_counts.append(max(1, len(retrieved)))

        # 过去时：取代链应带回全部历史值
        past = system.retrieve(f"{chain['person']}以前住在哪")
        past_ids = set(c.id for c in past.chunks)
        history_recalls.append(len(superseded & past_ids) / len(superseded))

    return {
        "name": "knowledge_update_probe",
        "seed": seed,
        "chains": len(chains),
        "cr_acc@1": sum(cr_hits) / len(cr_hits),
        "stale_rate": sum(stale_counts) / sum(result_counts),
        "history_recall": sum(history_recalls) / len(history_recalls),
    }


# ============ 探针 3：容量门 ============

def capacity_probe(data_dir: str, seed: int = 42, store_size: int = 1000,
                   num_gold: int = 40, top_k: int = 10) -> Dict:
    """
    store_size 条记忆（含 num_gold 条带独特关键词的黄金事实），
    度量黄金事实的 Recall@10 / MRR 与检索延迟。
    """
    rng = random.Random(seed)
    system = _fresh_system(data_dir)

    gold = []
    for i in range(num_gold):
        person = rng.choice(PERSONS)
        location = rng.choice(LOCATIONS)
        token = f"项目Ω{i:03d}"
        content = f"{person}在{location}负责{token}的验收，结论是通过"
        gold.append({"query": f"{token}的验收结论", "content": content})

    inserts = [g["content"] for g in gold]
    while len(inserts) < store_size:
        inserts.append(
            f"{rng.choice(PERSONS)}在{rng.choice(LOCATIONS)}"
            f"{rng.choice(ACTIVITIES)}，{rng.choice(DISTRACTOR_SNIPPETS)}"
        )
    rng.shuffle(inserts)

    content_to_id = {}
    for content in inserts:
        cid = system.add_memory(content=content)
        content_to_id[content] = cid

    recalls, mrrs, latencies = [], [], []
    for g in gold:
        expected = {content_to_id[g["content"]]}
        t0 = time.time()
        result = system.retrieve(g["query"])
        latencies.append(time.time() - t0)
        retrieved = [c.id for c in result.chunks]
        recalls.append(recall_at_k(retrieved, expected, top_k))
        mrrs.append(mrr(retrieved, expected))

    return {
        "name": "capacity_probe",
        "seed": seed,
        "store_size": store_size,
        "gold_facts": num_gold,
        f"recall@{top_k}": sum(recalls) / len(recalls),
        "mrr": sum(mrrs) / len(mrrs),
        "p50_ms": p50_latency(latencies) * 1000,
        "p95_ms": p95_latency(latencies) * 1000,
    }


def run_all_probes(base_dir: str, seed: int = 42) -> List[Dict]:
    """运行全部探针，各用独立数据目录"""
    results = []
    results.append(temporal_probe(os.path.join(base_dir, "probe_temporal"), seed=seed))
    results.append(knowledge_update_probe(os.path.join(base_dir, "probe_knowledge"), seed=seed))
    results.append(capacity_probe(os.path.join(base_dir, "probe_capacity"), seed=seed))
    return results


def probes_report_section(results: List[Dict]) -> str:
    lines = ["", "## 确定性探针（无 LLM 评审，种子可复现）", ""]
    for r in results:
        lines.append(f"### {r['name']}")
        lines.append("")
        for key, value in r.items():
            if key == "name":
                continue
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.4f}")
            else:
                lines.append(f"- {key}: {value}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description="运行 ToT-lite 确定性探针")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        for probe_result in run_all_probes(tmp, seed=args.seed):
            print()
            for k, v in probe_result.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
