"""
Benchmark 指标计算

实现常用的检索和记忆系统评测指标。
"""

import math
from typing import Dict, List, Set


def recall_at_k(retrieved: List[str], expected: Set[str], k: int) -> float:
    """
    Recall@K

    在前 K 个检索结果中，有多少是预期的。

    Args:
        retrieved: 检索结果 ID 列表（按相关性排序）
        expected: 预期结果 ID 集合
        k: 取前 K 个结果

    Returns:
        Recall@K 值（0.0 - 1.0）
    """
    if not expected:
        return 0.0

    retrieved_k = set(retrieved[:k])
    hits = len(retrieved_k & expected)
    return hits / len(expected)


def precision_at_k(retrieved: List[str], expected: Set[str], k: int) -> float:
    """
    Precision@K

    在前 K 个检索结果中，有多少是正确的。

    Args:
        retrieved: 检索结果 ID 列表
        expected: 预期结果 ID 集合
        k: 取前 K 个结果

    Returns:
        Precision@K 值（0.0 - 1.0）
    """
    if k == 0:
        return 0.0

    retrieved_k = set(retrieved[:k])
    hits = len(retrieved_k & expected)
    return hits / k


def mrr(retrieved: List[str], expected: Set[str]) -> float:
    """
    Mean Reciprocal Rank (MRR)

    第一个正确结果的排名的倒数。

    Args:
        retrieved: 检索结果 ID 列表
        expected: 预期结果 ID 集合

    Returns:
        MRR 值（0.0 - 1.0）
    """
    for i, doc_id in enumerate(retrieved):
        if doc_id in expected:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: List[str], expected: Set[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain (NDCG@K)

    Args:
        retrieved: 检索结果 ID 列表
        expected: 预期结果 ID 集合
        k: 取前 K 个结果

    Returns:
        NDCG@K 值（0.0 - 1.0）
    """
    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in expected:
            dcg += 1.0 / math.log2(i + 2)  # i+2 因为 log2(1) = 0

    # Ideal DCG
    ideal_hits = min(len(expected), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def f1_at_k(retrieved: List[str], expected: Set[str], k: int) -> float:
    """
    F1@K

    Precision@K 和 Recall@K 的调和平均。

    Args:
        retrieved: 检索结果 ID 列表
        expected: 预期结果 ID 集合
        k: 取前 K 个结果

    Returns:
        F1@K 值（0.0 - 1.0）
    """
    p = precision_at_k(retrieved, expected, k)
    r = recall_at_k(retrieved, expected, k)

    if p + r == 0:
        return 0.0

    return 2 * p * r / (p + r)


def mean_latency(latencies: List[float]) -> float:
    """平均延迟"""
    if not latencies:
        return 0.0
    return sum(latencies) / len(latencies)


def p50_latency(latencies: List[float]) -> float:
    """P50 延迟"""
    if not latencies:
        return 0.0
    sorted_lat = sorted(latencies)
    idx = int(len(sorted_lat) * 0.5)
    return sorted_lat[idx]


def p95_latency(latencies: List[float]) -> float:
    """P95 延迟"""
    if not latencies:
        return 0.0
    sorted_lat = sorted(latencies)
    idx = int(len(sorted_lat) * 0.95)
    return sorted_lat[min(idx, len(sorted_lat) - 1)]


def p99_latency(latencies: List[float]) -> float:
    """P99 延迟"""
    if not latencies:
        return 0.0
    sorted_lat = sorted(latencies)
    idx = int(len(sorted_lat) * 0.99)
    return sorted_lat[min(idx, len(sorted_lat) - 1)]


def calculate_all_metrics(
    retrieved: List[str],
    expected: Set[str],
    latency: float,
    k: int = 10,
) -> Dict[str, float]:
    """
    计算所有指标

    Args:
        retrieved: 检索结果 ID 列表
        expected: 预期结果 ID 集合
        latency: 检索延迟（秒）
        k: 取前 K 个结果

    Returns:
        指标字典
    """
    return {
        f"recall@{k}": recall_at_k(retrieved, expected, k),
        f"precision@{k}": precision_at_k(retrieved, expected, k),
        "mrr": mrr(retrieved, expected),
        f"ndcg@{k}": ndcg_at_k(retrieved, expected, k),
        f"f1@{k}": f1_at_k(retrieved, expected, k),
        "latency": latency,
    }


def aggregate_metrics(results: List[Dict[str, float]]) -> Dict[str, float]:
    """
    聚合多个查询的指标

    Args:
        results: 指标字典列表

    Returns:
        聚合后的指标字典
    """
    if not results:
        return {}

    aggregated = {}
    keys = results[0].keys()

    for key in keys:
        # 只聚合数值指标：结果字典里还携带 query/category 等文本字段，
        # 对字符串求和会直接 TypeError
        values = [
            r[key] for r in results
            if key in r and isinstance(r[key], (int, float)) and not isinstance(r[key], bool)
        ]
        if not values:
            continue
        aggregated[f"mean_{key}"] = sum(values) / len(values)
        aggregated[f"min_{key}"] = min(values)
        aggregated[f"max_{key}"] = max(values)

    return aggregated
