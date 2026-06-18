"""
Reciprocal Rank Fusion (RRF) 融合器

将多个检索器的结果融合为一个统一排序。

RRF 公式：
    score(d) = Σ 1 / (k + rank_i(d))

其中：
- k 是常数（默认 60）
- rank_i(d) 是文档 d 在第 i 个检索器中的排名
"""

from typing import Dict, List, Tuple


def reciprocal_rank_fusion(
    rankings: List[List[Tuple[str, float]]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion

    Args:
        rankings: 多个检索器的排序结果
            每个元素是 [(doc_id, score), ...] 按分数降序
        k: RRF 参数（默认 60，控制排名靠后的文档权重衰减速度）

    Returns:
        融合后的 [(doc_id, rrf_score), ...] 按 RRF 分数降序
    """
    scores: Dict[str, float] = {}

    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking):
            # RRF 公式：1 / (k + rank + 1)
            # rank 从 0 开始，所以加 1
            rrf_score = 1.0 / (k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score

    # 按 RRF 分数降序排序
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


def weighted_reciprocal_rank_fusion(
    rankings: List[List[Tuple[str, float]]],
    weights: List[float],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    加权 Reciprocal Rank Fusion

    Args:
        rankings: 多个检索器的排序结果
        weights: 每个检索器的权重
        k: RRF 参数

    Returns:
        融合后的 [(doc_id, rrf_score), ...] 按 RRF 分数降序
    """
    if len(rankings) != len(weights):
        raise ValueError("rankings 和 weights 长度必须相同")

    scores: Dict[str, float] = {}

    for ranking, weight in zip(rankings, weights):
        for rank, (doc_id, _) in enumerate(ranking):
            rrf_score = weight / (k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


def merge_and_deduplicate(
    rankings: List[List[Tuple[str, float]]],
) -> List[Tuple[str, float]]:
    """
    合并多个排序结果并去重

    对于重复的文档，保留最高分数

    Args:
        rankings: 多个检索器的排序结果

    Returns:
        合并去重后的 [(doc_id, max_score), ...] 按分数降序
    """
    best_scores: Dict[str, float] = {}

    for ranking in rankings:
        for doc_id, score in ranking:
            if doc_id not in best_scores or score > best_scores[doc_id]:
                best_scores[doc_id] = score

    sorted_results = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results
