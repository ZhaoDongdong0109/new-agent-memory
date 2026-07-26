"""
查询规划器 (Query Planner)

协调多个检索器并行检索，使用 RRF 融合结果。

检索流程：
1. Metadata/Time Filter (现有倒排索引)
2. BM25 Lexical Retrieval
3. Dense Vector Retrieval
4. RRF Fusion
5. Weight-based Reranking
"""

from typing import Dict, List, Optional, Tuple, Any

from memory_chunk import MemoryChunk, MemoryLayer
from core.bm25_retriever import BM25Retriever
from core.dense_retriever import DenseRetriever
from core.rrf_fusion import reciprocal_rank_fusion, weighted_reciprocal_rank_fusion


class QueryPlanner:
    """
    查询规划器

    协调多个检索器并行检索，使用 RRF 融合结果。
    """

    def __init__(
        self,
        core_layer=None,
        forgotten_layer=None,
        bm25: Optional[BM25Retriever] = None,
        dense: Optional[DenseRetriever] = None,
        # 融合权重
        metadata_weight: float = 0.4,
        bm25_weight: float = 0.3,
        dense_weight: float = 0.3,
        # RRF 参数
        rrf_k: int = 60,
    ):
        """
        Args:
            core_layer: 核心记忆层
            forgotten_layer: 伪遗忘层
            bm25: BM25 检索器
            dense: Dense 向量检索器
            metadata_weight: Metadata 检索权重
            bm25_weight: BM25 检索权重
            dense_weight: Dense 检索权重
            rrf_k: RRF 参数
        """
        self.core = core_layer
        self.forgotten = forgotten_layer
        self.bm25 = bm25
        self.dense = dense

        self.metadata_weight = metadata_weight
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k

    def plan_and_retrieve(
        self,
        query: str,
        query_tags: Dict[str, Any] = None,
        limit: int = 10,
    ) -> List[Tuple[MemoryChunk, float]]:
        """
        混合检索主流程

        Args:
            query: 查询文本
            query_tags: 查询标签（用于 metadata filter）
            limit: 返回结果数量

        Returns:
            [(chunk, score), ...] 按分数降序
        """
        rankings = []
        weights = []

        # Step 1: Metadata filter
        if query_tags and self.core:
            metadata_results = self._metadata_filter(query_tags)
            if metadata_results:
                rankings.append(metadata_results)
                weights.append(self.metadata_weight)

        # Step 2: BM25
        if self.bm25:
            bm25_results = self.bm25.search(query, top_k=limit * 2)
            if bm25_results:
                rankings.append(bm25_results)
                weights.append(self.bm25_weight)

        # Step 3: Dense
        if self.dense:
            dense_results = self.dense.search(query, top_k=limit * 2)
            if dense_results:
                rankings.append(dense_results)
                weights.append(self.dense_weight)

        # 如果没有任何检索结果，返回空
        if not rankings:
            return []

        # Step 4: RRF Fusion
        if len(weights) > 0 and abs(sum(weights) - 1.0) < 0.01:
            # 使用加权 RRF
            fused = weighted_reciprocal_rank_fusion(rankings, weights, self.rrf_k)
        else:
            # 使用普通 RRF
            fused = reciprocal_rank_fusion(rankings, self.rrf_k)

        # Step 5: 获取 chunk 并用权重重新排序
        #
        # 量纲对齐：RRF 原始分数的上界约为 sum(weights)/(k+1) ≈ 0.016，
        # 而记忆权重在 0~1。直接线性混合时权重项会以 10-20 倍压倒相关性，
        # "70% RRF + 30% 权重"名不符实。先把 RRF 分数按本批最大值归一化
        # 到 0~1，再混合，比例才是真实的。
        max_rrf = fused[0][1] if fused else 0.0

        results = []
        for chunk_id, rrf_score in fused[:limit * 2]:
            chunk = self._get_chunk(chunk_id)
            if chunk:
                # 计算最终权重
                if self.core and chunk.layer == MemoryLayer.CORE:
                    weight_factors = self.core.calc_weight(chunk)
                    weight = weight_factors.final
                else:
                    weight = 0.1  # 伪遗忘层的默认权重

                relevance = rrf_score / max_rrf if max_rrf > 0 else 0.0

                # 混合分数：70% 相关性 + 30% 记忆权重
                final_score = 0.7 * relevance + 0.3 * weight
                results.append((chunk, final_score))

        # 排序并返回
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    # 真正可用作元数据锚点的键：只有查询里出现这些键时元数据腿才参与融合
    _ANCHOR_KEYS = (
        "time_absolute", "time_relative", "time_context",
        "topics", "location", "persons",
    )

    def _metadata_filter(self, query_tags: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Metadata/Time Filter

        使用现有的倒排索引进行候选选择，并按记忆权重排序。

        两个关键约束：
        1. 查询没有任何索引锚点（只有 emotion 之类）时不参与融合——
           否则 _select_candidates 会退回全库扫描，把整个存储当成
           "命中结果"灌进 RRF 的最大权重腿。
        2. RRF 是基于名次的融合，无序候选集的名次毫无意义，
           必须先按记忆权重排出真实顺序。

        Returns:
            [(chunk_id, weight), ...] 按权重降序
        """
        if not self.core:
            return []

        if not any(key in query_tags for key in self._ANCHOR_KEYS):
            return []

        # 使用 core 的 _select_candidates 方法
        candidate_ids = self.core._select_candidates(query_tags)

        # 按记忆权重排序，让 RRF 的名次有真实含义
        scored = []
        for cid in candidate_ids:
            chunk = self.core.get(cid)
            if chunk is None:
                continue
            scored.append((cid, self.core.calc_weight(chunk).final))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _get_chunk(self, chunk_id: str) -> Optional[MemoryChunk]:
        """获取 chunk（先查 core，再查 forgotten）"""
        if self.core:
            chunk = self.core.get(chunk_id)
            if chunk:
                return chunk

        if self.forgotten:
            chunk = self.forgotten.get(chunk_id)
            if chunk:
                return chunk

        return None

    def index_chunks(self, chunks: Dict[str, MemoryChunk]) -> None:
        """
        索引所有 chunks 到 BM25 和 Dense 检索器

        Args:
            chunks: {chunk_id: MemoryChunk} 字典
        """
        for chunk_id, chunk in chunks.items():
            # 构建索引文本：content + summary + keywords
            text = chunk.content
            if chunk.summary:
                text += " " + chunk.summary
            if chunk.keywords:
                text += " " + " ".join(chunk.keywords)

            # 索引到 BM25
            if self.bm25:
                self.bm25.index(chunk_id, text)

            # 索引到 Dense
            if self.dense:
                self.dense.index(chunk_id, text)

    def add_chunk(self, chunk: MemoryChunk) -> None:
        """
        添加一个 chunk 到索引

        Args:
            chunk: MemoryChunk 对象
        """
        text = chunk.content
        if chunk.summary:
            text += " " + chunk.summary
        if chunk.keywords:
            text += " " + " ".join(chunk.keywords)

        if self.bm25:
            self.bm25.index(chunk.id, text)

        if self.dense:
            self.dense.index(chunk.id, text)

    def remove_chunk(self, chunk_id: str) -> None:
        """
        从索引中删除一个 chunk

        Args:
            chunk_id: chunk ID
        """
        if self.bm25:
            self.bm25.remove(chunk_id)

        if self.dense:
            self.dense.remove(chunk_id)

    def get_stats(self) -> Dict:
        """获取检索器统计信息"""
        stats = {}

        if self.bm25:
            stats["bm25"] = self.bm25.get_stats()

        if self.dense:
            stats["dense"] = self.dense.get_stats()

        return stats
