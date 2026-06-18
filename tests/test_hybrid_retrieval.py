"""
混合检索测试

测试 BM25、Dense、RRF 和混合检索效果。
"""

import sys
import time
import tempfile

sys.path.insert(0, '.')

from memory_chunk import MemoryChunk, MemoryLayer, MemoryType
from core.bm25_retriever import BM25Retriever
from core.dense_retriever import DenseRetriever
from core.rrf_fusion import reciprocal_rank_fusion, weighted_reciprocal_rank_fusion
from core.query_planner import QueryPlanner


def make_chunk(chunk_id: str, content: str, **kwargs) -> MemoryChunk:
    """创建测试用 MemoryChunk"""
    return MemoryChunk(
        id=chunk_id,
        content=content,
        summary=kwargs.get("summary", ""),
        memory_type=MemoryType.INTERACTION,
        time_absolute=kwargs.get("time_absolute", "2026-06-18"),
        location=kwargs.get("location", None),
        persons=kwargs.get("persons", set()),
        topics=kwargs.get("topics", set()),
        keywords=kwargs.get("keywords", set()),
        importance=kwargs.get("importance", 0.5),
        layer=MemoryLayer.CORE,
        created_at=time.time(),
        updated_at=time.time(),
        last_accessed=time.time(),
    )


def test_bm25_retriever():
    """测试 BM25 检索器"""
    print("=== 测试 BM25 检索器 ===")

    retriever = BM25Retriever()

    # 索引一些文档
    retriever.index("doc1", "今天中午在北京吃了烤鸭")
    retriever.index("doc2", "昨天在上海参加了技术会议")
    retriever.index("doc3", "上周去北京出差见了客户")
    retriever.index("doc4", "在家里看了一部电影")
    retriever.index("doc5", "北京的天气很好")

    # 测试检索
    results = retriever.search("北京", top_k=3)
    print(f"查询 '北京': {results}")
    assert len(results) > 0
    assert results[0][0] in ["doc1", "doc3", "doc5"]  # 包含"北京"的文档

    results = retriever.search("会议", top_k=3)
    print(f"查询 '会议': {results}")
    assert len(results) > 0
    assert results[0][0] == "doc2"

    # 测试增量更新
    retriever.index("doc6", "北京烤鸭很好吃")
    results = retriever.search("烤鸭", top_k=3)
    print(f"查询 '烤鸭': {results}")
    assert len(results) > 0
    assert results[0][0] in ["doc1", "doc6"]

    # 测试删除
    retriever.remove("doc1")
    results = retriever.search("烤鸭", top_k=3)
    print(f"删除 doc1 后查询 '烤鸭': {results}")
    assert len(results) > 0
    assert results[0][0] == "doc6"

    print("✅ BM25 检索器测试通过\n")


def test_dense_retriever():
    """测试 Dense 向量检索器"""
    print("=== 测试 Dense 向量检索器 ===")

    retriever = DenseRetriever(dimension=128)

    # 索引一些文档
    retriever.index("doc1", "今天中午在北京吃了烤鸭")
    retriever.index("doc2", "昨天在上海参加了技术会议")
    retriever.index("doc3", "上周去北京出差见了客户")
    retriever.index("doc4", "在家里看了一部电影")
    retriever.index("doc5", "北京的天气很好")

    # 测试检索
    results = retriever.search("北京", top_k=3)
    print(f"查询 '北京': {results}")
    assert len(results) > 0

    results = retriever.search("会议", top_k=3)
    print(f"查询 '会议': {results}")
    assert len(results) > 0

    # 测试增量更新
    retriever.index("doc6", "北京烤鸭很好吃")
    results = retriever.search("烤鸭", top_k=3)
    print(f"查询 '烤鸭': {results}")
    assert len(results) > 0

    # 测试删除
    retriever.remove("doc1")
    results = retriever.search("烤鸭", top_k=3)
    print(f"删除 doc1 后查询 '烤鸭': {results}")
    assert len(results) > 0

    print("✅ Dense 向量检索器测试通过\n")


def test_rrf_fusion():
    """测试 RRF 融合"""
    print("=== 测试 RRF 融合 ===")

    # 模拟两个检索器的结果
    ranking1 = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
    ranking2 = [("doc2", 0.95), ("doc3", 0.85), ("doc4", 0.75)]

    # 测试普通 RRF
    fused = reciprocal_rank_fusion([ranking1, ranking2], k=60)
    print(f"普通 RRF: {fused}")

    # doc1 和 doc2 都在排名中，但 doc2 在两个排名中都靠前
    # 验证融合结果包含所有文档
    doc_ids = [doc_id for doc_id, _ in fused]
    assert "doc1" in doc_ids
    assert "doc2" in doc_ids
    assert "doc3" in doc_ids
    assert "doc4" in doc_ids

    # 测试加权 RRF
    fused_weighted = weighted_reciprocal_rank_fusion(
        [ranking1, ranking2],
        weights=[0.6, 0.4],
        k=60,
    )
    print(f"加权 RRF: {fused_weighted}")

    print("✅ RRF 融合测试通过\n")


def test_query_planner():
    """测试 Query Planner"""
    print("=== 测试 Query Planner ===")

    # 创建检索器
    bm25 = BM25Retriever()
    dense = DenseRetriever(dimension=128)

    # 创建一个简单的 mock core layer
    class MockCoreLayer:
        def __init__(self):
            self._chunks = {}

        def get(self, chunk_id):
            return self._chunks.get(chunk_id)

        def calc_weight(self, chunk):
            class MockWeightFactors:
                def __init__(self):
                    self.final = 0.5
            return MockWeightFactors()

        def _select_candidates(self, query_tags):
            return list(self._chunks.keys())

    mock_core = MockCoreLayer()

    # 创建 QueryPlanner
    planner = QueryPlanner(
        core_layer=mock_core,
        bm25=bm25,
        dense=dense,
        metadata_weight=0.0,  # 不使用 metadata filter
        bm25_weight=0.5,
        dense_weight=0.5,
    )

    # 创建测试数据
    chunks = {
        "chunk1": make_chunk("chunk1", "今天中午在北京吃了烤鸭", location="北京"),
        "chunk2": make_chunk("chunk2", "昨天在上海参加了技术会议", location="上海"),
        "chunk3": make_chunk("chunk3", "上周去北京出差见了客户", location="北京"),
        "chunk4": make_chunk("chunk4", "在家里看了一部电影", location="家里"),
        "chunk5": make_chunk("chunk5", "北京的天气很好", location="北京"),
    }

    # 添加到 mock core
    mock_core._chunks.update(chunks)

    # 索引所有 chunks
    planner.index_chunks(chunks)

    # 测试检索
    results = planner.plan_and_retrieve(
        query="北京",
        limit=3,
    )
    print(f"查询 '北京': {[(c.id, s) for c, s in results]}")
    assert len(results) > 0, f"查询 '北京' 返回空结果"

    results = planner.plan_and_retrieve(
        query="技术会议",
        limit=3,
    )
    print(f"查询 '技术会议': {[(c.id, s) for c, s in results]}")
    assert len(results) > 0, f"查询 '技术会议' 返回空结果"

    # 测试增量更新
    chunk6 = make_chunk("chunk6", "北京烤鸭很好吃", location="北京")
    planner.add_chunk(chunk6)
    mock_core._chunks["chunk6"] = chunk6
    results = planner.plan_and_retrieve(
        query="烤鸭",
        limit=3,
    )
    print(f"查询 '烤鸭': {[(c.id, s) for c, s in results]}")
    assert len(results) > 0, f"查询 '烤鸭' 返回空结果"

    # 测试删除
    planner.remove_chunk("chunk6")
    del mock_core._chunks["chunk6"]
    results = planner.plan_and_retrieve(
        query="烤鸭",
        limit=3,
    )
    print(f"删除 chunk6 后查询 '烤鸭': {[(c.id, s) for c, s in results]}")

    # 测试统计
    stats = planner.get_stats()
    print(f"统计信息: {stats}")

    print("✅ Query Planner 测试通过\n")


def test_hybrid_vs_traditional():
    """测试混合检索 vs 传统检索"""
    print("=== 测试混合检索 vs 传统检索 ===")

    # 创建测试数据
    chunks = {
        "chunk1": make_chunk("chunk1", "今天中午在北京吃了烤鸭", location="北京"),
        "chunk2": make_chunk("chunk2", "昨天在上海参加了技术会议", location="上海"),
        "chunk3": make_chunk("chunk3", "上周去北京出差见了客户", location="北京"),
        "chunk4": make_chunk("chunk4", "在家里看了一部电影", location="家里"),
        "chunk5": make_chunk("chunk5", "北京的天气很好", location="北京"),
        "chunk6": make_chunk("chunk6", "烤鸭是北京的特色美食", location="北京"),
    }

    # 传统检索（只有 metadata filter）
    print("传统检索（metadata only）:")
    for query in ["北京", "烤鸭", "会议"]:
        # 模拟 metadata filter
        results = []
        for chunk_id, chunk in chunks.items():
            if chunk.location and chunk.location in query:
                results.append((chunk_id, 1.0))
        print(f"  查询 '{query}': {results}")

    # 混合检索
    print("\n混合检索（BM25 + Dense）:")
    bm25 = BM25Retriever()
    dense = DenseRetriever(dimension=128)
    planner = QueryPlanner(bm25=bm25, dense=dense)
    planner.index_chunks(chunks)

    for query in ["北京", "烤鸭", "会议"]:
        results = planner.plan_and_retrieve(query=query, limit=3)
        print(f"  查询 '{query}': {[(c.id, f'{s:.3f}') for c, s in results]}")

    print("\n✅ 混合检索 vs 传统检索测试通过\n")


if __name__ == "__main__":
    test_bm25_retriever()
    test_dense_retriever()
    test_rrf_fusion()
    test_query_planner()
    test_hybrid_vs_traditional()
    print("🎉 所有测试通过！")
