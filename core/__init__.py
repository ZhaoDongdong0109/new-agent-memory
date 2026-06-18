"""Internal core modules for new-agent-memory.

Public imports are exposed from `new_agent_memory`.
Keeping this package initializer lightweight avoids circular imports between
memory chunks, attention, and agent runtime modules.
"""

# 使用延迟导入避免循环依赖
# 具体的类在需要时通过 from core.xxx import YYY 导入

__all__ = [
    "MemoryStore",
    "IndexStore",
    "KeyValueStore",
    "JsonMemoryStore",
    "JsonKeyValueStore",
    "SqliteMemoryStore",
    "BM25Retriever",
    "DenseRetriever",
    "QueryPlanner",
    "reciprocal_rank_fusion",
    "MemorySpec",
    "ExtractionResult",
    "EntityExtractor",
    "MemoryExtractor",
    "ConsolidationEngine",
    "ContextCompressor",
    "PIIHandler",
    "DataManager",
    "AuditLogger",
]


def __getattr__(name):
    """延迟导入，避免循环依赖"""
    if name in ("MemoryStore", "IndexStore", "KeyValueStore"):
        from core.store import MemoryStore, IndexStore, KeyValueStore
        return {"MemoryStore": MemoryStore, "IndexStore": IndexStore, "KeyValueStore": KeyValueStore}[name]
    elif name in ("JsonMemoryStore", "JsonKeyValueStore"):
        from core.json_store import JsonMemoryStore, JsonKeyValueStore
        return {"JsonMemoryStore": JsonMemoryStore, "JsonKeyValueStore": JsonKeyValueStore}[name]
    elif name == "SqliteMemoryStore":
        from core.sqlite_store import SqliteMemoryStore
        return SqliteMemoryStore
    elif name == "BM25Retriever":
        from core.bm25_retriever import BM25Retriever
        return BM25Retriever
    elif name == "DenseRetriever":
        from core.dense_retriever import DenseRetriever
        return DenseRetriever
    elif name == "QueryPlanner":
        from core.query_planner import QueryPlanner
        return QueryPlanner
    elif name == "reciprocal_rank_fusion":
        from core.rrf_fusion import reciprocal_rank_fusion
        return reciprocal_rank_fusion
    elif name in ("MemorySpec", "ExtractionResult"):
        from core.memory_spec import MemorySpec, ExtractionResult
        return {"MemorySpec": MemorySpec, "ExtractionResult": ExtractionResult}[name]
    elif name == "EntityExtractor":
        from core.entity_extractor import EntityExtractor
        return EntityExtractor
    elif name == "MemoryExtractor":
        from core.memory_extractor import MemoryExtractor
        return MemoryExtractor
    elif name == "ConsolidationEngine":
        from core.consolidation_engine import ConsolidationEngine
        return ConsolidationEngine
    elif name == "ContextCompressor":
        from core.context_compressor import ContextCompressor
        return ContextCompressor
    elif name == "PIIHandler":
        from core.pii_handler import PIIHandler
        return PIIHandler
    elif name == "DataManager":
        from core.data_manager import DataManager
        return DataManager
    elif name == "AuditLogger":
        from core.audit_logger import AuditLogger
        return AuditLogger
    raise AttributeError(f"module 'core' has no attribute {name!r}")
