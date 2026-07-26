"""
双时态事实取代 - 写入时的确定性决策表

行业现状（2025-2026 调研）：知识更新是所有已发表记忆系统最弱的能力
（MemoryAgentBench 显示 SOTA 在多跳整合上接近随机）。主流方案
（Mem0）用 LLM 工具调用做写入仲裁——黑盒，不可复现。

本模块是确定性的版本：每次写入 FACT / PREFERENCE 记忆时，
与既有记忆比对后从具名规则表中选择一个操作：

    NOOP       近重复：强化既有记忆，不新增
    UPDATE     同一事实槽的扩展：就地更新，旧内容进 history
    SUPERSEDE  同一事实槽的新值：旧记忆 invalid_at 置为现在，
               归档到伪遗忘层（"过时"成为伪遗忘的一种正当理由），
               新记忆 parent_id 指向旧记忆——完整的双时态链条
    ADD        新事实：正常写入，重要性按惊奇度缩放

这激活了 schema 中一直休眠的字段：valid_at / invalid_at / version /
parent_id。"记得你以前住在里斯本"从此是真实行为：
现在时查询只返回当前有效事实，过去时查询可以唤醒被取代的旧事实。

每个决策都带 rule_id 与相似度分数，可完整审计复现。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional, Set

from memory_chunk import MemoryChunk
from core.weight_system import MemoryType

# 参与取代管理的记忆类型：事实与偏好有"当前值"语义；
# 故事/交互是事件记录，不存在"被新值取代"
MANAGED_TYPES = (MemoryType.FACT, MemoryType.PREFERENCE)

# 决策阈值（具名，可解释）
NOOP_JACCARD = 0.8        # 近重复判定
SLOT_MIN_JACCARD = 0.25   # 低于此相似度视为不同事实，直接 ADD
UPDATE_CONTAINMENT = 0.8  # 旧内容被新内容覆盖到此比例视为扩展


def _tokenize(text: str) -> Set[str]:
    """与注意力层一致的中英混合分词（词 + 中文 bigram）"""
    if not text:
        return set()
    lowered = text.lower()
    tokens = set(re.findall(r"[a-zA-Z0-9_]+", lowered))
    spans = re.findall(r"[一-鿿]{2,}", lowered)
    tokens.update(spans)
    for span in spans:
        tokens.update(span[i:i + 2] for i in range(max(0, len(span) - 1)))
    return tokens


def _chunk_tokens(chunk: MemoryChunk) -> Set[str]:
    tokens = _tokenize(chunk.content)
    for kw in chunk.keywords:
        tokens |= _tokenize(kw)
    return tokens


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _containment(inner: Set[str], outer: Set[str]) -> float:
    """inner 被 outer 覆盖的比例"""
    if not inner:
        return 0.0
    return len(inner & outer) / len(inner)


@dataclass
class Decision:
    """一次写入决策（完整可审计）"""
    op: str                      # noop / update / supersede / add
    rule_id: str                 # 命中的具名规则
    target_id: Optional[str]     # noop/update/supersede 的目标记忆
    similarity: float            # 与最相似候选的 Jaccard
    surprise: float              # 1 - max_similarity（惊奇度）

    def to_audit(self) -> dict:
        return {
            "op": self.op,
            "rule_id": self.rule_id,
            "target_id": self.target_id,
            "similarity": round(self.similarity, 4),
            "surprise": round(self.surprise, 4),
        }


class SupersessionEngine:
    """写入时事实取代决策引擎"""

    def __init__(self, core_layer, max_candidates: int = 10):
        self.core = core_layer
        self.max_candidates = max_candidates

    def _slot_candidates(self, chunk: MemoryChunk, now: float):
        """
        同一"事实槽"的候选：同类型、同用户、当前有效、
        且共享至少一个锚点（人物或主题）。
        """
        candidates = []
        for other in self.core.chunks.values():
            if other.id == chunk.id:
                continue
            if other.memory_type != chunk.memory_type:
                continue
            if other.user_id != chunk.user_id:
                continue
            if other.invalid_at is not None and other.invalid_at <= now:
                continue  # 已被取代的事实不再参与槽竞争

            person_overlap = bool(chunk.persons & other.persons)
            topic_overlap = bool(chunk.topics & other.topics)
            if chunk.persons and other.persons:
                if not person_overlap:
                    continue
            elif not topic_overlap:
                continue

            candidates.append(other)
        return candidates

    def decide(self, chunk: MemoryChunk, now: Optional[float] = None) -> Decision:
        """
        为一条待写入的 FACT/PREFERENCE 记忆选择操作。

        非管理类型直接 ADD（惊奇度 1.0，不缩放）。
        """
        now = now or time.time()

        if chunk.memory_type not in MANAGED_TYPES:
            return Decision(op="add", rule_id="R0_unmanaged_type",
                            target_id=None, similarity=0.0, surprise=1.0)

        new_tokens = _chunk_tokens(chunk)
        best = None
        best_sim = 0.0
        for other in self._slot_candidates(chunk, now):
            sim = _jaccard(new_tokens, _chunk_tokens(other))
            if sim > best_sim:
                best, best_sim = other, sim

        surprise = 1.0 - best_sim

        if best is None or best_sim < SLOT_MIN_JACCARD:
            return Decision(op="add", rule_id="R4_new_fact",
                            target_id=None, similarity=best_sim, surprise=surprise)

        if best_sim >= NOOP_JACCARD:
            return Decision(op="noop", rule_id="R1_near_duplicate",
                            target_id=best.id, similarity=best_sim, surprise=surprise)

        old_tokens = _chunk_tokens(best)
        if _containment(old_tokens, new_tokens) >= UPDATE_CONTAINMENT:
            # 新内容完整覆盖旧内容并有扩展 -> 同一事实的更完整版本
            return Decision(op="update", rule_id="R2_extension",
                            target_id=best.id, similarity=best_sim, surprise=surprise)

        # 同一事实槽、内容显著不同 -> 新值取代旧值
        return Decision(op="supersede", rule_id="R3_value_change",
                        target_id=best.id, similarity=best_sim, surprise=surprise)


def surprise_scaled_importance(importance: float, surprise: float) -> float:
    """
    惊奇度门控编码（Titans 思想的可移植内核）：
    越出乎意料的信息越值得记住。scale ∈ [0.7, 1.3]，确定性可打印。
    """
    return max(0.0, min(1.0, importance * (0.7 + 0.6 * surprise)))
