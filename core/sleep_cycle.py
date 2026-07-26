"""
确定性睡眠周期 - 优先回放的情景到语义巩固

人脑在睡眠中回放白天的经历，把零散的情景记忆抽象成持久的语义
知识——个体情景逐渐淡忘，而"要点"留下来。本模块是它的确定性
版本（融合 CLS 理论、Letta sleep-time compute、Generative Agents
反思触发器的可移植内核，全部无 LLM、可审计）：

    1. 优先回放选择：按 重要性 + 情绪强度 + 近因 + 惊奇度 打分，
       取 top-N 情景记忆（INTERACTION / STORY）
    2. 锚点聚类：共享人物/主题（Jaccard >= 0.25）或 Hebbian 强边
       （>= 0.3）连通分量，规模 >= 3 才值得抽象
    3. 要点合成：池化簇内关键词，按词频给句子打分，抽取 top-3 句
       构成语义记忆（IDEA 类型，衰减最慢档之一），逐句分数进审计
       记录——每一行摘要都能追溯到来源句
    4. 来源归档：情景记忆降入伪遗忘层（metadata.consolidated_into
       指向要点），线索仍可唤醒——抽象是可逆的，没有任何销毁
    5. 图卫生：修剪近零 Hebbian 边，限制单点边数

触发（全部确定性）：main.py 维护重要性累积器，写入累计重要性
达到阈值（Generative Agents 的 150 按 0~1 重要性折算为 7.5）时
在下一次 maintain() 中触发；也可显式调用 system.sleep()。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from memory_chunk import MemoryChunk
from core.weight_system import MemoryType
from core.supersession import _tokenize

# 参与回放的情景类型（语义类型 FACT/PREFERENCE/IDEA 不再抽象）
EPISODIC_TYPES = (MemoryType.INTERACTION, MemoryType.STORY)

# 聚类参数
CLUSTER_ANCHOR_JACCARD = 0.25   # 人物∪主题 的最低重叠
CLUSTER_EDGE_MIN = 0.3          # 或 Hebbian 边强度达到此值
MIN_CLUSTER_SIZE = 3            # 少于 3 条不值得抽象

# 图卫生参数
EDGE_PRUNE_EPSILON = 0.05       # 低于此强度的边直接修剪
MAX_EDGES_PER_CHUNK = 20        # 单点边数上限（保留最强的）

# 要点句数
GIST_SENTENCES = 3

# 图式强化：新簇与既有要点的锚点 Jaccard 达到此值时，增强既有
# 要点而不是重复抽象。阈值高于聚类阈值（0.25）——归并要点比归并
# 情景要求更强的主题一致性，避免不同主题被吞进同一条要点。
GIST_REINFORCE_JACCARD = 0.5


@dataclass
class SleepReport:
    """一次睡眠周期的完整审计记录"""
    replayed: int = 0               # 参与回放的情景数
    clusters: int = 0               # 形成的簇数
    gists_created: List[str] = field(default_factory=list)   # 新要点 id
    gists_reinforced: List[str] = field(default_factory=list)  # 被强化的既有要点
    sources_archived: List[str] = field(default_factory=list)
    edges_pruned: int = 0
    details: List[Dict] = field(default_factory=list)  # 每簇的可溯源明细

    def summary(self) -> str:
        return (
            f"replayed={self.replayed} clusters={self.clusters} "
            f"gists={len(self.gists_created)} "
            f"reinforced={len(self.gists_reinforced)} "
            f"archived={len(self.sources_archived)} "
            f"edges_pruned={self.edges_pruned}"
        )


class SleepCycle:
    """确定性睡眠巩固"""

    def __init__(self, core_layer, forgotten_layer, planner=None,
                 replay_limit: int = 50):
        self.core = core_layer
        self.forgotten = forgotten_layer
        self.planner = planner
        self.replay_limit = replay_limit

    # ---------- 1. 优先回放选择 ----------

    def _replay_priority(self, chunk: MemoryChunk, now: float) -> float:
        """回放优先级：重要 + 情绪化 + 最近用过 + 出乎意料 的经历先回放"""
        recency = 1.0 / (1.0 + (now - chunk.last_accessed) / (7 * 24 * 3600))
        surprise = float(chunk.metadata.get("encoding_surprise", 0.0))
        return (
            chunk.importance
            + abs(chunk.emotion_valence) * chunk.emotion_intensity
            + 0.5 * recency
            + 0.5 * surprise
        )

    def _select_replay(self, now: float) -> List[MemoryChunk]:
        candidates = [
            c for c in self.core.chunks.values()
            if c.memory_type in EPISODIC_TYPES
            and not c.metadata.get("consolidated_into")
        ]
        candidates.sort(key=lambda c: (-self._replay_priority(c, now), c.id))
        return candidates[: self.replay_limit]

    # ---------- 2. 锚点聚类（连通分量） ----------

    @staticmethod
    def _anchors(chunk: MemoryChunk) -> Set[str]:
        return set(chunk.persons) | set(chunk.topics)

    def _related(self, a: MemoryChunk, b: MemoryChunk) -> bool:
        anchors_a, anchors_b = self._anchors(a), self._anchors(b)
        if anchors_a and anchors_b:
            union = anchors_a | anchors_b
            if len(anchors_a & anchors_b) / len(union) >= CLUSTER_ANCHOR_JACCARD:
                return True
        return a.associations.get(b.id, 0.0) >= CLUSTER_EDGE_MIN

    def _cluster(self, chunks: List[MemoryChunk]) -> List[List[MemoryChunk]]:
        n = len(chunks)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(n):
            for j in range(i + 1, n):
                if self._related(chunks[i], chunks[j]):
                    parent[find(i)] = find(j)

        groups: Dict[int, List[MemoryChunk]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(chunks[i])
        clusters = [g for g in groups.values() if len(g) >= MIN_CLUSTER_SIZE]
        # 时间顺序，让要点叙述有方向
        for g in clusters:
            g.sort(key=lambda c: c.created_at)
        return sorted(clusters, key=lambda g: g[0].created_at)

    # ---------- 3. 抽取式要点合成 ----------

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r"[。！？!?；;\n]+", text)
        return [p.strip() for p in parts if p.strip()]

    def _synthesize_gist(self, cluster: List[MemoryChunk], llm_fn=None) -> Dict:
        """
        池化簇内关键词 -> 按关键词词频给每个来源句打分 -> 抽取 top 句。
        返回 {content, sentence_scores}；每一行摘要都可追溯到来源句。
        """
        pooled: Dict[str, int] = {}
        for chunk in cluster:
            for token in _tokenize(chunk.content) | set(chunk.keywords):
                pooled[token] = pooled.get(token, 0) + 1

        scored = []
        for chunk in cluster:
            for sentence in self._split_sentences(chunk.content):
                tokens = _tokenize(sentence)
                if not tokens:
                    continue
                score = sum(pooled.get(t, 0) for t in tokens) / len(tokens)
                scored.append({
                    "sentence": sentence,
                    "score": round(score, 4),
                    "source_id": chunk.id,
                })
        scored.sort(key=lambda s: (-s["score"], s["source_id"]))

        top = []
        seen_sentences = set()
        for item in scored:
            if item["sentence"] in seen_sentences:
                continue
            seen_sentences.add(item["sentence"])
            top.append(item)
            if len(top) >= GIST_SENTENCES:
                break

        body = "；".join(item["sentence"] for item in top)
        content = f"经验要点（{len(cluster)}次相关经历）：{body}"

        # llm_fn 只允许润色文字，不允许改变事实来源
        if llm_fn is not None:
            try:
                polished = llm_fn(
                    "把下面的记忆要点改写成一段连贯的话，"
                    f"不得增删事实：{body}"
                )
                if polished and polished.strip():
                    content = f"经验要点（{len(cluster)}次相关经历）：{polished.strip()}"
            except Exception:
                pass  # 润色失败保留抽取式版本

        return {"content": content, "sentence_scores": top}

    # ---------- 3.5 图式强化：新簇归并进既有要点 ----------

    def _gist_candidates(self) -> List[MemoryChunk]:
        """既有要点候选，一次睡眠只取一遍（SQLite 后端 get_all 是
        全表反序列化，逐簇全库扫描会成为大库睡眠的主要开销）"""
        return [
            c for c in self.core.chunks.values()
            if c.memory_type == MemoryType.IDEA and c.source == "consolidation"
        ]

    def _find_matching_gist(
        self,
        cluster: List[MemoryChunk],
        candidates: List[MemoryChunk],
    ) -> Optional[MemoryChunk]:
        """
        找与新簇主题一致的既有要点（图式强化，要点支持计数）。

        记忆科学依据：后续匹配情景应该增强既有图式（schema
        reinforcement），而不是每次睡眠都重复抽象出一条新要点——
        否则"和老王开会"这类反复出现的主题每周都会多一条近重复
        的要点，语义层被自己的抽象淹没。
        """
        anchors: Set[str] = set()
        for c in cluster:
            anchors |= self._anchors(c)
        if not anchors:
            return None

        best, best_sim = None, 0.0
        for other in candidates:
            if other.user_id != cluster[0].user_id:
                continue
            gist_anchors = self._anchors(other)
            if not gist_anchors:
                continue
            sim = len(anchors & gist_anchors) / len(anchors | gist_anchors)
            if sim > best_sim or (sim == best_sim and best and other.id < best.id):
                best, best_sim = other, sim
        return best if best_sim >= GIST_REINFORCE_JACCARD else None

    def _reinforce_gist(
        self,
        gist: MemoryChunk,
        cluster: List[MemoryChunk],
        now: float,
        report: SleepReport,
    ) -> None:
        """支持计数 + 重要性随支持情景总数增长 + 来源并入归档"""
        prev_total = int(gist.metadata.get(
            "supporting_episodes", len(gist.metadata.get("source_ids", []))
        ))
        total = prev_total + len(cluster)
        gist.metadata["supporting_episodes"] = total
        gist.metadata["support_count"] = int(gist.metadata.get("support_count", 1)) + 1
        gist.metadata.setdefault("reinforced_at", []).append(now)
        gist.metadata["source_ids"] = list(gist.metadata.get("source_ids", [])) + [
            c.id for c in cluster
        ]
        # 重要性与内容前缀都由支持情景总数决定（与新建要点同一公式）
        gist.importance = min(1.0, 0.6 + 0.05 * total)
        gist.content = re.sub(
            r"^经验要点（\d+次相关经历）：",
            f"经验要点（{total}次相关经历）：",
            gist.content,
        )
        for c in cluster:
            gist.topics |= c.topics
            gist.persons |= c.persons
            gist.keywords |= c.keywords
            c.metadata["consolidated_into"] = gist.id
            self.core.remove(c.id)
            if self.planner:
                self.planner.remove_chunk(c.id)
            self.forgotten.archive(c)
            report.sources_archived.append(c.id)
            gist.associations[c.id] = 0.6
        gist.updated_at = now
        # 必须走 core.add 而不是裸 _store.put：锚点集合变了，核心层
        # 内存倒排索引（topic_index/person_index）只在 add() 时按
        # _index_keys 快照回滚重建——裸 put 会让新锚点在常驻进程里
        # 检索不到直到重启（对抗审查在双后端实证复现的缺陷）
        self.core.add(gist)
        if self.planner:
            # 锚点/内容有变，刷新 BM25/Dense 检索索引
            self.planner.remove_chunk(gist.id)
            self.planner.add_chunk(gist)
        report.gists_reinforced.append(gist.id)
        report.details.append({
            "gist_id": gist.id,
            "reinforced": True,
            "cluster_size": len(cluster),
            "source_ids": [c.id for c in cluster],
            "supporting_episodes": total,
            "support_count": gist.metadata["support_count"],
        })

    # ---------- 4/5. 主流程 ----------

    def sleep(self, llm_fn=None, now: Optional[float] = None) -> SleepReport:
        """执行一次睡眠巩固，返回完整审计报告"""
        now = now or time.time()
        report = SleepReport()

        replayed = self._select_replay(now)
        report.replayed = len(replayed)

        clusters = self._cluster(replayed)
        report.clusters = len(clusters)

        # 既有要点候选一次取够；本次新建/强化的要点就地维护进列表，
        # 后续簇仍能匹配到最新状态
        candidates = self._gist_candidates()

        for cluster in clusters:
            # 图式强化优先：同主题的既有要点被增强而不是重复抽象
            existing = self._find_matching_gist(cluster, candidates)
            if existing is not None:
                self._reinforce_gist(existing, cluster, now, report)
                continue

            gist_info = self._synthesize_gist(cluster, llm_fn=llm_fn)

            pooled_topics: Set[str] = set()
            pooled_persons: Set[str] = set()
            pooled_keywords: Set[str] = set()
            for c in cluster:
                pooled_topics |= c.topics
                pooled_persons |= c.persons
                pooled_keywords |= c.keywords

            gist = MemoryChunk(
                content=gist_info["content"],
                memory_type=MemoryType.IDEA,  # 语义要点：慢衰减档
                topics=pooled_topics,
                persons=pooled_persons,
                keywords=pooled_keywords,
                importance=min(1.0, 0.6 + 0.05 * len(cluster)),
                source="consolidation",
                user_id=cluster[0].user_id,
                metadata={
                    "source_ids": [c.id for c in cluster],
                    "sentence_scores": gist_info["sentence_scores"],
                    "consolidated_at": now,
                    # 图式强化的计数起点（后续匹配簇会累加）
                    "supporting_episodes": len(cluster),
                    "support_count": 1,
                },
            )
            self.core.add(gist)
            if self.planner:
                self.planner.add_chunk(gist)
            report.gists_created.append(gist.id)
            candidates.append(gist)

            # 来源归档：抽象是可逆的——线索仍可唤醒具体情景
            for c in cluster:
                c.metadata["consolidated_into"] = gist.id
                self.core.remove(c.id)
                if self.planner:
                    self.planner.remove_chunk(c.id)
                self.forgotten.archive(c)
                report.sources_archived.append(c.id)
                # 要点与来源保持关联：联想回忆可以从要点走回情景
                gist.associations[c.id] = 0.6
            self.core._store.put(gist)

            report.details.append({
                "gist_id": gist.id,
                "cluster_size": len(cluster),
                "source_ids": [c.id for c in cluster],
                "sentences": gist_info["sentence_scores"],
            })

        report.edges_pruned = self._prune_graph()
        return report

    def _prune_graph(self) -> int:
        """Hebbian 图卫生：修剪近零边，限制单点边数"""
        pruned = 0
        for chunk in list(self.core.chunks.values()):
            edges = chunk.associations
            weak = [k for k, v in edges.items() if v < EDGE_PRUNE_EPSILON]
            for k in weak:
                del edges[k]
            pruned += len(weak)
            if len(edges) > MAX_EDGES_PER_CHUNK:
                keep = sorted(edges.items(), key=lambda kv: (-kv[1], kv[0]))
                removed = len(edges) - MAX_EDGES_PER_CHUNK
                chunk.associations = dict(keep[:MAX_EDGES_PER_CHUNK])
                pruned += removed
            if weak or len(edges) > MAX_EDGES_PER_CHUNK:
                self.core._store.put(chunk)
                self.core._invalidate_weight(chunk.id)
        return pruned
