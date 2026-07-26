"""
检索与重建模块 - 类人记忆系统

核心流程：
1. 解析输入（问句/照片/外部信息）→ 信息锚点
2. 核心层检索
3. 若失败 → 伪遗忘层唤醒（信息锚点触发）
4. 碎片重组
5. 审阅（判断记忆是否合理）
6. 输出
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import time
import re

from memory_chunk import MemoryChunk, MemoryLayer
from memory_layer_core import MemoryLayerCore
from forgotten_layer import ForgottenLayer
from core.topic_vocab import expand_topics, extract_query_topics


class ReviewResult(Enum):
    """审阅结果"""
    APPROVED = "approved"           # 合理，直接输出
    MODIFIED = "modified"           # 需要修正
    QUESTIONABLE = "questionable"   # 存疑，谨慎输出
    REJECTED = "rejected"           # 不合理，标记


@dataclass
class QueryContext:
    """查询上下文（信息锚点）"""
    # 原始输入
    raw_query: str = ""
    
    # 时间维度
    time_absolute: Optional[str] = None   # "2026-04-29"
    time_relative: Optional[str] = None   # "10年前", "昨天"
    time_context: Optional[str] = None    # "中午", "出差时"
    
    # 地点维度
    location: Optional[str] = None
    
    # 人物维度
    persons: Set[str] = None
    
    # 主题/语义维度
    topics: Set[str] = None
    keywords: Set[str] = None
    
    # 情绪维度
    emotion_valence: Optional[float] = None

    # 时间窗口与时态（core/time_parser.py 解析）
    time_window: Optional[Tuple[float, float]] = None  # [t_start, t_end] epoch 秒
    tense: Optional[str] = None  # "present" / "past" / None

    # 元信息
    source_type: str = "query"  # query / photo / audio / ...
    
    def __post_init__(self):
        self.persons = self.persons or set()
        self.topics = self.topics or set()
        self.keywords = self.keywords or set()
    
    def to_tags(self) -> Dict[str, Any]:
        """转换为检索标签"""
        tags = {}
        if self.time_absolute:
            tags["time_absolute"] = self.time_absolute
        if self.time_relative:
            tags["time_relative"] = self.time_relative
        if self.time_context:
            tags["time_context"] = self.time_context
        if self.location:
            tags["location"] = self.location
        if self.persons:
            tags["persons"] = self.persons
        if self.topics:
            tags["topics"] = self.topics
        if self.emotion_valence is not None:
            tags["emotion_valence"] = self.emotion_valence
        return tags


@dataclass
class ReconstructionResult:
    """重建结果"""
    success: bool
    chunks: List[MemoryChunk]          # 参与的碎片
    assembled_content: str             # 组装后的内容
    review_result: ReviewResult        # 审阅结果
    review_note: str = ""              # 审阅备注
    retrieval_path: str = ""          # 检索路径：core / forgotten / both
    confidence: float = 0.0           # 置信度 0~1
    
    def summary(self) -> str:
        return (
            f"[{self.review_result.value}] "
            f"path={self.retrieval_path} "
            f"chunks={len(self.chunks)} "
            f"confidence={self.confidence:.2f} | "
            f"{self.assembled_content[:100]}"
        )


class MemoryRetrieval:
    """
    记忆检索与重建系统

    工作流程：
    1. 解析输入 → QueryContext
    2. 核心层检索（支持混合检索）
    3. 若失败 → 伪遗忘层唤醒
    4. 碎片组装
    5. 审阅
    6. 输出

    重构：支持混合检索（BM25 + Dense + Metadata）
    """

    def __init__(
        self,
        core_layer: MemoryLayerCore,
        forgotten_layer: ForgottenLayer,

        # 混合检索器（可选）
        planner=None,

        # 检索参数
        core_min_weight: float = 0.2,
        core_limit: int = 10,
        forgotten_min_match: int = 2,

        # 组装参数
        assembly_method: str = "chronological",  # chronological / relevance / hybrid

        # 审阅参数
        review_confidence_threshold: float = 0.5,  # 低于此值标记为 questionable

        # 唤醒提升参数：唤醒临时权重达到该值的记忆会被提升回核心层
        promote_threshold: float = 0.55,

        # 联想唤醒提升参数：扩散激活达到该值的归档记忆会被提升
        # （1 跳、边权 0.5 时激活为 0.25——只有牢固关联才够格）
        assoc_promote_threshold: float = 0.25,
    ):
        self.core = core_layer
        self.forgotten = forgotten_layer
        self.planner = planner  # QueryPlanner 实例
        self.core_min_weight = core_min_weight
        self.core_limit = core_limit
        self.forgotten_min_match = forgotten_min_match
        self.assembly_method = assembly_method
        self.review_confidence_threshold = review_confidence_threshold
        self.promote_threshold = promote_threshold
        self.assoc_promote_threshold = assoc_promote_threshold

        # 统计
        self.total_retrievals = 0
        self.core_hit = 0
        self.forgotten_hit = 0
        self.both_hit = 0
        self.total_promoted = 0
        self.total_assoc_recalled = 0
        self.total_assoc_wakes = 0

        # 最近一次扩散激活的轨迹（联想回忆的可解释审计）
        self.last_activation_trace: List[Tuple[str, float]] = []

    # 词汇覆盖率警戒阈值。
    #
    # 狗粮期实测教训：曾按"负例 0.38 vs 最低正例 0.54"设硬弃答阈值
    # 0.45，随即被真实改述查询打脸——"中英文词汇割裂问题"（正例，
    # 记忆原文说的是"零交集/双语同义词表"）覆盖率仅 0.29，比负例
    # 还低。无嵌入的词面覆盖不可作硬弃答依据，只能做诚实标注：
    # 低覆盖 -> QUESTIONABLE + 明确警告，把判断权交给调用方。
    VOCAB_CAUTION_BELOW = 0.60   # 低于此值：低覆盖，降置信度并标注
    VOCAB_WARNING_BELOW = 0.45   # 低于此值：强警告（结果很可能不相关）

    def _vocab_coverage(self, query: str) -> Optional[float]:
        """
        查询词汇被记忆库索引覆盖的比例。

        BM25 的中文 bigram 几乎总能部分匹配到点什么（虚词碎片），
        导致库里完全没有的主题也会得到一个"自信"的回答——狗粮期
        实测弃答正确率为 0。词汇覆盖率是确定性的界外检测信号：
        查询的大部分 token 从未出现在任何记忆里，说明这个主题
        根本不在库中，任何命中都是碎片噪声。
        """
        if not self.planner or not getattr(self.planner, "bm25", None):
            return None
        bm25 = self.planner.bm25
        if not bm25.total_docs:
            return None
        tokens = bm25._tokenize(query)
        if not tokens:
            return None
        known = sum(1 for t in set(tokens) if t in bm25.doc_freqs)
        return known / len(set(tokens))

    def promote_woken(self, forgotten_results: List[tuple]) -> List[MemoryChunk]:
        """
        把唤醒结果中锚点足够强的记忆提升回核心层，
        返回完整的唤醒列表（被提升的条目已替换为核心层的新对象）。

        这是"遗忘-唤醒"生命周期的关键闭环：
        降级(maintain) -> 归档(forgotten) -> 线索唤醒(try_wake)
          -> 提升(promote) -> 重新进入核心层与检索索引

        弱唤醒（低于 promote_threshold）保持归档状态，但 try_wake
        已经为它们记录了唤醒痕迹。

        注意必须使用返回值而不是原 forgotten_results 里的对象：
        SQLite 后端返回副本，提升后原对象的 layer 标记已经过期。
        """
        ids = [
            chunk.id for chunk, temp_weight in forgotten_results
            if temp_weight >= self.promote_threshold
        ]
        promoted = {c.id: c for c in self.forgotten.promote(ids)}
        for chunk in promoted.values():
            self.core.add(chunk)
            # 提升即访问：给再巩固的记忆一个新鲜的近因信号
            self.core.access(chunk.id)
            if self.planner:
                self.planner.add_chunk(chunk)
        self.total_promoted += len(promoted)
        # 被提升的条目替换为核心层新对象，未提升的保留原对象
        return [promoted.get(chunk.id, chunk) for chunk, _tw in forgotten_results]
    
    # ============ 查询解析 ============
    
    def parse_query(self, query: str) -> QueryContext:
        """
        解析自然语言查询为 QueryContext
        
        目前是简化版规则解析，未来可以换成LLM
        """
        ctx = QueryContext(raw_query=query)

        # 相对时间解析
        #
        # 注意：保留原始表述（"10年前"），不换算成"2016年"这类年份字符串——
        # 写入侧存储的就是用户给出的原始相对时间，换算后的表述任何写入路径
        # 都不会存储，等值匹配将永远失败。
        time_relative_patterns = [
            r"\d+年前",
            r"昨天",
            r"上周",
            r"上个月",
            r"去年",
        ]

        for pattern in time_relative_patterns:
            match = re.search(pattern, query)
            if match:
                ctx.time_relative = match.group(0)
                break

        # 时间上下文解析
        if "中午" in query or "午饭" in query or "午餐" in query:
            ctx.time_context = ctx.time_context or "中午"

        # 地点解析（简化）
        locations = ["北京", "上海", "家里", "公司", "餐厅", "酒店", "机场"]
        for loc in locations:
            if loc in query:
                ctx.location = loc
                break

        # 人物解析（简化）
        person_pattern = r"和(.+?)(一起|吃的|去的|见的)"
        match = re.search(person_pattern, query)
        if match:
            ctx.persons.add(match.group(1))

        # 已知人物锚点：记忆库里出现过的人名在查询中出现即提取。
        # 人名是最强的检索锚点（"小李现在住在哪"必须锁定小李），
        # 而句式模板无法穷举——用人物索引的键做确定性匹配。
        for person in self.core.person_index.keys():
            if person and person in query:
                ctx.persons.add(person)

        # 主题解析：使用统一双语词汇表（core/topic_vocab.py），
        # 扩展出的标签能同时命中中文与英文写入侧的主题
        ctx.topics.update(extract_query_topics(query))

        # 时间窗口与时态：确定性解析（无时间表达时保持 None，
        # 管线行为与原来完全一致）
        from core.time_parser import parse_query_window, query_tense
        ctx.time_window = parse_query_window(query)
        ctx.tense = query_tense(query)
        
        # 情绪解析（简化）
        positive_words = ["开心", "高兴", "快乐", "愉快", "棒", "好"]
        negative_words = ["难过", "伤心", "痛苦", "糟糕", "差"]
        
        for pw in positive_words:
            if pw in query:
                ctx.emotion_valence = 0.5
                break
        for nw in negative_words:
            if nw in query:
                ctx.emotion_valence = -0.5
                break
        
        return ctx
    
    def parse_photo_info(self, photo_info: Dict[str, Any]) -> QueryContext:
        """
        解析照片信息为 QueryContext
        
        photo_info 可能包含：
        - 时间：photo_info.get("timestamp")
        - 地点：photo_info.get("location")
        - 人物：photo_info.get("faces", [])
        - 内容标签：photo_info.get("labels", [])
        """
        ctx = QueryContext()
        ctx.source_type = "photo"
        
        # 时间
        timestamp = photo_info.get("timestamp")
        if timestamp:
            # 假设是时间戳
            ctx.time_absolute = time.strftime("%Y-%m-%d", time.localtime(timestamp))
            # 时钟时间没有任何写入路径会存储（记忆的 time_context 是"中午"这类
            # 语义时段），必须换算成时段词才可能匹配
            ctx.time_context = self._hour_to_daypart(time.localtime(timestamp).tm_hour)

        # 地点
        location = photo_info.get("location")
        if location:
            ctx.location = location

        # 人物
        faces = photo_info.get("faces", [])
        ctx.persons.update(faces)

        # 标签（扩展同义主题，兼容中英文写入侧）
        labels = photo_info.get("labels", [])
        ctx.topics.update(expand_topics(labels))

        return ctx

    @staticmethod
    def _hour_to_daypart(hour: int) -> str:
        """把小时映射为语义时段（与记忆的 time_context 词汇一致）"""
        if 5 <= hour < 8:
            return "早上"
        if 8 <= hour < 11:
            return "上午"
        if 11 <= hour < 14:
            return "中午"
        if 14 <= hour < 18:
            return "下午"
        if 18 <= hour < 23:
            return "晚上"
        return "深夜"
    
    # ============ 检索 ============
    
    def retrieve(
        self,
        query: str,
        allow_forgotten: bool = True,
    ) -> ReconstructionResult:
        """
        主检索入口

        流程：
        1. 解析查询
        2. 混合检索（如果 planner 可用）或传统检索
        3. 伪遗忘层唤醒（如需要）
        4. 组装 + 审阅
        5. 返回结果
        """
        self.total_retrievals += 1

        # Step 1: 解析
        ctx = self.parse_query(query)
        vocab_cov = self._vocab_coverage(query)

        # Step 2: 检索（混合或传统）
        retrieval_path = ""
        all_chunks = []

        if self.planner:
            # 使用混合检索
            hybrid_results = self.planner.plan_and_retrieve(
                query=query,
                query_tags=ctx.to_tags(),
                limit=self.core_limit,
            )
            if hybrid_results:
                retrieval_path = "hybrid"
                self.core_hit += 1
                all_chunks = [chunk for chunk, _ in hybrid_results]
        else:
            # 传统检索（向后兼容）
            core_results = self.core.retrieve(
                ctx.to_tags(),
                min_weight=self.core_min_weight,
                limit=self.core_limit,
            )
            if core_results:
                retrieval_path = "core"
                self.core_hit += 1
                all_chunks = [chunk for chunk, _ in core_results]

        # Step 3: 伪遗忘层唤醒
        #
        # 关键：唤醒不是"核心未命中时的备胎"，而是与核心检索并行的通路。
        # 混合检索几乎总能返回点什么（词法部分匹配），如果唤醒只在
        # 核心全空时运行，遗忘-唤醒生命周期在实际部署中就永远不可达。
        # 人类回忆也是如此：强线索既召回新记忆，也能翻出尘封的旧记忆。
        if allow_forgotten:
            forgotten_results = self.forgotten.try_wake(ctx.to_tags(), limit=5)
            if forgotten_results:
                self.forgotten_hit += 1
                # 锚点足够强的唤醒记忆提升回核心层（遗忘-唤醒闭环）；
                # 返回值里被提升的对象已替换为核心层版本（layer 已更新）
                woken_chunks = self.promote_woken(forgotten_results)
                if all_chunks:
                    retrieval_path = "both"
                    self.both_hit += 1
                    seen = {c.id for c in all_chunks}
                    for c in woken_chunks:
                        if c.id not in seen:
                            all_chunks.append(c)
                            seen.add(c.id)
                else:
                    retrieval_path = "forgotten"
                    all_chunks = woken_chunks

        if not all_chunks:
            retrieval_path = "none"
        
        # 如果都没有命中
        if not all_chunks:
            note = "没有找到相关记忆"
            if vocab_cov is not None and vocab_cov < self.VOCAB_WARNING_BELOW:
                note = (
                    f"没有找到相关记忆（查询词汇覆盖率仅 {vocab_cov:.0%}，"
                    f"该主题可能不在记忆库中）"
                )
            return ReconstructionResult(
                success=False,
                chunks=[],
                assembled_content="",
                review_result=ReviewResult.REJECTED,
                retrieval_path="none",
                confidence=0.0,
                review_note=note,
            )

        # Step 3.2: 时态路由（双时态事实取代的读取侧）
        outdated_only = False
        now = time.time()
        if ctx.tense == "past":
            # 过去时查询：沿取代链（parent_id）把被取代的历史事实带回来
            all_chunks = self._follow_supersession_chains(all_chunks)
        else:
            # 现在时/无时态查询：排除已失效（被取代）的事实；
            # 若全部失效则保留并在审阅中标注"已过时"，不冒充现状
            valid = [
                c for c in all_chunks
                if c.invalid_at is None or c.invalid_at > now
            ]
            if valid:
                all_chunks = valid
            elif all_chunks:
                outdated_only = True

        # Step 3.3: 时间窗口过滤（查询含显式时间表达时）
        if ctx.time_window:
            from core.time_parser import chunk_time_range, windows_overlap
            in_window = []
            for c in all_chunks:
                # 数值窗口重叠，或相对时间标签精确匹配（写入侧存的
                # 是"10年前"这类原始表述时，标签匹配仍然有效）
                if windows_overlap(chunk_time_range(c), ctx.time_window):
                    in_window.append(c)
                elif ctx.time_relative and ctx.time_relative in (c.time_relative, c.time_context):
                    in_window.append(c)
            if in_window:
                all_chunks = in_window
            else:
                # 可审计弃答：窗口内没有记忆时，给出最接近的记忆时间，
                # 而不是拿窗口外的内容冒充答案
                nearest = min(
                    all_chunks,
                    key=lambda c: min(
                        abs(chunk_time_range(c)[0] - ctx.time_window[0]),
                        abs(chunk_time_range(c)[1] - ctx.time_window[1]),
                    ),
                )
                nearest_day = time.strftime(
                    "%Y-%m-%d", time.localtime(chunk_time_range(nearest)[0])
                )
                return ReconstructionResult(
                    success=False,
                    chunks=[],
                    assembled_content="",
                    review_result=ReviewResult.REJECTED,
                    retrieval_path=retrieval_path,
                    confidence=0.0,
                    review_note=(
                        f"询问的时间范围内没有记忆；最接近的相关记忆在 {nearest_day}"
                    ),
                )

        # Step 3.5: 联想回忆（PPR 扩散激活）
        # 由命中记忆沿 概念隶属 + Hebbian 关联图扩散，把"因为想起 A
        # 而想起 B"变成真实行为；被强激活的归档记忆会被联想唤醒甚至提升。
        #
        # 联想追加的结果必须通过与主结果相同的时态/时间窗口过滤——
        # 否则被窗口/失效过滤排除的记忆会从联想这条后门溜回结果。
        associated = self._associative_recall(all_chunks, allow_forgotten=allow_forgotten)
        seen_ids = {c.id for c in all_chunks}
        for c in associated:
            if c.id in seen_ids:
                continue
            if ctx.tense != "past" and c.invalid_at is not None and c.invalid_at <= now:
                continue
            if ctx.time_window:
                from core.time_parser import chunk_time_range, windows_overlap
                if not windows_overlap(chunk_time_range(c), ctx.time_window) and not (
                    ctx.time_relative and ctx.time_relative in (c.time_relative, c.time_context)
                ):
                    continue
            all_chunks.append(c)
            seen_ids.add(c.id)

        # Step 4: 组装
        assembled = self._assemble(all_chunks, ctx)

        # Step 5: 审阅
        review_result, confidence = self._review(all_chunks, assembled, ctx)

        # 低词汇覆盖：命中可能只是词面碎片重合，诚实标注而不是硬拒绝
        # （改述查询的覆盖率可以比界外查询更低——狗粮期实测教训）
        review_note = ""
        if vocab_cov is not None and vocab_cov < self.VOCAB_WARNING_BELOW:
            confidence *= 0.5
            review_result = ReviewResult.QUESTIONABLE
            review_note = (
                f"警告：查询词汇与记忆库重合度很低（{vocab_cov:.0%}），"
                f"以下结果可能不相关"
            )
        elif vocab_cov is not None and vocab_cov < self.VOCAB_CAUTION_BELOW:
            confidence *= 0.7
            review_note = (
                f"查询词汇覆盖率较低（{vocab_cov:.0%}），结果谨慎参考"
            )
            if confidence < self.review_confidence_threshold:
                review_result = ReviewResult.QUESTIONABLE

        # 全部结果都是被取代的旧事实：明确标注"已过时"，
        # 置信度打折，绝不冒充当前状态
        if outdated_only:
            superseded_by = all_chunks[0].metadata.get("superseded_by", "")
            review_note = (
                f"内容已过时（被 {superseded_by or '更新的记忆'} 取代），"
                "以下是历史状态而非现状"
            )
            review_result = ReviewResult.QUESTIONABLE
            confidence *= 0.6

        result = ReconstructionResult(
            success=True,
            chunks=all_chunks,
            assembled_content=assembled,
            review_result=review_result,
            retrieval_path=retrieval_path,
            confidence=confidence,
            review_note=review_note,
        )

        # Hebbian 共激活：一起被检索到的记忆互相连线
        # （"fire together, wire together"——这是关联图的主要生长途径）
        self._coactivate(all_chunks)

        # 反馈给核心层
        for chunk in all_chunks:
            if chunk.layer == MemoryLayer.CORE:
                self.core.access(chunk.id)
        
        return result
    
    # PPR 子图规模上限（确定性截断，防止大库上的构图开销失控）
    _PPR_MAX_CONCEPT_MEMBERS = 20   # 每个概念节点纳入的成员数上限
    _PPR_MAX_EDGES_PER_CHUNK = 10   # 每条记忆纳入的 Hebbian 边上限
    _PPR_MAX_NODES = 300            # 子图节点总数上限

    def _associative_recall(
        self,
        seed_chunks: List[MemoryChunk],
        damping: float = 0.5,
        max_iterations: int = 20,
        convergence_eps: float = 1e-6,
        activation_gain: float = 3.0,
        activation_threshold: float = 0.15,
        limit: int = 5,
        allow_forgotten: bool = True,
    ) -> List[MemoryChunk]:
        """
        Personalized PageRank 联想回忆（HippoRAG 的海马体索引思想，
        确定性幂迭代实现）。

        图结构（按需从种子邻域构建）：
        - 记忆节点：种子 + 概念成员 + Hebbian 邻居
        - 概念节点：种子记忆携带的人物/主题/地点标签（新皮层-海马体
          的双向投射：两条记忆即使没有显式 Hebbian 边，共享"老王"
          也会通过概念节点变成两跳邻居）
        - 边权：概念隶属 1.0，Hebbian 边取存储的权重

        p = (1-d)·r + d·Wᵀp，restart 质量均分在种子上，
        列归一化的 Wᵀ 天然实现扇出阻尼（ACT-R fan effect）：
        连接 50 条记忆的常见概念每条只分到 1/50 的质量，
        稀有线索的联想强度远高于烂大街的线索。

        相比旧的固定跳数扩散激活：多路径汇聚由平稳分布原理性处理
        （不存在重复传播类缺陷），任意深度传播自动收敛，无需 hop 上限。

        校准锚点（activation = gain × p / max(种子 p)）：
        - 单条 0.8 强 Hebbian 边 -> ~0.4：召回且可提升归档记忆（>=0.25）
        - 单条 0.1 弱边 -> ~0.07：低于收录阈值（<0.15），联想有选择性
        - 归档记忆只能经 Hebbian 边进入子图，概念隶属不桥接归档层
          （伪遗忘层不参与主动检索的原则不变）

        激活轨迹保存在 self.last_activation_trace，便于审计解释。
        """
        if not seed_chunks:
            return []

        seed_ids = {c.id for c in seed_chunks}

        # ---------- 构建种子邻域子图 ----------
        # 节点 id 约定：记忆节点用 chunk.id，概念节点用 "concept::<tag>"
        chunks_in_graph: Dict[str, MemoryChunk] = {c.id: c for c in seed_chunks}
        edges: Dict[str, Dict[str, float]] = {}

        def _add_edge(a: str, b: str, weight: float):
            edges.setdefault(a, {})[b] = max(edges.get(a, {}).get(b, 0.0), weight)
            edges.setdefault(b, {})[a] = max(edges.get(b, {}).get(a, 0.0), weight)

        def _concept_members(index: Dict[str, set], tag: str) -> List[str]:
            bucket = index.get(tag, set())
            return sorted(bucket)[: self._PPR_MAX_CONCEPT_MEMBERS]

        def _include_chunk(chunk_id: str) -> Optional[MemoryChunk]:
            if chunk_id in chunks_in_graph:
                return chunks_in_graph[chunk_id]
            if len(chunks_in_graph) >= self._PPR_MAX_NODES:
                return None
            target = self.core.get(chunk_id)
            if target is None and allow_forgotten:
                # 归档记忆只能经 Hebbian 边进入（调用方控制）
                target = self.forgotten.get(chunk_id)
            if target is not None:
                chunks_in_graph[chunk_id] = target
            return target

        # 第一层：种子的概念隶属 + Hebbian 边
        frontier_ids = list(seed_ids)
        for _depth in range(2):
            next_frontier: List[str] = []
            for cid in sorted(frontier_ids):
                chunk = chunks_in_graph.get(cid)
                if chunk is None:
                    continue

                # 概念隶属边（只对核心层记忆展开成员——
                # 伪遗忘层不通过概念参与主动联想）
                if chunk.layer == MemoryLayer.CORE:
                    concept_specs = (
                        [("person", p, self.core.person_index) for p in sorted(chunk.persons)]
                        + [("topic", t, self.core.topic_index) for t in sorted(chunk.topics)]
                        + ([("location", chunk.location, self.core.location_index)]
                           if chunk.location else [])
                    )
                    for kind, tag, index in concept_specs:
                        concept_node = f"concept::{kind}::{tag}"
                        _add_edge(cid, concept_node, 1.0)
                        for member_id in _concept_members(index, tag):
                            if member_id == cid:
                                continue
                            member = _include_chunk(member_id)
                            if member is not None:
                                _add_edge(concept_node, member_id, 1.0)
                                next_frontier.append(member_id)

                # Hebbian 边（可通向归档记忆——关联本身是唤醒线索）
                hebbian = sorted(
                    chunk.associations.items(), key=lambda x: (-x[1], x[0]),
                )[: self._PPR_MAX_EDGES_PER_CHUNK]
                for assoc_id, weight in hebbian:
                    if weight <= 0.0:
                        continue
                    target = self.core.get(assoc_id)
                    if target is None:
                        if not allow_forgotten:
                            continue
                        target = self.forgotten.get(assoc_id)
                    if target is None:
                        continue
                    if assoc_id not in chunks_in_graph:
                        if len(chunks_in_graph) >= self._PPR_MAX_NODES:
                            continue
                        chunks_in_graph[assoc_id] = target
                    _add_edge(cid, assoc_id, weight)
                    next_frontier.append(assoc_id)
            frontier_ids = [i for i in next_frontier if i not in seed_ids]

        # ---------- 幂迭代 ----------
        nodes = sorted(edges.keys() | chunks_in_graph.keys())
        restart = {n: (1.0 / len(seed_ids) if n in seed_ids else 0.0) for n in nodes}
        # 列归一化：每个节点向外推送的质量按边权比例分配
        out_weight = {n: sum(edges.get(n, {}).values()) for n in nodes}

        p = dict(restart)
        for _ in range(max_iterations):
            nxt = {n: (1.0 - damping) * restart[n] for n in nodes}
            for src in nodes:
                mass = p[src]
                total = out_weight[src]
                if mass <= 0.0 or total <= 0.0:
                    continue
                for dst, weight in edges.get(src, {}).items():
                    nxt[dst] += damping * mass * (weight / total)
            delta = sum(abs(nxt[n] - p[n]) for n in nodes)
            p = nxt
            if delta < convergence_eps:
                break

        # ---------- 激活换算（相对种子平稳质量） ----------
        max_seed_mass = max((p.get(sid, 0.0) for sid in seed_ids), default=0.0)
        collected: Dict[str, MemoryChunk] = {}
        activation: Dict[str, float] = {}
        if max_seed_mass > 0:
            for cid, chunk in chunks_in_graph.items():
                if cid in seed_ids:
                    continue
                act = min(1.0, activation_gain * p.get(cid, 0.0) / max_seed_mass)
                if act > 0.0:
                    activation[cid] = act
                    collected[cid] = chunk

        # 审计轨迹（只记录非种子的联想激活）
        self.last_activation_trace = sorted(
            ((cid, activation[cid]) for cid in collected),
            key=lambda x: (-x[1], x[0]),
        )

        # 取激活最强的若干条
        ranked = [
            (collected[cid], act) for cid, act in self.last_activation_trace
            if act >= activation_threshold
        ][:limit]

        recalled: List[MemoryChunk] = []
        to_promote: List[str] = []
        for chunk, act in ranked:
            if chunk.layer == MemoryLayer.FORGOTTEN:
                # 联想唤醒：关联本身就是线索
                self.forgotten.record_wake(chunk.id)
                self.total_assoc_wakes += 1
                if act >= self.assoc_promote_threshold:
                    to_promote.append(chunk.id)
            recalled.append(chunk)

        # 被强激活的归档记忆提升回核心层
        if to_promote:
            promoted = {c.id: c for c in self.forgotten.promote(to_promote)}
            for chunk_id, chunk in promoted.items():
                self.core.add(chunk)
                self.core.access(chunk_id)
                if self.planner:
                    self.planner.add_chunk(chunk)
            self.total_promoted += len(promoted)
            # 用提升后的对象（layer 已置回 CORE）替换返回列表里的旧引用
            recalled = [promoted.get(c.id, c) for c in recalled]

        self.total_assoc_recalled += len(recalled)
        return recalled

    def _follow_supersession_chains(self, chunks: List[MemoryChunk], max_depth: int = 5) -> List[MemoryChunk]:
        """
        过去时查询的取代链追溯。

        双时态链条本身就是检索路径："我以前住在哪"先命中当前事实
        （奥斯陆），再沿 parent_id 走回被取代的历史值（柏林 -> 里斯本）。
        被取代的记忆在归档层，链条追溯是它们最可靠的唤醒线索。
        """
        result = list(chunks)
        seen = {c.id for c in chunks}
        for chunk in chunks:
            parent_id = chunk.parent_id
            depth = 0
            while parent_id and parent_id not in seen and depth < max_depth:
                ancestor = self.core.get(parent_id) or self.forgotten.get(parent_id)
                if ancestor is None:
                    break
                result.append(ancestor)
                seen.add(ancestor.id)
                # 历史事实被想起也留下唤醒痕迹
                if ancestor.layer == MemoryLayer.FORGOTTEN:
                    self.forgotten.record_wake(ancestor.id)
                parent_id = ancestor.parent_id
                depth += 1
        return result

    # 共激活去抖窗口：同一对记忆在窗口内的重复共现只加强一次
    COACTIVATION_DEBOUNCE_SECONDS = 60.0

    def _coactivate(self, chunks: List[MemoryChunk], max_wired: int = 4, strength: float = 0.05):
        """
        Hebbian 共激活：同一次检索里一起出现的记忆互相加强关联。

        这是关联图的主要生长途径——没有它，扩散激活面对的是一张空图。
        只连线前几条核心层记忆：人类的共激活也是选择性的，
        全连接会让关联图退化成噪声。

        60 秒去抖：狗粮期实测一轮评测就让边权疯长——短窗口内的
        重复共现是同一次"共同经历"，不应重复加强。
        """
        now = time.time()
        if not hasattr(self, "_recent_coactivations"):
            self._recent_coactivations: Dict[tuple, float] = {}

        core_chunks = [c for c in chunks if c.layer == MemoryLayer.CORE][:max_wired]
        for i, chunk_a in enumerate(core_chunks):
            for chunk_b in core_chunks[i + 1:]:
                pair = tuple(sorted((chunk_a.id, chunk_b.id)))
                last = self._recent_coactivations.get(pair, 0.0)
                if now - last < self.COACTIVATION_DEBOUNCE_SECONDS:
                    continue
                self._recent_coactivations[pair] = now
                self.core.strengthen_association(chunk_a.id, chunk_b.id, strength=strength)

        # 防止去抖记录无限增长
        if len(self._recent_coactivations) > 2000:
            cutoff = now - self.COACTIVATION_DEBOUNCE_SECONDS
            self._recent_coactivations = {
                k: v for k, v in self._recent_coactivations.items() if v > cutoff
            }
    
    def _assemble(
        self,
        chunks: List[MemoryChunk],
        ctx: QueryContext,
    ) -> str:
        """
        将碎片组装成连贯的叙述
        
        目前是简化版，按时间/重要性排序后拼接
        未来可以换成LLM生成
        """
        if not chunks:
            return ""
        
        if len(chunks) == 1:
            return chunks[0].content
        
        # 按时间排序
        if self.assembly_method == "chronological":
            sorted_chunks = sorted(
                chunks,
                key=lambda c: c.created_at,
                reverse=True,
            )
        elif self.assembly_method == "relevance":
            # 按重要性排序
            sorted_chunks = sorted(
                chunks,
                key=lambda c: c.importance,
                reverse=True,
            )
        else:  # hybrid
            # 综合排序
            sorted_chunks = sorted(
                chunks,
                key=lambda c: c.importance * 0.5 + c.access_count * 0.3 + c.successful_recall_count * 0.2,
                reverse=True,
            )
        
        # 拼接内容（简化版）
        # 未来用LLM根据上下文生成连贯叙述
        parts = [c.content for c in sorted_chunks if c.content]
        
        if len(parts) == 1:
            return parts[0]
        
        # 简单的拼接
        assembled = "；".join(parts)
        
        # 如果上下文提示了具体时间，加上
        if ctx.time_relative:
            assembled = f"关于{ctx.time_relative}的记忆：{assembled}"
        
        return assembled
    
    def _review(
        self,
        chunks: List[MemoryChunk],
        assembled: str,
        ctx: QueryContext,
    ) -> Tuple[ReviewResult, float]:
        """
        审阅组装的记忆是否合理
        
        简化版审阅：
        1. 时间线检查（如果多个碎片，时间线是否矛盾）
        2. 情绪一致性检查
        3. 重要性加权置信度
        """
        if not chunks:
            return ReviewResult.REJECTED, 0.0
        
        confidence = 0.0
        issues = []
        
        # 1. 基础置信度（基于碎片数量和质量）
        if len(chunks) == 1:
            confidence += 0.3
        elif len(chunks) <= 3:
            confidence += 0.4
        else:
            confidence += 0.5
        
        # 2. 基于平均重要性
        avg_importance = sum(c.importance for c in chunks) / len(chunks)
        confidence += avg_importance * 0.3
        
        # 3. 基于成功回忆次数
        total_recalls = sum(c.successful_recall_count for c in chunks)
        confidence += min(0.2, total_recalls * 0.05)
        
        # 4. 检查情绪一致性（如果有上下文的话）
        if ctx.emotion_valence is not None:
            avg_emotion = sum(c.emotion_valence for c in chunks) / len(chunks)
            emotion_diff = abs(avg_emotion - ctx.emotion_valence)
            if emotion_diff > 0.5:
                issues.append("情绪不一致")
                confidence -= 0.15
        
        # 5. 检查时间合理性
        time_contexts = [c.time_context for c in chunks if c.time_context]
        if ctx.time_context and ctx.time_context not in time_contexts:
            # 问的是中午，但记忆没有中午的上下文
            issues.append("时间上下文可能不匹配")
            confidence -= 0.1
        
        # 6. 审阅状态检查
        statuses = [c.review_status for c in chunks]
        if "rejected" in statuses:
            return ReviewResult.REJECTED, confidence * 0.3
        if "questionable" in statuses:
            issues.append("部分记忆被标记为存疑")
            confidence *= 0.7
        
        # 最终置信度
        confidence = max(0.0, min(1.0, confidence))
        
        # 审阅结论
        if confidence >= self.review_confidence_threshold:
            if issues:
                return ReviewResult.MODIFIED, confidence
            return ReviewResult.APPROVED, confidence
        else:
            return ReviewResult.QUESTIONABLE, confidence
    
    # ============ 反馈 ============
    
    def feedback(
        self,
        query: str,
        accepted: bool,
        corrected_content: Optional[str] = None,
    ):
        """
        用户反馈
        
        accepted=True: 输出被接受了
        accepted=False: 输出被纠正了，corrected_content是正确的内容
        """
        result = self.retrieve(query, allow_forgotten=False)

        if not result.success:
            return

        # 反馈只作用于头部结果：用户对答案的评价主要由排名靠前的
        # 记忆驱动，把惩罚摊到整个结果列表会永久压低碰巧被带出的
        # 无关记忆（recall_bias 是持久的）。
        for chunk in result.chunks[:3]:
            if chunk.layer == MemoryLayer.CORE:
                self.core.adjust_after_recall(
                    chunk.id,
                    success=accepted,
                    feedback_emotion=result.confidence,
                )
        
        # 如果被纠正，可能需要更新内容
        if not accepted and corrected_content:
            # 这是更高级的功能，涉及记忆修正
            pass
    
    # ============ 统计 ============
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_retrievals": self.total_retrievals,
            "core_hit_rate": self.core_hit / max(1, self.total_retrievals),
            "forgotten_hit_rate": self.forgotten_hit / max(1, self.total_retrievals),
            "total_promoted": self.total_promoted,
            "total_assoc_recalled": self.total_assoc_recalled,
            "total_assoc_wakes": self.total_assoc_wakes,
            "core_chunks": len(self.core),
            "forgotten_chunks": len(self.forgotten),
        }
