"""
ConsolidationEngine - 记忆巩固引擎

职责：
1. 合并重复记忆
2. 失效过时记忆
3. 升级/降级重要性
4. 跨 episode 综合
"""

import time
from typing import Dict, List, Optional, Any

from core.memory_spec import MemorySpec, ExtractionResult
from core.memory_extractor import MemoryExtractor


class ConsolidationEngine:
    """
    记忆巩固引擎

    将 episode 转换为结构化记忆，并进行去重、合并、巩固。
    """

    def __init__(
        self,
        memory_system,
        extractor: MemoryExtractor,
        similarity_threshold: float = 0.8,
    ):
        """
        Args:
            memory_system: HumanLikeMemorySystem 实例
            extractor: MemoryExtractor 实例
            similarity_threshold: 相似度阈值（用于去重）
        """
        self.memory = memory_system
        self.extractor = extractor
        self.similarity_threshold = similarity_threshold

    def consolidate_episode(self, episode) -> List[str]:
        """
        巩固单个 episode

        1. 使用 extractor 提取多个 MemorySpec
        2. 检查与现有记忆的重复
        3. 合并或创建新记忆
        4. 更新关联

        Args:
            episode: ExperienceEpisode 对象

        Returns:
            创建/更新的 memory ID 列表
        """
        # 提取记忆
        extraction_result = self.extractor.extract(episode)

        memory_ids = []

        for spec in extraction_result.specs:
            # 检查重复
            existing = self._find_similar(spec)

            if existing:
                # 合并
                self._merge_memory(existing, spec)
                memory_ids.append(existing.id)
            else:
                # 创建新记忆
                memory_id = self._create_memory(spec, episode)
                if memory_id:
                    memory_ids.append(memory_id)

        return memory_ids

    def consolidate_batch(self, episodes: List) -> List[str]:
        """
        批量巩固多个 episode

        1. 提取所有 episode 的记忆
        2. 跨 episode 去重
        3. 综合相似记忆

        Args:
            episodes: ExperienceEpisode 列表

        Returns:
            创建/更新的 memory ID 列表
        """
        all_results = []

        # 提取所有 episode 的记忆
        for episode in episodes:
            result = self.extractor.extract(episode)
            all_results.append((result, episode))

        # 跨 episode 去重
        merged_specs = self._deduplicate_specs(all_results)

        # 创建/合并记忆
        memory_ids = []
        for spec, episode in merged_specs:
            existing = self._find_similar(spec)

            if existing:
                self._merge_memory(existing, spec)
                memory_ids.append(existing.id)
            else:
                memory_id = self._create_memory(spec, episode)
                if memory_id:
                    memory_ids.append(memory_id)

        return memory_ids

    def _find_similar(self, spec: MemorySpec) -> Optional[Any]:
        """
        查找相似的现有记忆

        使用内容相似度检测。
        """
        # 使用检索系统查找相似记忆
        if hasattr(self.memory, 'retrieval') and self.memory.retrieval:
            try:
                result = self.memory.retrieval.retrieve(spec.content, allow_forgotten=True)
                if result.success and result.chunks:
                    # 检查最相似的记忆
                    for chunk in result.chunks:
                        similarity = self._calculate_similarity(spec.content, chunk.content)
                        if similarity >= self.similarity_threshold:
                            return chunk
            except Exception:
                pass

        return None

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算文本相似度

        简单实现：基于字符重叠。
        """
        if not text1 or not text2:
            return 0.0

        # 转换为字符集合
        chars1 = set(text1)
        chars2 = set(text2)

        # 计算 Jaccard 相似度
        intersection = chars1 & chars2
        union = chars1 | chars2

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def _merge_memory(self, existing: Any, new_spec: MemorySpec):
        """
        合并新信息到现有记忆

        Args:
            existing: 现有 MemoryChunk
            new_spec: 新的 MemorySpec
        """
        # 更新内容（追加新信息）
        if new_spec.content and new_spec.content not in existing.content:
            existing.content = existing.content + "\n" + new_spec.content

        # 更新重要性（取较高值）
        if new_spec.importance > existing.importance:
            existing.importance = new_spec.importance

        # 更新人物（合并）
        if new_spec.persons:
            existing.persons = existing.persons | new_spec.persons

        # 更新主题（合并）
        if new_spec.topics:
            existing.topics = existing.topics | new_spec.topics

        # 更新关键词（合并）
        if new_spec.keywords:
            existing.keywords = existing.keywords | new_spec.keywords

        # 更新时间（保留更精确的）
        if new_spec.time_absolute and not existing.time_absolute:
            existing.time_absolute = new_spec.time_absolute
        if new_spec.time_relative and not existing.time_relative:
            existing.time_relative = new_spec.time_relative
        if new_spec.time_context and not existing.time_context:
            existing.time_context = new_spec.time_context

        # 更新地点（保留更精确的）
        if new_spec.location and not existing.location:
            existing.location = new_spec.location

        # 更新情绪（取较强的情绪）
        if abs(new_spec.emotion_intensity) > abs(existing.emotion_intensity):
            existing.emotion_valence = new_spec.emotion_valence
            existing.emotion_intensity = new_spec.emotion_intensity

        # 更新时间戳
        existing.updated_at = time.time()

        # 保存到存储
        if hasattr(self.memory, 'core') and self.memory.core:
            self.memory.core._store.put(existing)

    def _create_memory(self, spec: MemorySpec, episode) -> Optional[str]:
        """
        创建新记忆

        Args:
            spec: MemorySpec
            episode: ExperienceEpisode

        Returns:
            memory ID 或 None
        """
        try:
            memory_id = self.memory.add_memory(
                content=spec.content,
                memory_type=spec.memory_type,
                time_absolute=spec.time_absolute,
                time_relative=spec.time_relative,
                time_context=spec.time_context,
                location=spec.location,
                persons=list(spec.persons) if spec.persons else None,
                topics=list(spec.topics) if spec.topics else None,
                keywords=list(spec.keywords) if spec.keywords else None,
                emotion_valence=spec.emotion_valence,
                emotion_intensity=spec.emotion_intensity,
                importance=spec.importance,
                metadata={
                    **spec.metadata,
                    "episode_id": episode.id if episode else None,
                    "extraction_time": time.time(),
                },
            )
            return memory_id
        except Exception as e:
            print(f"[ConsolidationEngine] 创建记忆失败: {e}")
            return None

    def _deduplicate_specs(
        self, results: List[tuple]
    ) -> List[tuple]:
        """
        跨 episode 去重

        Args:
            results: [(ExtractionResult, episode), ...]

        Returns:
            [(MemorySpec, episode), ...] 去重后的列表
        """
        merged = []
        seen_contents = set()

        for result, episode in results:
            for spec in result.specs:
                # 检查是否已见过相似内容
                is_duplicate = False
                for seen_content in seen_contents:
                    similarity = self._calculate_similarity(spec.content, seen_content)
                    if similarity >= self.similarity_threshold:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    merged.append((spec, episode))
                    seen_contents.add(spec.content)

        return merged

    def invalidate_outdated(self, days_threshold: int = 30) -> List[str]:
        """
        失效过时记忆

        Args:
            days_threshold: 天数阈值

        Returns:
            失效的记忆 ID 列表
        """
        invalidated = []

        if not hasattr(self.memory, 'core'):
            return invalidated

        now = time.time()
        threshold_seconds = days_threshold * 24 * 3600

        for chunk_id, chunk in self.memory.core._store.get_all().items():
            # 检查是否过时
            age = now - chunk.updated_at
            if age > threshold_seconds:
                # 降低重要性
                decay_factor = 0.99 ** (age / threshold_seconds)
                chunk.importance = chunk.importance * decay_factor

                # 如果重要性太低，降级到伪遗忘层
                if chunk.importance < 0.1:
                    self.memory.core.remove(chunk_id)
                    self.memory.forgotten.archive(chunk)
                    invalidated.append(chunk_id)
                else:
                    # 更新重要性
                    self.memory.core._store.put(chunk)

        return invalidated

    def upgrade_important(self, recall_threshold: int = 5) -> List[str]:
        """
        升级重要记忆

        被频繁回忆的记忆应该升级重要性。

        Args:
            recall_threshold: 回忆次数阈值

        Returns:
            升级的记忆 ID 列表
        """
        upgraded = []

        if not hasattr(self.memory, 'core'):
            return upgraded

        for chunk_id, chunk in self.memory.core._store.get_all().items():
            # 检查回忆次数
            if chunk.access_count >= recall_threshold:
                # 提升重要性
                boost = min(0.1, chunk.access_count * 0.01)
                chunk.importance = min(1.0, chunk.importance + boost)

                # 更新
                self.memory.core._store.put(chunk)
                upgraded.append(chunk_id)

        return upgraded
