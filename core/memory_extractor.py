"""
MemoryExtractor - 从 episode 提取结构化记忆

策略：
1. 规则抽取（默认，无 LLM 依赖）
2. LLM 抽取（可选，更丰富）
"""

import time
from typing import Callable, List, Optional

from core.memory_spec import MemorySpec, ExtractionResult
from core.entity_extractor import EntityExtractor
from core.weight_system import MemoryType


class MemoryExtractor:
    """
    从 episode 提取结构化记忆

    支持两种模式：
    1. 规则抽取（默认）：基于模板和规则，无外部依赖
    2. LLM 抽取（可选）：使用 LLM 进行更丰富的语义抽取
    """

    def __init__(self, llm_fn: Optional[Callable[[str], str]] = None):
        """
        Args:
            llm_fn: LLM 函数，接受 prompt 返回文本（可选）
        """
        self.llm_fn = llm_fn
        self.entity_extractor = EntityExtractor()

    def extract(self, episode) -> ExtractionResult:
        """
        从 episode 提取多个记忆

        Args:
            episode: ExperienceEpisode 对象

        Returns:
            ExtractionResult 包含多个 MemorySpec
        """
        start_time = time.time()

        result = ExtractionResult(
            episode_id=episode.id,
            extractor_type="llm" if self.llm_fn else "rule",
        )

        if self.llm_fn:
            specs = self._llm_extract(episode)
        else:
            specs = self._rule_extract(episode)

        for spec in specs:
            result.add_spec(spec)

        result.extraction_time = time.time() - start_time
        return result

    def _rule_extract(self, episode) -> List[MemorySpec]:
        """规则抽取"""
        specs = []

        # 1. 事实记忆（从 observation 提取）
        fact_spec = self._extract_fact(episode)
        if fact_spec:
            specs.append(fact_spec)

        # 2. 经验记忆（从 action + result 提取）
        experience_spec = self._extract_experience(episode)
        if experience_spec:
            specs.append(experience_spec)

        # 3. 程序记忆（从 next_policy 提取）
        procedure_spec = self._extract_procedure(episode)
        if procedure_spec:
            specs.append(procedure_spec)

        # 如果没有提取到任何记忆，创建一个默认的
        if not specs:
            default_spec = self._extract_default(episode)
            if default_spec:
                specs.append(default_spec)

        return specs

    def _extract_fact(self, episode) -> Optional[MemorySpec]:
        """
        提取事实记忆

        从 observation 中提取用户输入的事实信息。
        """
        observation = episode.observation
        if not observation or not observation.content:
            return None

        # 使用实体抽取器提取信息
        entities = self.entity_extractor.extract_all(observation.content)

        # 如果没有提取到任何实体，跳过
        if not entities["persons"] and not entities["location"] and not entities["topics"]:
            return None

        # 构建内容
        content = observation.content

        # 构建摘要
        summary = self._generate_summary(content, "fact")

        return MemorySpec(
            content=content,
            summary=summary,
            memory_type=MemoryType.FACT,
            time_absolute=entities["time_absolute"],
            time_relative=entities["time_relative"],
            time_context=entities["time_context"],
            location=entities["location"],
            persons=entities["persons"],
            topics=entities["topics"],
            keywords=entities["keywords"],
            emotion_valence=entities["emotion_valence"],
            emotion_intensity=entities["emotion_intensity"],
            importance=self._calculate_importance(episode, "fact"),
            metadata={"kind": "fact", "source": observation.source},
        )

    def _extract_experience(self, episode) -> Optional[MemorySpec]:
        """
        提取经验记忆

        从 action + result 中提取经验信息。
        """
        action = episode.action
        result = episode.result

        if not action or not result:
            return None

        # 构建内容
        content_parts = []

        if episode.goal:
            content_parts.append(f"目标：{episode.goal}")

        content_parts.append(f"执行：{action.name}")
        if action.rationale:
            content_parts.append(f"原因：{action.rationale}")

        if result.success:
            content_parts.append("结果：成功")
            if result.output:
                content_parts.append(f"输出：{result.output[:200]}")
        else:
            content_parts.append("结果：失败")
            if result.output:
                content_parts.append(f"错误：{result.output[:200]}")

        if episode.lesson:
            content_parts.append(f"教训：{episode.lesson}")

        content = "\n".join(content_parts)

        # 使用实体抽取器提取信息
        entities = self.entity_extractor.extract_all(content)

        # 构建摘要
        summary = self._generate_summary(content, "experience")

        return MemorySpec(
            content=content,
            summary=summary,
            memory_type=MemoryType.STORY,
            time_absolute=entities["time_absolute"],
            time_relative=entities["time_relative"],
            time_context=entities["time_context"],
            location=entities["location"],
            persons=entities["persons"],
            topics=entities["topics"] | {"经验", action.name},
            keywords=entities["keywords"] | {action.name},
            emotion_valence=entities["emotion_valence"],
            emotion_intensity=entities["emotion_intensity"],
            importance=self._calculate_importance(episode, "experience"),
            metadata={
                "kind": "experience",
                "action": action.name,
                "success": result.success,
                "reward": episode.reward,
            },
        )

    def _extract_procedure(self, episode) -> Optional[MemorySpec]:
        """
        提取程序记忆

        从 next_policy 中提取程序性知识。
        """
        if not episode.next_policy:
            return None

        action = episode.action
        result = episode.result

        # 构建内容
        content_parts = []
        content_parts.append(f"程序：{episode.next_policy}")

        if action:
            content_parts.append(f"适用场景：{action.name}")

        if episode.lesson:
            content_parts.append(f"背景：{episode.lesson}")

        content = "\n".join(content_parts)

        # 使用实体抽取器提取信息
        entities = self.entity_extractor.extract_all(content)

        # 构建摘要
        summary = self._generate_summary(content, "procedure")

        return MemorySpec(
            content=content,
            summary=summary,
            memory_type=MemoryType.IDEA,
            time_absolute=entities["time_absolute"],
            time_relative=entities["time_relative"],
            time_context=entities["time_context"],
            location=entities["location"],
            persons=entities["persons"],
            topics=entities["topics"] | {"程序", "策略"},
            keywords=entities["keywords"] | {"策略", "方法"},
            emotion_valence=entities["emotion_valence"],
            emotion_intensity=entities["emotion_intensity"],
            importance=self._calculate_importance(episode, "procedure"),
            metadata={
                "kind": "procedure",
                "action": action.name if action else None,
                "success": result.success if result else None,
            },
        )

    def _extract_default(self, episode) -> Optional[MemorySpec]:
        """默认抽取（当其他抽取都失败时）"""
        observation = episode.observation
        if not observation or not observation.content:
            return None

        # 使用实体抽取器提取信息
        entities = self.entity_extractor.extract_all(observation.content)

        return MemorySpec(
            content=observation.content,
            summary=observation.content[:100],
            memory_type=MemoryType.INTERACTION,
            time_absolute=entities["time_absolute"],
            time_relative=entities["time_relative"],
            time_context=entities["time_context"],
            location=entities["location"],
            persons=entities["persons"],
            topics=entities["topics"],
            keywords=entities["keywords"],
            emotion_valence=entities["emotion_valence"],
            emotion_intensity=entities["emotion_intensity"],
            importance=self._calculate_importance(episode, "default"),
            metadata={"kind": "default"},
        )

    def _llm_extract(self, episode) -> List[MemorySpec]:
        """
        LLM 抽取

        使用 LLM 进行更丰富的语义抽取。
        """
        # 构建 prompt
        prompt = self._build_extraction_prompt(episode)

        # 调用 LLM
        try:
            response = self.llm_fn(prompt)
            specs = self._parse_llm_response(response, episode)
        except Exception as e:
            # LLM 失败时回退到规则抽取
            print(f"[MemoryExtractor] LLM 抽取失败: {e}")
            return self._rule_extract(episode)

        # 返回内容无法解析成任何 spec 也按失败处理：
        # 静默返回空列表意味着这个 episode 什么都没学到
        if not specs:
            print("[MemoryExtractor] LLM 返回无法解析出记忆，回退到规则抽取")
            return self._rule_extract(episode)
        return specs

    def _build_extraction_prompt(self, episode) -> str:
        """构建 LLM 抽取 prompt"""
        observation = episode.observation
        action = episode.action
        result = episode.result

        prompt = f"""请从以下对话/事件中提取结构化记忆信息。

## 输入信息
- 用户输入：{observation.content if observation else '无'}
- 执行动作：{action.name if action else '无'}
- 动作结果：{'成功' if result and result.success else '失败'}
- 输出内容：{result.output[:200] if result and result.output else '无'}
- 经验教训：{episode.lesson or '无'}
- 下一步策略：{episode.next_policy or '无'}

## 请提取以下信息
1. 事实记忆：用户提到的事实信息
2. 经验记忆：这次经历的教训
3. 程序记忆：可复用的策略/方法

## 输出格式（JSON）
```json
{{
    "facts": [
        {{
            "content": "事实内容",
            "persons": ["人物1", "人物2"],
            "location": "地点",
            "topics": ["主题1", "主题2"],
            "importance": 0.5
        }}
    ],
    "experiences": [
        {{
            "content": "经验内容",
            "lesson": "教训",
            "importance": 0.5
        }}
    ],
    "procedures": [
        {{
            "content": "程序内容",
            "trigger": "触发条件",
            "importance": 0.5
        }}
    ]
}}
```
"""
        return prompt

    def _parse_llm_response(self, response: str, episode) -> List[MemorySpec]:
        """解析 LLM 响应"""
        import json

        specs = []

        try:
            # 尝试提取 JSON
            json_match = response
            if "```json" in response:
                json_match = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_match = response.split("```")[1].split("```")[0]

            data = json.loads(json_match)

            # 解析事实记忆
            for fact in data.get("facts", []):
                spec = MemorySpec(
                    content=fact.get("content", ""),
                    memory_type=MemoryType.FACT,
                    persons=set(fact.get("persons", [])),
                    location=fact.get("location"),
                    topics=set(fact.get("topics", [])),
                    importance=fact.get("importance", 0.5),
                    metadata={"kind": "fact", "source": "llm"},
                )
                specs.append(spec)

            # 解析经验记忆
            for exp in data.get("experiences", []):
                spec = MemorySpec(
                    content=exp.get("content", ""),
                    memory_type=MemoryType.STORY,
                    importance=exp.get("importance", 0.5),
                    metadata={"kind": "experience", "source": "llm"},
                )
                specs.append(spec)

            # 解析程序记忆
            for proc in data.get("procedures", []):
                spec = MemorySpec(
                    content=proc.get("content", ""),
                    memory_type=MemoryType.IDEA,
                    importance=proc.get("importance", 0.5),
                    metadata={"kind": "procedure", "source": "llm"},
                )
                specs.append(spec)

        except Exception as e:
            print(f"[MemoryExtractor] 解析 LLM 响应失败: {e}")

        return specs

    def _generate_summary(self, content: str, kind: str) -> str:
        """生成摘要"""
        # 简单实现：截取前 100 字符
        if len(content) <= 100:
            return content
        return content[:100] + "..."

    def _calculate_importance(self, episode, kind: str) -> float:
        """计算重要性"""
        base_importance = 0.5

        # 根据 reward 调整
        reward_factor = episode.reward * 0.3

        # 根据类型调整
        type_factors = {
            "fact": 0.1,
            "experience": 0.2,
            "procedure": 0.15,
            "default": 0.0,
        }
        type_factor = type_factors.get(kind, 0.0)

        # 根据成功/失败调整
        success_factor = 0.1 if episode.result and episode.result.success else -0.1

        importance = base_importance + reward_factor + type_factor + success_factor
        return max(0.1, min(1.0, importance))
