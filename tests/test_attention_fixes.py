"""注意力/认知层缺陷回归测试（第一轮审计遗留项）

1. Jaccard 并集归一化惩罚长记忆：完整包含查询的段落级记忆得分趋近 0
2. 程序记忆零相关也能凭 importance+confidence 越过工作区门槛
3. _looks_uncertain 子串匹配把普通陈述误标为不确定
4. Persona 行为评估只在第 10 个信号运行一次，enabled 永久冻结
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.attention_system import AttentionScorer, Goal, ProcedureMemory
from core.cognitive_state import CognitiveState
from core.persona_layer import BehaviorType, PersonaLayer
from memory_chunk import MemoryChunk


# ============ 1. 长记忆不再被长度惩罚 ============

def test_long_memory_containing_query_scores_high():
    scorer = AttentionScorer()
    query = "pytest 失败后怎么继续"
    long_content = (
        "上次运行 pytest 失败后我们总结的流程是：先看失败的断言定位模块，"
        "再运行单个测试文件缩小范围，检查最近的 git diff 是否引入回归，"
        "必要时用 bisect 定位提交；确认修复后重新运行完整测试套件，"
        "并检查覆盖率没有下降。这个流程在三个项目上都验证有效。"
        "pytest 失败后怎么继续就按这个来。"
    )
    chunk = MemoryChunk(content=long_content, importance=0.5)

    score = scorer.score_memory(chunk, query, active_goal=None)
    assert score.query_relevance > 0.5, (
        f"完整包含查询的长记忆 query_relevance={score.query_relevance:.4f}——"
        "并集归一化的长度惩罚仍然存在"
    )
    assert score.final_score >= scorer.memory_threshold


def test_unrelated_memory_still_penalized():
    scorer = AttentionScorer()
    chunk = MemoryChunk(content="昨天买了一箱橘子，很甜。", importance=0.9)
    score = scorer.score_memory(chunk, "pytest 失败后怎么继续", active_goal=None)
    assert score.distraction_penalty > 0
    assert score.final_score < scorer.memory_threshold


# ============ 2. 程序门槛需要真实相关性 ============

def test_zero_relevance_procedure_blocked():
    scorer = AttentionScorer()
    proc = ProcedureMemory(
        title="发布 npm 包",
        steps=["构建", "打 tag", "npm publish"],
        triggers=["npm", "发布"],
        importance=0.9,
        confidence=0.9,
    )
    score = scorer.score_procedure(proc, "怎么煮出好喝的手冲咖啡", active_goal=None)
    assert score.distraction_penalty > 0
    assert score.final_score < scorer.procedure_threshold, (
        f"零相关程序凭 importance+confidence 得分 {score.final_score:.3f} "
        f"越过门槛 {scorer.procedure_threshold}"
    )


def test_triggered_procedure_passes():
    scorer = AttentionScorer()
    proc = ProcedureMemory(
        title="Safe PR workflow",
        steps=["run pytest", "check git diff", "commit", "push"],
        triggers=["pytest", "PR", "push"],
        importance=0.85,
        confidence=0.8,
    )
    goal = Goal(objective="修复测试失败并安全推送 PR")
    score = scorer.score_procedure(proc, "pytest 失败后怎么继续", active_goal=goal)
    assert score.final_score >= scorer.procedure_threshold


# ============ 3. 不确定性检测的词边界 ============

def test_statements_not_flagged_uncertain():
    state = CognitiveState()
    # "show"/"whatever"/"somewhat" 含 how/what 子串；普通中文陈述句
    for text in [
        "The demo will show the results tomorrow.",
        "Whatever happens, the pipeline is green.",
        "这个方案我们已经确认过了。",
        "他买了一台新电脑。",
    ]:
        assert not state._looks_uncertain(text), f"陈述句被误标为不确定: {text}"


def test_questions_flagged_uncertain():
    state = CognitiveState()
    for text in [
        "how do we proceed?",
        "Why did the test fail",
        "这个接口怎么调用",
        "为什么权重没有衰减",
        "明天可以上线吗",
        "结果对吗？",
    ]:
        assert state._looks_uncertain(text), f"疑问句未被识别: {text}"


# ============ 4. Persona 周期性重评 ============

def test_persona_reevaluates_after_initial_window():
    persona = PersonaLayer()
    behavior = BehaviorType.ACTIVE_RECALL

    # 前 10 个正信号 -> 兴趣度高 -> enabled=True
    for _ in range(10):
        persona.record_signal(behavior, "explicit_positive")
    pref = persona.profile.behaviors[behavior.value]
    assert pref.enabled is True

    # 随后持续负信号：到第 40 个信号（周期性重评点）时必须翻转
    for _ in range(30):
        persona.record_signal(behavior, "explicit_negative")
    pref = persona.profile.behaviors[behavior.value]
    assert pref.interest_score < 0.3
    assert pref.enabled is False, (
        "持续负反馈后 enabled 仍为 True——周期性重评从未运行，偏好被冻结"
    )
