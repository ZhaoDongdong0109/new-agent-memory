"""大脑主循环 v1 测试：会聊天、会调工具、会用自己的记忆

盘点确诊的四条断脉，各自的接通验证：
1. 多轮对话记忆：REPL 曾每轮失忆——ConversationBuffer 滚动保留
   近轮原文，历史进规划提示，溢出轮归档进长期记忆
2. 召回主通路：agent 曾只走 focus() 词元重叠、绕过生产检索且
   不记访问——现在 retrieve() 命中并入工作区，ACT-R/间隔效应
   在 agent 路径生效
3. 自动编码：encode 环节曾只有显式 remember 工具——现在每轮
   规则抽取事实进 add_memory 决策表，聊天改口走取代链
4. 工具参数 Schema：模型不用再猜参数名

全部测试无 LLM 依赖（规则规划器 + 捕获提示的假 LLM）。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import HumanLikeMemorySystem
from core.conversation import ConversationBuffer
from core.weight_system import MemoryType


def _make_system(tmp_path, **kwargs):
    kwargs.setdefault("enable_pii_detection", False)
    kwargs.setdefault("enable_audit_log", False)
    return HumanLikeMemorySystem(data_dir=str(tmp_path), **kwargs)


# ============ 会话工作记忆 ============

def test_conversation_buffer_rolls_and_renders():
    buf = ConversationBuffer(max_turns=4)
    for i in range(3):
        buf.add("user", f"问题{i}")
        buf.add("assistant", f"回答{i}")
    overflow = buf.pop_overflow()
    assert len(buf.turns) == 4
    assert len(overflow) == 2
    assert overflow[0].content == "问题0"
    rendered = buf.render_context()
    assert "问题2" in rendered and "回答2" in rendered
    assert rendered.index("问题1") < rendered.index("问题2"), "渲染应从旧到新"
    msgs = buf.to_messages()
    assert msgs[-1] == {"role": "assistant", "content": "回答2"}


def test_agent_keeps_multi_turn_context(tmp_path):
    """会话缓冲跨轮保留原文；历史块只服务提示、不残留 episode"""
    system = _make_system(tmp_path)
    agent = system.create_agent()

    ep1 = agent.run_turn("我在调试记忆系统的检索模块")
    assert agent.conversation.turns[0].content == "我在调试记忆系统的检索模块"
    assert ep1.result is not None

    ep2 = agent.run_turn("刚才那个模块叫什么")
    rendered = agent.conversation.render_context()
    assert "检索模块" in rendered and "刚才那个模块" in rendered
    # 历史块用完即弃：留在 metadata 里会随 episode 持久化，
    # 长聊时重复存储几十 KB（对抗审查实测）
    assert "conversation" not in ep2.observation.metadata


def test_overflow_turns_archived_to_memory(tmp_path):
    """溢出的旧对话轮归档为 INTERACTION 记忆（自然衰减）"""
    system = _make_system(tmp_path)
    agent = system.create_agent(auto_consolidate=False)
    agent.conversation.max_turns = 2
    agent.auto_extract = False

    agent.run_turn("第一句话谈到了滑雪计划")
    agent.run_turn("第二句话谈到了滑雪装备")

    archived = [
        c for c in system.core.chunks.values()
        if c.content.startswith("对话片段")
    ]
    assert archived, "溢出轮未归档进长期记忆"
    assert all(c.memory_type == MemoryType.INTERACTION for c in archived)


# ============ 生产检索接入召回 ============

def test_agent_recall_uses_production_retrieval(tmp_path):
    """agent 召回走 retrieve() 主通路：命中并入工作区且记访问"""
    system = _make_system(tmp_path)
    fact_id = system.add_memory(
        content="发布流程要先跑全部测试再用 workflow_dispatch 触发",
        memory_type=MemoryType.FACT, topics=["发布"], keywords=["发布", "流程"],
    )
    agent = system.create_agent(auto_consolidate=False)
    agent.auto_extract = False

    before = system.core.get(fact_id).access_count
    episode = agent.run_turn("发布流程怎么走")

    assert "workflow_dispatch" in episode.focus_context, "生产检索命中未进工作区"
    after = system.core.get(fact_id).access_count
    assert after > before, "agent 召回未记访问——ACT-R 频率/间隔效应在 agent 路径失效"


def test_recall_audit_recorded(tmp_path):
    """工作区审计里有生产检索的可解释记录"""
    system = _make_system(tmp_path)
    system.add_memory(content="崇礼的雪季是十一月到三月", topics=["滑雪"], keywords=["雪季"])
    agent = system.create_agent(auto_consolidate=False)
    agent.auto_extract = False
    obs = agent.observe("雪季是什么时候")
    workspace = agent.build_workspace(obs)
    stages = [a.get("stage") for a in workspace.audit if isinstance(a, dict)]
    assert "production_recall" in stages


# ============ 自动编码与取代链 ============

def test_chat_facts_auto_encoded_with_supersession(tmp_path):
    """聊天中陈述的事实自动进长期记忆；改口走取代链"""
    system = _make_system(tmp_path)
    agent = system.create_agent(auto_consolidate=False)

    agent.run_turn("小李现在住在里斯本")
    v1 = [c for c in system.core.chunks.values() if "里斯本" in c.content]
    assert v1, "聊天事实未被自动编码"
    assert v1[0].memory_type == MemoryType.FACT

    agent.run_turn("小李现在住在柏林")
    # 旧值被取代归档，新值为当前
    assert system.core.get(v1[0].id) is None, "旧事实未被取代"
    old = system.forgotten.get(v1[0].id)
    assert old is not None and old.metadata.get("superseded_by")

    result = system.retrieve("小李现在住在哪")
    contents = " ".join(c.content for c in result.chunks)
    assert "柏林" in contents and "里斯本" not in contents


def test_auto_encode_can_be_disabled(tmp_path):
    system = _make_system(tmp_path)
    agent = system.create_agent(auto_consolidate=False)
    agent.auto_extract = False
    agent.run_turn("小李现在住在里斯本")
    assert not [c for c in system.core.chunks.values() if "里斯本" in c.content]


# ============ 工具参数 Schema ============

def test_tool_parameters_flow_into_prompt(tmp_path):
    """注册带 Schema 的工具后，规划提示携带参数说明"""
    from core.llm_planner import LLMPlanner

    system = _make_system(tmp_path)
    agent = system.create_agent(auto_consolidate=False)
    agent.auto_extract = False
    agent.add_tool(
        "weather", "查询指定城市天气", lambda args: None,
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
    described = {t["name"]: t for t in agent.tools.describe()}
    assert described["weather"]["parameters"]["required"] == ["city"]
    assert "parameters" in described["respond"], "默认工具也应有 Schema"

    captured = {}

    def fake_llm(prompt, **kwargs):
        captured["prompt"] = prompt
        return '{"name": "respond", "arguments": {"message": "好的"}, "rationale": "test"}'

    planner = LLMPlanner(fake_llm)
    obs = agent.observe("北京天气怎么样")
    workspace = agent.build_workspace(obs)
    planner(obs, workspace, agent.tools)
    assert '"city"' in captured["prompt"], "参数 Schema 未进规划提示"


def test_conversation_block_in_planner_prompt(tmp_path):
    """多轮历史以独立块进提示，且不再混进 metadata JSON"""
    from core.llm_planner import LLMPlanner

    system = _make_system(tmp_path)
    agent = system.create_agent(auto_consolidate=False)
    agent.auto_extract = False

    captured = {}

    def fake_llm(prompt, **kwargs):
        captured["prompt"] = prompt
        return '{"name": "respond", "arguments": {"message": "好的"}, "rationale": "test"}'

    agent.planner = LLMPlanner(fake_llm)
    agent.run_turn("我在调试检索模块")
    agent.run_turn("它为什么慢")

    prompt = captured["prompt"]
    assert "Recent conversation" in prompt
    assert "调试检索模块" in prompt
    # metadata json 不应再包含整段历史（独立成块，避免双份）
    meta_line = [ln for ln in prompt.splitlines() if ln.startswith("metadata=")]
    assert meta_line and "调试检索模块" not in meta_line[0]


# ============ 对抗审查硬化回归 ============

def test_auto_encode_respects_negation(tmp_path):
    """"不要记住"的内容一个字也不进长期记忆"""
    system = _make_system(tmp_path)
    agent = system.create_agent(auto_consolidate=False)
    agent.run_turn("不要记住这件事，小王昨天去了医院")
    facts = [
        c for c in system.core.chunks.values()
        if c.memory_type == MemoryType.FACT
    ]
    assert not facts, "否定意图被自动编码绕过"


def test_auto_encode_skips_questions_and_requests(tmp_path):
    """疑问/祈使句不是事实，不得成为永久 FACT"""
    system = _make_system(tmp_path)
    agent = system.create_agent(auto_consolidate=False)
    agent.run_turn("小李现在住在哪？")
    agent.run_turn("帮我查一下小李的地址")
    facts = [
        c for c in system.core.chunks.values()
        if c.memory_type == MemoryType.FACT
    ]
    assert not facts, f"非陈述句被编码：{[c.content for c in facts]}"


def test_auto_encode_keeps_anchored_clause_only(tmp_path):
    """只有承载人物锚点的子句入库——填充语不得污染取代判定

    对抗审查复现过：'小李住在里斯本，周末我们约了饭' 与
    '小李养了一只猫，周末我们约了饭' 因共享填充语被误判同槽，
    养猫事实把住址事实取代掉了。
    """
    system = _make_system(tmp_path)
    agent = system.create_agent(auto_consolidate=False)
    agent.run_turn("小李现在住在里斯本，周末我们约了饭")
    agent.run_turn("小李最近养了一只猫，周末我们约了饭")

    contents = [c.content for c in system.core.chunks.values()
                if c.memory_type == MemoryType.FACT]
    assert any("里斯本" in c for c in contents), "住址事实丢失（被误取代）"
    assert any("猫" in c for c in contents), "养猫事实未编码"
    assert all("约了饭" not in c for c in contents), "填充语混入事实内容"


def test_auto_encode_provenance_and_modest_importance(tmp_path):
    """自动编码带来源标记，重要性/置信度低于显式写入"""
    system = _make_system(tmp_path)
    agent = system.create_agent(auto_consolidate=False)
    agent.run_turn("小李现在住在里斯本")
    fact = next(c for c in system.core.chunks.values() if "里斯本" in c.content)
    assert fact.source == "system_extract"
    assert fact.metadata.get("auto_encoded") is True
    assert fact.confidence < 0.8
    assert fact.importance <= 0.55, "自动编码的重要性不得超过显式 remember"


def test_archived_dialogue_not_recalled_into_workspace(tmp_path):
    """归档对话片段不被生产召回捞回——防自激励污染回环"""
    system = _make_system(tmp_path)
    agent = system.create_agent(auto_consolidate=False)
    agent.auto_extract = False
    agent.conversation.max_turns = 2
    agent.run_turn("崇礼滑雪场的雪季什么时候开始")
    agent.run_turn("滑雪装备去哪里租比较好")

    archives = [c for c in system.core.chunks.values()
                if c.metadata.get("conversation_archive")]
    assert archives, "溢出轮应已归档"

    obs = agent.observe("崇礼滑雪场什么时候开门")
    workspace = agent.build_workspace(obs)
    merged = [m for m in workspace.memories if m.reason == "hybrid-retrieval"]
    archive_ids = {c.id for c in archives}
    assert not [m for m in merged if m.id in archive_ids], (
        "归档对话片段被生产召回捞回工作区"
    )


def test_runtime_scaffold_uses_original_task_for_recall(tmp_path):
    """runtime 脚手架文本不得作为检索查询——用原始任务查

    多 KB 样板文本做查询会给无关记忆刷访问计数、连虚假
    Hebbian 边（对抗审查端到端复现）。
    """
    system = _make_system(tmp_path)
    agent = system.create_agent(auto_consolidate=False)
    agent.auto_extract = False

    queries = []
    original_retrieve = system.retrieve

    def spy(query, *args, **kwargs):
        queries.append(query)
        return original_retrieve(query, *args, **kwargs)

    system.retrieve = spy
    scaffold = (
        "CognitiveRuntime step 2/4\nOriginal user task: 缓存策略怎么选\n"
        "Runtime rules: respond, finish, remember, introspect\n" * 30
    )
    obs = agent.observe(scaffold, metadata={"runtime": True, "original_task": "缓存策略怎么选"})
    agent.build_workspace(obs)
    assert queries == ["缓存策略怎么选"], (
        f"检索查询应是原始任务，实际：{[q[:60] for q in queries]}"
    )
