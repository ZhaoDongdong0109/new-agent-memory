"""Memory-R1 协议适配器测试：

验证将 ACL 2026 Memory-R1 的两阶段（原子抽取 + 整数 ID 映射 ADD/UPDATE/DELETE/NONE）
与 new-agent-memory 的双时态链、伪遗忘层归档及 ACT-R 访问强化对接后的真实效果。
"""

from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.llm_planner import LLMError  # noqa: E402
from core.memory_r1_manager import MemoryR1Manager  # noqa: E402
from main import HumanLikeMemorySystem  # noqa: E402


def _make_system(tmp_path):
    return HumanLikeMemorySystem(
        data_dir=str(tmp_path / "mem"),
        enable_pii_detection=False,
        enable_audit_log=False,
    )


def test_memory_r1_paraphrase_update_preserves_bitemporal_chain(tmp_path):
    """核心痛点验证：词面零城市重叠的改述（规则 Jaccard 盲区），经 Memory-R1 UPDATE
    既完成当前值更新，又保留双时态历史链（原版 Memory-R1 原地覆写会丢失旧值）。"""
    system = _make_system(tmp_path)
    turn = {"n": 0}

    def fake_extractor(prompt):
        turn["n"] += 1
        if turn["n"] == 1:
            return '{"facts": ["用户目前在上海浦东租房住"]}'
        return '{"facts": ["用户已经搬去帝都海淀定居了"]}'

    def fake_manager(prompt):
        if "帝都海淀" not in prompt:
            return json.dumps(
                {
                    "memory": [
                        {
                            "id": "0",
                            "text": "用户目前在上海浦东租房住",
                            "event": "ADD",
                            "memory_type": "fact",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        # 校验整数 ID 映射确实把旧记忆编码成了 "0"
        assert '"id": "0"' in prompt and "上海浦东" in prompt
        return json.dumps(
            {
                "memory": [
                    {
                        "id": "0",
                        "text": "用户目前在北京海淀定居",
                        "event": "UPDATE",
                        "memory_type": "fact",
                        "old_memory": "用户目前在上海浦东租房住",
                    }
                ]
            },
            ensure_ascii=False,
        )

    mgr = MemoryR1Manager(system, manager_llm=fake_manager, extractor_llm=fake_extractor)

    r1 = mgr.ingest_turn("我这阵子在魔都浦东那边租房子住")
    assert len(r1.operations) == 1 and r1.operations[0].event == "ADD"
    old_id = r1.operations[0].chunk_id

    r2 = mgr.ingest_turn("换工作啦，上礼拜彻底搬去帝都海淀安家了")
    assert len(r2.operations) == 1 and r2.operations[0].event == "UPDATE"
    new_id = r2.operations[0].chunk_id

    # 当前核心层只有新事实，且 parent_id 指向归档在伪遗忘层的旧事实
    assert system.core.get(old_id) is None
    new_chunk = system.core.get(new_id)
    old_archived = system.forgotten.get(old_id)
    assert new_chunk is not None and new_chunk.parent_id == old_id
    assert old_archived is not None and old_archived.invalid_at is not None
    assert old_archived.metadata.get("superseded_by") == new_id


def test_memory_r1_delete_is_auditable_soft_archive_and_hallucinated_id_ignored(tmp_path):
    """DELETE 走伪遗忘层软归档而非物理销毁；模型幻觉出的越界整数 ID 被安全忽略。"""
    system = _make_system(tmp_path)
    seed_id = system.add_memory(content="用户对花生严重过敏")

    mgr = MemoryR1Manager(
        system,
        extractor_llm=lambda p: '{"facts": ["脱敏治疗结束，不再对花生过敏"]}',
        manager_llm=lambda p: json.dumps(
            {
                "memory": [
                    {"id": "0", "text": "用户对花生严重过敏", "event": "DELETE"},
                    {"id": "999", "text": "幻觉出的不存在条目", "event": "UPDATE"},
                ]
            },
            ensure_ascii=False,
        ),
    )

    rep = mgr.ingest_turn("我已经做完脱敏治疗，现在对花生不过敏了")
    assert [op.event for op in rep.operations] == ["DELETE"]
    assert system.core.get(seed_id) is None
    archived = system.forgotten.get(seed_id)
    assert archived is not None
    assert archived.metadata.get("retracted_by") == "memory_r1"


def test_agent_with_memory_r1_manager_and_fallback_to_rules_on_error(tmp_path):
    """挂载到 CognitiveAgent 后：正常时走 Memory-R1 跨会话合并，故障时无感回退本地规则抽取。"""
    system = _make_system(tmp_path)
    broken = {"on": False}

    def extractor(prompt):
        if broken["on"]:
            raise LLMError("vllm server down")
        return '{"facts": ["喜欢芝士披萨和鸡肉披萨"]}'

    def manager(prompt):
        return json.dumps(
            {
                "memory": [
                    {
                        "id": "0",
                        "text": "喜欢芝士披萨和鸡肉披萨",
                        "event": "ADD",
                        "memory_type": "preference",
                    }
                ]
            },
            ensure_ascii=False,
        )

    r1_mgr = MemoryR1Manager(system, manager_llm=manager, extractor_llm=extractor)
    agent = system.create_agent(auto_consolidate=False)
    agent.memory_manager = r1_mgr

    agent.run_turn("我超爱芝士披萨，最近也迷上了鸡肉披萨")
    assert any("芝士披萨和鸡肉披萨" in c.content for c in system.core.chunks.values())

    # 否定意图依然在入口处被硬门控拦截，绝不触发外部模型写入
    before_cnt = len(system.core.chunks)
    agent.run_turn("不要记住这句话：我的临时密码是 1234")
    assert len(system.core.chunks) == before_cnt

    # 外部 vLLM 故障时自动降级回本地规则抽取，对话不中断且事实照常入库
    broken["on"] = True
    agent.run_turn("小李现在住在里斯本")
    assert any("里斯本" in c.content for c in system.core.chunks.values())


def test_memory_r1_distill_chunks_filters_noise(tmp_path):
    """Stage 2 Answer-Agent 记忆蒸馏：剔除与当前问题无关的检索噪声块。"""
    system = _make_system(tmp_path)
    id_keep = system.add_memory(content="项目发布日定在周五下午三点")
    system.add_memory(content="周五中午食堂有红烧肉")

    mgr = MemoryR1Manager(
        system,
        manager_llm=lambda p: '{"selected_ids": ["0"]}',
    )
    candidates = [system.core.get(id_keep), [c for c in system.core.chunks.values() if c.id != id_keep][0]]
    distilled = mgr.distill_chunks("项目什么时候发布？", candidates)
    assert len(distilled) == 1
    assert distilled[0].id == id_keep
