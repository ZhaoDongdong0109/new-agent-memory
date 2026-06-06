from new_agent_memory import HumanLikeMemorySystem, MemoryType


def test_focus_workspace_prefers_goal_relevant_memory_and_procedure():
    system = HumanLikeMemorySystem()
    system.start_goal(
        "修复 pytest failing 后推送 PR",
        constraints=["先跑测试", "不要破坏 main"],
        open_loops=["PR 还没创建"],
    )

    relevant_id = system.add_memory(
        content="上次 pytest 失败是因为 maintain 调用了不存在的方法。",
        memory_type=MemoryType.FACT,
        topics=["pytest", "tests", "maintenance"],
        importance=0.7,
    )
    irrelevant_id = system.add_memory(
        content="用户喜欢短句回复。",
        memory_type=MemoryType.PREFERENCE,
        topics=["style"],
        importance=1.0,
    )
    procedure = system.add_procedure(
        title="Push-safe test loop",
        steps=["run pytest", "run examples", "check git diff", "push branch"],
        triggers=["pytest", "tests", "push", "PR"],
        importance=0.8,
        confidence=0.8,
    )

    workspace = system.focus("pytest 失败后怎么处理")

    memory_ids = {item.id for item in workspace.memories}
    procedure_ids = {item.id for item in workspace.procedures}

    assert relevant_id in memory_ids
    assert irrelevant_id not in memory_ids
    assert procedure.id in procedure_ids
    assert workspace.active_goal is not None
    assert "pytest" in workspace.to_prompt_context()


def test_attention_state_round_trips(tmp_path):
    data_dir = tmp_path / "memory_data"
    system = HumanLikeMemorySystem(data_dir=str(data_dir))
    goal = system.start_goal("整理 README 并发布 v0.1")
    procedure = system.add_procedure(
        title="Release checklist",
        steps=["run tests", "tag release", "write changelog"],
        triggers=["release", "v0.1"],
    )
    system.save()

    loaded = HumanLikeMemorySystem(data_dir=str(data_dir))

    assert loaded.load() is True
    summary = loaded.get_attention_summary()
    assert summary["active_goal"]["id"] == goal.id
    assert summary["procedures"][0]["id"] == procedure.id
