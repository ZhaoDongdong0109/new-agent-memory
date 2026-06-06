"""Goal-driven attention workspace example."""

from new_agent_memory import HumanLikeMemorySystem, MemoryType


def main():
    system = HumanLikeMemorySystem()

    system.start_goal(
        "修复测试失败并安全推送 PR",
        constraints=["先跑 pytest", "检查 git diff", "不要直接改 main"],
        open_loops=["PR 还没创建"],
    )

    system.add_memory(
        content="上次 maintain() 失败是因为核心层缺少 decay_all_unused()。",
        memory_type=MemoryType.FACT,
        topics=["pytest", "maintenance", "bugfix"],
        importance=0.75,
    )
    system.add_memory(
        content="用户喜欢直接给出结论，不喜欢太长的解释。",
        memory_type=MemoryType.PREFERENCE,
        topics=["style"],
        importance=0.9,
    )

    system.add_procedure(
        title="Safe PR workflow",
        steps=["run pytest", "run examples", "check git diff", "commit", "push branch"],
        triggers=["pytest", "tests", "PR", "push"],
        importance=0.85,
        confidence=0.8,
    )

    workspace = system.focus("pytest 失败后怎么继续")

    print(workspace.to_prompt_context())
    print("\nAudit:")
    for item in workspace.audit[:5]:
        print(item)


if __name__ == "__main__":
    main()
