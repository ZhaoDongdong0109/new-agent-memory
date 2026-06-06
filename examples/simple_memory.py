"""Minimal example: add a memory and retrieve it."""

from new_agent_memory import HumanLikeMemorySystem, MemoryType


def main():
    memory = HumanLikeMemorySystem()

    memory.add_memory(
        content="今天中午和客户在北京餐厅吃了烤鸭，聊了项目预算。",
        memory_type=MemoryType.INTERACTION,
        time_absolute="2026-04-29",
        time_context="中午",
        location="北京",
        persons=["客户"],
        topics=["food", "business", "project"],
        keywords=["烤鸭", "预算"],
        emotion_valence=0.3,
        emotion_intensity=0.6,
        importance=0.8,
    )

    result = memory.retrieve("中午在北京吃了什么")

    print(result.summary())
    print(result.assembled_content)


if __name__ == "__main__":
    main()
