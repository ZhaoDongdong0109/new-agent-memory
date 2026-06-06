"""Pseudo-forgotten recall example.

A weak memory is archived into the forgotten layer. It is not part of the
active core search, but a strong cue can still wake it.
"""

from new_agent_memory import HumanLikeMemorySystem, MemoryLayer, MemoryType


def main():
    memory = HumanLikeMemorySystem()

    memory.add_memory(
        content="去年在上海机场遇到老同学，聊起了大学时一起做机器人比赛。",
        memory_type=MemoryType.STORY,
        time_relative="去年",
        location="机场",
        persons=["老同学"],
        topics=["travel", "school", "robotics"],
        importance=0.25,
        target_layer=MemoryLayer.FORGOTTEN,
    )

    result = memory.retrieve("去年在机场和老同学一起聊了什么")

    print(result.summary())
    print(result.assembled_content)


if __name__ == "__main__":
    main()
