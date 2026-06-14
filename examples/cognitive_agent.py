"""Minimal cognitive agent loop.

This example shows a digital body loop:

Observe -> Focus -> Act -> Evaluate -> Remember -> Consolidate
"""

from new_agent_memory import ActionResult, AgentAction, CognitiveAgent, HumanLikeMemorySystem


def main():
    memory = HumanLikeMemorySystem()
    memory.start_goal(
        "帮助用户把项目推进到可运行的 Agent 原型",
        constraints=["先构建注意力工作区", "行动后记录经验", "把成功经验巩固成程序记忆"],
        open_loops=["需要证明闭环能跑通"],
    )

    agent = memory.create_agent(name="hermes-seed")

    def planner(observation, workspace, tools):
        if "检查" in observation.content:
            return AgentAction(
                name="inspect",
                arguments={
                    "input": observation.content,
                    "focus": workspace.to_prompt_context(),
                },
                rationale="用户要求检查，因此调用 inspect 工具。",
            )
        return AgentAction(name="respond", arguments={"message": observation.content})

    def inspect_tool(arguments):
        focus = arguments.get("focus", "")
        return ActionResult(
            success=True,
            output="检查完成。当前注意力上下文长度：" + str(len(focus)),
            cost=0.05,
        )

    agent.planner = planner
    agent.add_tool("inspect", "Inspect the current focus workspace.", inspect_tool, cost=0.05)

    episode = agent.run_turn("请检查当前 Agent 闭环是否能工作")

    print("Action:", episode.action.to_dict())
    print("Result:", episode.result.to_dict())
    print("Reward:", episode.reward)
    print("Lesson:", episode.lesson)
    print("Next policy:", episode.next_policy)
    print("Memory stats:", memory.get_memory_stats()["core_chunks"])
    print("Procedures:", len(memory.get_attention_summary()["procedures"]))


if __name__ == "__main__":
    main()
