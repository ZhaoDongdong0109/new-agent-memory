"""LLMPlanner example with a fake LLM.

Replace `fake_llm` with OpenAI, Hermes, or a local model call:

    planner = LLMPlanner(lambda prompt: hermes.generate(prompt))

or:

    from openai import OpenAI
    planner = LLMPlanner.from_openai_responses(OpenAI(), "gpt-4.1-mini")
"""

from new_agent_memory import ActionResult, CognitiveAgent, HumanLikeMemorySystem, LLMPlanner


def main():
    memory = HumanLikeMemorySystem()
    memory.start_goal(
        "用 LLM 决策下一步行动",
        constraints=["只返回可用工具", "行动后记录经验"],
    )

    agent = memory.create_agent(name="llm-agent-seed", auto_consolidate=False)

    def inspect_tool(arguments):
        return ActionResult(
            success=True,
            output=f"inspected: {arguments.get('target', 'unknown')}",
            cost=0.05,
        )

    agent.add_tool("inspect", "Inspect a target string.", inspect_tool)

    def fake_llm(prompt):
        # A real LLM should inspect the prompt and choose from Available tools.
        return """
{
  "name": "inspect",
  "arguments": {"target": "focus workspace"},
  "rationale": "The user asked for inspection, so use the inspect tool."
}
"""

    agent.planner = LLMPlanner(fake_llm)
    episode = agent.run_turn("请检查当前注意力工作区")

    print("Action:", episode.action.to_dict())
    print("Result:", episode.result.to_dict())
    print("Lesson:", episode.lesson)


if __name__ == "__main__":
    main()
