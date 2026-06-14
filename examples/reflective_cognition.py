"""Reflective self/world model example."""

from new_agent_memory import CognitiveAgent, HumanLikeMemorySystem


def main():
    memory = HumanLikeMemorySystem()
    memory.start_goal(
        "Grow a memory-driven agent into a self-correcting intelligence",
        constraints=["observe before acting", "prefer reversible actions", "learn from outcomes"],
    )

    agent = CognitiveAgent(memory_system=memory, auto_consolidate=False)

    first = agent.run_turn("How should I safely test a new agent capability?")
    print("first action:", first.action.name)
    print("first reward:", first.reward)
    print()

    second = agent.run_turn("please introspect current state")
    print(second.result.output)
    print()

    workspace = memory.focus("What should the agent improve next?")
    print(workspace.to_prompt_context())


if __name__ == "__main__":
    main()
