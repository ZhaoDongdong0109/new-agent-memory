"""OpenAI-compatible ChatCompletions agent example.

Create a .env file first:

    OPENAI_API_KEY=your-key-or-local-placeholder
    OPENAI_BASE_URL=https://api.openai.com/v1
    OPENAI_MODEL=gpt-4.1-mini

For local gateways, set OPENAI_BASE_URL to your /v1 endpoint, for example
http://localhost:1234/v1.
"""

from new_agent_memory import HumanLikeMemorySystem


def main():
    memory = HumanLikeMemorySystem()
    memory.start_goal(
        "Help the user evolve this project into a downloadable memory-driven agent.",
        constraints=["use memory", "inspect cognitive state", "prefer small reversible steps"],
    )

    agent = memory.create_openai_agent(auto_consolidate=True)
    episode = agent.run_turn("你现在是谁？请用一句话说明你能做什么。")

    print(episode.result.output)
    memory.save()


if __name__ == "__main__":
    main()
