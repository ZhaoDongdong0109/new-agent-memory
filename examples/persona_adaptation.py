"""Persona adaptation example."""

from new_agent_memory import HumanLikeMemorySystem


def main():
    memory = HumanLikeMemorySystem()

    print("initial:", memory.get_persona_summary()["active_recall_preference"])

    memory.on_active_recall_explicit_positive()
    memory.on_active_recall_explicit_positive()
    print("after positive feedback:", memory.get_persona_summary()["active_recall_preference"])

    for _ in range(10):
        memory.on_active_recall_explicit_negative()

    print("after repeated negative feedback:", memory.get_persona_summary()["active_recall_preference"])
    print("should trigger active recall:", memory.should_trigger_active_recall())


if __name__ == "__main__":
    main()
