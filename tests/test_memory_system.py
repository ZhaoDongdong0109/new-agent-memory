import time

from new_agent_memory import HumanLikeMemorySystem, MemoryLayer, MemoryType


def test_add_and_retrieve_memory():
    system = HumanLikeMemorySystem()

    system.add_memory(
        content="今天中午和客户在北京餐厅吃了烤鸭，聊了项目预算。",
        memory_type=MemoryType.INTERACTION,
        time_absolute="2026-04-29",
        time_context="中午",
        location="北京",
        persons=["客户"],
        topics=["food", "business", "project"],
        importance=0.8,
    )

    result = system.retrieve("中午在北京吃了什么")

    assert result.success is True
    assert result.retrieval_path == "core"
    assert "烤鸭" in result.assembled_content


def test_maintain_is_safe_and_degrades_weak_old_memory():
    system = HumanLikeMemorySystem()
    memory_id = system.add_memory(
        content="一条很久没有访问、几乎没有连接价值的临时记忆。",
        memory_type=MemoryType.INTERACTION,
        topics=["temporary"],
        importance=0.0,
    )

    chunk = system.core.get(memory_id)
    assert chunk is not None
    old_time = time.time() - 90 * 24 * 3600
    chunk.created_at = old_time
    chunk.last_accessed = old_time
    chunk.connection_value = 0.0
    chunk.emotion_valence = 0.0
    chunk.emotion_intensity = 0.0

    system.maintain()

    assert system.core.get(memory_id) is None
    assert system.forgotten.get(memory_id) is not None


def test_forgotten_memory_can_be_woken_by_cues():
    system = HumanLikeMemorySystem()

    system.add_memory(
        content="去年在上海机场遇到老同学，聊起大学机器人比赛。",
        memory_type=MemoryType.STORY,
        time_relative="去年",
        location="机场",
        persons=["老同学"],
        topics=["travel", "school", "robotics"],
        importance=0.25,
        target_layer=MemoryLayer.FORGOTTEN,
    )

    result = system.retrieve("去年在机场和老同学一起聊了什么")

    assert result.success is True
    assert result.retrieval_path == "forgotten"
    assert "机器人比赛" in result.assembled_content


def test_save_and_load_round_trip(tmp_path):
    data_dir = tmp_path / "memory_data"
    system = HumanLikeMemorySystem(data_dir=str(data_dir))
    system.add_memory(content="用户喜欢短句回复。", memory_type=MemoryType.PREFERENCE, topics=["preference"])
    system.save()

    loaded = HumanLikeMemorySystem(data_dir=str(data_dir))

    assert loaded.load() is True
    assert loaded.get_memory_stats()["core_chunks"] == 1


def test_positive_persona_feedback_increases_interest():
    system = HumanLikeMemorySystem()
    before = system.get_persona_summary()["active_recall_preference"]["interest_score"]

    after = system.on_active_recall_explicit_positive()

    assert after > before
