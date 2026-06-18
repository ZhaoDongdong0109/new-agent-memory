"""
抽取系统测试

测试 EntityExtractor、MemoryExtractor、ConsolidationEngine、ContextCompressor。
"""

import sys
import time

sys.path.insert(0, '.')

from core.entity_extractor import EntityExtractor
from core.memory_spec import MemorySpec, ExtractionResult
from core.memory_extractor import MemoryExtractor
from core.context_compressor import ContextCompressor
from core.weight_system import MemoryType


def test_entity_extractor():
    """测试实体抽取器"""
    print("=== 测试实体抽取器 ===")

    extractor = EntityExtractor()

    # 测试人物抽取
    text1 = "今天中午和张三一起在北京吃了烤鸭"
    persons = extractor.extract_persons(text1)
    print(f"人物抽取: {text1} -> {persons}")
    assert "张三" in persons

    # 测试地点抽取
    location = extractor.extract_location(text1)
    print(f"地点抽取: {text1} -> {location}")
    assert location == "北京"

    # 测试时间抽取
    text2 = "昨天下午3点在上海参加了技术会议"
    time_abs, time_rel, time_ctx = extractor.extract_time(text2)
    print(f"时间抽取: {text2} -> abs={time_abs}, rel={time_rel}, ctx={time_ctx}")
    assert time_rel == "昨天"
    assert time_ctx == "下午"

    # 测试主题抽取
    topics = extractor.extract_topics(text1)
    print(f"主题抽取: {text1} -> {topics}")
    assert "美食" in topics or "餐饮" in topics

    # 测试关键词抽取
    keywords = extractor.extract_keywords(text1, top_k=5)
    print(f"关键词抽取: {text1} -> {keywords}")
    assert len(keywords) > 0

    # 测试情绪抽取
    text3 = "今天很开心，项目成功了！"
    valence, intensity = extractor.extract_emotion(text3)
    print(f"情绪抽取: {text3} -> valence={valence}, intensity={intensity}")
    assert valence > 0

    # 测试综合抽取
    result = extractor.extract_all(text1)
    print(f"综合抽取: {result}")
    assert "张三" in result["persons"]
    assert result["location"] == "北京"

    print("✅ 实体抽取器测试通过\n")


def test_memory_spec():
    """测试 MemorySpec"""
    print("=== 测试 MemorySpec ===")

    # 创建 MemorySpec
    spec = MemorySpec(
        content="今天中午和张三在北京吃了烤鸭",
        summary="和张三吃烤鸭",
        memory_type=MemoryType.FACT,
        time_absolute="2026-06-18",
        time_context="中午",
        location="北京",
        persons={"张三"},
        topics={"美食", "餐饮"},
        keywords={"烤鸭", "北京"},
        emotion_valence=0.5,
        emotion_intensity=0.7,
        importance=0.8,
    )

    # 测试 to_dict
    d = spec.to_dict()
    print(f"to_dict: {d}")
    assert d["content"] == "今天中午和张三在北京吃了烤鸭"
    assert "张三" in d["persons"]

    # 测试 from_dict
    spec2 = MemorySpec.from_dict(d)
    print(f"from_dict: {spec2}")
    assert spec2.content == spec.content
    assert spec2.persons == spec.persons

    print("✅ MemorySpec 测试通过\n")


def test_extraction_result():
    """测试 ExtractionResult"""
    print("=== 测试 ExtractionResult ===")

    result = ExtractionResult(episode_id="exp_001")

    # 添加 spec
    spec1 = MemorySpec(content="事实1", memory_type=MemoryType.FACT)
    spec2 = MemorySpec(content="经验1", memory_type=MemoryType.STORY)

    result.add_spec(spec1)
    result.add_spec(spec2)

    print(f"长度: {len(result)}")
    assert len(result) == 2

    # 测试迭代
    specs = list(result)
    print(f"迭代: {[s.content for s in specs]}")
    assert len(specs) == 2

    print("✅ ExtractionResult 测试通过\n")


def test_memory_extractor():
    """测试 MemoryExtractor"""
    print("=== 测试 MemoryExtractor ===")

    extractor = MemoryExtractor()

    # 模拟 episode
    class MockObservation:
        def __init__(self):
            self.content = "今天中午和张三在北京吃了烤鸭，很开心"
            self.source = "user"
            self.metadata = {}
            self.timestamp = time.time()

    class MockAction:
        def __init__(self):
            self.name = "respond"
            self.arguments = {}
            self.rationale = "用户分享了午餐经历"

    class MockResult:
        def __init__(self):
            self.success = True
            self.output = "收到！听起来很好吃"
            self.cost = 0.1
            self.metadata = {}

    class MockEpisode:
        def __init__(self):
            self.id = "exp_001"
            self.goal = "回应用户"
            self.observation = MockObservation()
            self.action = MockAction()
            self.result = MockResult()
            self.reward = 0.8
            self.lesson = "用户喜欢分享美食经历"
            self.next_policy = "当用户分享美食时，表示兴趣并询问细节"
            self.focus_context = ""
            self.created_at = time.time()

    episode = MockEpisode()

    # 测试抽取
    result = extractor.extract(episode)
    print(f"抽取结果: {len(result)} 个记忆")

    for spec in result.specs:
        print(f"  - 类型: {spec.memory_type.value}, 内容: {spec.content[:50]}...")
        print(f"    人物: {spec.persons}, 地点: {spec.location}")
        print(f"    主题: {spec.topics}, 重要性: {spec.importance}")

    assert len(result) > 0
    assert any(spec.memory_type == MemoryType.FACT for spec in result.specs) or \
           any(spec.memory_type == MemoryType.STORY for spec in result.specs)

    print("✅ MemoryExtractor 测试通过\n")


def test_context_compressor():
    """测试 ContextCompressor"""
    print("=== 测试 ContextCompressor ===")

    compressor = ContextCompressor(max_tokens=500)

    # 测试消息压缩
    messages = [
        {"message_type": "text", "sender_name": "用户", "content": "今天中午吃了什么？"},
        {"message_type": "text", "sender_name": "Claude", "content": "你吃了烤鸭！"},
        {"message_type": "tool_call", "metadata": {"tool_name": "search"}, "content": "搜索烤鸭"},
        {"message_type": "tool_result", "metadata": {"tool_name": "search", "success": True}, "content": "找到3条结果"},
        {"message_type": "text", "sender_name": "用户", "content": "对，很好吃！"},
        {"message_type": "text", "sender_name": "Claude", "content": "下次一起去！"},
    ]

    compressed = compressor.compress(messages)
    print(f"压缩结果:\n{compressed}")
    assert len(compressed) > 0
    assert "工具日志" in compressed or "对话" in compressed

    # 测试 episode 压缩
    episodes = [
        {"goal": "回应用户", "action": "respond", "success": True, "lesson": "用户喜欢美食"},
        {"goal": "搜索信息", "action": "search", "success": True, "lesson": "搜索需要具体关键词"},
    ]

    compressed_episodes = compressor.compress_episodes(episodes)
    print(f"\nEpisode 压缩:\n{compressed_episodes}")
    assert len(compressed_episodes) > 0

    print("✅ ContextCompressor 测试通过\n")


if __name__ == "__main__":
    test_entity_extractor()
    test_memory_spec()
    test_extraction_result()
    test_memory_extractor()
    test_context_compressor()
    print("🎉 所有测试通过！")
