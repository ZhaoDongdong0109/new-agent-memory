#!/usr/bin/env python3
"""
使用本地 LLM 进行记忆抽取示例

使用方式：
    # 1. 配置 .env 文件
    LOCAL_LLM_API_BASE=http://localhost:8080/v1
    LOCAL_LLM_MODEL=your-4b-model

    # 2. 运行示例
    python examples/local_llm_extraction.py
"""

import os
import sys

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.local_llm import LocalLLM, create_memory_extractor_llm
from core.memory_extractor import MemoryExtractor


def main():
    print("=" * 60)
    print("本地 LLM 记忆抽取示例")
    print("=" * 60)
    print()

    # 检查配置
    api_base = os.environ.get("LOCAL_LLM_API_BASE")
    if not api_base:
        print("请配置 LOCAL_LLM_API_BASE 环境变量")
        print("示例：export LOCAL_LLM_API_BASE=http://localhost:8080/v1")
        return

    print(f"API 地址: {api_base}")
    print(f"模型: {os.environ.get('LOCAL_LLM_MODEL', 'local-model')}")
    print()

    # 创建 LLM 函数
    llm_fn = create_memory_extractor_llm()
    if not llm_fn:
        print("创建 LLM 函数失败")
        return

    # 创建 MemoryExtractor
    extractor = MemoryExtractor(llm_fn=llm_fn)

    # 测试文本
    test_texts = [
        "今天中午和张三在北京餐厅吃了烤鸭，很开心。张三说这家店的烤鸭是最正宗的。",
        "昨天下午3点在上海参加了技术会议，讨论了新项目的架构设计。会议决定使用微服务架构。",
        "上周出差去深圳见了客户李总，签了一个50万的合同。李总对我们的方案很满意。",
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"--- 测试 {i} ---")
        print(f"输入: {text}")
        print()

        # 模拟 episode
        class MockObservation:
            def __init__(self, content):
                self.content = content
                self.source = "user"
                self.metadata = {}
                self.timestamp = 1234567890.0

        class MockAction:
            def __init__(self):
                self.name = "respond"
                self.arguments = {}
                self.rationale = ""

        class MockResult:
            def __init__(self):
                self.success = True
                self.output = "收到"
                self.cost = 0.1
                self.metadata = {}

        class MockEpisode:
            def __init__(self, content):
                self.id = f"exp_{i}"
                self.goal = "记录用户分享"
                self.observation = MockObservation(content)
                self.action = MockAction()
                self.result = MockResult()
                self.reward = 0.8
                self.lesson = ""
                self.next_policy = ""
                self.focus_context = ""
                self.created_at = 1234567890.0

        episode = MockEpisode(text)

        # 抽取记忆
        result = extractor.extract(episode)

        print(f"抽取结果: {len(result)} 个记忆")
        for spec in result.specs:
            print(f"  类型: {spec.memory_type.value}")
            print(f"  内容: {spec.content[:100]}...")
            print(f"  人物: {spec.persons}")
            print(f"  地点: {spec.location}")
            print(f"  主题: {spec.topics}")
            print(f"  重要性: {spec.importance}")
            print()

    print("=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
