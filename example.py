#!/usr/bin/env python3
"""
AIMemorySystem - 完整示例
演示如何使用重构后的 AI 记忆系统
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_memory_system import AIMemorySystem


def main():
    print("=" * 70)
    print("AI 记忆系统 - 完整示例")
    print("=" * 70)
    
    print("\n[初始化系统...")
    system = AIMemorySystem(max_context_tokens=4096)
    
    print("\n[1] 添加示例对话...")
    conversations = [
        ("我叫张三，我喜欢吃火锅，尤其是麻辣火锅",
         "你好张三！很高兴认识你，我也很喜欢火锅！"),
        
        ("我上周刚去成都旅游，吃了当地的火锅，还逛了宽窄巷子和春熙路",
         "听起来很精彩！成都美食和景点都很棒！"),
        
        ("我在成都拍了很多照片，记录了当地的美食和风景",
         "太棒了！成都的美食和风景确实值得记录"),
        
        ("我正在学习Python编程，希望能够做一些有趣的项目",
         "Python是很棒的编程语言！有什么具体的项目想法吗？"),
        
        ("我想做一个聊天机器人，还想了解AI记忆系统",
         "非常有意思的想法！"),
        
        ("我喜欢打羽毛球，每周都和朋友一起运动",
         "运动很健康！"),
        
        ("我最近在看一本书，书名叫《未来简史》，讲的是AI的未来发展",
         "尤瓦尔·赫拉利的书都很有深度！")
    ]
    
    for user, assistant in conversations:
        system.add_conversation(user, assistant)
    
    print(f"  已添加 {len(conversations)} 轮对话")
    
    print("\n[2] 测试相关记忆检索...")
    test_queries = [
        "我叫什么名字？",
        "我喜欢吃什么？",
        "我最近在做什么项目？",
        "我旅游去了哪里？"
    ]
    
    for query in test_queries:
        print(f"\n  Query: {query}")
        response = system.get_relevant_context(query)
        print(f"  找到 {len(response.context_window.items)} 条相关记忆")
        print(f"  使用了 {response.context_window.total_tokens} tokens")
        print(f"  节省: {response.stats['context_savings_percent']:.1f}% 上下文")
    
    print("\n[3] 构建 LLM 提示词示例...")
    prompt = system.build_llm_prompt(
        "你记得我喜欢做什么运动？",
        system_prompt="请根据用户的历史记忆回答问题"
    )
    print("构建的提示词:")
    print("-" * 50)
    print(prompt[:500])
    print("-" * 50)
    
    print("\n[4] 系统统计信息...")
    stats = system.get_stats()
    print(f"  工作记忆: {stats['hierarchy']['working_memory_count']}")
    print(f"  近期记忆: {stats['hierarchy']['recent_memory_count']}")
    print(f"  总记忆数: {stats['hierarchy']['total_memories']}")
    print(f"  上下文节省: {stats['context_savings_percent']:.1f}%")
    
    print("\n" + "=" * 70)
    print("系统运行成功！")
    print("=" * 70)
    
    print("\n快速开始指南:")
    print("  1. system = AIMemorySystem()")
    print("  2. system.add_conversation(user_input, assistant_response)")
    print("  3. prompt = system.build_llm_prompt(\"用户问题\")")
    print("  4. 把 prompt 发给 LLM 即可！")


if __name__ == "__main__":
    main()
