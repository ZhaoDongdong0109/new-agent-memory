#!/usr/bin/env python3
"""
Cognitive Memory System - 颠覆 Transformer 的新范式
演示认知记忆系统 vs Transformer 的核心区别
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cognitive_memory_system import (
    CognitiveMemorySystem,
    EmotionState,
    EmotionType
)


def main():
    print("=" * 70)
    print("🧠 认知记忆系统 vs Transformer - 核心对比")
    print("=" * 70)
    
    # 初始化认知系统
    cognitive = CognitiveMemorySystem()
    
    print("\n[场景] 用户与 AI 进行多轮对话...")
    
    # 对话 1
    print("\n📝 对话 1:")
    print("用户: 我叫张三，我喜欢编程，特别喜欢 Python")
    print("AI: 你好张三！Python 是一门很棒的语言")
    
    cognitive.process_conversation(
        "我叫张三，我喜欢编程，特别喜欢 Python",
        "你好张三！Python 是一门很棒的语言",
        emotion=EmotionState(EmotionType.JOY, 0.7, 0.8)
    )
    
    # 对话 2
    print("\n📝 对话 2:")
    print("用户: 我想做一个聊天机器人")
    print("AI: 很棒的想法！可以用 Python 来实现")
    
    cognitive.process_conversation(
        "我想做一个聊天机器人",
        "很棒的想法！可以用 Python 来实现",
        emotion=EmotionState(EmotionType.ANTICIPATION, 0.6, 0.7)
    )
    
    # 对话 3
    print("\n📝 对话 3:")
    print("用户: 我最近在学习机器学习")
    print("AI: 机器学习很有前景！")
    
    cognitive.process_conversation(
        "我最近在学习机器学习",
        "机器学习很有前景！",
        emotion=EmotionState(EmotionType.TRUST, 0.5, 0.6)
    )
    
    # 对话 4 - 不相关
    print("\n📝 对话 4:")
    print("用户: 今天天气不错")
    print("AI: 是的，天气很好")
    
    cognitive.process_conversation(
        "今天天气不错",
        "是的，天气很好",
        emotion=EmotionState(EmotionType.NEUTRAL, 0.1, 0.0)
    )
    
    # 查询测试
    print("\n" + "=" * 70)
    print("🔍 查询测试: '我是谁？我想做什么？'")
    print("=" * 70)
    
    results = cognitive.query("我是谁？我想做什么？")
    
    print(f"\n找到 {len(results)} 个相关认知节点:")
    for i, node in enumerate(results, 1):
        print(f"\n  [{i}] 概念: {node.concept}")
        print(f"      重要性: {node.importance:.2f}")
        print(f"      关系: {list(node.relationships.keys())[:3]}")
        print(f"      访问次数: {node.access_count}")
    
    # 认知摘要
    print("\n" + "=" * 70)
    print("📊 认知系统状态")
    print("=" * 70)
    
    summary = cognitive.get_cognitive_summary()
    print(f"\n总概念数: {summary['total_concepts']}")
    print(f"认知密度: {summary['cognitive_density']:.2%}")
    print(f"活跃意图: {summary['active_intentions']}")
    print(f"当前情绪: {summary['current_emotion']}")
    print(f"平均重要性: {summary['average_importance']:.2f}")
    
    # 对比分析
    print("\n" + "=" * 70)
    print("⚖️  Transformer vs 认知记忆系统")
    print("=" * 70)
    
    print("""
    ┌────────────────────┬────────────────────┬────────────────────┐
    │      特性           │    Transformer     │   认知记忆系统      │
    ├────────────────────┼────────────────────┼────────────────────┤
    │ 存储方式           │ 原始对话文本        │ 结构化概念图        │
    │ 上下文依赖         │ 100% (全部加载)     │ 10% (只加载相关)    │
    │ 知识表示           │ Token 序列          │ 关系图 + 重要性     │
    │ 遗忘机制           │ ❌ 无              │ ✅ 主动遗忘        │
    │ 情感理解           │ ❌ 无              │ ✅ 情感驱动        │
    │ 意图追踪           │ ❌ 无              │ ✅ 意图导向        │
    │ 一致性维护         │ 困难                │ 人物层保证         │
    │ 推理能力           │ 有限 (统计)        │ 关系图推理         │
    └────────────────────┴────────────────────┴────────────────────┘
    """)
    
    print("\n🎯 颠覆 Transformer 的核心点:")
    print("   1. 不是存储'说了什么'，而是理解'是什么'")
    print("   2. 情感作为第一优先级，影响所有决策")
    print("   3. 遗忘不是缺陷，而是必要的优化")
    print("   4. 意图驱动主动学习，而非被动存储")
    
    print("\n" + "=" * 70)
    print("✅ 认知记忆系统运行成功！")
    print("=" * 70)


if __name__ == "__main__":
    main()
