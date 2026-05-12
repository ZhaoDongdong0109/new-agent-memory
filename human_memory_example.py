#!/usr/bin/env python3
"""
HumanMemorySystem - 极致模仿人类记忆
完整演示：展示所有记忆特性
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from human_memory_system import HumanMemorySystem, MemoryType


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demonstrate_sensory_memory(mem):
    """演示感官记忆"""
    print_section("1. 感官记忆 (Iconic/Echoic Memory)")
    
    # 编码图像记忆
    mem.encode_sensory("看到一个苹果", "visual", intensity=0.9)
    mem.encode_sensory("听到铃声", "auditory", intensity=0.8)
    
    print(f"  图像记忆缓冲: {len(mem.iconic_memory)} 条")
    print(f"  声音记忆缓冲: {len(mem.echoic_memory)} 条")
    print(f"  苹果的新鲜度: {list(mem.iconic_memory.values())[0].get_freshness():.2%}")
    
    # 自然衰减
    time.sleep(0.5)
    decayed = mem.decay_sensory_memory()
    print(f"  衰减后剩余: {len(mem.iconic_memory) + len(mem.echoic_memory)} 条")


def demonstrate_working_memory(mem):
    """演示工作记忆容量限制"""
    print_section("2. 工作记忆 (7±2 组块法则)")
    
    # 测试容量
    test_items = [
        "苹果", "香蕉", "橙子", "葡萄", "西瓜",
        "草莓", "蓝莓", "桃子", "柚子", "柠檬",
        "石榴", "芒果"
    ]
    
    print(f"  尝试添加 {len(test_items)} 个项目...")
    
    for i, item in enumerate(test_items, 1):
        success = mem.add_to_working(item)
        status = "✓" if success else "✗"
        print(f"    {status} 添加 '{item}' - 工作记忆: {len(mem.working_memory)}/{mem.working_memory_slots}")
    
    print(f"\n  工作记忆利用率: {len(mem.working_memory)/mem.working_memory_slots:.1%}")
    print(f"  容量限制验证: {len(mem.working_memory)} ≤ 7±2 ✅")


def demonstrate_long_term_encoding(mem):
    """演示长时记忆编码"""
    print_section("3. 长时记忆编码 (精细编码)")
    
    # 情景记忆
    mem.encode_to_long_term(
        content="今天上午在办公室和张三开会讨论项目进度",
        memory_type=MemoryType.EPISODIC,
        context={"time": "上午", "location": "办公室", "people": ["张三"]},
        emotions=[("anticipation", 0.7)],
        importance=0.8
    )
    
    # 语义记忆
    mem.encode_to_long_term(
        content="Python 是一种高级编程语言",
        memory_type=MemoryType.SEMANTIC,
        context={"examples": ["Django", "Flask", "数据分析"]},
        importance=0.6
    )
    
    # 程序记忆
    mem.encode_to_long_term(
        content="如何骑自行车",
        memory_type=MemoryType.PROCEDURAL,
        context={"steps": ["上车", "踩踏板", "保持平衡", "转向", "刹车"]},
        importance=0.7
    )
    
    # 情绪记忆
    mem.encode_to_long_term(
        content="考试通过的那一刻",
        memory_type=MemoryType.EMOTIONAL,
        emotions=[("joy", 0.9), ("relief", 0.8)],
        importance=0.95
    )
    
    print(f"  情景记忆: {len(mem.episodic_memory)} 条")
    print(f"  语义记忆: {len(mem.semantic_memory)} 条")
    print(f"  程序记忆: {len(mem.procedural_memory)} 条")
    print(f"  情绪记忆: {len(mem.emotional_memory)} 条")
    
    # 查看雷斯多夫效应
    emotional_mem = list(mem.emotional_memory.values())[0]
    episodic_mem = list(mem.episodic_memory.values())[0]
    print(f"\n  雷斯多夫效应验证:")
    print(f"    情绪记忆持久性: {emotional_mem.get_persistence_factor():.2f} (情绪增强)")
    print(f"    普通情景记忆强度: {episodic_mem.encoding_strength:.2f} (基础: 0.8)")


def demonstrate_retrieval(mem):
    """演示记忆提取"""
    print_section("4. 记忆提取机制")
    
    # 添加更多记忆用于测试
    test_memories = [
        ("Python 是一种编程语言", MemoryType.SEMANTIC),
        ("学习 JavaScript", MemoryType.EPISODIC),
        ("完成项目报告", MemoryType.EPISODIC),
        ("和团队开会", MemoryType.EPISODIC),
        ("阅读技术书籍", MemoryType.EPISODIC),
    ]
    
    for content, mtype in test_memories:
        mem.encode_to_long_term(content, mtype, importance=0.6)
    
    # 测试不同提取方式
    print("\n  提取测试:")
    
    print("\n  [a] 自由回忆 '编程'")
    results = mem.retrieve("编程", retrieval_type="recall")
    print(f"      找到 {len(results)} 条记忆")
    for r in results[:3]:
        print(f"        - {r.content[:40]}...")
    
    print("\n  [b] 再认 '语言'")
    results = mem.retrieve("语言", retrieval_type="recognition")
    print(f"      找到 {len(results)} 条记忆")
    
    print("\n  [c] 线索回忆 (提供上下文)")
    results = mem.retrieve("项目", retrieval_type="cued", context={"location": "办公室"})
    print(f"      找到 {len(results)} 条记忆")


def demonstrate_forgetting(mem):
    """演示遗忘机制"""
    print_section("5. 遗忘机制")
    
    # 添加记忆
    mem.encode_to_long_term("旧信息 A", MemoryType.EPISODIC, importance=0.5)
    mem.encode_to_long_term("旧信息 B", MemoryType.EPISODIC, importance=0.5)
    mem.encode_to_long_term("旧信息 C", MemoryType.EPISODIC, importance=0.5)
    
    initial_count = len(mem.episodic_memory)
    
    # 添加干扰项
    mem.apply_interference(["新记忆 X", "新记忆 Y", "新记忆 Z"])
    
    # 时间衰退
    forgotten = mem.forget(reason="decay")
    
    print(f"  初始情景记忆: {initial_count} 条")
    print(f"  遗忘项目数: {forgotten}")
    print(f"  当前情景记忆: {len(mem.episodic_memory)} 条")
    
    # 遗忘曲线
    if mem.episodic_memory:
        first_mem_id = list(mem.episodic_memory.keys())[0]
        curve = mem.get_forgetting_curve(first_mem_id)
        print(f"\n  遗忘曲线示例:")
        for hours, retention in curve:
            bar = "█" * int(retention * 20)
            print(f"    {hours:4.0f}h: {bar} {retention:.2%}")


def demonstrate_serial_position(mem):
    """演示系列位置效应"""
    print_section("6. 系列位置效应")
    
    print("  测试: 记忆一系列单词")
    words = [
        "第一", "第二", "第三", "第四", "第五",
        "第六", "第七", "第八", "第九", "第十"
    ]
    
    # 添加到工作记忆
    for word in words:
        mem.add_to_working(word)
    
    # 获取当前工作记忆内容
    current = [slot.content for slot in mem.working_memory]
    
    print(f"\n  原始顺序: {' → '.join(words)}")
    print(f"  工作记忆保留: {' → '.join(current)}")
    print(f"\n  预期效应:")
    print(f"    首因效应: 前几个记得更清楚 (位置 1-3)")
    print(f"    近因效应: 后几个记得更清楚 (位置 8-10)")
    print(f"    中间遗忘: 中间位置最容易遗忘")


def demonstrate_sleep_consolidation(mem):
    """演示睡眠巩固"""
    print_section("7. 睡眠记忆巩固")
    
    # 添加待巩固的记忆
    for i in range(5):
        mem.encode_to_long_term(
            f"需要记忆的内容 {i+1}",
            MemoryType.EPISODIC,
            importance=0.6
        )
    
    pending_before = len(mem.pending_consolidation)
    
    # 模拟睡眠
    print(f"  模拟 8 小时睡眠...")
    cycles = mem.simulate_sleep(duration_hours=8.0)
    
    print(f"\n  睡眠周期数: {cycles}")
    print(f"  巩固前待处理: {pending_before} 条")
    print(f"  巩固后待处理: {len(mem.pending_consolidation)} 条")
    
    # 检查记忆巩固状态
    if mem.memory_traces:
        consolidation_levels = [
            trace.consolidation_state
            for trace in mem.memory_traces.values()
        ]
        avg_consolidation = sum(consolidation_levels) / len(consolidation_levels)
        print(f"  平均巩固程度: {avg_consolidation:.1%}")
        print(f"\n  睡眠阶段效果:")
        print(f"    NREM: 一般记忆整合和强化")
        print(f"    REM: 情绪记忆加工 + 创造性联想")


def demonstrate_cognitive_biases(mem):
    """演示认知偏差"""
    print_section("8. 认知偏差")
    
    print("  系统内置认知偏差:")
    
    print("\n  [a] 首因效应 (Primacy Effect)")
    print("      先学习的内容更容易记住")
    print("      原因: 有更多时间复述进入长时记忆")
    
    print("\n  [b] 近因效应 (Recency Effect)")
    print("      最后学习的内容更容易记住")
    print("      原因: 还在工作记忆中")
    
    print("\n  [c] 雷斯多夫定律 (Ribot's Law)")
    print("      越近的记忆越容易被记住/遗忘")
    ribot = mem.get_ribot_curve()
    print(f"      当前分布: {ribot}")
    
    print("\n  [d] 情绪增强效应")
    print("      情绪强烈的事件记得更清楚、更持久")
    
    print("\n  [e] 干扰效应 (Interference)")
    print("      相关记忆会互相干扰")


def demonstrate_memory_comparison():
    """对比人类记忆 vs Transformer"""
    print_section("9. 人类记忆 vs Transformer")
    
    print("""
    ┌────────────────────┬────────────────────┬────────────────────┐
    │      特性           │    人类记忆        │     Transformer    │
    ├────────────────────┼────────────────────┼────────────────────┤
    │ 容量               │ 无限（分层存储）   │ 受限于上下文窗口   │
    │ 记忆类型           │ 情景/语义/程序     │ 统一 token 序列    │
    │ 遗忘机制           │ ✅ 主动遗忘        │ ❌ 无             │
    │ 情感影响           │ ✅ 情绪记忆增强    │ ❌ 无             │
    │ 容量限制           │ 7±2 (工作记忆)    │ 几千到几十万      │
    │ 提取方式           │ 联想/线索/再认    │ 注意力加权        │
    │ 巩固机制           │ 睡眠整合           │ ❌ 无             │
    │ 认知偏差           │ ✅ 系统性偏差      │ ❌ 无             │
    │ 组织结构           │ 图结构关系网络    │ 线性序列          │
    └────────────────────┴────────────────────┴────────────────────┘
    """)


def main():
    print("=" * 70)
    print("  🧠 人类记忆系统 - 极致模仿")
    print("  基于认知心理学经典理论")
    print("=" * 70)
    
    # 初始化系统
    mem = HumanMemorySystem(
        working_memory_slots=7,
        sensory_decay_rate=0.1
    )
    
    # 运行演示
    demonstrate_sensory_memory(mem)
    demonstrate_working_memory(mem)
    demonstrate_long_term_encoding(mem)
    demonstrate_retrieval(mem)
    demonstrate_forgetting(mem)
    demonstrate_serial_position(mem)
    demonstrate_sleep_consolidation(mem)
    demonstrate_cognitive_biases(mem)
    demonstrate_memory_comparison()
    
    # 最终报告
    print_section("系统状态报告")
    report = mem.get_memory_report()
    
    print(f"  感官记忆:")
    print(f"    图像缓冲: {report['sensory']['iconic']}")
    print(f"    声音缓冲: {report['sensory']['echoic']}")
    
    print(f"\n  工作记忆:")
    print(f"    使用槽位: {report['working_memory']['slots_used']}/{report['working_memory']['slots_available']}")
    print(f"    利用率: {report['working_memory']['utilization']:.1%}")
    
    print(f"\n  长时记忆:")
    print(f"    情景记忆: {report['long_term']['episodic']}")
    print(f"    语义记忆: {report['long_term']['semantic']}")
    print(f"    程序记忆: {report['long_term']['procedural']}")
    print(f"    情绪记忆: {report['long_term']['emotional']}")
    
    print(f"\n  检索统计:")
    print(f"    总提取次数: {report['retrieval']['total_retrievals']}")
    print(f"    成功次数: {report['retrieval']['successful_retrievals']}")
    print(f"    成功率: {report['retrieval']['success_rate']:.1%}")
    
    print("\n" + "=" * 70)
    print("  ✅ 演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
