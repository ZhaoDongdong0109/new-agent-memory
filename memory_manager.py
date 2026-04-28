#!/usr/bin/env python3
"""
记忆管理系统 - 原型实验
完全独立，不影响 Hermes 的真实记忆系统
"""

import json
import os
from datetime import datetime
from pathlib import Path

# 存储目录
MEMORY_DIR = Path.home() / "memory-prototype" / "memories"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

def add_memory(subject: str, content: str) -> dict:
    """
    添加一条记忆
    - subject: 主体（记忆的索引/钥匙）
    - content: 原始内容（要记住的完整信息）
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"{timestamp}_{subject[:20].replace(' ', '_')}.json"
    
    memory = {
        "subject": subject,
        "content": content,
        "created_at": timestamp,
        "slices": slice_content(content),  # 自动切片
    }
    
    filepath = MEMORY_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)
    
    return {"success": True, "file": str(filepath), "slices": memory["slices"]}


def slice_content(content: str) -> list:
    """
    简单切片：按句子分割
    后续可以换成 LLM 来做智能切片
    """
    sentences = content.replace("！", "。").replace("？", "。").split("。")
    slices = [s.strip() for s in sentences if s.strip()]
    return slices


def recall(keyword: str) -> list:
    """
    检索记忆：通过关键词搜索主体和内容
    """
    results = []
    for filepath in MEMORY_DIR.glob("*.json"):
        with open(filepath, "r", encoding="utf-8") as f:
            memory = json.load(f)
        
        # 简单的关键词匹配
        if (keyword.lower() in memory["subject"].lower() or 
            keyword.lower() in memory["content"].lower()):
            results.append(memory)
    
    return results


def list_memories() -> list:
    """
    列出所有记忆
    """
    memories = []
    for filepath in sorted(MEMORY_DIR.glob("*.json")):
        with open(filepath, "r", encoding="utf-8") as f:
            memory = json.load(f)
        memories.append({
            "subject": memory["subject"],
            "created_at": memory["created_at"],
            "preview": memory["content"][:50] + "..." if len(memory["content"]) > 50 else memory["content"],
        })
    return memories


def rebuild(keyword: str) -> str:
    """
    重建记忆：根据关键词找到记忆，然后重建一个连贯的叙述
    """
    results = recall(keyword)
    if not results:
        return f"没有找到关于「{keyword}」的记忆"
    
    # 简单重建：把所有匹配的记忆内容拼接起来
    # 后续可以用 LLM 做更智能的重建
    rebuild_text = []
    for memory in results:
        rebuild_text.append(f"【{memory['subject']}】{memory['content']}")
    
    return "\n\n".join(rebuild_text)


if __name__ == "__main__":
    # 测试
    print("=== 记忆管理系统 - 原型测试 ===\n")
    
    # 1. 添加一条记忆
    print("1. 添加记忆...")
    result = add_memory(
        subject="第一次记忆实验",
        content="今天是我和 Hermes 开始做记忆原型实验的日子。我们创建了一个独立的文件夹 ~/memory-prototype/，完全不会影响 Hermes 原有的记忆系统。很开心能一起探索这个想法！"
    )
    print(f"   ✅ 添加成功！生成了 {len(result['slices'])} 个切片")
    print(f"   文件：{result['file']}\n")
    
    # 2. 列出所有记忆
    print("2. 列出所有记忆...")
    memories = list_memories()
    for m in memories:
        print(f"   - {m['subject']} ({m['created_at']})")
    print()
    
    # 3. 检索记忆
    print("3. 检索「记忆」相关的内容...")
    results = recall("记忆")
    print(f"   找到 {len(results)} 条记忆：")
    for r in results:
        print(f"   - {r['subject']}")
    print()
    
    # 4. 重建记忆
    print("4. 重建「第一次记忆实验」...")
    rebuilt = rebuild("第一次记忆实验")
    print(f"   {rebuilt}\n")
    
    print("=== 测试完成 ===")
