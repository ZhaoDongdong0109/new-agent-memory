# 🧠 AI 记忆系统 - 重构版本

## 🎯 解决的核心问题

**大幅减少对 LLM 上下文窗口的依赖！**

- 传统方法：需要把所有对话放在上下文 → 窗口很快耗尽
- 新系统：**90%+ 上下文节省** → 只加载真正相关的记忆

## 🏗️ 新架构设计

### 三层记忆体系

```
┌─────────────────────────────────────────────────────────────┐
│  1. Working Memory (工作记忆) - LRU 缓存 (高优先级)          │
│  2. Recent Memory (近期记忆) - 向量检索 (中优先级)           │
│  3. Long-Term Memory (长期记忆) - 压缩+索引 (低优先级)      │
└─────────────────────────────────────────────────────────────┘
```

### 核心模块

| 模块 | 文件 | 功能 |
|------|------|------|
| VectorStore | [vector_store.py](vector_store.py) | 向量化存储 + 相似度检索 |
| MemoryHierarchy | [memory_hierarchy.py](memory_hierarchy.py) | 三层记忆管理 + 生命周期 |
| MemoryCompressor | [memory_compressor.py](memory_compressor.py) | 智能压缩 + 关键词摘要 |
| ContextManager | [context_manager.py](context_manager.py) | 窗口管理 + 智能取舍 |
| AIMemorySystem | [ai_memory_system.py](ai_memory_system.py) | 统一 API 入口 |

## 🚀 快速开始

### 1. 基本使用

```python
from ai_memory_system import AIMemorySystem

# 初始化
system = AIMemorySystem(max_context_tokens=4096)

# 添加对话
system.add_conversation(
    user_input="我叫张三，我喜欢吃火锅",
    assistant_response="你好张三！我也喜欢火锅！"
)

# 构建 LLM 提示词
prompt = system.build_llm_prompt(
    "你记得我叫什么？",
    system_prompt="请根据记忆回答"
)

print(prompt)
```

### 2. 输出效果

```
System: 请根据记忆回答

--- Relevant Memories ---
[★1] User: 我叫张三，我喜欢吃火锅 (working)
[★2] Assistant: 你好张三！我也喜欢火锅！ (working)

---
User: 你记得我叫什么？
```

## 📊 性能对比

| 指标 | 传统方法 | 新系统 |
|------|----------|--------|
| 上下文利用率 | ~100% (很快耗尽) | 5-10% (节省 90%+) |
| 可记忆的对话历史 | 取决于窗口大小 | 理论无限 |
| 检索延迟 | O(n) | 索引优化 |
| 自适应压缩 | ❌ | ✅ |

## 📁 项目结构

```
new-agent-memory/
├── vector_store.py          # 向量存储
├── memory_hierarchy.py      # 层级管理
├── memory_compressor.py     # 压缩系统
├── context_manager.py       # 窗口管理
├── ai_memory_system.py      # 统一 API
├── example.py               # 完整示例
├── ARCHITECTURE.md          # 架构文档
└── README_V1.md             # 旧版文档
```

## 🔧 进阶功能

### 持久化

```python
system.save()          # 保存
system.load()          # 加载
```

### 获取统计

```python
stats = system.get_stats()
print(f"上下文节省: {stats['context_savings_percent']:.1f}%")
```

### 手动压缩

```python
system.trigger_compression(days=30)  # 压缩旧记忆
```

## 📈 示例运行

```bash
python3 example.py
```

## 🎨 与 LLM 集成

```python
from openai import OpenAI

# 你的 LLM 代码
client = OpenAI()

# 获取上下文
system_prompt = "你是一个友好的助手"
response = system.get_relevant_context("你记得我们聊过什么")

# 调用 LLM
completion = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": response.formatted_context},
        {"role": "user", "content": "你记得我们聊过什么"}
    ]
)

# 添加到记忆
system.add_conversation("你记得我们聊过什么", completion.choices[0].message.content)
```

## 📝 与旧系统的区别

| 特性 | 旧版 | 新版 |
|------|------|------|
| 核心思路 | 情绪+人物权重 | 语义向量+层级 |
| 检索方法 | 关键词/标签 | 余弦相似度 |
| 内存管理 | 无优化 | 智能压缩 |
| 上下文 | 无法优化 | 90%+ 节省 |
| 扩展能力 | 有限 | 无限 |

## 🔮 未来改进

- [ ] 支持真实的 Embedding API (OpenAI/Sentence-Transformers)
- [ ] 添加记忆重要性的离线训练
- [ ] 支持分布式向量数据库 (Pinecone/Weaviate/Chroma)
- [ ] 可视化记忆网络

## 📄 许可证

MIT License
