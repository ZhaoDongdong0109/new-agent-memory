# new-agent-memory

Explainable human-like memory layer for AI agents.

`new-agent-memory` 是一个探索类人长期记忆的 Python 原型。它不只是把所有内容塞进向量库，而是把记忆分成核心层和伪遗忘层，让记忆可以被强化、衰减、降级，并在合适的线索出现时重新唤醒。

> AI 应该像人一样慢慢积累、越用越懂你，而不是每次会话都从零开始。

## Why This Exists

大多数 Agent Memory 系统重点解决“怎么存”和“怎么搜”。这个项目更关注另一个问题：

- 什么记忆应该常驻核心上下文？
- 什么记忆应该暂时沉到伪遗忘层？
- 用户重新给出线索时，系统怎样把它唤醒？
- 用户不喜欢主动提旧事时，系统怎样逐渐收敛？

核心定位：**可解释的记忆衰减、线索唤醒与目标驱动注意力调度**。

## Features

- **核心记忆层**：高权重记忆常驻，检索路径直接、可解释。
- **伪遗忘层**：低权重记忆不主动占用核心空间，但可被地点、人物、主题等锚点唤醒。
- **自适应权重**：结合时间衰减、访问频率、近因效应、情绪强度、关联密度和重要性。
- **人格适应层**：通过用户反馈学习是否应该主动提及旧记忆。
- **GoalStack**：维护当前目标、约束、证据和未闭环事项，让系统知道“现在正在干什么”。
- **程序记忆**：保存可复用的做事方式，例如测试、提交、发布、排查问题的流程。
- **注意力工作区**：按目标相关度、查询相关度、重要性、近因、频率和干扰惩罚选择上下文。
- **审计轨迹**：每次进入工作区的记忆和程序都会带有分数拆解，便于解释为什么被选中。
- **可持久化**：核心层、伪遗忘层、人格偏好和注意力状态可以保存到 JSON。
- **轻量高性能检索**：核心层内置倒排索引、短期权重缓存、候选集限制和早期退出。

## Install

```bash
git clone https://github.com/ZhaoDongdong0109/new-agent-memory.git
cd new-agent-memory
python -m pip install -e ".[dev]"
```

项目当前没有强制运行时依赖；`dev` extra 只安装测试工具。

## Quickstart

```python
from new_agent_memory import HumanLikeMemorySystem, MemoryType

memory = HumanLikeMemorySystem()

memory.add_memory(
    content="今天中午和客户在北京餐厅吃了烤鸭，聊了项目预算。",
    memory_type=MemoryType.INTERACTION,
    time_absolute="2026-04-29",
    time_context="中午",
    location="北京",
    persons=["客户"],
    topics=["food", "business", "project"],
    keywords=["烤鸭", "预算"],
    emotion_valence=0.3,
    emotion_intensity=0.6,
    importance=0.8,
)

result = memory.retrieve("中午在北京吃了什么")
print(result.summary())
print(result.assembled_content)
```

输出类似：

```text
[approved] path=core chunks=1 confidence=0.54 | 今天中午和客户在北京餐厅吃了烤鸭，聊了项目预算。
今天中午和客户在北京餐厅吃了烤鸭，聊了项目预算。
```

## Attention Workspace

如果普通检索回答的是“记得什么”，注意力工作区回答的是“现在该想什么”。

```python
from new_agent_memory import HumanLikeMemorySystem, MemoryType

memory = HumanLikeMemorySystem()

memory.start_goal(
    "修复测试失败并安全推送 PR",
    constraints=["先跑 pytest", "检查 git diff", "不要直接改 main"],
    open_loops=["PR 还没创建"],
)

memory.add_memory(
    content="上次 maintain() 失败是因为核心层缺少 decay_all_unused()。",
    memory_type=MemoryType.FACT,
    topics=["pytest", "maintenance", "bugfix"],
    importance=0.75,
)

memory.add_procedure(
    title="Safe PR workflow",
    steps=["run pytest", "run examples", "check git diff", "commit", "push branch"],
    triggers=["pytest", "tests", "PR", "push"],
    importance=0.85,
    confidence=0.8,
)

workspace = memory.focus("pytest 失败后怎么继续")
print(workspace.to_prompt_context())
```

输出会包含当前目标、应该带入上下文的程序记忆、以及真正相关的记忆。无关但重要的偏好会被审计记录下来，但不会自动进入工作区。

## Examples

```bash
python examples/simple_memory.py
python examples/forgotten_recall.py
python examples/persona_adaptation.py
python examples/attention_workspace.py
```

示例覆盖：

- 添加记忆并检索
- 伪遗忘层被线索唤醒
- 用户反馈改变主动回忆偏好
- 目标驱动的注意力工作区

## Architecture

```text
用户输入
  |
  v
目标栈 + 意图/线索解析
  |
  v
注意力门控 --选择--> 工作区上下文
  |
  +--> 程序记忆
  +--> 核心记忆层
  +--> 必要时唤醒伪遗忘层
  |
  v
组装与审阅 --> 输出
  |
  v
反馈更新：人格偏好 / 程序置信度 / 记忆权重
```

模块结构：

```text
new-agent-memory/
├── main.py                    # 统一 API：HumanLikeMemorySystem
├── memory_chunk.py            # 记忆碎片数据结构
├── memory_layer_core.py       # 核心记忆层、索引、缓存、权重计算
├── forgotten_layer.py         # 伪遗忘层与线索唤醒
├── retrieval.py               # 查询解析、检索、组装、审阅
├── core/
│   ├── attention_system.py    # GoalStack、程序记忆、注意力评分与工作区
│   ├── weight_system.py       # 自适应权重系统实验
│   ├── emotion_engine.py      # 情绪推断与情绪系数采样
│   └── persona_layer.py       # 行为反馈与人格适应
├── examples/                  # 可运行示例
└── tests/                     # 行为测试
```

## Memory Model

权重由多种信号共同决定：

```text
effective_weight =
  time_decay
  + access_frequency
  + recency
  + emotion_boost
  + association_density
  + importance
  + connection_value
```

当核心记忆权重低于阈值时，它会降级到伪遗忘层。伪遗忘层不会主动参与普通检索，但当查询里出现足够强的锚点，例如地点、人物、主题，系统可以重新唤醒这段记忆。

## Attention Model

注意力工作区使用可解释分数，而不是黑盒排序：

```text
attention_score =
  goal_relevance
  + query_relevance
  + user_importance
  + recency
  + frequency
  + emotional_salience
  + uncertainty_need
  + novelty
  - distraction_penalty
  - staleness_penalty
```

这让系统可以做到：

- 当前目标相关的记忆优先进入上下文
- 程序记忆在触发词出现时被带入，例如 `pytest`、`release`、`PR`
- 重要但无关的记忆会被干扰惩罚挡住
- 每次选择都有 audit，可解释分数来源

## Performance Notes

核心层已合入实验中的主要优化：

- 多级倒排索引：时间、主题、地点、人物
- 权重缓存：避免重复计算同一批候选
- 候选集限制：默认最多扫描 200 条候选
- 早期退出：找到足够候选后停止扫描

历史实验报告见 [experiments/OPTIMIZATION_REPORT.md](experiments/OPTIMIZATION_REPORT.md)。

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m compileall .
```

当前测试覆盖：

- 核心层添加与检索
- `maintain()` 安全运行与弱记忆降级
- 伪遗忘层线索唤醒
- 保存与加载
- 人格正反馈提升兴趣度
- 目标驱动注意力门控
- 注意力状态持久化

## Roadmap

- [x] 标准 Python 包入口
- [x] 核心层索引、缓存和早期退出
- [x] 基础测试与 GitHub Actions
- [x] 可运行 examples
- [x] GoalStack、程序记忆和注意力工作区
- [ ] CLI：`memory add/search/stats`
- [ ] SQLite 持久化后端
- [ ] 更强的自然语言线索解析
- [ ] LLM 驱动的碎片组装与审阅
- [ ] 向量检索 / BM25 / 图关联的混合检索
- [ ] MCP 或 LangGraph 集成示例

## Positioning

这个项目不是 Mem0、Zep、LangChain Memory 的替代品，而是一个更小、更可解释的类人记忆与注意力机制实验。它适合用于：

- 学习长期记忆系统怎么分层、衰减、唤醒
- 研究 Agent 如何在有限上下文里决定“现在该想什么”
- 给个人 Agent 增加可解释的长期记忆原型
- 研究“遗忘机制”本身，而不是只研究语义检索

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
