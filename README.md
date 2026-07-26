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
- **CognitiveAgent**：提供 `Observe -> Focus -> Act -> Evaluate -> Remember -> Consolidate` 的最小数字身体闭环。
- **LLMPlanner**：把 OpenAI、Hermes 或本地模型接成 Agent 的决策核心，让模型根据目标、记忆和工具返回下一步行动。
- **CognitiveState**：维护自我模型、世界信念、驱动力、行动预期和反思记录，让 Agent 知道“我是谁、我能做什么、我不确定什么”。
- **经验层**：把行动、结果、奖励、教训和下一次策略记录为经验 episode，并可巩固成记忆/程序。
- **可持久化**：核心层、伪遗忘层、人格偏好和注意力状态可以保存到 JSON。
- **轻量高性能检索**：核心层内置倒排索引、短期权重缓存、候选集限制和早期退出。

## Install

```bash
git clone https://github.com/ZhaoDongdong0109/new-agent-memory.git
cd new-agent-memory
python -m pip install -e ".[dev]"
```

项目当前没有强制运行时依赖；`dev` extra 只安装测试工具。

## OpenAI-Compatible Quickstart

配置 `.env`：

```bash
OPENAI_API_KEY=your-key-or-local-placeholder
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4.1-mini
```

如果你用 Hermes、LM Studio、vLLM、Ollama 网关或其他 OpenAI-compatible 服务，把 `OPENAI_BASE_URL` 改成对应的 `/v1` 地址即可，例如：

```bash
OPENAI_BASE_URL=http://localhost:1234/v1
OPENAI_MODEL=hermes-3
```

单轮对话：

```bash
new-agent-memory ask "你现在是谁？用一句话说明你能做什么。"
```

交互式会话：

```bash
new-agent-memory chat --goal "帮助用户把记忆系统进化成可下载可运行的智能体"
```

Python 中直接创建：

```python
from new_agent_memory import HumanLikeMemorySystem

memory = HumanLikeMemorySystem()
agent = memory.create_openai_agent()
episode = agent.run_turn("你现在能做什么？")
print(episode.result.output)
memory.save()
```

### Cognitive Runtime

Use the multi-step runtime when a task needs more than one internal action:

```bash
new-agent-memory ask --runtime --max-steps 4 \
  "Answer first, then call introspect, then finish with two next questions."
```

Python API:

```python
from new_agent_memory import HumanLikeMemorySystem

memory = HumanLikeMemorySystem()
runtime = memory.create_openai_runtime(max_steps=4)
run = runtime.run("Answer first, then call introspect, then finish.")

print(run.result.output)
print([step.action.name for step in run.steps])
```

`CognitiveRuntime` adds a small executive loop above `CognitiveAgent`:

```text
Observe -> Goal -> Focus -> Plan -> Act -> Reflect -> Continue/Finish
```

It registers a `finish` control tool, records each step as an `ExperienceEpisode`,
guards against finishing before required tools actually run, and keeps runtime
trace text separate from the user's original intent.

支持的环境变量：

```text
OPENAI_API_KEY
OPENAI_BASE_URL
OPENAI_MODEL
OPENAI_TEMPERATURE
OPENAI_MAX_TOKENS
OPENAI_TIMEOUT
OPENAI_USE_ENV_PROXY
```

也可以使用 `OPENAI_COMPATIBLE_*` 前缀，例如 `OPENAI_COMPATIBLE_BASE_URL`。

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

## Cognitive Agent Loop

`CognitiveAgent` 是这个项目迈向通用 Agent 的第一具“数字身体”。它不假装已经拥有物理感受，但它能在一个可观察、可行动、有反馈、可累积后果的环境里运行。

```python
from new_agent_memory import ActionResult, AgentAction, HumanLikeMemorySystem

memory = HumanLikeMemorySystem()
memory.start_goal(
    "帮助用户把项目推进到可运行的 Agent 原型",
    constraints=["先观察", "再行动", "最后复盘"],
)

agent = memory.create_agent(name="hermes-seed")

def planner(observation, workspace, tools):
    return AgentAction(
        name="inspect",
        arguments={
            "input": observation.content,
            "focus": workspace.to_prompt_context(),
        },
        rationale="用户要求检查，因此调用 inspect 工具。",
    )

def inspect_tool(arguments):
    return ActionResult(
        success=True,
        output="检查完成。当前注意力上下文长度：" + str(len(arguments.get("focus", ""))),
        cost=0.05,
    )

agent.planner = planner
agent.add_tool("inspect", "Inspect the current focus workspace.", inspect_tool)

episode = agent.run_turn("请检查当前 Agent 闭环是否能工作")
print(episode.lesson)
print(episode.next_policy)
```

每一轮会产生一个 `ExperienceEpisode`：

```text
Observation -> FocusWorkspace -> AgentAction -> ActionResult -> reward
  -> lesson -> next_policy -> memory/procedure consolidation
```

这让系统开始记录“我做了什么、结果如何、下次该怎么做”，而不只是保存对话文本。

## LLM Planner

`new-agent-memory` 本身是认知层，不是大模型。要让 Agent 做复杂推理，需要给 `CognitiveAgent` 接一个 LLM planner。

最小形式是传入一个 `prompt -> text` 的函数：

```python
from new_agent_memory import CognitiveAgent, HumanLikeMemorySystem, LLMPlanner

memory = HumanLikeMemorySystem()
agent = CognitiveAgent(memory_system=memory)

def call_hermes_or_local_model(prompt: str) -> str:
    # 返回必须包含一个 JSON action
    return """
{
  "name": "respond",
  "arguments": {"message": "我会根据当前目标和记忆回答。"},
  "rationale": "No external tool is needed."
}
"""

agent.planner = LLMPlanner(call_hermes_or_local_model)
episode = agent.run_turn("下一步怎么做？")
```

如果使用 OpenAI-compatible `/v1/chat/completions` 服务，可以直接从环境变量创建：

```python
from new_agent_memory import HumanLikeMemorySystem

memory = HumanLikeMemorySystem()
agent = memory.create_openai_agent()
episode = agent.run_turn("请读取当前记忆和认知状态，然后回复我。")
```

也可以手动构造：

```python
from new_agent_memory import LLMPlanner, OpenAICompatibleConfig

config = OpenAICompatibleConfig(
    api_key="local-or-real-key",
    base_url="http://localhost:1234/v1",
    model="hermes-3",
)
planner = LLMPlanner.from_openai_compatible(config)
```

如果使用 OpenAI Responses 风格客户端，可以这样包一层：

```python
from openai import OpenAI
from new_agent_memory import LLMPlanner

client = OpenAI()
planner = LLMPlanner.from_openai_responses(
    client=client,
    model="gpt-4.1-mini",
)
```

模型需要返回的 JSON：

```json
{
  "name": "tool_name",
  "arguments": {},
  "rationale": "short reason"
}
```

`name` 必须是当前 Agent 已注册工具之一，例如 `respond`、`remember` 或你自己用 `agent.add_tool(...)` 添加的工具。

## Reflective Cognition

`CognitiveState` 是这次新增的“内在状态”层。它不负责替代 LLM 推理，而是给推理核心提供一个稳定、可保存、可审计的自我/世界模型：

- **Self model**：身份、使命、能力边界和限制。
- **Drive state**：连贯性、胜任感、好奇心、有用性、安全性等 homeostatic drives。
- **World beliefs**：从观察、反馈和行动结果中形成带置信度的信念。
- **Action priors**：根据工具历史结果预测某个行动的成功率和成本。
- **Reflection notes**：每次行动后记录 outcome、insight、uncertainty 和 next question。

完整 Agent loop 现在会变成：

```text
Observe -> update world model -> Focus -> attach CognitiveState
  -> Predict -> Act -> Evaluate -> Reflect -> Remember -> Consolidate
```

默认工具里新增了 `introspect`：

```python
from new_agent_memory import CognitiveAgent, HumanLikeMemorySystem

memory = HumanLikeMemorySystem()
agent = CognitiveAgent(memory_system=memory, auto_consolidate=False)

episode = agent.run_turn("please introspect current state")
print(episode.result.output)
```

这让 Agent 可以主动读取自己的驱动力、开放问题、工具先验和最近反思。`focus()` 也会把 `CognitiveState` 注入 `FocusWorkspace.to_prompt_context()`，所以 `LLMPlanner` 会自然看到这层上下文。

## Examples

```bash
python examples/simple_memory.py
python examples/forgotten_recall.py
python examples/persona_adaptation.py
python examples/attention_workspace.py
python examples/cognitive_agent.py
python examples/llm_planner.py
python examples/openai_compatible_agent.py
python examples/reflective_cognition.py
```

示例覆盖：

- 添加记忆并检索
- 伪遗忘层被线索唤醒
- 用户反馈改变主动回忆偏好
- 目标驱动的注意力工作区
- 数字身体闭环和经验巩固
- LLM 决策器如何选择工具行动
- OpenAI-compatible API 如何接入 Agent
- 自我/世界模型、反思记录和 `introspect` 工具

## Architecture

```text
用户输入
  |
  v
目标栈 + 观察输入
  |
  v
注意力门控 --选择--> 工作区上下文
  |
  +--> 程序记忆
  +--> 核心记忆层
  +--> 必要时唤醒伪遗忘层
  |
  v
Agent 决策 -> 工具行动 -> 结果评价
  |
  v
经验记录与巩固：记忆 / 程序 / 偏好 / 目标
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
│   ├── agent_system.py        # CognitiveAgent、工具、经验层与行动闭环
│   ├── llm_planner.py         # 模型无关 LLMPlanner 和 OpenAI Responses 包装
│   ├── cognitive_state.py     # 自我模型、世界信念、驱动力、行动预期与反思
│   ├── attention_system.py    # GoalStack、程序记忆、注意力评分与工作区
│   ├── weight_system.py       # 记忆类型与按类型分层的衰减半衰期
│   ├── topic_vocab.py         # 查询/写入共享的双语主题词汇表
│   └── persona_layer.py       # 行为反馈与人格适应
├── examples/                  # 可运行示例
└── tests/                     # 行为测试
```

## Memory Model

记忆强度核心是 **ACT-R 基线激活**（Anderson & Schooler，30 年认知科学验证的方程），一个公式同时产生幂律遗忘、频率效应和近因效应：

```text
B = ln( Σ_j t_j^(-d) )              # t_j = 距第 j 次使用的时间
P = 1 / (1 + exp(-(B - τ) / s))     # 保持率（检索概率）

effective_weight =
  W_r * P
  + P * (emotion_boost + importance + connection_value + association_density)
  + recall_bias        # 回忆反馈偏置：被确认的记忆上浮，被纠错的下沉
```

- 每条记忆精确保留最近 8 次使用时间戳，更早的使用以 Petrov O(k) 积分近似
- 静态因子被保持率 P 门控：不被使用的记忆无论多"重要"，P 趋零后权重也趋零——遗忘可达是生命周期的前提
- 关联强度渐进减缓衰减速率 d（上限 10%），不构成永久权重下限

衰减半衰期按记忆类型分层（程序性/叙事性记忆比一次性交互持久，
继承记忆科学的遗忘曲线分层）：

```text
STORY 3.33x > IDEA 2.67x > PREFERENCE 2.0x > FACT 1.33x > INTERACTION 1.0x
```

完整的遗忘-唤醒生命周期：

```text
add -> 使用中（访问/反馈强化）
  -> 长期不用权重衰减 -> maintain() 降级到伪遗忘层
  -> 伪遗忘层不参与主动检索
  -> 查询携带足够强的锚点（人物/地点/主题/时间）-> try_wake 唤醒
  -> 锚点足够强 -> promote 提升回核心层（带再巩固奖励）
  -> 继续被使用则留下，不用则再次自然衰减
```

每一步都可解释：`WeightFactors` 给出权重的逐因子拆解，唤醒有得分与锚点计数，提升有统计口径（`retrieval.total_promoted`）。

## Retrieval Model

生产检索路径是三腿混合检索 + RRF 融合：

```text
query -> QueryContext（时间/地点/人物/主题锚点）
  |-- Metadata leg: 倒排索引候选，按记忆权重排序
  |-- BM25 leg:     词法相关性（中文字符级 + 关键词）
  |-- Dense leg:    哈希 TF-IDF 余弦相似度
  -> RRF 融合（按批内最大值归一化）
  -> 70% 相关性 + 30% 记忆权重 混合排序
  -> 未命中 -> 伪遗忘层锚点唤醒
```

查询侧与写入侧共享一张双语主题词汇表（`core/topic_vocab.py`），中文查询可以命中英文标签的记忆，反之亦然。

### Bi-temporal Facts（双时态事实取代）

FACT / PREFERENCE 记忆的写入经过确定性决策表（`core/supersession.py`），
每个决策带具名规则 ID 可审计：

```text
写入新事实
  ├── 近重复（Jaccard >= 0.8）      -> NOOP：强化既有记忆
  ├── 同槽扩展（旧内容被覆盖 >= 0.8）-> UPDATE：就地更新，旧内容进 history
  ├── 同槽换值（相似度 >= 0.25）     -> SUPERSEDE：旧值 invalid_at 置为现在，
  │                                    归档到伪遗忘层，新值 parent_id 指向旧值
  └── 新事实                        -> ADD：重要性按惊奇度缩放（Titans 思想）
```

读取侧按时态路由：

```text
"小李现在住在哪"  -> 只返回当前有效值（invalid_at 未设置）
"小李以前住在哪"  -> 沿 parent_id 取代链追溯：奥斯陆 -> 柏林 -> 里斯本
"2024年3月做了什么" -> 确定性时间窗口过滤（core/time_parser.py）
窗口内没有记忆     -> 可审计弃答："最接近的相关记忆在 2026-06-20"
```

"记得你以前住在里斯本"是真实的运行时行为——这正是 2025-2026 各大
记忆系统评测中最弱的知识更新能力（MemoryAgentBench 显示 SOTA 接近随机），
而这里的每一步都是确定性规则，无需 LLM 仲裁。

### Sleep Cycle（确定性睡眠巩固）

人脑在睡眠中把零散的情景抽象成持久的语义知识。`system.sleep()`
（或写入累计重要性达到阈值后 `maintain()` 自动触发）执行确定性版本：

```text
优先回放选择（重要性 + 情绪 + 近因 + 惊奇度）
  -> 锚点聚类（共享人物/主题 或 Hebbian 强边，连通分量 >= 3）
  -> 抽取式要点合成（池化关键词给句子打分，top-3 句，逐句可溯源）
  -> 要点入核心层（IDEA 类型，慢衰减，重要性随簇规模上浮）
  -> 来源归档到伪遗忘层（线索仍可唤醒——抽象完全可逆）
  -> Hebbian 图卫生（修剪近零边，限制单点边数）
```

之后检索"和老王的项目会议"命中的是抽象要点（常驻），而强线索
仍能唤醒某次会议的具体细节（归档但未销毁）——要点与来源之间
保留关联边，联想回忆可以从抽象走回具体。llm_fn 可选地润色要点
文字，但不允许增删事实；没有 LLM 时抽取式模板同样工作。

### Associative Recall（联想回忆）

检索命中之后还有一步扩散激活（spreading activation）：

```text
命中记忆（activation = 1.0）
  -> 沿 Hebbian 关联边传播：contribution = 激活 × 边权 × 0.5/跳
  -> 多路径汇聚的记忆激活累积（更容易被想起）
  -> 强激活的归档记忆被联想唤醒，甚至提升回核心层
```

关联图的生长途径是检索时的 Hebbian 共激活：同一次检索里一起出现的
记忆互相连线（fire together, wire together）。这让系统表现出
"因为想起 A 而想起 B"，包括把尘封在伪遗忘层的 B 一并带回来。
每次联想都有激活轨迹审计（`retrieval.last_activation_trace`）。

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
- CognitiveAgent 行动闭环
- 经验 episode 巩固为长期记忆和程序记忆
- LLMPlanner JSON 行动解析、fallback 和工具选择
- CognitiveState 自我/世界模型、行动预期、反思和持久化

## Roadmap

- [x] 标准 Python 包入口
- [x] 核心层索引、缓存和早期退出
- [x] 基础测试与 GitHub Actions
- [x] 可运行 examples
- [x] GoalStack、程序记忆和注意力工作区
- [x] CognitiveAgent 数字身体闭环
- [x] 模型无关 LLMPlanner
- [x] CognitiveState 自我/世界模型与反思闭环
- [x] Hermes / OpenAI-compatible Agent adapter
- [x] CLI：`memory add/search/stats`
- [x] 自动记忆提取器 `MemoryExtractor`
- [x] SQLite 持久化后端
- [x] 向量检索 / BM25 / RRF 混合检索（已接入生产检索路径）
- [x] 遗忘-唤醒生命周期闭环（降级 → 线索唤醒 → 提升回核心层）
- [x] 联想回忆：Hebbian 共激活 + 扩散激活 + 联想唤醒归档记忆
- [x] 双时态事实取代：ADD/UPDATE/SUPERSEDE/NOOP 确定性决策表
- [x] 时间感知检索：确定性时间表达解析 + 窗口过滤 + 时态路由
- [x] 可审计弃答：窗口外/已过时的结果不冒充答案
- [x] 惊奇度门控编码：意外信息编码更强（Titans 思想的确定性内核）
- [x] MCP Server 集成
- [x] 确定性评测套件：ToT-lite 时序生成器 + 知识更新探针 + 容量门（研究议程 #2）
- [x] ACT-R 基线激活方程替换手调权重因子（研究议程 #5，Petrov O(k) 近似）
- [ ] Pavlik 激活依赖衰减扩展（间隔效应，需按次衰减状态）
- [x] 确定性睡眠周期：优先回放的情景→语义巩固（研究议程 #7，session-rollup 切片）
- [ ] Personalized PageRank 联想回忆 + 扇出效应阻尼（研究议程 #6，HippoRAG）
- [ ] 睡眠周期第二切片：当前值槽索引 + 槽内乱序矛盾重扫 + MCP 维护工具
- [ ] 要点支持计数：后续匹配情景增强既有要点而非重复抽象
- [ ] 经验复盘 `ConsolidationEngine` 接入 Agent 主循环
- [ ] LLM 驱动的碎片组装与审阅（可选插件）

## Positioning

这个项目不是 Mem0、Zep、LangChain Memory 的替代品，而是一个更小、更可解释的类人记忆与注意力机制实验。它适合用于：

- 学习长期记忆系统怎么分层、衰减、唤醒
- 研究 Agent 如何在有限上下文里决定“现在该想什么”
- 给个人 Agent 增加可解释的长期记忆原型
- 研究“遗忘机制”本身，而不是只研究语义检索

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
