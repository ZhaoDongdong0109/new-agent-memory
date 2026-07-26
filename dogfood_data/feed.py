"""狗粮期 Phase 1：把本项目开发史的真实知识喂进记忆系统。

全程走 MCP 工具面（MemoryMCPServer._call_tool）——真实客户端的路径。
内容全部是真实发生过的事实/偏好/事件，用自然语言喂（测试规则抽取质量）。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MEMORY_DATA_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "store")

import mcp_server
mcp_server.create_llm_fn = lambda: None  # 无本地 LLM，走规则抽取（真实默认场景）
server = mcp_server.MemoryMCPServer()


def call(name, **arguments):
    # 走真实 JSON-RPC 路径（第二轮教训：_call_tool 捷径绕过了
    # handle_request 的白名单，第一轮没咬出分发缺陷）
    resp = server.handle_request({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    })
    if "error" in resp:
        return resp["error"]["message"], True
    result = resp["result"]
    text = result["content"][0]["text"]
    return text, result.get("isError", False)


# ============ 真实项目知识（自然语言，测试抽取） ============

FEED = [
    # —— 事实：项目与架构 ——
    dict(content="new-agent-memory 项目的定位是可解释的类人记忆内核，零运行时依赖，每个权重和决策都有审计轨迹", memory_type="fact"),
    dict(content="main 分支是老赵的原始架构，只有15个文件的纯净记忆内核", importance=0.8, memory_type="fact"),
    dict(content="记忆强度模型用的是 ACT-R 基线激活方程，参数表在 core/weight_system.py，STORY 类型衰减速率 d=0.38 最持久，INTERACTION d=0.50 最容易忘", importance=0.8, memory_type="fact"),
    dict(content="混合检索是 BM25 加哈希 TF-IDF 加 RRF 融合，融合时每腿分数要 min-max 归一化，否则精确命中和模板相似在名次上只差1.6%", importance=0.7, memory_type="fact"),
    dict(content="伪遗忘层的唤醒需要至少2个匹配锚点且得分超过0.3，唤醒临时权重达到0.55才提升回核心层", memory_type="fact"),
    dict(content="睡眠巩固的触发条件是写入累计重要性达到7.5，这个数字是 Generative Agents 论文的150按0到1重要性折算来的", importance=0.7, memory_type="fact"),
    # —— 事实：发布与 CI ——
    dict(content="2026年7月26日 v0.2.0 发布到了 PyPI，用的是 Trusted Publisher，包名 new-agent-memory", time_note="2026-07-26", importance=0.9, memory_type="fact"),
    dict(content="发布必须用 workflow_dispatch 手动调度 publish 工作流，因为这个环境的 git 凭据代理不允许推标签", importance=0.9, memory_type="fact"),
    dict(content="PyPI 的 Trusted Publisher 是按工作流文件名 publish.yml 加 environment pypi 匹配的，跟触发方式无关", importance=0.8, memory_type="fact"),
    dict(content="CI 的 ruff 规则集固定为 E4 E7 E9 F，因为 CI 装的是未固定版本的 ruff，新版默认规则漂移曾报出770个风格错误", importance=0.8, memory_type="fact"),
    dict(content="bandit 新版把 MD5 判为高危，审计日志里的 md5 都加了 usedforsecurity=False", memory_type="fact"),
    dict(content="2026年6月18日 v0.1.0-alpha 发布，那是第一次用 publish 工作流", time_note="2026-06-18", memory_type="fact"),
    # —— 事实：拆分 ——
    dict(content="agent-hub 是从 new-agent-memory 拆分出去的独立仓库，hub 是老赵的另一个项目，多 agent 交流用的，之前混进记忆项目了", importance=0.9, memory_type="fact"),
    dict(content="agent-hub 的记忆桥接是可选依赖，pip install agent-hub[memory] 会装 new-agent-memory 0.2.0 以上", memory_type="fact"),
    dict(content="拆分后 new-agent-memory 剩200项测试，agent-hub 带走28项单元测试和4项集成测试", memory_type="fact"),
    # —— 偏好：老赵的工作方式 ——
    dict(content="老赵不想自己做狗粮测试，让我自己喂狗粮自己当真实用户", importance=0.8, memory_type="preference"),
    dict(content="老赵说过跟进的事情全权交给我处理，他只要好结果", importance=0.9, memory_type="preference"),
    dict(content="老赵的指令风格是简短的中文，比如确认拆、发布0.2.0、继续", importance=0.6, memory_type="preference"),
    dict(content="提交信息用中文写详细说明是这个项目的惯例，每个决策要写清为什么", memory_type="preference"),
    # —— 事件：修复史 ——
    dict(content="第一轮审计发现权重公式有0.195的永久下限高于0.15降级阈值，导致任何记忆都无法被遗忘，招牌功能是虚构的", importance=0.9, memory_type="story"),
    dict(content="对抗审查曾抓到扩散激活重复传播的缺陷，同一节点被多路径激活后向外传播多次，后来换成 PPR 幂迭代从结构上消灭了这类缺陷", importance=0.7, memory_type="story"),
    dict(content="查询侧英文标签和抽取侧中文标签曾经零交集，自然语言查询永远命中不了自动抽取的记忆，用双语同义词表打通的", importance=0.8, memory_type="story"),
    dict(content="MCP 服务器曾经把日志打到 stdout 污染 JSON-RPC 协议，还在客户端请求前发未经请求的 id 0 响应", memory_type="story"),
    dict(content="CLI 的 --fresh 参数曾经会静默清空磁盘上的记忆，跑一次测试就把积累的记忆全丢了", importance=0.7, memory_type="story"),
    dict(content="容量探针第一次跑出 MRR 只有0.27，根因是 RRF 纯名次融合抛弃了分数差距，修复后到0.99", importance=0.7, memory_type="story"),
    # —— 日常琐事（低重要性，应该被自然遗忘的那类） ——
    dict(content="今天下午跑了三次 sleep 80 等 CI，每次都是80秒左右", importance=0.2),
    dict(content="scratchpad 目录在 /tmp 下面一长串路径", importance=0.2),
    dict(content="git worktree 模拟合并后忘了先退出目录导致 getcwd 报错一次", importance=0.2),
    dict(content="中途用户切换过两次模型又切回来了", importance=0.1),
]

results = []
for item in FEED:
    kwargs = {"content": item["content"]}
    if "importance" in item:
        kwargs["importance"] = item["importance"]
    if "memory_type" in item:
        kwargs["memory_type"] = item["memory_type"]
    text, err = call("memory_add", **kwargs)
    results.append({"content": item["content"][:50], "result": text, "error": err})

print(json.dumps({"fed": len(results), "errors": sum(r["error"] for r in results)}, ensure_ascii=False))
for r in results:
    print(("ERR " if r["error"] else "ok  ") + r["result"][:60] + " | " + r["content"])

stats, _ = call("memory_stats")
print("\n=== stats ===")
print(stats[:500])
