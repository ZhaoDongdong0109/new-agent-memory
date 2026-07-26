"""狗粮期第二轮：带着第一轮的库继续真实使用。

场景：一周后的续用。追加新知识（这次用 memory_type 参数——
第一轮修复的成果），问真实的跟进问题，并量化"结果太多"的摩擦：
每个查询统计返回条数。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MEMORY_DATA_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "store")

import mcp_server
mcp_server.create_llm_fn = lambda: None
server = mcp_server.MemoryMCPServer()


def call(name, **arguments):
    resp = server.handle_request({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    })
    if "error" in resp:
        return resp["error"]["message"], True
    result = resp["result"]
    return result["content"][0]["text"], result.get("isError", False)


# ============ Phase A：追加本周真实新知识（用上 memory_type） ============

FEED2 = [
    dict(content="PR #7 已经用 rebase 方式合并进 main，狗粮期第一轮的4个修复全部进主干",
         memory_type="fact", importance=0.8),
    dict(content="狗粮期第一轮的完整报告在 docs/dogfood_report_2026-07.md，含修复前后指标对比",
         memory_type="fact", importance=0.7),
    dict(content="现在全部测试是214项，狗粮期新增了6个回归测试",
         memory_type="fact", importance=0.7),
    dict(content="老赵合并 PR 的惯例是 rebase 方式，保持线性历史",
         memory_type="preference", importance=0.7),
    dict(content="v0.2.0 的 git 标签还没打，环境代理不让推标签，需要老赵在 GitHub 网页上建 Release",
         memory_type="fact", importance=0.8),
    dict(content="下一轮优化候选：检索结果最终截断排序，以及 MCP 的程序记忆入口",
         memory_type="idea", importance=0.7),
    # 程序性知识（第二轮修复：procedure 类型，最慢衰减 + 取代管理）
    dict(content="发布新版本的操作流程：先本地跑全部测试确认绿，再用 workflow_dispatch 手动触发 publish 工作流，Trusted Publisher 按 publish.yml 加 environment pypi 匹配",
         memory_type="procedure", importance=0.9),
]

print("=== Phase A: 追加 ===")
for item in FEED2:
    text, err = call("memory_add", **item)
    print(("ERR " if err else "ok  ") + text[:70])

# ============ Phase B：真实跟进查询 + 噪声量化 ============

QUERIES2 = [
    ("上一轮狗粮测试发现了什么问题", "狗粮"),
    ("PR 7 合并了吗", "rebase"),
    ("还有什么优化没做", "截断"),
    ("怎么发布新版本", "workflow_dispatch"),
    ("老赵喜欢怎么合并 PR", "rebase"),
    ("现在有多少项测试", "214"),
    ("git 标签的问题解决了吗", "Release"),
    ("狗粮报告在哪", "dogfood_report"),
    ("睡眠巩固的触发阈值", "7.5"),
    ("为什么融合要归一化", "1.6"),
]

print("\n=== Phase B: 查询（含返回条数） ===")
hits1 = hits3 = 0
counts = []
for query, expect in QUERIES2:
    text, _ = call("memory_search", query=query)
    lines = [ln for ln in text.splitlines() if ln.startswith("[mem_")]
    counts.append(len(lines))
    top1 = lines[0] if lines else ""
    top3 = " ".join(lines[:3])
    in1, in3 = expect in top1, expect in top3
    hits1 += in1
    hits3 += in3
    mark = "✓" if in1 else ("△top3" if in3 else "✗")
    print(f"{mark} [{len(lines):2d}条] {query}")
    if not in3:
        print(f"    期望含: {expect}")
        for ln in lines[:3]:
            print("    " + ln[:100])

print(f"\ntop1 {hits1}/{len(QUERIES2)}, top3 {hits3}/{len(QUERIES2)}")
print(f"每查询返回条数: min={min(counts)} max={max(counts)} avg={sum(counts)/len(counts):.1f}")

# ============ Phase C：生命周期 ============

print("\n=== Phase C: 生命周期 ===")
text, _ = call("memory_sleep")
print("sleep: " + text[:200].replace("\n", " | "))
text, _ = call("memory_stats")
print("stats: " + text[:300].replace("\n", " | "))
