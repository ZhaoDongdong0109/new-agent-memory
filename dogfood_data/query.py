"""狗粮期 Phase 2：真实查询。

这些是未来会话里我真正会问的问题。每条标注期望命中的关键内容，
诚实评分：top-3 内有没有；top-1 对不对。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MEMORY_DATA_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "store")

import mcp_server
mcp_server.create_llm_fn = lambda: None
server = mcp_server.MemoryMCPServer()


def search(query, **extra):
    resp = server.handle_request({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "memory_search", "arguments": {"query": query, **extra}},
    })
    return resp["result"]["content"][0]["text"]


# (查询, 期望出现在结果里的关键子串)
QUERIES = [
    ("发布流程怎么走", "workflow_dispatch"),
    ("为什么不能推 git 标签", "凭据代理"),
    ("Trusted Publisher 是怎么匹配的", "publish.yml"),
    ("ruff 的规则集为什么要固定", "770"),
    ("ACT-R 的参数在哪个文件", "weight_system"),
    ("睡眠巩固什么时候触发", "7.5"),
    ("agent-hub 是什么", "拆分"),
    ("老赵对跟进工作的要求", "好结果"),
    ("老赵想自己做狗粮测试吗", "自己喂"),
    ("权重下限的缺陷是什么", "0.195"),
    ("重复传播的缺陷怎么解决的", "PPR"),
    ("中英文词汇割裂问题", "双语"),
    ("v0.2.0 是什么时候发布的", "7月26"),
    ("拆分后还剩多少测试", "200"),
    ("MD5 为什么要加参数", "usedforsecurity"),
    ("唤醒需要什么条件", "锚点"),
    ("2026年6月发生了什么", "v0.1.0"),
    # 应该弃答的：库里没有的知识
    ("数据库分库分表的方案是什么", None),
    ("明年的融资计划", None),
]

hits3, hits1, abstain_ok, total_pos, total_neg = 0, 0, 0, 0, 0
for query, expect in QUERIES:
    text = search(query)
    lines = [ln for ln in text.splitlines() if ln.startswith("[mem_")]
    if expect is None:
        total_neg += 1
        # 正确处置 = 拒答，或返回但带覆盖率软标注（强警告/谨慎均可：
        # 无嵌入的词面覆盖不能硬弃答——第一轮教训，判断权交给调用方）
        ok = ("No relevant memories" in text or not lines
              or "警告" in text or "谨慎" in text)
        abstain_ok += ok
        print(f"{'✓' if ok else '✗ 应弃答却回答了'} [负例] {query}")
        if not ok:
            print("    top1: " + (lines[0][:100] if lines else text[:100]))
    else:
        total_pos += 1
        top3 = " ".join(lines[:3])
        top1 = lines[0] if lines else ""
        in3, in1 = expect in top3, expect in top1
        hits3 += in3
        hits1 += in1
        mark = "✓" if in1 else ("△top3" if in3 else "✗")
        print(f"{mark} {query}")
        if not in3:
            print(f"    期望含: {expect}")
            print("    top1: " + (top1[:110] if top1 else text.splitlines()[0][:110]))

print(f"\n=== 评分: top1 {hits1}/{total_pos}, top3 {hits3}/{total_pos}, 弃答正确 {abstain_ok}/{total_neg} ===")
