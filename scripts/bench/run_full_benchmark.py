#!/usr/bin/env python3
"""
完整 Benchmark 测试

测试记忆系统的检索质量、延迟和写入性能。
"""

import json
import os
import sys
import time

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from main import HumanLikeMemorySystem
from scripts.bench.benchmark_runner import BenchmarkRunner
from scripts.bench.metrics import (
    recall_at_k,
    precision_at_k,
    mrr,
    ndcg_at_k,
    mean_latency,
    p50_latency,
    p95_latency,
    p99_latency,
)


def create_test_memories(system: HumanLikeMemorySystem) -> dict:
    """
    创建测试记忆

    Returns:
        query -> expected_ids 映射
    """
    memories = [
        {
            "content": "今天中午和同事在公司食堂吃了红烧肉，味道不错",
            "persons": ["同事"],
            "location": "公司",
            "topics": ["美食", "餐饮"],
            "time_context": "中午",
            "importance": 0.6,
        },
        {
            "content": "上周出差去了上海，拜访了客户讨论新项目",
            "persons": ["客户"],
            "location": "上海",
            "topics": ["工作", "出差"],
            "time_relative": "上周",
            "importance": 0.8,
        },
        {
            "content": "和张三一起去了电影院看了科幻电影，很精彩",
            "persons": ["张三"],
            "location": "电影院",
            "topics": ["娱乐", "电影"],
            "importance": 0.5,
        },
        {
            "content": "北京今天下雪了，路上很滑，开车要小心",
            "location": "北京",
            "topics": ["天气"],
            "emotion_valence": -0.2,
            "importance": 0.4,
        },
        {
            "content": "项目进展顺利，预计下周完成第一阶段",
            "topics": ["工作", "项目"],
            "importance": 0.9,
        },
        {
            "content": "昨天晚上看了电影《星际穿越》，很有深度",
            "topics": ["娱乐", "电影"],
            "time_relative": "昨天",
            "time_context": "晚上",
            "importance": 0.5,
        },
        {
            "content": "学习了 Python 的装饰器，终于理解了它的原理",
            "topics": ["学习", "编程"],
            "importance": 0.7,
        },
        {
            "content": "早上跑步跑了5公里，感觉精神很好",
            "topics": ["运动", "健康"],
            "time_context": "早上",
            "emotion_valence": 0.8,
            "importance": 0.6,
        },
        {
            "content": "和客户讨论了合同细节，双方达成一致",
            "persons": ["客户"],
            "topics": ["工作", "商务"],
            "importance": 0.9,
        },
        {
            "content": "晚上在家做了意大利面，家人很喜欢",
            "location": "家",
            "topics": ["美食", "餐饮"],
            "time_context": "晚上",
            "importance": 0.5,
        },
    ]

    # 创建记忆并记录 ID
    memory_map = {}
    for i, mem in enumerate(memories):
        memory_id = system.add_raw_memory(
            text=mem["content"],
            user_id="benchmark_user",
            importance=mem.get("importance", 0.5),
            check_duplicate=False,  # benchmark 不检查重复
        )
        memory_map[i] = memory_id
        print(f"  创建记忆 {i+1}: {memory_id}")

    # 查询映射
    query_expected = {
        "中午吃了什么": [memory_map[0]],
        "出差去了哪里": [memory_map[1]],
        "和张三一起做了什么": [memory_map[2]],
        "北京的天气": [memory_map[3]],
        "项目进展": [memory_map[4]],
        "看电影": [memory_map[5], memory_map[2]],  # 两个电影相关记忆
        "学习新知识": [memory_map[6]],
        "运动锻炼": [memory_map[7]],
        "和客户讨论": [memory_map[8]],
        "晚餐吃了什么": [memory_map[9]],
    }

    return query_expected


def run_retrieval_benchmark(system: HumanLikeMemorySystem, query_expected: dict) -> dict:
    """
    运行检索 benchmark

    Returns:
        benchmark 结果
    """
    results = []
    latencies = []

    for query, expected_ids in query_expected.items():
        # 运行检索
        start = time.time()
        result = system.retrieve(query, user_id="benchmark_user")
        latency = time.time() - start
        latencies.append(latency)

        # 获取检索到的 ID
        retrieved_ids = [c.id for c in result.chunks]

        # 计算指标
        metrics = {
            "query": query,
            "recall@1": recall_at_k(retrieved_ids, set(expected_ids), 1),
            "recall@3": recall_at_k(retrieved_ids, set(expected_ids), 3),
            "recall@5": recall_at_k(retrieved_ids, set(expected_ids), 5),
            "precision@1": precision_at_k(retrieved_ids, set(expected_ids), 1),
            "precision@3": precision_at_k(retrieved_ids, set(expected_ids), 3),
            "mrr": mrr(retrieved_ids, set(expected_ids)),
            "ndcg@5": ndcg_at_k(retrieved_ids, set(expected_ids), 5),
            "latency": latency,
            "num_results": len(retrieved_ids),
        }
        results.append(metrics)

    # 聚合指标
    aggregated = {
        "total_queries": len(results),
        "mean_recall@1": sum(r["recall@1"] for r in results) / len(results),
        "mean_recall@3": sum(r["recall@3"] for r in results) / len(results),
        "mean_recall@5": sum(r["recall@5"] for r in results) / len(results),
        "mean_precision@1": sum(r["precision@1"] for r in results) / len(results),
        "mean_precision@3": sum(r["precision@3"] for r in results) / len(results),
        "mean_mrr": sum(r["mrr"] for r in results) / len(results),
        "mean_ndcg@5": sum(r["ndcg@5"] for r in results) / len(results),
        "latency_mean": mean_latency(latencies),
        "latency_p50": p50_latency(latencies),
        "latency_p95": p95_latency(latencies),
        "latency_p99": p99_latency(latencies),
    }

    return {
        "type": "retrieval",
        "metrics": aggregated,
        "details": results,
    }


def run_write_benchmark(system: HumanLikeMemorySystem, num_memories: int = 100) -> dict:
    """
    运行写入 benchmark

    Returns:
        benchmark 结果
    """
    latencies = []

    for i in range(num_memories):
        start = time.time()
        system.add_raw_memory(
            text=f"测试记忆 {i}: 这是第 {i} 条测试记忆，用于 benchmark 测试",
            user_id="benchmark_user",
            importance=0.5,
            check_duplicate=False,
        )
        latency = time.time() - start
        latencies.append(latency)

    return {
        "type": "write",
        "num_memories": num_memories,
        "latency_mean": mean_latency(latencies),
        "latency_p50": p50_latency(latencies),
        "latency_p95": p95_latency(latencies),
        "latency_p99": p99_latency(latencies),
        "throughput": num_memories / sum(latencies),
    }


def generate_report(results: list) -> str:
    """
    生成 benchmark 报告

    Returns:
        报告文本
    """
    lines = []
    lines.append("# Memory System Benchmark Report")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    for result in results:
        lines.append(f"## {result['type'].title()} Benchmark")
        lines.append("")

        if result["type"] == "retrieval":
            metrics = result["metrics"]
            lines.append("### Retrieval Quality")
            lines.append(f"- Recall@1: {metrics['mean_recall@1']:.4f}")
            lines.append(f"- Recall@3: {metrics['mean_recall@3']:.4f}")
            lines.append(f"- Recall@5: {metrics['mean_recall@5']:.4f}")
            lines.append(f"- Precision@1: {metrics['mean_precision@1']:.4f}")
            lines.append(f"- Precision@3: {metrics['mean_precision@3']:.4f}")
            lines.append(f"- MRR: {metrics['mean_mrr']:.4f}")
            lines.append(f"- NDCG@5: {metrics['mean_ndcg@5']:.4f}")
            lines.append("")
            lines.append("### Latency")
            lines.append(f"- Mean: {metrics['latency_mean']*1000:.2f}ms")
            lines.append(f"- P50: {metrics['latency_p50']*1000:.2f}ms")
            lines.append(f"- P95: {metrics['latency_p95']*1000:.2f}ms")
            lines.append(f"- P99: {metrics['latency_p99']*1000:.2f}ms")
            lines.append("")

        elif result["type"] == "write":
            lines.append("### Write Performance")
            lines.append(f"- Total memories: {result['num_memories']}")
            lines.append(f"- Throughput: {result['throughput']:.2f} memories/sec")
            lines.append("")
            lines.append("### Latency")
            lines.append(f"- Mean: {result['latency_mean']*1000:.2f}ms")
            lines.append(f"- P50: {result['latency_p50']*1000:.2f}ms")
            lines.append(f"- P95: {result['latency_p95']*1000:.2f}ms")
            lines.append(f"- P99: {result['latency_p99']*1000:.2f}ms")
            lines.append("")

    # 详细结果
    lines.append("## Detailed Results")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(results, indent=2, ensure_ascii=False))
    lines.append("```")

    return "\n".join(lines)


def main():
    """主函数"""
    print("=" * 60)
    print("Memory System Benchmark")
    print("=" * 60)

    # 创建临时目录
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_dir = os.path.join(tmp_dir, "benchmark_data")

        # 初始化系统
        print("\n1. 初始化记忆系统...")
        system = HumanLikeMemorySystem(
            store_backend="sqlite",
            data_dir=data_dir,
            enable_pii_detection=True,
            enable_audit_log=True,
        )

        # 创建测试记忆
        print("\n2. 创建测试记忆...")
        query_expected = create_test_memories(system)
        print(f"   创建了 {len(query_expected)} 组测试记忆")

        # 运行检索 benchmark
        print("\n3. 运行检索 benchmark...")
        retrieval_result = run_retrieval_benchmark(system, query_expected)
        print(f"   完成 {retrieval_result['metrics']['total_queries']} 次查询")

        # 运行写入 benchmark
        print("\n4. 运行写入 benchmark...")
        write_result = run_write_benchmark(system, num_memories=50)
        print(f"   写入 {write_result['num_memories']} 条记忆")

        # 生成报告
        print("\n5. 生成报告...")
        results = [retrieval_result, write_result]
        report = generate_report(results)

        # 保存报告
        report_file = os.path.join(os.path.dirname(__file__), "benchmark_report.md")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"   报告已保存到: {report_file}")

        # 打印摘要
        print("\n" + "=" * 60)
        print("Benchmark Summary")
        print("=" * 60)
        print(f"\nRetrieval Quality:")
        print(f"  Recall@1: {retrieval_result['metrics']['mean_recall@1']:.4f}")
        print(f"  Recall@3: {retrieval_result['metrics']['mean_recall@3']:.4f}")
        print(f"  MRR: {retrieval_result['metrics']['mean_mrr']:.4f}")
        print(f"\nLatency:")
        print(f"  Retrieval P50: {retrieval_result['metrics']['latency_p50']*1000:.2f}ms")
        print(f"  Retrieval P95: {retrieval_result['metrics']['latency_p95']*1000:.2f}ms")
        print(f"  Write P50: {write_result['latency_p50']*1000:.2f}ms")
        print(f"  Write Throughput: {write_result['throughput']:.2f} memories/sec")

    print("\n✅ Benchmark 完成!")


if __name__ == "__main__":
    main()
