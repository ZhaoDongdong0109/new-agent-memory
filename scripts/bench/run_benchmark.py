#!/usr/bin/env python3
"""
Benchmark 运行脚本

运行检索和巩固 benchmark，生成报告。

使用方式：
    python scripts/bench/run_benchmark.py [--data-dir ./memory_data] [--output report.md]
"""

import argparse
import os
import sys

# 添加项目根目录到 path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from scripts.bench.benchmark_runner import BenchmarkRunner, load_dataset  # noqa: E402
from main import HumanLikeMemorySystem  # noqa: E402


def create_test_memories(memory_system, num_memories=100):
    """创建测试记忆"""
    for i in range(num_memories):
        memory_system.add_memory(
            content=f"测试记忆 {i}: 这是第 {i} 条测试记忆",
            importance=0.5 + (i % 5) * 0.1,
        )


def main():
    parser = argparse.ArgumentParser(
        description="运行 new-agent-memory benchmark"
    )
    parser.add_argument(
        "--data-dir",
        default="./memory_data",
        help="数据目录路径"
    )
    parser.add_argument(
        "--output",
        default="benchmark_report.md",
        help="输出报告文件"
    )
    parser.add_argument(
        "--num-memories",
        type=int,
        default=100,
        help="测试记忆数量"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("new-agent-memory Benchmark")
    print("=" * 60)
    print()

    # 创建记忆系统
    print("初始化记忆系统...")
    memory_system = HumanLikeMemorySystem(data_dir=args.data_dir)

    # 创建测试记忆
    print(f"创建 {args.num_memories} 条测试记忆...")
    create_test_memories(memory_system, args.num_memories)

    # 创建 benchmark 运行器
    runner = BenchmarkRunner(memory_system)

    # 运行检索 benchmark
    print("\n运行检索 benchmark...")
    dataset_path = os.path.join(project_root, "scripts/bench/datasets/retrieval.json")
    if os.path.exists(dataset_path):
        dataset = load_dataset(dataset_path)
        retrieval_result = runner.run_retrieval_benchmark(dataset, top_k=10)
        runner.results.append(retrieval_result)
        print(f"  完成 {retrieval_result['total_queries']} 个查询")
        print(f"  平均 Recall@10: {retrieval_result['metrics'].get('mean_recall@10', 0):.4f}")
        print(f"  平均 MRR: {retrieval_result['metrics'].get('mean_mrr', 0):.4f}")
    else:
        print(f"  数据集不存在: {dataset_path}")

    # 运行权重 benchmark
    print("\n运行权重计算 benchmark...")
    weight_result = runner.run_weight_benchmark(num_memories=args.num_memories)
    runner.results.append(weight_result)
    print(f"  测试 {weight_result.get('tested', 0)} 个记忆")
    print(f"  平均延迟: {weight_result.get('latency_mean', 0)*1000:.2f}ms")

    # 运行确定性探针（种子可复现，无 LLM 评审）
    print("\n运行确定性探针（ToT-lite / 知识更新 / 容量门）...")
    import tempfile
    from scripts.bench.tot_lite import probes_report_section, run_all_probes
    with tempfile.TemporaryDirectory() as probe_dir:
        probe_results = run_all_probes(probe_dir, seed=42)
    for probe in probe_results:
        print(f"  {probe['name']}: " + ", ".join(
            f"{k}={v:.3f}" for k, v in probe.items()
            if isinstance(v, float) and not k.endswith("_ms")
        ))

    # 生成报告
    print(f"\n生成报告: {args.output}")
    report = runner.generate_report(args.output)
    report += probes_report_section(probe_results)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)

    print("\n" + "=" * 60)
    print("Benchmark 完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
