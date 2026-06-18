"""
Benchmark 运行器

运行检索和巩固 benchmark，生成报告。
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Set

from scripts.bench.metrics import (
    calculate_all_metrics,
    aggregate_metrics,
    mean_latency,
    p50_latency,
    p95_latency,
    p99_latency,
)


class BenchmarkRunner:
    """
    Benchmark 运行器

    运行检索和巩固 benchmark，生成报告。
    """

    def __init__(self, memory_system=None):
        """
        Args:
            memory_system: HumanLikeMemorySystem 实例
        """
        self.memory = memory_system
        self.results = []

    def run_retrieval_benchmark(
        self,
        dataset: List[Dict],
        top_k: int = 10,
    ) -> Dict:
        """
        运行检索 benchmark

        Args:
            dataset: 测试数据集，每个元素是 {"query": str, "expected_ids": List[str]}
            top_k: 取前 K 个结果

        Returns:
            benchmark 结果
        """
        results = []
        latencies = []

        for item in dataset:
            query = item["query"]
            expected_ids = set(item["expected_ids"])
            category = item.get("category", "general")

            # 运行检索
            start = time.time()
            if self.memory:
                result = self.memory.retrieve(query, allow_forgotten=True)
                retrieved_ids = [c.id for c in result.chunks]
            else:
                retrieved_ids = []
            latency = time.time() - start

            latencies.append(latency)

            # 计算指标
            metrics = calculate_all_metrics(retrieved_ids, expected_ids, latency, top_k)
            metrics["query"] = query
            metrics["category"] = category
            results.append(metrics)

        # 聚合指标
        aggregated = aggregate_metrics(results)

        # 添加延迟统计
        aggregated["latency_p50"] = p50_latency(latencies)
        aggregated["latency_p95"] = p95_latency(latencies)
        aggregated["latency_p99"] = p99_latency(latencies)

        return {
            "type": "retrieval",
            "total_queries": len(dataset),
            "metrics": aggregated,
            "details": results,
        }

    def run_consolidation_benchmark(
        self,
        dataset: List[Dict],
    ) -> Dict:
        """
        运行巩固 benchmark

        Args:
            dataset: 测试数据集

        Returns:
            benchmark 结果
        """
        results = []

        for item in dataset:
            episodes = item["episodes"]
            expected_count = item.get("expected_memory_count", 0)
            expected_types = item.get("expected_types", [])

            # 运行巩固
            start = time.time()
            if self.memory:
                # 模拟巩固
                memory_ids = []
                for episode in episodes:
                    # 简化：直接添加记忆
                    memory_id = self.memory.add_memory(
                        content=episode.get("content", ""),
                        importance=episode.get("importance", 0.5),
                    )
                    if memory_id:
                        memory_ids.append(memory_id)
            else:
                memory_ids = []
            latency = time.time() - start

            # 检查结果
            actual_count = len(memory_ids)
            count_ok = actual_count == expected_count

            results.append({
                "expected_count": expected_count,
                "actual_count": actual_count,
                "count_ok": count_ok,
                "latency": latency,
            })

        # 聚合
        total = len(results)
        correct = sum(1 for r in results if r["count_ok"])
        latencies = [r["latency"] for r in results]

        return {
            "type": "consolidation",
            "total_cases": total,
            "correct_cases": correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "latency_mean": mean_latency(latencies),
            "latency_p50": p50_latency(latencies),
            "latency_p95": p95_latency(latencies),
            "details": results,
        }

    def run_weight_benchmark(
        self,
        num_memories: int = 1000,
    ) -> Dict:
        """
        运行权重计算 benchmark

        Args:
            num_memories: 记忆数量

        Returns:
            benchmark 结果
        """
        if not self.memory or not hasattr(self.memory, 'core'):
            return {"type": "weight", "error": "No memory system"}

        # 生成测试记忆
        for i in range(num_memories):
            self.memory.add_memory(
                content=f"测试记忆 {i}",
                importance=0.5,
            )

        # 测试权重计算
        latencies = []
        chunks = self.memory.core._store.get_all()

        for chunk_id, chunk in list(chunks.items())[:100]:  # 测试 100 个
            start = time.time()
            self.memory.core.calc_weight(chunk)
            latency = time.time() - start
            latencies.append(latency)

        return {
            "type": "weight",
            "num_memories": num_memories,
            "tested": len(latencies),
            "latency_mean": mean_latency(latencies),
            "latency_p50": p50_latency(latencies),
            "latency_p95": p95_latency(latencies),
            "latency_p99": p99_latency(latencies),
        }

    def generate_report(self, output_file: str = None) -> str:
        """
        生成报告

        Args:
            output_file: 输出文件路径（可选）

        Returns:
            报告文本
        """
        report_lines = []
        report_lines.append("# Benchmark Report")
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        for result in self.results:
            report_lines.append(f"## {result.get('type', 'Unknown').title()} Benchmark")
            report_lines.append("")

            # 指标
            metrics = result.get("metrics", {})
            if metrics:
                report_lines.append("### Metrics")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        report_lines.append(f"- {key}: {value:.4f}")
                    else:
                        report_lines.append(f"- {key}: {value}")
                report_lines.append("")

            # 延迟
            if "latency_p50" in result:
                report_lines.append("### Latency")
                report_lines.append(f"- P50: {result['latency_p50']*1000:.2f}ms")
                report_lines.append(f"- P95: {result['latency_p95']*1000:.2f}ms")
                report_lines.append(f"- P99: {result['latency_p99']*1000:.2f}ms")
                report_lines.append("")

        report_text = "\n".join(report_lines)

        # 保存到文件
        if output_file:
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report_text)

        return report_text


def load_dataset(file_path: str) -> List[Dict]:
    """
    加载测试数据集

    Args:
        file_path: 数据集文件路径

    Returns:
        数据集列表
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
