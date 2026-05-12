"""
性能基准测试 - 类人记忆系统

测试项目：
1. 记忆添加性能
2. 检索性能（无索引 vs 有索引）
3. 权重计算性能（缓存效果）
4. 批量操作性能
5. 内存使用
"""

import time
import random
import sys
import tracemalloc
from typing import List, Dict, Any

from main import HumanLikeMemorySystem
from memory_layer_core import MemoryLayerCore
from memory_chunk import MemoryChunk
from core.weight_system import MemoryType


def generate_test_memories(count: int) -> List[Dict[str, Any]]:
    """生成测试记忆数据"""
    locations = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安"]
    persons = ["张三", "李四", "王五", "赵六", "钱七", "孙八", "周九", "吴十"]
    topics = ["工作", "生活", "学习", "旅行", "娱乐", "美食", "运动", "音乐"]
    emotions = ["开心", "平静", "焦虑", "兴奋", "难过", "期待"]
    
    memories = []
    for i in range(count):
        memories.append({
            "content": f"测试记忆 {i}: 这是一条用于性能测试的记忆内容",
            "memory_type": random.choice(list(MemoryType)),
            "time_absolute": f"2026-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            "time_context": random.choice(["上午", "中午", "下午", "晚上"]),
            "location": random.choice(locations),
            "persons": random.sample(persons, random.randint(1, 3)),
            "topics": set(random.sample(topics, random.randint(1, 4))),
            "emotion_valence": random.uniform(-1, 1),
            "emotion_intensity": random.uniform(0, 1),
            "importance": random.uniform(0.3, 1.0),
        })
    return memories


def benchmark_add_memories(count: int) -> Dict[str, float]:
    """测试记忆添加性能"""
    system = HumanLikeMemorySystem()
    memories = generate_test_memories(count)
    
    start = time.perf_counter()
    for mem in memories:
        system.add_memory(**mem)
    elapsed = time.perf_counter() - start
    
    return {
        "total_time": elapsed,
        "avg_time": elapsed / count,
        "memories_per_second": count / elapsed,
    }


def benchmark_batch_add(count: int) -> Dict[str, float]:
    """测试批量添加性能"""
    system = HumanLikeMemorySystem()
    memories = generate_test_memories(count)
    
    start = time.perf_counter()
    system.add_memories_batch(memories)
    elapsed = time.perf_counter() - start
    
    return {
        "total_time": elapsed,
        "avg_time": elapsed / count,
        "memories_per_second": count / elapsed,
        "speedup_vs_single": benchmark_add_memories(100)["total_time"] / elapsed if count == 100 else None,
    }


def benchmark_retrieval(system: HumanLikeMemorySystem, query_count: int) -> Dict[str, float]:
    """测试检索性能"""
    test_queries = [
        {"location": "北京"},
        {"persons": {"张三"}},
        {"topics": {"工作", "学习"}},
        {"time_context": "上午"},
        {"location": "上海", "persons": {"李四"}},
        {"topics": {"美食"}, "time_context": "中午"},
    ]
    
    start = time.perf_counter()
    for _ in range(query_count):
        query = random.choice(test_queries)
        system.retrieve("测试查询", allow_forgotten=False)
    elapsed = time.perf_counter() - start
    
    return {
        "total_time": elapsed,
        "avg_time": elapsed / query_count,
        "queries_per_second": query_count / elapsed,
    }


def benchmark_weight_calculation(count: int) -> Dict[str, float]:
    """测试权重计算性能"""
    core = MemoryLayerCore()
    
    for i in range(count):
        chunk = MemoryChunk(
            content=f"测试记忆 {i}",
            location=f"地点{i % 10}",
            topics={f"主题{i % 20}"}
        )
        core.add(chunk)
    
    start = time.perf_counter()
    for chunk in core.chunks.values():
        core.calc_weight(chunk)
    elapsed_no_cache = time.perf_counter() - start
    
    start = time.perf_counter()
    for _ in range(3):
        for chunk in core.chunks.values():
            core.calc_weight(chunk)
    elapsed_with_cache = time.perf_counter() - start
    
    cache_hits = core._stats["cache_hits"]
    cache_misses = core._stats["cache_misses"]
    
    return {
        "first_calculation_time": elapsed_no_cache,
        "cached_calculations_time": elapsed_with_cache,
        "cache_hit_rate": cache_hits / max(1, cache_hits + cache_misses),
        "speedup_factor": elapsed_no_cache * 3 / elapsed_with_cache if elapsed_with_cache > 0 else 1,
    }


def benchmark_memory_usage(count: int) -> Dict[str, float]:
    """测试内存使用"""
    tracemalloc.start()
    
    system = HumanLikeMemorySystem()
    memories = generate_test_memories(count)
    system.add_memories_batch(memories)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        "current_mb": current / 1024 / 1024,
        "peak_mb": peak / 1024 / 1024,
        "memory_per_memory_kb": peak / 1024 / count,
    }


def benchmark_index_vs_fullscan(count: int) -> Dict[str, float]:
    """对比有索引和无索引的检索性能"""
    system = HumanLikeMemorySystem()
    memories = generate_test_memories(count)
    system.add_memories_batch(memories)
    
    stats = system.core.get_performance_stats()
    stats["core_memory_chunks"] = len(system.core)
    
    return stats


def run_all_benchmarks():
    """运行所有基准测试"""
    print("=" * 70)
    print("类人记忆系统 - 性能基准测试")
    print("=" * 70)
    
    results = {}
    
    print("\n[1] 记忆添加性能测试...")
    for count in [100, 500, 1000]:
        result = benchmark_add_memories(count)
        results[f"add_{count}"] = result
        print(f"  添加 {count} 条记忆:")
        print(f"    总耗时: {result['total_time']*1000:.2f}ms")
        print(f"    平均耗时: {result['avg_time']*1000:.3f}ms")
        print(f"    吞吐量: {result['memories_per_second']:.0f} 条/秒")
    
    print("\n[2] 批量添加性能测试...")
    result = benchmark_batch_add(100)
    results["batch_add_100"] = result
    print(f"  批量添加 100 条记忆:")
    print(f"    总耗时: {result['total_time']*1000:.2f}ms")
    print(f"    吞吐量: {result['memories_per_second']:.0f} 条/秒")
    if result["speedup_vs_single"]:
        print(f"    相比单条添加加速: {result['speedup_vs_single']:.2f}x")
    
    print("\n[3] 检索性能测试...")
    system = HumanLikeMemorySystem()
    memories = generate_test_memories(500)
    system.add_memories_batch(memories)
    
    result = benchmark_retrieval(system, 100)
    results["retrieval"] = result
    print(f"  100 次检索:")
    print(f"    总耗时: {result['total_time']*1000:.2f}ms")
    print(f"    平均耗时: {result['avg_time']*1000:.3f}ms")
    print(f"    吞吐量: {result['queries_per_second']:.0f} 次/秒")
    
    print("\n[4] 权重计算性能测试（含缓存）...")
    result = benchmark_weight_calculation(100)
    results["weight_calc"] = result
    print(f"  100 条记忆，3次完整计算:")
    print(f"    首次计算耗时: {result['first_calculation_time']*1000:.2f}ms")
    print(f"    缓存后3次计算耗时: {result['cached_calculations_time']*1000:.2f}ms")
    print(f"    缓存命中率: {result['cache_hit_rate']*100:.1f}%")
    print(f"    加速因子: {result['speedup_factor']:.2f}x")
    
    print("\n[5] 索引性能对比...")
    result = benchmark_index_vs_fullscan(200)
    results["index_perf"] = result
    print(f"  200 条记忆的检索统计:")
    print(f"    索命查询次数: {result['index_lookups']}")
    print(f"    全表扫描次数: {result['full_scans']}")
    print(f"    缓存命中率: {result['cache_hit_rate']*100:.1f}%")
    
    print("\n[6] 内存使用测试...")
    for count in [100, 500, 1000]:
        result = benchmark_memory_usage(count)
        results[f"memory_{count}"] = result
        print(f"  {count} 条记忆:")
        print(f"    当前内存: {result['current_mb']:.2f}MB")
        print(f"    峰值内存: {result['peak_mb']:.2f}MB")
        print(f"    每条记忆: {result['memory_per_memory_kb']:.2f}KB")
    
    print("\n" + "=" * 70)
    print("基准测试完成")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_all_benchmarks()
