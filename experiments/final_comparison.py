"""
最终对比测试：原始版本 vs 优化v2
"""

import sys
sys.path.insert(0, '/tmp/memory-prototype')
sys.path.insert(0, '/root/new-agent-memory/experiments')

import time
import random
import statistics
from memory_chunk import MemoryChunk
from memory_layer_core import MemoryLayerCore
from optimization3_v2 import OptimizedV2MemoryLayerCore


def generate_chunk(chunk_id: int) -> MemoryChunk:
    topics = ["food", "travel", "work", "life", "study", "entertainment", "sports", "tech"]
    locations = ["北京", "上海", "广州", "深圳", "杭州", "成都", "家里", "公司"]
    persons = ["我", "朋友A", "朋友B", "同事", "家人", "客户", "老师"]
    emotions = [-0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8]
    
    return MemoryChunk(
        id=f"mem_{chunk_id}",
        content=f"测试记忆 {chunk_id}",
        memory_type="test",
        time_absolute=f"202{random.randint(0,6)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        time_relative=random.choice(["最近", "上周", "上个月", "去年", "几年前"]),
        time_context=random.choice(["上午", "中午", "下午", "晚上"]),
        location=random.choice(locations),
        persons=set(random.sample(persons, k=random.randint(1, 3))),
        topics=set(random.sample(topics, k=random.randint(1, 3))),
        emotion_valence=random.choice(emotions),
        importance=random.uniform(0.2, 0.9),
    )


def benchmark_retrieval(core, queries: list, n_iterations: int = 100):
    latencies = []
    for _ in range(n_iterations):
        query = random.choice(queries)
        start = time.perf_counter()
        results = core.retrieve(query, min_weight=0.1, limit=10)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
    
    latencies.sort()
    return {
        "mean_ms": statistics.mean(latencies),
        "p50_ms": latencies[len(latencies)//2],
        "p95_ms": latencies[int(len(latencies) * 0.95)],
        "p99_ms": latencies[int(len(latencies) * 0.99)],
    }


def run_final_comparison():
    print("=" * 70)
    print("最终性能对比测试：原始版本 vs 优化v2")
    print("=" * 70)
    
    # 查询集（包含有索引和无索引的查询）
    queries = [
        {"topics": {"food"}},
        {"topics": {"travel"}},
        {"topics": {"work"}},
        {"location": "北京"},
        {"location": "上海"},
        {"persons": {"我"}},
        {"time_relative": "去年"},
        {"time_relative": "最近"},
        {"topics": {"work", "tech"}, "emotion_valence_min": 0.3},
        {"topics": {"food"}, "location": "北京"},
    ]
    
    scales = [500, 1000, 2000, 5000, 10000]
    results = []
    
    for scale in scales:
        print(f"\n{'='*60}")
        print(f"[规模: {scale:,} 条记忆]")
        print("-" * 60)
        
        # ========== 原始版本 ==========
        print("  初始化原始版本...")
        t0 = time.perf_counter()
        original = MemoryLayerCore()
        for i in range(scale):
            original.add(generate_chunk(i))
        
        # 建立关联
        chunk_ids = list(original.chunks.keys())
        for cid in chunk_ids:
            for _ in range(random.randint(3, 8)):
                target = random.choice(chunk_ids)
                if target != cid:
                    original.strengthen_association(cid, target)
        init_time_orig = time.perf_counter() - t0
        
        # ========== 优化版本 ==========
        print("  初始化优化版本...")
        t0 = time.perf_counter()
        optimized = OptimizedV2MemoryLayerCore(
            max_scan_candidates=min(200, scale),
            early_exit_k=20,
        )
        for i in range(scale):
            optimized.add(generate_chunk(i + scale))
        
        # 建立关联
        chunk_ids_opt = list(optimized.chunks.keys())
        for cid in chunk_ids_opt:
            for _ in range(random.randint(3, 8)):
                target = random.choice(chunk_ids_opt)
                if target != cid:
                    optimized.strengthen_association(cid, target)
        init_time_opt = time.perf_counter() - t0
        
        # ========== 检索测试 ==========
        print(f"  运行检索测试 ({scale * 10} 次查询)...")
        
        orig_stats = benchmark_retrieval(original, queries, n_iterations=100)
        opt_stats = benchmark_retrieval(optimized, queries, n_iterations=100)
        
        # 统计
        orig_assocs = sum(len(c.associations) for c in original.chunks.values())
        opt_stats_info = optimized.get_stats()
        
        print(f"\n  原始版本:")
        print(f"    初始化: {init_time_orig:.2f}s")
        print(f"    关联数: {orig_assocs:,}")
        print(f"    检索延迟: mean={orig_stats['mean_ms']:.2f}ms, p95={orig_stats['p95_ms']:.2f}ms, p99={orig_stats['p99_ms']:.2f}ms")
        
        print(f"\n  优化版本:")
        print(f"    初始化: {init_time_opt:.2f}s")
        print(f"    关联数: {sum(len(c.associations) for c in optimized.chunks.values()):,}")
        print(f"    缓存大小: {opt_stats_info['weight_cache_size']:,}")
        print(f"    检索延迟: mean={opt_stats['mean_ms']:.2f}ms, p95={opt_stats['p95_ms']:.2f}ms, p99={opt_stats['p99_ms']:.2f}ms")
        
        p95_improvement = (1 - opt_stats['p95_ms'] / max(orig_stats['p95_ms'], 0.001)) * 100
        mean_improvement = (1 - opt_stats['mean_ms'] / max(orig_stats['mean_ms'], 0.001)) * 100
        
        print(f"\n  性能提升:")
        print(f"    平均延迟: {mean_improvement:+.1f}%")
        print(f"    P95延迟:  {p95_improvement:+.1f}%")
        
        results.append({
            "scale": scale,
            "orig_p95": orig_stats['p95_ms'],
            "opt_p95": opt_stats['p95_ms'],
            "p95_improvement": p95_improvement,
            "orig_mean": orig_stats['mean_ms'],
            "opt_mean": opt_stats['mean_ms'],
            "mean_improvement": mean_improvement,
        })
    
    # 总结
    print("\n" + "=" * 70)
    print("最终总结")
    print("=" * 70)
    
    print(f"\n{'规模':>8} | {'原始p95':>12} | {'优化p95':>12} | {'提升':>10}")
    print("-" * 50)
    for r in results:
        sign = "+" if r['p95_improvement'] > 0 else ""
        print(f"{r['scale']:>8} | "
              f"{r['orig_p95']:>10.2f}ms | "
              f"{r['opt_p95']:>10.2f}ms | "
              f"{sign}{r['p95_improvement']:>8.1f}%")
    
    avg_improvement = sum(r['p95_improvement'] for r in results) / len(results)
    print(f"\n平均P95性能变化: {avg_improvement:+.1f}%")
    
    if avg_improvement > 10:
        print("\n✓ 优化版本有明显提升！")
    elif avg_improvement > 0:
        print("\n✓ 优化版本略有提升")
    else:
        print("\n⚠ 优化版本无提升，需要进一步分析")


if __name__ == "__main__":
    run_final_comparison()
