# Memory System Benchmark Report
Generated: 2026-06-19 13:06:14

## Retrieval Benchmark

### Retrieval Quality
- Recall@1: 0.2500
- Recall@3: 0.3500
- Recall@5: 0.5500
- Precision@1: 0.3000
- Precision@3: 0.1333
- MRR: 0.3950
- NDCG@5: 0.4062

### Latency
- Mean: 1.17ms
- P50: 0.67ms
- P95: 7.36ms
- P99: 7.36ms

## Write Benchmark

### Write Performance
- Total memories: 50
- Throughput: 5607.51 memories/sec

### Latency
- Mean: 0.18ms
- P50: 0.15ms
- P95: 0.35ms
- P99: 1.00ms

## Detailed Results

```json
[
  {
    "type": "retrieval",
    "metrics": {
      "total_queries": 10,
      "mean_recall@1": 0.25,
      "mean_recall@3": 0.35,
      "mean_recall@5": 0.55,
      "mean_precision@1": 0.3,
      "mean_precision@3": 0.13333333333333333,
      "mean_mrr": 0.395,
      "mean_ndcg@5": 0.4061606311644851,
      "latency_mean": 0.0011695623397827148,
      "latency_p50": 0.0006666183471679688,
      "latency_p95": 0.0073626041412353516,
      "latency_p99": 0.0073626041412353516
    },
    "details": [
      {
        "query": "中午吃了什么",
        "recall@1": 0.0,
        "recall@3": 0.0,
        "recall@5": 0.0,
        "precision@1": 0.0,
        "precision@3": 0.0,
        "mrr": 0.0,
        "ndcg@5": 0.0,
        "latency": 0.0073626041412353516,
        "num_results": 0
      },
      {
        "query": "出差去了哪里",
        "recall@1": 0.0,
        "recall@3": 0.0,
        "recall@5": 0.0,
        "precision@1": 0.0,
        "precision@3": 0.0,
        "mrr": 0.0,
        "ndcg@5": 0.0,
        "latency": 0.0004189014434814453,
        "num_results": 0
      },
      {
        "query": "和张三一起做了什么",
        "recall@1": 1.0,
        "recall@3": 1.0,
        "recall@5": 1.0,
        "precision@1": 1.0,
        "precision@3": 0.3333333333333333,
        "mrr": 1.0,
        "ndcg@5": 1.0,
        "latency": 0.00019598007202148438,
        "num_results": 1
      },
      {
        "query": "北京的天气",
        "recall@1": 1.0,
        "recall@3": 1.0,
        "recall@5": 1.0,
        "precision@1": 1.0,
        "precision@3": 0.3333333333333333,
        "mrr": 1.0,
        "ndcg@5": 1.0,
        "latency": 0.00012803077697753906,
        "num_results": 1
      },
      {
        "query": "项目进展",
        "recall@1": 0.0,
        "recall@3": 0.0,
        "recall@5": 0.0,
        "precision@1": 0.0,
        "precision@3": 0.0,
        "mrr": 0.0,
        "ndcg@5": 0.0,
        "latency": 0.00030541419982910156,
        "num_results": 0
      },
      {
        "query": "看电影",
        "recall@1": 0.5,
        "recall@3": 0.5,
        "recall@5": 0.5,
        "precision@1": 1.0,
        "precision@3": 0.3333333333333333,
        "mrr": 1.0,
        "ndcg@5": 0.6131471927654584,
        "latency": 0.0007183551788330078,
        "num_results": 10
      },
      {
        "query": "学习新知识",
        "recall@1": 0.0,
        "recall@3": 0.0,
        "recall@5": 1.0,
        "precision@1": 0.0,
        "precision@3": 0.0,
        "mrr": 0.2,
        "ndcg@5": 0.38685280723454163,
        "latency": 0.00081634521484375,
        "num_results": 10
      },
      {
        "query": "运动锻炼",
        "recall@1": 0.0,
        "recall@3": 0.0,
        "recall@5": 1.0,
        "precision@1": 0.0,
        "precision@3": 0.0,
        "mrr": 0.25,
        "ndcg@5": 0.43067655807339306,
        "latency": 0.0006897449493408203,
        "num_results": 10
      },
      {
        "query": "和客户讨论",
        "recall@1": 0.0,
        "recall@3": 1.0,
        "recall@5": 1.0,
        "precision@1": 0.0,
        "precision@3": 0.3333333333333333,
        "mrr": 0.5,
        "ndcg@5": 0.6309297535714575,
        "latency": 0.0006666183471679688,
        "num_results": 10
      },
      {
        "query": "晚餐吃了什么",
        "recall@1": 0.0,
        "recall@3": 0.0,
        "recall@5": 0.0,
        "precision@1": 0.0,
        "precision@3": 0.0,
        "mrr": 0.0,
        "ndcg@5": 0.0,
        "latency": 0.0003936290740966797,
        "num_results": 0
      }
    ]
  },
  {
    "type": "write",
    "num_memories": 50,
    "latency_mean": 0.00017833232879638672,
    "latency_p50": 0.00014543533325195312,
    "latency_p95": 0.0003464221954345703,
    "latency_p99": 0.001003265380859375,
    "throughput": 5607.5082221449775
  }
]
```