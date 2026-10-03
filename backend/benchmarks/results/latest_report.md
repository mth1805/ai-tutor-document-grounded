# RAG Evaluation Report

Dataset version: 1.0  
Queries: 30  
Captured runs: 0  
Labeled queries: 0

## Retrieval baselines

| Configuration | Labeled queries | Recall@1 | Recall@5 | Recall@10 | Precision@5 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|
| Dense only | 0 | n/a | n/a | n/a | n/a | n/a |
| Lexical only | 0 | n/a | n/a | n/a | n/a | n/a |
| Hybrid + RRF | 0 | n/a | n/a | n/a | n/a | n/a |
| Hybrid + RRF + Cross-Encoder | 0 | n/a | n/a | n/a | n/a | n/a |
| Hybrid + RRF + Cross-Encoder + Evidence Gate | 0 | n/a | n/a | n/a | n/a | n/a |
| Full system + Web fallback | 0 | n/a | n/a | n/a | n/a | n/a |

## Evidence and web routing

Evidence gate: `{"false_insufficient_rate": null, "false_negatives": 0, "false_positives": 0, "false_sufficient_rate": null, "sufficient_precision": null, "sufficient_recall": null, "true_negatives": 0, "true_positives": 0}`  
Web fallback: `{"correct_no_web_decisions": 0, "correct_web_triggers": 0, "false_web_triggers": 0, "no_web_count": 0, "routing_accuracy": null, "web_positive_count": 0}`

## Latency (milliseconds)

| Stage | Count | Mean | Median | p95 |
|---|---:|---:|---:|---:|
| query_embedding_ms | 0 | n/a | n/a | n/a |
| dense_retrieval_ms | 0 | n/a | n/a | n/a |
| lexical_retrieval_ms | 0 | n/a | n/a | n/a |
| rrf_ms | 0 | n/a | n/a | n/a |
| rerank_ms | 0 | n/a | n/a | n/a |
| total_retrieval_ms | 0 | n/a | n/a | n/a |
| web_search_ms | 0 | n/a | n/a | n/a |
| ttft_ms | 0 | n/a | n/a | n/a |
| generation_ms | 0 | n/a | n/a | n/a |

## End-to-end latency by configuration (milliseconds)

| Configuration | Count | Mean | Median | p95 |
|---|---:|---:|---:|---:|

## Limitations

- No captured run records supplied; retrieval, gate, citation, and latency metrics are unavailable.
