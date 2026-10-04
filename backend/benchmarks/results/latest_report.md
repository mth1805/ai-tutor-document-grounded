# RAG Evaluation Report

Dataset version: 1.0  
Queries: 30  
Captured runs: 30  
Labeled queries: 24  
Executed retrieval configuration: `{"candidate_pool_size": 20, "dense_top_k": 20, "embedding_model": "BAAI/bge-m3", "lexical_top_k": 20, "min_answerable_rerank_score": 0.55, "relevance_threshold": 0.35, "rerank_top_k": 10, "reranker_model": "BAAI/bge-reranker-v2-m3", "routing_mode": "always_quality", "rrf_k": 60}`

Configuration metadata reconciliation: the prior report configuration differed from the captured runs. Metrics below remain calculated from the historical capture records; configuration is now sourced from those records.

## Retrieval baselines

| Configuration | Labeled queries | Recall@1 | Recall@5 | Recall@10 | Precision@5 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|
| Dense only | 24 | 0.958 | 1.000 | 1.000 | 0.267 | 0.972 |
| Lexical only | 24 | 1.000 | 1.000 | 1.000 | 0.283 | 1.000 |
| Hybrid + RRF | 24 | 1.000 | 1.000 | 1.000 | 0.283 | 1.000 |
| Hybrid + RRF + Cross-Encoder | 24 | 0.958 | 1.000 | 1.000 | 0.283 | 0.979 |
| Hybrid + RRF + Cross-Encoder + Evidence Gate | 24 | 0.958 | 1.000 | 1.000 | 0.283 | 0.979 |
| Full system + Web fallback | 24 | 0.958 | 1.000 | 1.000 | 0.283 | 0.979 |

## Evidence and web routing

Evidence gate: `{"false_insufficient_rate": 0.125, "false_negatives": 3, "false_positives": 0, "false_sufficient_rate": 0.0, "sufficient_precision": 1.0, "sufficient_recall": 0.875, "true_negatives": 6, "true_positives": 21}`  
Web fallback: `{"correct_no_web_decisions": 21, "correct_web_triggers": 6, "false_web_triggers": 3, "no_web_count": 24, "routing_accuracy": 0.9, "web_positive_count": 6}`

## Latency (milliseconds)

| Stage | Count | Mean | Median | p95 |
|---|---:|---:|---:|---:|
| query_embedding_ms | 30 | 1299.53 | 730.17 | 3517.05 |
| dense_retrieval_ms | 30 | 13.99 | 9.87 | 32.82 |
| lexical_retrieval_ms | 30 | 1.26 | 1.08 | 2.76 |
| rrf_ms | 30 | 0.92 | 0.35 | 3.17 |
| rerank_ms | 30 | 11461.22 | 10829.22 | 17274.23 |
| total_retrieval_ms | 30 | 12791.94 | 12031.01 | 19842.21 |
| web_search_ms | 30 | 0.75 | 0.00 | 5.95 |
| ttft_ms | 0 | n/a | n/a | n/a |
| generation_ms | 0 | n/a | n/a | n/a |

## End-to-end latency by configuration (milliseconds)

| Configuration | Count | Mean | Median | p95 |
|---|---:|---:|---:|---:|
| dense_only | 30 | 2364.60 | 2442.07 | 5301.91 |
| lexical_only | 30 | 9.34 | 6.74 | 12.29 |
| hybrid_rrf | 30 | 2374.48 | 2450.39 | 5309.21 |
| reranked | 30 | 12791.94 | 12031.01 | 19842.21 |
| evidence_gate | 30 | 12791.94 | 12031.01 | 19842.21 |
| full_web_fallback | 30 | 12792.69 | 12031.01 | 19843.35 |

## Limitations

- Answer generation and citation correctness are not evaluated by the offline capture.
