# Offline RAG evaluation

`eval_dataset.jsonl` contains 30 reviewable queries grouped as factual (15), comparison (5), multi-hop (4), insufficient-document (2), and web-fallback (4). The evidence IDs refer to topic passages in the existing `benchmark_retrieval.py` fixture, not to production database UUIDs: `artificial_intelligence`, `machine_learning`, `deep_learning`, `transformer`, `photosynthesis`, `chlorophyll`, `cellular_respiration`, `newton_laws`, `relational_database`, and `hnsw`. The fixture's pages are all page 1. Replace the fixture aliases and gold page/document labels with reviewed IDs from a local evaluation corpus before using quality metrics for a release decision. Queries without gold evidence are intentionally unscored for retrieval metrics.

## Captured run contract

Run retrieval against a local, seeded corpus with production embedding/reranker providers as desired, then save one JSONL record per dataset `id` and pass it with `--records`. The report command itself never connects to PostgreSQL, Gemini, or Tavily. A record can contain:

```json
{"eval_id":"eval-001","rankings":{"dense_only":["fixture:machine_learning"],"lexical_only":[],"hybrid_rrf":[],"reranked":[]},"evidence_sufficient":true,"used_web_fallback":false,"citations":[],"timings":{"query_embedding_ms":4.2,"dense_retrieval_ms":8.1,"lexical_retrieval_ms":3.0,"rrf_ms":0.2,"rerank_ms":12.0,"total_retrieval_ms":30.0}}
```

`Recall@K` is a per-query hit rate (any gold ID in the first K); `MRR@10` is the reciprocal rank of the first gold ID, or zero; `Precision@5` is relevant unique IDs in top five divided by five. Queries with no gold IDs are excluded. The baseline table only scores supplied rankings; absent captures are `n/a`, not zero. Latency reports aggregate supplied values using mean, median, and nearest-rank p95. Local stages and external web/LLM timings are distinct fields.

## Commands

From `backend/`:

```powershell
python -m benchmarks.capture_rag_eval
python -m benchmarks.run_rag_eval
python -m benchmarks.run_rag_eval --records path/to/captured_runs.jsonl
```

Capture uses the BGE-M3 and Cross-Encoder providers from the existing local benchmark, the fixture chunks in memory, and `MockWebSearchProvider`. It requires local model weights but makes no Gemini/Tavily/Ragas calls. Captures are written to `benchmarks/results/captured_runs.jsonl`; each run contains ranking aliases, UUIDs from that execution, chunk ranks, scores, gate flags, mock web-call status, and timings. Aliases are explicitly mapped to fixture chunk indices and are never treated as production UUIDs. If any gold alias is not in that mapping, its run is marked `unmapped_alias` and excluded from retrieval metrics.

The report command defaults to that capture file and writes `benchmarks/results/latest_results.json` and `latest_report.md`. Dataset, thresholds, model configuration, runtime, and timestamp are recorded. No credentials are written. Queries with no fixture evidence remain unlabeled for retrieval; they still have routing labels.

## Optional Ragas evaluation

Ragas is not a normal dependency and the ordinary benchmark/tests do not call it. Install the isolated profile with `pip install -r benchmarks/requirements-ragas.txt`, set `OPENAI_API_KEY`, then run:

```powershell
python -m benchmarks.run_ragas_eval path/to/prepared_answers.jsonl
```

Prepared answer JSONL records need `question`, `answer`, `contexts` (string list), and `reference`. This optional run uses Ragas v0.4's metrics collections (`ascore` API) for Faithfulness, Answer Relevance, Context Precision, and Context Recall. It makes external evaluator calls and is explicitly an LLM-as-a-judge result; it does not generate answers. No LangChain dependency is used.

The default evaluation cannot establish current production quality without captured rankings, gate decisions, citation payloads, and stage timings from a real local corpus. Mock/provider availability is not treated as a routing-quality result.
