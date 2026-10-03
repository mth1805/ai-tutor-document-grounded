"""Aggregate offline evaluation metrics from captured pipeline run records.

This command deliberately does not query production retrieval, Gemini, or Tavily.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.citation_eval import evaluate_citations
from benchmarks.evidence_gate_eval import confusion_matrix, web_routing_metrics
from benchmarks.metrics import aggregate_retrieval, latency_summary, retrieval_metrics

ROOT = Path(__file__).resolve().parent
CONFIGS = {
    "Dense only": "dense_only",
    "Lexical only": "lexical_only",
    "Hybrid + RRF": "hybrid_rrf",
    "Hybrid + RRF + Cross-Encoder": "reranked",
    "Hybrid + RRF + Cross-Encoder + Evidence Gate": "reranked",
    "Full system + Web fallback": "reranked",
}
TIMING_KEYS = ("query_embedding_ms", "dense_retrieval_ms", "lexical_retrieval_ms",
               "rrf_ms", "rerank_ms", "total_retrieval_ms", "web_search_ms",
               "ttft_ms", "generation_ms")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def build_report(dataset: list[dict], records: dict[str, dict], config: dict) -> dict:
    baselines = {}
    for display, key in CONFIGS.items():
        scored = []
        for item in dataset:
            run = records.get(item["id"], {})
            ranking = run.get("rankings", {}).get(key)
            if ranking is not None and run.get("gold_label_status") != "unmapped_alias":
                scored.append(retrieval_metrics(ranking, item.get("gold_evidence_ids", [])))
        baselines[display] = aggregate_retrieval(scored)

    paired = []
    gate_rows = []
    citation_results = []
    timings = {key: [] for key in TIMING_KEYS}
    for item in dataset:
        run = records.get(item["id"], {})
        rankings = run.get("rankings", {})
        if "hybrid_rrf" in rankings and "reranked" in rankings:
            paired.append({"before": retrieval_metrics(rankings["hybrid_rrf"], item.get("gold_evidence_ids", [])),
                           "after": retrieval_metrics(rankings["reranked"], item.get("gold_evidence_ids", [])),
                           "rerank_latency_ms": run.get("timings", {}).get("rerank_ms")})
        if "evidence_sufficient" in run:
            gate_rows.append({"gold_sufficient": not item["should_use_web"],
                              "predicted_sufficient": bool(run["evidence_sufficient"])})
        if "used_web_fallback" in run:
            gate_rows.append({"should_use_web": item["should_use_web"],
                              "used_web_fallback": bool(run["used_web_fallback"])})
        if "citations" in run:
            citation_results.append(evaluate_citations(
                run["citations"], item.get("gold_documents", []),
                item.get("gold_web_urls", []), item.get("expected_fact_ids", [])))
        for key in TIMING_KEYS:
            value = run.get("timings", {}).get(key)
            if isinstance(value, (int, float)):
                timings[key].append(float(value))

    gate_only = [row for row in gate_rows if "gold_sufficient" in row]
    web_only = [row for row in gate_rows if "should_use_web" in row]
    before_metrics = aggregate_retrieval([row["before"] for row in paired])
    after_metrics = aggregate_retrieval([row["after"] for row in paired])
    gate_metrics = confusion_matrix(gate_only, gold_key="gold_sufficient", prediction_key="predicted_sufficient")
    web_metrics = web_routing_metrics(web_only)
    baselines["Hybrid + RRF + Cross-Encoder + Evidence Gate"]["evidence_gate"] = gate_metrics
    baselines["Full system + Web fallback"]["web_fallback"] = web_metrics
    citation_correctness = [row["citation_correctness"] for row in citation_results
                            if row["citation_correctness"] is not None]
    citation_completeness = [row["citation_completeness"] for row in citation_results
                             if row["citation_completeness"] is not None]
    configuration_latencies: dict[str, list[float]] = {}
    for run in records.values():
        for key, value in run.get("configuration_timings_ms", {}).items():
            configuration_latencies.setdefault(key, []).append(float(value))
    return {
        "metadata": {"dataset_version": "1.0", "query_count": len(dataset),
                     "query_categories": dict(Counter(row.get("query_type", "unknown") for row in dataset)),
                     "captured_run_count": len(records), "generated_at": datetime.now(timezone.utc).isoformat(),
                     "labeled_query_count": sum(run.get("gold_label_status") == "mapped" for run in records.values()),
                     "routing_labeled_query_count": sum("should_use_web" in run for run in records.values()),
                     "python": sys.version.split()[0], "platform": platform.platform(),
                     "configuration": config},
        "baselines": baselines,
        "reranker_comparison": {"paired_query_count": len(paired), "before": before_metrics,
                                "after": after_metrics, "per_query": paired,
                                "rerank_latency": latency_summary(p["rerank_latency_ms"] for p in paired
                                                                   if p["rerank_latency_ms"] is not None)},
        "evidence_gate": gate_metrics,
        "web_fallback": web_metrics,
        "citations": {"evaluated_queries": len(citation_results),
                      "citation_correctness_mean": sum(citation_correctness) / len(citation_correctness) if citation_correctness else None,
                      "citation_completeness_mean": sum(citation_completeness) / len(citation_completeness) if citation_completeness else None,
                      "results": citation_results},
        "latency_ms": {key: latency_summary(values) for key, values in timings.items()},
        "latency_by_configuration_ms": {key: latency_summary(values) for key, values in configuration_latencies.items()},
        "unmapped_query_ids": [run.get("eval_id") for run in records.values()
                               if run.get("gold_label_status") == "unmapped_alias"],
        "limitations": (["No captured run records supplied; retrieval, gate, citation, and latency metrics are unavailable."]
                        if not records else ["Answer generation and citation correctness are not evaluated by the offline capture."]),
    }


def markdown_report(report: dict) -> str:
    lines = ["# RAG Evaluation Report", "", f"Dataset version: {report['metadata']['dataset_version']}  ",
             f"Queries: {report['metadata']['query_count']}  ",
             f"Captured runs: {report['metadata']['captured_run_count']}  ",
             f"Labeled queries: {report['metadata']['labeled_query_count']}", "",
             "## Retrieval baselines", "", "| Configuration | Labeled queries | Recall@1 | Recall@5 | Recall@10 | Precision@5 | MRR@10 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, values in report["baselines"].items():
        def fmt(value): return "n/a" if value is None else f"{value:.3f}"
        lines.append(f"| {name} | {values['labeled_query_count']} | {fmt(values['recall@1'])} | {fmt(values['recall@5'])} | {fmt(values['recall@10'])} | {fmt(values['precision@5'])} | {fmt(values['mrr@10'])} |")
    lines += ["", "## Evidence and web routing", "", f"Evidence gate: `{json.dumps(report['evidence_gate'], sort_keys=True)}`  ",
              f"Web fallback: `{json.dumps(report['web_fallback'], sort_keys=True)}`", "",
              "## Latency (milliseconds)", "", "| Stage | Count | Mean | Median | p95 |", "|---|---:|---:|---:|---:|"]
    for key, stats in report["latency_ms"].items():
        values = ["n/a" if stats[field] is None else f"{stats[field]:.2f}" for field in ("mean", "median", "p95")]
        lines.append(f"| {key} | {stats['count']} | " + " | ".join(values) + " |")
    lines += ["", "## End-to-end latency by configuration (milliseconds)", "",
              "| Configuration | Count | Mean | Median | p95 |", "|---|---:|---:|---:|---:|"]
    for key, stats in report["latency_by_configuration_ms"].items():
        values = ["n/a" if stats[field] is None else f"{stats[field]:.2f}" for field in ("mean", "median", "p95")]
        lines.append(f"| {key} | {stats['count']} | " + " | ".join(values) + " |")
    lines += ["", "## Limitations", ""]
    lines += [f"- {item}" for item in report["limitations"]] or ["- Scores describe only supplied captured runs; no live services were called."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=ROOT / "results" / "captured_runs.jsonl",
                        help="Captured run JSONL (default: benchmarks/results/captured_runs.jsonl).")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    dataset = read_jsonl(ROOT / "eval_dataset.jsonl")
    runs = read_jsonl(args.records) if args.records.is_file() else []
    by_id = {record["eval_id"]: record for record in runs}
    from app.core.config import settings
    config = {"retrieval": "dense + FTS + RRF + configured routing/reranker",
              "dense_top_k": settings.DENSE_TOP_K,
              "lexical_top_k": settings.LEXICAL_TOP_K,
              "rrf_k": settings.RRF_K,
              "candidate_pool_size": settings.CANDIDATE_POOL_SIZE,
              "rerank_top_k": settings.RERANK_TOP_K,
              "retrieval_routing_mode": settings.RETRIEVAL_ROUTING_MODE,
              "reranker_model": settings.RERANKER_MODEL_NAME,
              "relevance_threshold": settings.RELEVANCE_THRESHOLD,
              "min_answerable_rerank_score": settings.MIN_ANSWERABLE_RERANK_SCORE,
              "gemini_model": settings.GEMINI_MODEL}
    report = build_report(dataset, by_id, config)
    report["metadata"]["capture_file"] = str(args.records) if args.records.is_file() else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "latest_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "latest_report.md").write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report))


if __name__ == "__main__":
    main()
