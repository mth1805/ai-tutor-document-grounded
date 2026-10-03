"""Deterministic metrics for captured retrieval and latency records."""
from __future__ import annotations

from statistics import mean, median
from typing import Iterable, Mapping, Sequence


def retrieval_metrics(ranked_ids: Sequence[str], gold_ids: Iterable[str]) -> dict[str, float | None]:
    """Compute hit Recall@K, fixed-denominator Precision@5 and MRR@10.

    Recall@K here is per-query hit rate: 1 if any gold ID is in the first K,
    otherwise 0. Queries without gold IDs are excluded (None).
    """
    gold = set(gold_ids)
    if not gold:
        return {"recall@1": None, "recall@5": None, "recall@10": None,
                "precision@5": None, "mrr@10": None}
    ranked = list(ranked_ids)
    metrics: dict[str, float | None] = {
        f"recall@{k}": float(bool(set(ranked[:k]) & gold)) for k in (1, 5, 10)
    }
    metrics["precision@5"] = len(set(ranked[:5]) & gold) / 5
    first_rank = next((i for i, item in enumerate(ranked[:10], 1) if item in gold), None)
    metrics["mrr@10"] = 1 / first_rank if first_rank is not None else 0.0
    return metrics


def aggregate_retrieval(rows: Sequence[Mapping]) -> dict[str, float | int | None]:
    keys = ("recall@1", "recall@5", "recall@10", "precision@5", "mrr@10")
    out: dict[str, float | int | None] = {"query_count": len(rows)}
    for key in keys:
        values = [row[key] for row in rows if row.get(key) is not None]
        out[key] = mean(values) if values else None
    out["labeled_query_count"] = sum(row.get("recall@1") is not None for row in rows)
    return out


def latency_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    samples = sorted(float(value) for value in values)
    if not samples:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    # Nearest-rank percentile: rank=ceil(0.95*n), converted to zero-based index.
    rank = max(1, int(0.95 * len(samples) + 0.999999))
    return {"count": len(samples), "mean": mean(samples), "median": median(samples),
            "p95": samples[rank - 1]}


def compare_reranking(before: Sequence[str], after: Sequence[str], gold_ids: Iterable[str],
                      rerank_latency_ms: float | None = None) -> dict:
    """Return before/after retrieval measures with the observed rerank latency."""
    return {"before": retrieval_metrics(before, gold_ids),
            "after": retrieval_metrics(after, gold_ids),
            "rerank_latency_ms": rerank_latency_ms}
