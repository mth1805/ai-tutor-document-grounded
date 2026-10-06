"""Helpers for fixture-grounded Answer Evaluation records and citations."""
from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path

from benchmarks.fixture_mapping import FIXTURE_ALIAS_TO_INDEX

ROOT = Path(__file__).resolve().parent


def load_answer_dataset(path: Path | None = None) -> list[dict]:
    source = path or ROOT / "answer_eval_dataset.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    original = {row["id"]: row for row in
                (json.loads(line) for line in (ROOT / "eval_dataset.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    if len(rows) != 24 or len({row["eval_id"] for row in rows}) != len(rows):
        raise ValueError("Answer evaluation dataset must contain 24 unique local eval IDs.")
    for row in rows:
        source_row = original.get(row["eval_id"])
        if (source_row is None or source_row["should_use_web"] or not source_row["gold_evidence_ids"]
                or row["question"] != source_row["question"]
                or row["gold_evidence_ids"] != source_row["gold_evidence_ids"]
                or row["reference_source_aliases"] != source_row["gold_evidence_ids"]
                or not row["reference_answer"].strip()):
            raise ValueError(f"Answer dataset row {row.get('eval_id')} does not match a labeled local source query.")
        unknown = set(row["gold_evidence_ids"]) - FIXTURE_ALIAS_TO_INDEX.keys()
        if unknown:
            raise ValueError(f"Unknown benchmark fixture aliases in {row['eval_id']}: {sorted(unknown)}")
    return rows


def evaluate_answer_citations(citations: list[dict], evidence: list[dict], gold_aliases: list[str]) -> dict:
    """Score emitted document citations by fixture chunk identity and page provenance."""
    source_by_chunk = {str(item["chunk_id"]): item for item in evidence}
    seen: set[tuple[str, int, int]] = set()
    cited_aliases: set[str] = set()
    correct = 0
    malformed = 0
    for citation in citations:
        try:
            chunk_id = str(citation["chunk_id"])
            source = source_by_chunk[chunk_id]
            start, end = int(citation["page_start"]), int(citation["page_end"])
            if start < 1 or end < start or start < int(source["page_start"]) or end > int(source["page_end"]):
                raise ValueError("citation page range is outside source provenance")
            alias = source["alias"]
            key = (chunk_id, start, end)
            if key in seen:
                continue
            seen.add(key)
            is_correct = alias in gold_aliases and str(citation["document_id"]) == str(source["document_id"])
            correct += int(is_correct)
            if is_correct:
                cited_aliases.add(alias)
        except (KeyError, TypeError, ValueError):
            malformed += 1
    return {
        "citation_count": len(seen), "correct_count": correct, "malformed_count": malformed,
        "citation_correctness": correct / len(seen) if seen else None,
        "citation_completeness": len(cited_aliases & set(gold_aliases)) / len(set(gold_aliases)) if gold_aliases else None,
        "covered_gold_aliases": sorted(cited_aliases),
    }


def validate_answer_record(record: dict) -> dict:
    required = {"eval_id", "question", "query_type", "retrieved_evidence", "gold_evidence_ids",
                "final_generated_answer", "document_citations", "citation_metadata",
                "evidence_gate", "ttft_ms", "generation_latency_ms", "total_answer_latency_ms",
                "model_configuration"}
    missing = required - record.keys()
    if missing:
        raise ValueError("Answer record missing fields: " + ", ".join(sorted(missing)))
    if not isinstance(record["retrieved_evidence"], list) or not isinstance(record["document_citations"], list):
        raise ValueError("retrieved_evidence and document_citations must be lists")
    return record


def load_captured_answers(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("Captured answer results must be a JSON array of objects")
    records = [validate_answer_record(record) for record in records]
    if len({row["eval_id"] for row in records}) != len(records):
        raise ValueError("Captured answer results must contain one record per unique eval_id")
    return records


def answer_report(records: list[dict], ragas_results: dict | None = None,
                  configured_model: str | None = None) -> tuple[dict, str]:
    try:
        installed_ragas_version = metadata.version("ragas")
    except metadata.PackageNotFoundError:
        installed_ragas_version = None
    def average(key: str):
        values = [row["citation_metrics"][key] for row in records
                  if isinstance(row.get("citation_metrics", {}).get(key), (int, float))]
        return sum(values) / len(values) if values else None
    timings = {}
    for key in ("ttft_ms", "generation_latency_ms", "failure_latency_ms", "total_answer_latency_ms"):
        values = []
        for row in records:
            if key == "generation_latency_ms" and row.get("generation_status") not in (None, "generated", "success"):
                continue
            value = row.get(key)
            # Earlier captures stored failed-call elapsed time under generation_latency_ms.
            if key == "failure_latency_ms" and value is None and row.get("generation_status") == "generation_failed":
                value = row.get("generation_latency_ms")
            if isinstance(value, (int, float)):
                values.append(value)
        timings[key] = {"count": len(values), "mean": sum(values) / len(values) if values else None}
    metrics = (ragas_results or {}).get("metrics", {})
    report = {"metadata": {"dataset_size": 24, "captured_run_count": len(records),
                           "gemini_model": records[0]["model_configuration"].get("model") if records else None,
                           "configured_gemini_model": configured_model,
                           "ragas_version": (ragas_results or {}).get("ragas_version") or installed_ragas_version,
                           "web_search_enabled": False},
              "citation_metrics": {"citation_correctness_mean": average("citation_correctness"),
                                   "citation_completeness_mean": average("citation_completeness")},
              "latency_ms": timings,
              "ragas_metrics": {name: metrics.get(name, {}).get("mean") for name in
                                ("faithfulness", "answer_relevance", "context_precision", "context_recall")},
              "limitations": ([] if ragas_results else ["Ragas scoring was not run; metrics are n/a."])}
    if ragas_results and not ragas_results.get("sample_count"):
        report["limitations"].append("Ragas had no successfully generated answer records to score; metrics are n/a.")
    if not records:
        report["limitations"].append("No answer runs were captured; answer, citation, and latency metrics are n/a.")
    lines = ["# Answer Evaluation Report", "", f"Dataset queries: 24  ",
             f"Captured runs: {len(records)}  ", f"Gemini model: {report['metadata']['gemini_model'] or 'n/a'}  ",
             f"Configured Gemini model: {configured_model or 'n/a'}  ",
             f"Ragas version: {report['metadata']['ragas_version'] or 'n/a'}", "",
             "## Deterministic citation evaluation", "",
             f"Citation correctness: {fmt(report['citation_metrics']['citation_correctness_mean'])}  ",
             f"Citation completeness: {fmt(report['citation_metrics']['citation_completeness_mean'])}", "",
             "## Latency (milliseconds)", "",
             "| Metric | Count | Mean |", "|---|---:|---:|"]
    for name, stat in timings.items():
        lines.append(f"| {name} | {stat['count']} | {fmt(stat['mean'])} |")
    lines += ["", "## Ragas answer quality", "", "| Metric | Mean |", "|---|---:|"]
    for name in ("faithfulness", "answer_relevance", "context_precision", "context_recall"):
        lines.append(f"| {name} | {fmt(report['ragas_metrics'].get(name))} |")
    lines += ["", "## Limitations", ""]
    lines += [f"- {item}" for item in report["limitations"]] or ["- Web search was disabled; only local fixture evidence was used."]
    return report, "\n".join(lines) + "\n"


def fmt(value):
    return "n/a" if value is None else f"{value:.3f}"
