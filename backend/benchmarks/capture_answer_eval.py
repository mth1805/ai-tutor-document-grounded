"""Capture fixture-grounded answers using production retrieval, prompt and citation services.

This explicit benchmark command calls Gemini. It never calls web search.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

from benchmarks.answer_eval import (ROOT, answer_report, evaluate_answer_citations,
                                    load_answer_dataset, validate_answer_record)
from benchmarks.request_scheduler import (RequestScheduler, RetryPolicy,
                                          generate_stream_with_retries)

MIN_REQUEST_INTERVAL_SECONDS = 15.0
PRETTY_FIELD_ORDER = (
    "eval_id", "question", "query_type",
    "generation_status", "generation_error", "final_generated_answer",
    "retry_count", "retry_delays_ms", "transient_error_codes", "final_failure_classification",
    "evidence_gate", "retrieved_evidence",
    "document_citations", "citation_metadata", "citation_metrics",
    "ttft_ms", "generation_latency_ms", "retrieval_latency_ms", "total_answer_latency_ms",
    "web_search_enabled", "model_configuration", "retrieval_configuration",
)


def evidence_gate_is_sufficient(response) -> bool:
    """Mirror RAGService's two-tier gate, including FAST path handling."""
    from app.core.config import settings
    quality_path = response.routing_path == "QUALITY"
    top_score = max((chunk.rerank_score for chunk in response.results if chunk.rerank_score is not None), default=0.0)
    tier2_passes = not quality_path or top_score >= settings.MIN_ANSWERABLE_RERANK_SCORE
    return bool(response.has_sufficient_evidence and response.results and tier2_passes)


def safe_generation_diagnostic(exc: Exception, api_key: str | None = None) -> dict:
    """Extract provider diagnostics; keep only safe, explicit fields in captures."""
    details = getattr(exc, "details", {}) or {}
    fields = ("exception_class", "http_status", "provider_status", "provider_message",
              "classification", "retryable", "retry_delay_seconds", "quota_scope")
    diagnostic = {field: details[field] for field in fields if field in details}
    if not diagnostic:
        cause = exc.__cause__ or exc
        diagnostic = {"exception_class": type(cause).__name__,
                      "http_status": getattr(cause, "code", None),
                      "provider_status": getattr(cause, "status", None),
                      "provider_message": getattr(cause, "message", None) or str(cause),
                      "classification": "unknown", "retryable": False}
    if api_key and diagnostic.get("provider_message"):
        diagnostic["provider_message"] = str(diagnostic["provider_message"]).replace(api_key, "[REDACTED]")
    return diagnostic


async def capture_answer_dataset(dataset: list[dict], embedding_provider, reranker_provider,
                                 llm_provider, limit: int | None = None,
                                 retry_policy: RetryPolicy | None = None,
                                 request_scheduler: RequestScheduler | None = None) -> list[dict]:
    """Run the existing retrieval flow and production prompt/citation components on local fixture data."""
    from benchmarks.capture_rag_eval import build_fixture
    from app.core.config import settings
    from app.rag.citation_service import CitationService
    from app.rag.prompt_builder import ChatMode, PromptBuilder
    from app.schemas.retrieval import RetrievalRequest
    from app.services.retrieval_service import RetrievalService
    from app.services.ingestion_service import _IN_MEMORY_CHUNKS
    dataset = dataset[:limit] if limit else dataset
    if len({item["eval_id"] for item in dataset}) != len(dataset):
        raise ValueError("Answer evaluation dataset contains duplicate eval_id values.")
    retry_policy = retry_policy or RetryPolicy(
        min_request_interval_seconds=MIN_REQUEST_INTERVAL_SECONDS,
    )
    request_scheduler = request_scheduler or RequestScheduler(
        retry_policy.min_request_interval_seconds,
    )
    workspace_id, user_id = uuid.uuid4(), uuid.uuid4()
    document_id, chunks_by_alias = build_fixture(workspace_id, user_id, embedding_provider)
    alias_by_chunk_id = {str(chunk.id): alias for alias, chunk in chunks_by_alias.items()}
    records: list[dict] = []
    try:
        for item in dataset:
            total_started = time.perf_counter()
            retrieval = await RetrievalService.retrieve(
                db=None, workspace_id=workspace_id, user_id=user_id,
                request=RetrievalRequest(query=item["question"]),
                embedding_provider=embedding_provider, reranker_provider=reranker_provider,
            )
            gate_sufficient = evidence_gate_is_sufficient(retrieval)
            evidence_chunks = [chunk for chunk in retrieval.results if chunk.passed_relevance_gate]
            if not evidence_chunks:
                evidence_chunks = retrieval.results[:settings.RERANK_TOP_K]
            evidence = [{
                "alias": alias_by_chunk_id[str(chunk.chunk_id)], "chunk_id": str(chunk.chunk_id),
                "document_id": str(chunk.document_id), "content": chunk.content,
                "page_start": chunk.page_number_start, "page_end": chunk.page_number_end,
                "rank": chunk.final_rank, "rerank_score": chunk.rerank_score,
                "passed_relevance_gate": chunk.passed_relevance_gate,
                "retrieval_sources": chunk.retrieval_sources,
            } for chunk in retrieval.results]
            retrieved_for_prompt = [chunk for chunk in evidence_chunks]
            assembled = PromptBuilder.assemble(
                query=item["question"], evidence_chunks=retrieved_for_prompt,
                document_names={document_id: "Benchmark fixture"},
                chat_mode=ChatMode.DETAILED_GUIDANCE, conversation_history=[],
            )

            status = "skipped_insufficient_evidence"
            generation = {"success": False, "answer": "", "ttft_ms": None,
                          "generation_latency_ms": None, "failure_latency_ms": None,
                          "partial_generation_text": None, "generation_error": None,
                          "retry_count": 0, "retry_delays_ms": [],
                          "transient_error_codes": [], "final_failure_classification": None,
                          "attempt_count": 0}
            if gate_sufficient:
                generation = await generate_stream_with_retries(
                    llm_provider, assembled.prompt, assembled.system_instruction,
                    request_scheduler, retry_policy, getattr(llm_provider, "api_key", None),
                )
                status = "success" if generation["success"] else "generation_failed"
            raw_answer = generation["answer"] or generation["partial_generation_text"] or ""
            citations, normalized_answer = CitationService.extract_and_validate_citations(
                text=raw_answer, sources=assembled.sources,
            ) if raw_answer and status == "success" else ([], "")
            citation_payload = [citation.to_dict() for citation in citations]
            citation_metrics = evaluate_answer_citations(citation_payload, evidence, item["gold_evidence_ids"])
            if status != "success":
                citation_metrics = {"citation_count": 0, "correct_count": 0, "malformed_count": 0,
                                    "citation_correctness": None, "citation_completeness": None,
                                    "covered_gold_aliases": []}
            alias_metadata = {str(source.chunk_id): alias_by_chunk_id[str(source.chunk_id)]
                              for source in assembled.sources}
            record = {
                "eval_id": item["eval_id"], "question": item["question"],
                "query_type": item["query_type"], "retrieved_evidence": evidence,
                "gold_evidence_ids": item["gold_evidence_ids"],
                "reference_answer": item["reference_answer"],
                "ragas_contexts": [chunk.content for chunk in retrieved_for_prompt],
                "final_generated_answer": normalized_answer if status == "success" else "",
                "partial_generation_text": generation["partial_generation_text"],
                "document_citations": citation_payload,
                "citation_metadata": {"chunk_id_to_fixture_alias": alias_metadata,
                                      "source_count": len(assembled.sources)},
                "citation_metrics": citation_metrics,
                "evidence_gate": {"has_sufficient_evidence": retrieval.has_sufficient_evidence,
                                  "passed": gate_sufficient, "routing_path": retrieval.routing_path,
                                  "relevance_threshold": retrieval.relevance_threshold,
                                  "minimum_answerable_rerank_score": settings.MIN_ANSWERABLE_RERANK_SCORE},
                "ttft_ms": round(generation["ttft_ms"], 3) if generation["ttft_ms"] is not None else None,
                "generation_latency_ms": (round(generation["generation_latency_ms"], 3)
                                          if generation["generation_latency_ms"] is not None else None),
                "failure_latency_ms": (round(generation["failure_latency_ms"], 3)
                                       if generation["failure_latency_ms"] is not None else None),
                "retrieval_latency_ms": retrieval.timings.total_retrieval_ms,
                "total_answer_latency_ms": round((time.perf_counter() - total_started) * 1000, 3),
                "generation_status": status, "generation_error": generation["generation_error"],
                "retry_count": generation["retry_count"],
                "retry_delays_ms": generation["retry_delays_ms"],
                "transient_error_codes": generation["transient_error_codes"],
                "final_failure_classification": generation["final_failure_classification"],
                "generation_attempt_count": generation["attempt_count"],
                "generation_retry_policy": {"concurrency": 1,
                                            "max_retries": retry_policy.max_retries,
                                            "min_request_interval_seconds": request_scheduler.min_interval_seconds,
                                            "backoff_base_seconds": retry_policy.backoff_base_seconds,
                                            "backoff_max_seconds": retry_policy.backoff_max_seconds},
                "web_search_enabled": False,
                "model_configuration": {"provider": "gemini", "model": llm_provider.model_name,
                                        "thinking_level": llm_provider.thinking_level,
                                        "max_output_tokens": llm_provider.max_output_tokens},
                "retrieval_configuration": {"dense_top_k": settings.DENSE_TOP_K,
                                            "lexical_top_k": settings.LEXICAL_TOP_K,
                                            "candidate_pool_size": settings.CANDIDATE_POOL_SIZE,
                                            "rerank_top_k": settings.RERANK_TOP_K,
                                            "routing_mode": settings.RETRIEVAL_ROUTING_MODE,
                                            "rrf_k": settings.RRF_K,
                                            "embedding_model": getattr(embedding_provider, "model_name", type(embedding_provider).__name__),
                                            "reranker_model": getattr(reranker_provider, "model_name", type(reranker_provider).__name__)},
            }
            records.append(validate_answer_record(record))
    finally:
        _IN_MEMORY_CHUNKS.pop(document_id, None)
    return records


def _ordered_record(record: dict) -> dict:
    """Put familiar fields first while preserving every value and extra field."""
    ordered = {key: record[key] for key in PRETTY_FIELD_ORDER if key in record}
    ordered.update((key, value) for key, value in record.items() if key not in ordered)
    return ordered


def _summary_markdown(records: list[dict], configured_model: str | None = None) -> str:
    def cell(value) -> str:
        return str(value if value is not None else "n/a").replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    def fmt_ms(value) -> str:
        return f"{value:.3f}" if isinstance(value, (int, float)) else "n/a"

    lines = ["# Answer Evaluation Capture Summary", "",
             "| Eval ID | Query | Status | Retry Count | Retrieval ms | Generation ms |",
             "|---|---|---|---:|---:|---:|"]
    for row in records:
        lines.append("| " + " | ".join((
            cell(row.get("eval_id")), cell(row.get("question")), cell(row.get("generation_status")),
            cell(row.get("retry_count", 0)), fmt_ms(row.get("retrieval_latency_ms")),
            fmt_ms(row.get("generation_latency_ms")),
        )) + " |")

    successful = [row for row in records if row.get("generation_status") in {"success", "generated"}]
    generation_values = [row["generation_latency_ms"] for row in successful
                         if isinstance(row.get("generation_latency_ms"), (int, float))]
    retrieval_values = [row["retrieval_latency_ms"] for row in records
                        if isinstance(row.get("retrieval_latency_ms"), (int, float))]
    retry_total = sum(row.get("retry_count", 0) for row in records
                      if isinstance(row.get("retry_count", 0), (int, float)))
    model = (records[0].get("model_configuration", {}).get("model") if records else None) or configured_model
    lines.extend(["", f"Total queries: {len(records)}",
                  f"Successful generations: {len(successful)}",
                  f"Failed generations: {sum(row.get('generation_status') == 'generation_failed' for row in records)}",
                  f"Total retries: {retry_total}",
                  f"Mean generation latency (successful generations): {fmt_ms(sum(generation_values) / len(generation_values) if generation_values else None)} ms",
                  f"Mean retrieval latency: {fmt_ms(sum(retrieval_values) / len(retrieval_values) if retrieval_values else None)} ms",
                  f"Model: {cell(model)}", ""])
    return "\n".join(lines)


def write_outputs(records: list[dict], capture_path: Path, unavailable_reason: str | None = None,
                  configured_model: str | None = None,
                  summary_path: Path | None = None) -> None:
    if len({record["eval_id"] for record in records}) != len(records):
        raise ValueError("Capture must contain exactly one final record per eval_id")
    summary_path = summary_path or ROOT / "results" / "captured_answer_summary.md"
    latest_results_path = ROOT / "results" / "latest_answer_results.json"
    latest_report_path = ROOT / "results" / "latest_answer_report.md"
    output_paths = (capture_path, summary_path, latest_results_path, latest_report_path)
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    ordered_records = [_ordered_record(row) for row in records]
    with capture_path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(ordered_records, output, indent=2, ensure_ascii=False)
        output.write("\n")
    summary_path.write_text(_summary_markdown(records, configured_model), encoding="utf-8")
    report, markdown = answer_report(records, configured_model=configured_model)
    if unavailable_reason:
        report["metadata"]["capture_status"] = "unavailable"
        report["limitations"].append(unavailable_reason)
        markdown = markdown.replace("## Limitations", f"Generation unavailable: {unavailable_reason}\n\n## Limitations")
    latest_results_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    latest_report_path.write_text(markdown, encoding="utf-8")




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Capture the first N rows (use 3 for smoke run).")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "captured_answer_runs.json")
    parser.add_argument("--max-retries", type=int, default=6,
                        help="Maximum retries after the initial Gemini request (default: 6).")
    parser.add_argument("--min-request-interval", type=float, default=MIN_REQUEST_INTERVAL_SECONDS,
                        help="Minimum seconds between Gemini request starts (default: 15).")
    parser.add_argument("--backoff-base", type=float, default=5.0,
                        help="Base exponential retry delay in seconds (default: 5).")
    parser.add_argument("--backoff-max", type=float, default=60.0,
                        help="Maximum exponential retry delay in seconds (default: 60).")
    args = parser.parse_args()
    dataset = load_answer_dataset()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    try:
        retry_policy = RetryPolicy(args.max_retries, args.min_request_interval,
                                   args.backoff_base, args.backoff_max)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        from app.core.config import settings
        configured_model = settings.GEMINI_MODEL
    except Exception:
        configured_model = None
    try:
        from app.llm.gemini_provider import GeminiProvider
        from app.ml.bge_provider import BGEEmbeddingProvider
        from app.ml.reranker_provider import CrossEncoderRerankerProvider
        llm = GeminiProvider()
        embedding = BGEEmbeddingProvider()
        reranker = CrossEncoderRerankerProvider()
    except Exception as exc:
        reason = (f"Capture could not initialize ({type(exc).__name__}); check the backend dependencies "
                  "and Gemini configuration.")
        write_outputs([], args.output, reason, configured_model)
        raise SystemExit(reason)
    records = asyncio.run(capture_answer_dataset(dataset, embedding, reranker, llm, args.limit,
                                                 retry_policy=retry_policy))
    write_outputs(records, args.output, configured_model=configured_model)
    print(f"Captured {len(records)} answer records from {min(args.limit or len(dataset), len(dataset))} queries: {args.output}")


if __name__ == "__main__":
    main()
