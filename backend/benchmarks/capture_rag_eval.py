"""Execute the versioned query set against the local retrieval benchmark fixture.

The capture stage uses local embedding/reranker models and the existing
RetrievalService. It never creates an LLM provider or a real web provider.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

# The configured encoders must use already-cached checkpoints; no model download
# is allowed during this offline benchmark.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from benchmarks.benchmark_retrieval import BENCHMARK_CHUNKS
from app.core.config import settings
from app.ml.bge_provider import BGEEmbeddingProvider
from app.ml.reranker_provider import CrossEncoderRerankerProvider
from app.models.chunk import DocumentChunk
from app.schemas.retrieval import RetrievalRequest
from app.services.ingestion_service import _IN_MEMORY_CHUNKS
from app.services.retrieval_service import RetrievalService
from app.web_search.mock_provider import MockWebSearchProvider

ROOT = Path(__file__).resolve().parent

# Dataset aliases are tied explicitly to chunk_index in benchmark_retrieval.py.
# All other fixture chunks have stable positional aliases for captured rankings.
FIXTURE_ALIAS_TO_INDEX = {
    "fixture:artificial_intelligence": 0,
    "fixture:machine_learning": 1,
    "fixture:deep_learning": 2,
    "fixture:transformer": 3,
    "fixture:photosynthesis": 4,
    "fixture:chlorophyll": 5,
    "fixture:cellular_respiration": 6,
    "fixture:newton_laws": 7,
    "fixture:relational_database": 8,
    "fixture:hnsw": 9,
}
INDEX_TO_FIXTURE_ALIAS = {index: alias for alias, index in FIXTURE_ALIAS_TO_INDEX.items()}


def fixture_alias(chunk_index: int) -> str:
    return INDEX_TO_FIXTURE_ALIAS.get(chunk_index, f"fixture:chunk_{chunk_index}")


def load_dataset() -> list[dict]:
    with (ROOT / "eval_dataset.jsonl").open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def build_fixture(workspace_id: uuid.UUID, user_id: uuid.UUID,
                  embedding_provider) -> tuple[uuid.UUID, dict[str, DocumentChunk]]:
    document_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    fixture_vectors = embedding_provider.encode_batch(
        BENCHMARK_CHUNKS, normalize=settings.EMBEDDING_NORMALIZE, batch_size=16
    )
    chunks_by_alias: dict[str, DocumentChunk] = {}
    chunks: list[DocumentChunk] = []
    for index, (content, vector) in enumerate(zip(BENCHMARK_CHUNKS, fixture_vectors)):
        chunk = DocumentChunk(
            id=uuid.uuid4(), document_id=document_id, workspace_id=workspace_id,
            user_id=user_id, chunk_index=index, content=content,
            page_number_start=1, page_number_end=1, token_count=len(content.split()),
            embedding=vector,
            embedding_model=getattr(embedding_provider, "model_name", None),
            embedding_version=getattr(embedding_provider, "version", None),
            created_at=now, updated_at=now,
        )
        chunks.append(chunk)
        chunks_by_alias[fixture_alias(index)] = chunk
    _IN_MEMORY_CHUNKS[document_id] = chunks
    return document_id, chunks_by_alias


def _ranked_aliases(chunks) -> list[str]:
    return [fixture_alias(chunk.chunk_index) for chunk in chunks]


_RETRIEVED_RESULT_FIELDS = (
    "chunk_id", "document_id", "content", "page_number_start", "page_number_end",
    "chunk_index", "dense_score", "lexical_score", "rrf_score", "rerank_score",
    "final_rank", "passed_relevance_gate", "retrieval_sources",
)
_REQUIRED_RETRIEVED_RESULT_FIELDS = (
    "chunk_id", "document_id", "page_number_start", "page_number_end", "final_rank",
)


def _normalize_retrieved_result(result) -> dict:
    """Normalize RetrievalService RetrievedChunk models or dicts to schema fields."""
    if isinstance(result, Mapping):
        raw = dict(result)
    elif hasattr(result, "model_dump"):
        raw = result.model_dump()
    else:
        raw = {field: getattr(result, field) for field in _RETRIEVED_RESULT_FIELDS
               if hasattr(result, field)}

    missing = [field for field in _REQUIRED_RETRIEVED_RESULT_FIELDS if field not in raw]
    if missing:
        raise ValueError(
            "Retrieval result is missing RetrievedChunk fields: " + ", ".join(missing)
        )
    # Select only fields declared by the existing RetrievedChunk schema. Missing
    # optional fields remain None; no score or provenance values are synthesized.
    return {field: raw.get(field) for field in _RETRIEVED_RESULT_FIELDS}


def _result_record(result: dict, alias_by_id: dict) -> dict:
    alias = alias_by_id[str(result["chunk_id"])]
    return {
        "rank": result["final_rank"],
        "alias": alias,
        "chunk_id": str(result["chunk_id"]),
        "document_id": str(result["document_id"]),
        "page_start": result["page_number_start"],
        "page_end": result["page_number_end"],
        "rerank_score": result["rerank_score"],
        "passed_relevance_gate": result["passed_relevance_gate"],
    }


async def capture_dataset(dataset: list[dict], embedding_provider,
                         reranker_provider, web_provider=None) -> list[dict]:
    """Run all query rows against the in-memory fixture and return JSON-ready records."""
    web_provider = web_provider or MockWebSearchProvider()
    workspace_id, user_id = uuid.uuid4(), uuid.uuid4()
    document_id, chunks_by_alias = build_fixture(workspace_id, user_id, embedding_provider)
    alias_by_id = {str(chunk.id): alias for alias, chunk in chunks_by_alias.items()}
    records: list[dict] = []
    try:
        for item in dataset:
            query = item["question"]
            gold_aliases = item.get("gold_evidence_ids", [])
            unknown_aliases = [
                alias for alias in gold_aliases
                if alias not in FIXTURE_ALIAS_TO_INDEX
                or FIXTURE_ALIAS_TO_INDEX[alias] >= len(BENCHMARK_CHUNKS)
            ]
            label_status = "unmapped_alias" if unknown_aliases else (
                "mapped" if gold_aliases else "no_gold_evidence"
            )

            # Capture each channel for ranking baselines from the same fixture.
            stage_start = time.perf_counter()
            query_vector = embedding_provider.encode_batch(
                [query], normalize=settings.EMBEDDING_NORMALIZE, batch_size=1
            )[0]
            query_embedding_ms = (time.perf_counter() - stage_start) * 1000

            stage_start = time.perf_counter()
            dense = await RetrievalService.dense_retrieve(
                None, workspace_id, user_id, query_vector, len(BENCHMARK_CHUNKS)
            )
            dense_ms = (time.perf_counter() - stage_start) * 1000

            stage_start = time.perf_counter()
            lexical = await RetrievalService.lexical_retrieve(
                None, workspace_id, user_id, query, len(BENCHMARK_CHUNKS)
            )
            lexical_ms = (time.perf_counter() - stage_start) * 1000

            stage_start = time.perf_counter()
            fused = RetrievalService.reciprocal_rank_fusion(
                list(dense), list(lexical), settings.RRF_K, len(BENCHMARK_CHUNKS)
            )
            rrf_ms = (time.perf_counter() - stage_start) * 1000
            hybrid_ranking = list(fused)

            # Execute the production orchestrator in forced QUALITY mode for its
            # reranked candidates and existing relevance gate.
            request = RetrievalRequest(
                query=query,
                dense_top_k=len(BENCHMARK_CHUNKS),
                lexical_top_k=len(BENCHMARK_CHUNKS),
                candidate_pool_size=len(BENCHMARK_CHUNKS),
                rerank_top_k=min(10, len(BENCHMARK_CHUNKS)),
                routing_mode="always_quality",
            )
            response = await RetrievalService.retrieve(
                db=None, workspace_id=workspace_id, user_id=user_id, request=request,
                embedding_provider=embedding_provider, reranker_provider=reranker_provider,
            )
            reranked = [_normalize_retrieved_result(result) for result in response.results]
            top_score = max((result["rerank_score"] or 0.0 for result in reranked), default=0.0)
            answerable = bool(response.has_sufficient_evidence and reranked
                              and top_score >= settings.MIN_ANSWERABLE_RERANK_SCORE)

            web_search_ms = 0.0
            calls_before = web_provider.call_count if hasattr(web_provider, "call_count") else None
            if not answerable:
                web_start = time.perf_counter()
                # This provider is always the local mock in the command path.
                await web_provider.search(query, max_results=settings.TAVILY_MAX_RESULTS)
                web_search_ms = (time.perf_counter() - web_start) * 1000
            web_called = bool(not answerable and calls_before is not None
                              and web_provider.call_count == calls_before + 1)

            ranked_results = [
                _result_record(result, alias_by_id)
                for result in reranked
            ]
            timings = {
                "query_embedding_ms": round(response.timings.query_embedding_ms, 3),
                "dense_retrieval_ms": round(response.timings.dense_retrieval_ms, 3),
                "lexical_retrieval_ms": round(response.timings.lexical_retrieval_ms, 3),
                "rrf_ms": round(response.timings.rrf_ms, 3),
                "rerank_ms": round(response.timings.rerank_ms, 3),
                "total_retrieval_ms": round(response.timings.total_retrieval_ms, 3),
                "web_search_ms": round(web_search_ms, 3),
            }
            rankings = {
                "dense_only": _ranked_aliases(dense),
                "lexical_only": _ranked_aliases(lexical),
                "hybrid_rrf": _ranked_aliases(hybrid_ranking),
                "reranked": [result["alias"] for result in ranked_results],
            }
            configuration_timings = {
                "dense_only": query_embedding_ms + dense_ms,
                "lexical_only": lexical_ms,
                "hybrid_rrf": query_embedding_ms + dense_ms + lexical_ms + rrf_ms,
                "reranked": response.timings.total_retrieval_ms,
                "evidence_gate": response.timings.total_retrieval_ms,
                "full_web_fallback": response.timings.total_retrieval_ms + web_search_ms,
            }
            records.append({
                "eval_id": item["id"],
                "query_id": item["id"],
                "question": query,
                "query_type": item["query_type"],
                "retrieval_configuration": {
                    "dense_top_k": len(BENCHMARK_CHUNKS),
                    "lexical_top_k": len(BENCHMARK_CHUNKS),
                    "candidate_pool_size": len(BENCHMARK_CHUNKS),
                    "rerank_top_k": min(10, len(BENCHMARK_CHUNKS)),
                    "routing_mode": "always_quality",
                    "rrf_k": settings.RRF_K,
                    "relevance_threshold": settings.RELEVANCE_THRESHOLD,
                    "min_answerable_rerank_score": settings.MIN_ANSWERABLE_RERANK_SCORE,
                    "embedding_model": getattr(embedding_provider, "model_name", type(embedding_provider).__name__),
                    "reranker_model": getattr(reranker_provider, "model_name", type(reranker_provider).__name__),
                },
                "gold_label_status": label_status,
                "unmapped_gold_aliases": unknown_aliases,
                "gold_evidence_ids": gold_aliases,
                "gold_alias_to_fixture_index": {
                    alias: FIXTURE_ALIAS_TO_INDEX[alias]
                    for alias in gold_aliases if alias in FIXTURE_ALIAS_TO_INDEX
                    and FIXTURE_ALIAS_TO_INDEX[alias] < len(BENCHMARK_CHUNKS)
                },
                "gold_should_use_web": bool(item["should_use_web"]),
                "rankings": rankings,
                "retrieved_results": ranked_results,
                "has_sufficient_evidence": bool(response.has_sufficient_evidence),
                "evidence_sufficient": answerable,
                "top_rerank_score": top_score,
                "should_use_web": bool(item["should_use_web"]),
                "used_web_fallback": not answerable,
                "web_search_mocked": True,
                "mock_web_search_called": web_called,
                "timings": timings,
                "configuration_timings_ms": configuration_timings,
                "timings_are_measurements": True,
                "benchmark_note": "Local retrieval stages only; generation and citation capture are not run.",
            })
    finally:
        _IN_MEMORY_CHUNKS.pop(document_id, None)
    return records


async def _capture_with_models(dataset: list[dict]) -> list[dict]:
    embedding_provider = BGEEmbeddingProvider()
    reranker_provider = CrossEncoderRerankerProvider()
    return await capture_dataset(dataset, embedding_provider, reranker_provider, MockWebSearchProvider())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "captured_runs.jsonl")
    args = parser.parse_args()
    records = asyncio.run(_capture_with_models(load_dataset()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Captured {len(records)} runs from {len(load_dataset())} queries: {args.output}")


if __name__ == "__main__":
    main()
