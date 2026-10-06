"""Retrieval Service for Phase 7 Hybrid Retrieval + Cross-Encoder Reranking."""
import logging
import math
import re
import time
import uuid
import asyncio
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set, Tuple
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.concurrency import run_in_threadpool

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.rag.router import RetrievalRouter, RetrievalRoutingMode, RoutingDecision
from app.ml.loader import get_embedding_provider, get_reranker_provider
from app.ml.base import BaseEmbeddingProvider
from app.ml.reranker_base import BaseRerankerProvider
from app.schemas.retrieval import (
    RetrievalRequest,
    RetrievedChunk,
    RetrievalTimingMetrics,
    RetrievalResponse,
)
from app.services.ingestion_service import _IN_MEMORY_CHUNKS

logger = logging.getLogger(__name__)


@dataclass
class RetrievalCandidate:
    """Internal candidate representation across retrieval, fusion, and reranking stages."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    workspace_id: uuid.UUID
    user_id: uuid.UUID
    content: str
    page_number_start: int
    page_number_end: int
    chunk_index: int
    dense_score: Optional[float] = None
    lexical_score: Optional[float] = None
    dense_rank: Optional[int] = None
    lexical_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    retrieval_sources: Set[str] = field(default_factory=set)
    passed_relevance_gate: bool = False
    final_rank: int = 0


class RetrievalService:
    """Coordinates Dense pgvector search, PostgreSQL FTS lexical search, RRF, and Cross-Encoder reranking."""

    @classmethod
    async def dense_retrieve(
        cls,
        db: Optional[AsyncSession],
        workspace_id: uuid.UUID,
        user_id: uuid.UUID,
        query_vector: List[float],
        top_k: int,
    ) -> List[RetrievalCandidate]:
        """Performs semantic vector search against document_chunks.embedding using cosine similarity."""
        if db is not None:
            # Query pgvector via cosine distance (<=>). Cosine similarity = 1 - cosine distance.
            vec_literal = f"[{','.join(str(round(x, 7)) for x in query_vector)}]"
            stmt = text("""
                SELECT id, document_id, workspace_id, user_id, content,
                       page_number_start, page_number_end, chunk_index,
                       1.0 - (embedding <=> CAST(:vec_literal AS vector)) AS dense_score
                FROM public.document_chunks
                WHERE workspace_id = :workspace_id
                  AND user_id = :user_id
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> CAST(:vec_literal AS vector) ASC
                LIMIT :top_k
            """)
            try:
                result = await db.execute(
                    stmt,
                    {
                        "workspace_id": workspace_id,
                        "user_id": user_id,
                        "vec_literal": vec_literal,
                        "top_k": top_k,
                    },
                )
                rows = result.fetchall()
            except Exception as exc:
                try:
                    await db.rollback()
                except Exception:
                    pass
                raise exc

            candidates: List[RetrievalCandidate] = []
            for row in rows:
                c = RetrievalCandidate(
                    chunk_id=row.id,
                    document_id=row.document_id,
                    workspace_id=row.workspace_id,
                    user_id=row.user_id,
                    content=row.content,
                    page_number_start=row.page_number_start,
                    page_number_end=row.page_number_end,
                    chunk_index=row.chunk_index,
                    dense_score=round(float(row.dense_score), 4),
                    retrieval_sources={"dense"},
                )
                candidates.append(c)
            return candidates

        # In-memory fallback for test environments without a live PostgreSQL instance
        in_memory_candidates: List[Tuple[float, Any]] = []
        for chunks in _IN_MEMORY_CHUNKS.values():
            for chunk in chunks:
                if (
                    chunk.workspace_id == workspace_id
                    and chunk.user_id == user_id
                    and chunk.embedding is not None
                ):
                    # Compute cosine similarity between chunk.embedding and query_vector
                    c_vec = chunk.embedding
                    dot_product = sum(a * b for a, b in zip(query_vector, c_vec))
                    norm_q = math.sqrt(sum(a * a for a in query_vector))
                    norm_c = math.sqrt(sum(b * b for b in c_vec))
                    sim = dot_product / (norm_q * norm_c) if (norm_q > 1e-9 and norm_c > 1e-9) else 0.0
                    in_memory_candidates.append((sim, chunk))

        # Sort descending by cosine similarity
        in_memory_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = in_memory_candidates[:top_k]

        return [
            RetrievalCandidate(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                workspace_id=chunk.workspace_id,
                user_id=chunk.user_id,
                content=chunk.content,
                page_number_start=chunk.page_number_start,
                page_number_end=chunk.page_number_end,
                chunk_index=chunk.chunk_index,
                dense_score=round(sim, 4),
                retrieval_sources={"dense"},
            )
            for sim, chunk in top_candidates
        ]

    @classmethod
    async def lexical_retrieve(
        cls,
        db: Optional[AsyncSession],
        workspace_id: uuid.UUID,
        user_id: uuid.UUID,
        query: str,
        top_k: int,
    ) -> List[RetrievalCandidate]:
        """Performs PostgreSQL full-text search with cover-density ranking (ts_rank_cd).

        Supports Vietnamese and English via a dual tsvector ('simple' + 'english').
        'simple' preserves exact Vietnamese accented tokens and prevents stopword elimination.
        'english' provides Porter stemming for English terms.
        Uses cover-density ranking (ts_rank_cd) which evaluates term proximity, not canonical BM25.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        if db is not None:
            # Dual query: search via 'simple' (Vietnamese/exact) and 'english' (stemmed)
            stmt = text("""
                SELECT id, document_id, workspace_id, user_id, content,
                       page_number_start, page_number_end, chunk_index,
                       GREATEST(
                           ts_rank_cd(tsv, websearch_to_tsquery('simple', :query)),
                           ts_rank_cd(tsv, websearch_to_tsquery('english', :query))
                       ) AS lexical_score
                FROM public.document_chunks
                WHERE workspace_id = :workspace_id
                  AND user_id = :user_id
                  AND (
                      tsv @@ websearch_to_tsquery('simple', :query)
                      OR tsv @@ websearch_to_tsquery('english', :query)
                  )
                ORDER BY lexical_score DESC
                LIMIT :top_k
            """)
            try:
                result = await db.execute(
                    stmt,
                    {
                        "workspace_id": workspace_id,
                        "user_id": user_id,
                        "query": cleaned_query,
                        "top_k": top_k,
                    },
                )
                rows = result.fetchall()

                # If websearch_to_tsquery returned 0 rows, fallback to plainto_tsquery
                if not rows:
                    stmt_fallback = text("""
                        SELECT id, document_id, workspace_id, user_id, content,
                               page_number_start, page_number_end, chunk_index,
                               GREATEST(
                                   ts_rank_cd(tsv, plainto_tsquery('simple', :query)),
                                   ts_rank_cd(tsv, plainto_tsquery('english', :query))
                               ) AS lexical_score
                        FROM public.document_chunks
                        WHERE workspace_id = :workspace_id
                          AND user_id = :user_id
                          AND (
                              tsv @@ plainto_tsquery('simple', :query)
                              OR tsv @@ plainto_tsquery('english', :query)
                          )
                        ORDER BY lexical_score DESC
                        LIMIT :top_k
                    """)
                    res_fallback = await db.execute(
                        stmt_fallback,
                        {
                            "workspace_id": workspace_id,
                            "user_id": user_id,
                            "query": cleaned_query,
                            "top_k": top_k,
                        },
                    )
                    rows = res_fallback.fetchall()
            except Exception as exc:
                try:
                    await db.rollback()
                except Exception:
                    pass
                raise exc

            candidates: List[RetrievalCandidate] = []
            for row in rows:
                c = RetrievalCandidate(
                    chunk_id=row.id,
                    document_id=row.document_id,
                    workspace_id=row.workspace_id,
                    user_id=row.user_id,
                    content=row.content,
                    page_number_start=row.page_number_start,
                    page_number_end=row.page_number_end,
                    chunk_index=row.chunk_index,
                    lexical_score=round(float(row.lexical_score), 4),
                    retrieval_sources={"lexical"},
                )
                candidates.append(c)
            return candidates

        # In-memory fallback for test environments: Unicode-aware cover-density ranking
        query_words = [w.lower() for w in re.findall(r"\w+", cleaned_query)]
        if not query_words:
            return []

        query_set = set(query_words)
        in_memory_lexical: List[Tuple[float, Any]] = []

        for chunks in _IN_MEMORY_CHUNKS.values():
            for chunk in chunks:
                if chunk.workspace_id == workspace_id and chunk.user_id == user_id:
                    content_words = [w.lower() for w in re.findall(r"\w+", chunk.content)]
                    if not content_words:
                        continue

                    # Find matching word positions
                    matched_positions = [
                        (idx, word) for idx, word in enumerate(content_words) if word in query_set
                    ]
                    if not matched_positions:
                        continue

                    distinct_matches = len({word for _, word in matched_positions})
                    match_count = len(matched_positions)

                    # Cover density: calculate minimum span covering all distinct matched query terms
                    proximity_bonus = 0.0
                    if distinct_matches > 1:
                        # Find minimum distance between different matched query words
                        min_span = float("inf")
                        for i in range(len(matched_positions)):
                            seen_words = {matched_positions[i][1]}
                            for j in range(i, len(matched_positions)):
                                seen_words.add(matched_positions[j][1])
                                if len(seen_words) == distinct_matches:
                                    span = matched_positions[j][0] - matched_positions[i][0] + 1
                                    if span < min_span:
                                        min_span = span
                                    break
                        if min_span < float("inf"):
                            proximity_bonus = distinct_matches / max(min_span, 1.0)

                    # Cover density score combining term frequency and term proximity
                    score = (
                        (distinct_matches * 1.5)
                        + (match_count * 0.5)
                        + (proximity_bonus * 2.0)
                    ) / (len(content_words) + 10.0)

                    in_memory_lexical.append((score, chunk))

        in_memory_lexical.sort(key=lambda x: x[0], reverse=True)
        top_candidates = in_memory_lexical[:top_k]

        return [
            RetrievalCandidate(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                workspace_id=chunk.workspace_id,
                user_id=chunk.user_id,
                content=chunk.content,
                page_number_start=chunk.page_number_start,
                page_number_end=chunk.page_number_end,
                chunk_index=chunk.chunk_index,
                lexical_score=round(score, 4),
                retrieval_sources={"lexical"},
            )
            for score, chunk in top_candidates
        ]

    @classmethod
    def reciprocal_rank_fusion(
        cls,
        dense_candidates: List[RetrievalCandidate],
        lexical_candidates: List[RetrievalCandidate],
        rrf_k: int = 60,
        pool_size: int = 30,
    ) -> List[RetrievalCandidate]:
        """Fuses dense and lexical candidates using Reciprocal Rank Fusion: RRF(d) = sum(1 / (k + rank))."""
        # Dictionary mapping chunk_id to candidate record
        merged: Dict[uuid.UUID, RetrievalCandidate] = {}

        # 1. Process dense rankings (1-indexed)
        for rank_idx, candidate in enumerate(dense_candidates, start=1):
            candidate.dense_rank = rank_idx
            candidate.retrieval_sources.add("dense")
            merged[candidate.chunk_id] = candidate

        # 2. Process lexical rankings (1-indexed)
        for rank_idx, candidate in enumerate(lexical_candidates, start=1):
            candidate.lexical_rank = rank_idx
            if candidate.chunk_id in merged:
                # Chunk retrieved by both dense and lexical channels: merge scores and sources
                existing = merged[candidate.chunk_id]
                existing.lexical_score = candidate.lexical_score
                existing.lexical_rank = rank_idx
                existing.retrieval_sources.add("lexical")
            else:
                candidate.retrieval_sources.add("lexical")
                merged[candidate.chunk_id] = candidate

        # 3. Compute RRF score for every merged candidate
        for candidate in merged.values():
            rrf_score = 0.0
            if candidate.dense_rank is not None:
                rrf_score += 1.0 / (rrf_k + candidate.dense_rank)
            if candidate.lexical_rank is not None:
                rrf_score += 1.0 / (rrf_k + candidate.lexical_rank)
            candidate.rrf_score = round(rrf_score, 6)

        # 4. Sort candidates descending by RRF score with deterministic tie-breaking
        candidates_list = list(merged.values())
        candidates_list.sort(
            key=lambda c: (
                -(c.rrf_score or 0.0),
                c.chunk_index,
                str(c.chunk_id),
            )
        )

        # 5. Restrict candidate pool size for reranking
        return candidates_list[:pool_size]

    @classmethod
    async def rerank(
        cls,
        query: str,
        candidates: List[RetrievalCandidate],
        reranker: BaseRerankerProvider,
        batch_size: int = 16,
        top_k: int = 5,
    ) -> List[RetrievalCandidate]:
        """Reranks the candidate pool using Cross-Encoder inference executed in a threadpool."""
        if not candidates:
            return []

        # Prepare (query, chunk_text) pairs
        pairs = [(query, c.content) for c in candidates]

        # Execute blocking model inference in threadpool to keep the asyncio event loop responsive
        scores: List[float] = await run_in_threadpool(
            reranker.predict,
            pairs,
            batch_size=batch_size,
        )

        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = round(float(score), 4)

        # Sort descending by rerank score with deterministic tie-breakers
        candidates.sort(
            key=lambda c: (
                -(c.rerank_score or 0.0),
                -(c.rrf_score or 0.0),
                c.chunk_index,
                str(c.chunk_id),
            )
        )

        # Slices to final top-K
        reranked = candidates[:top_k]
        for idx, c in enumerate(reranked, start=1):
            c.final_rank = idx

        return reranked

    @classmethod
    def apply_relevance_gate(
        cls,
        candidates: List[RetrievalCandidate],
        threshold: float,
    ) -> Tuple[List[RetrievalCandidate], bool]:
        """Determines if retrieved chunks meet minimum calibrated relevance threshold."""
        has_sufficient_evidence = False

        for candidate in candidates:
            score = candidate.rerank_score if candidate.rerank_score is not None else 0.0
            is_passed = score >= threshold
            candidate.passed_relevance_gate = is_passed
            if is_passed:
                has_sufficient_evidence = True

        return candidates, has_sufficient_evidence

    @classmethod
    async def retrieve(
        cls,
        db: Optional[AsyncSession],
        workspace_id: uuid.UUID,
        user_id: uuid.UUID,
        request: RetrievalRequest,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        reranker_provider: Optional[BaseRerankerProvider] = None,
    ) -> RetrievalResponse:
        """End-to-end execution of the Phase 7 Hybrid Retrieval + Cross-Encoder Reranking pipeline."""
        t_start = time.perf_counter()
        diagnostics: Dict[str, Any] = {}

        # Resolve hyperparameters
        dense_k = request.dense_top_k or settings.DENSE_TOP_K
        lexical_k = request.lexical_top_k or settings.LEXICAL_TOP_K
        rrf_k = request.rrf_k or settings.RRF_K
        pool_size = request.candidate_pool_size or settings.CANDIDATE_POOL_SIZE
        rerank_k = request.rerank_top_k or settings.RERANK_TOP_K
        threshold = (
            request.relevance_threshold
            if request.relevance_threshold is not None
            else settings.RELEVANCE_THRESHOLD
        )

        # Model providers (reused singletons)
        emb_provider = embedding_provider or get_embedding_provider()
        rerank_provider = reranker_provider or get_reranker_provider()

        query_str = request.query.strip()

        # Step 1 & 2: Concurrent Dense and Lexical Retrieval
        async def _execute_dense() -> Tuple[List[RetrievalCandidate], float, float]:
            t_d0 = time.perf_counter()
            query_vectors = await run_in_threadpool(
                emb_provider.encode_batch,
                [query_str],
                normalize=settings.EMBEDDING_NORMALIZE,
                batch_size=1,
            )
            q_vec = query_vectors[0]
            t_embed = (time.perf_counter() - t_d0) * 1000

            t_d_search = time.perf_counter()
            if db is not None and AsyncSessionLocal is not None:
                async with AsyncSessionLocal() as dense_session:
                    dense_session.info["rls_user_id"] = user_id
                    cands = await cls.dense_retrieve(
                        db=dense_session,
                        workspace_id=workspace_id,
                        user_id=user_id,
                        query_vector=q_vec,
                        top_k=dense_k,
                    )
            else:
                cands = await cls.dense_retrieve(
                    db=db,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    query_vector=q_vec,
                    top_k=dense_k,
                )
            t_dense = (time.perf_counter() - t_d_search) * 1000
            return cands, t_embed, t_dense

        async def _execute_lexical() -> Tuple[List[RetrievalCandidate], float]:
            t_l0 = time.perf_counter()
            if db is not None and AsyncSessionLocal is not None:
                async with AsyncSessionLocal() as lex_session:
                    lex_session.info["rls_user_id"] = user_id
                    cands = await cls.lexical_retrieve(
                        db=lex_session,
                        workspace_id=workspace_id,
                        user_id=user_id,
                        query=query_str,
                        top_k=lexical_k,
                    )
            else:
                cands = await cls.lexical_retrieve(
                    db=db,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    query=query_str,
                    top_k=lexical_k,
                )
            t_lex = (time.perf_counter() - t_l0) * 1000
            return cands, t_lex

        dense_res, lexical_res = await asyncio.gather(
            _execute_dense(),
            _execute_lexical(),
            return_exceptions=True,
        )

        # If any channel failed, ensure db session transaction is rolled back so it cannot poison subsequent queries
        if db is not None and (isinstance(dense_res, Exception) or isinstance(lexical_res, Exception)):
            try:
                await db.rollback()
            except Exception:
                pass

        dense_candidates: List[RetrievalCandidate] = []
        query_embedding_ms = 0.0
        dense_retrieval_ms = 0.0
        if isinstance(dense_res, Exception):
            logger.warning("Dense retrieval channel failed: %s", dense_res)
            diagnostics["dense_error"] = str(dense_res)
        else:
            dense_candidates, query_embedding_ms, dense_retrieval_ms = dense_res

        lexical_candidates: List[RetrievalCandidate] = []
        lexical_retrieval_ms = 0.0
        if isinstance(lexical_res, Exception):
            logger.warning("Lexical retrieval channel failed: %s", lexical_res)
            diagnostics["lexical_error"] = str(lexical_res)
        else:
            lexical_candidates, lexical_retrieval_ms = lexical_res

        # Check for complete failure across both channels
        if not dense_candidates and not lexical_candidates and diagnostics:
            logger.error("Both dense and lexical retrieval channels encountered errors.")

        # Step 3: Reciprocal Rank Fusion
        t_rrf = time.perf_counter()
        fused_pool = cls.reciprocal_rank_fusion(
            dense_candidates=dense_candidates,
            lexical_candidates=lexical_candidates,
            rrf_k=rrf_k,
            pool_size=pool_size,
        )
        rrf_ms = (time.perf_counter() - t_rrf) * 1000

        # Step 4: Fast / Quality Path V1 Routing Decision
        routing_decision = RetrievalRouter.decide(
            candidates=fused_pool,
            mode=request.routing_mode,
        )
        diagnostics.update(routing_decision.to_dict())

        final_candidates: List[RetrievalCandidate] = []
        rerank_ms = 0.0

        if routing_decision.path == "FAST":
            # FAST PATH: Bypass Cross-Encoder model inference
            final_candidates = fused_pool[:rerank_k]
            for idx, c in enumerate(final_candidates, start=1):
                c.final_rank = idx
                c.rerank_score = c.rrf_score
                # On Fast path, chunks with dual consensus or top rank satisfy relevance
                c.passed_relevance_gate = (idx == 1) or (
                    "dense" in c.retrieval_sources and "lexical" in c.retrieval_sources
                )
            has_evidence = any(c.passed_relevance_gate for c in final_candidates)
        else:
            # QUALITY PATH: Cross-Encoder Reranking
            t_rerank = time.perf_counter()
            try:
                final_candidates = await cls.rerank(
                    query=query_str,
                    candidates=fused_pool,
                    reranker=rerank_provider,
                    batch_size=settings.RERANKER_BATCH_SIZE,
                    top_k=rerank_k,
                )
                rerank_ms = (time.perf_counter() - t_rerank) * 1000
            except Exception as e:
                logger.error("Cross-Encoder reranking failed: %s. Falling back to RRF ranking.", e)
                diagnostics["reranker_error"] = str(e)
                for idx, c in enumerate(fused_pool[:rerank_k], start=1):
                    c.rerank_score = c.rrf_score
                    c.final_rank = idx
                final_candidates = fused_pool[:rerank_k]
                rerank_ms = (time.perf_counter() - t_rerank) * 1000

            # Relevance Gating for Quality Path
            gated_candidates, has_evidence = cls.apply_relevance_gate(
                final_candidates,
                threshold=threshold,
            )
            final_candidates = gated_candidates

        total_retrieval_ms = (time.perf_counter() - t_start) * 1000

        # Diagnostics for sufficiency check
        if not has_evidence:
            max_score = max((c.rerank_score or 0.0 for c in final_candidates), default=0.0)
            diagnostics["relevance_gate_status"] = (
                f"No candidates passed relevance threshold {threshold}. "
                f"Top candidate score: {max_score:.4f}."
            )

        timings = RetrievalTimingMetrics(
            query_embedding_ms=round(query_embedding_ms, 2),
            dense_retrieval_ms=round(dense_retrieval_ms, 2),
            lexical_retrieval_ms=round(lexical_retrieval_ms, 2),
            rrf_ms=round(rrf_ms, 2),
            rerank_ms=round(rerank_ms, 2),
            total_retrieval_ms=round(total_retrieval_ms, 2),
        )

        results: List[RetrievedChunk] = [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                content=c.content,
                page_number_start=c.page_number_start,
                page_number_end=c.page_number_end,
                chunk_index=c.chunk_index,
                dense_score=c.dense_score,
                lexical_score=c.lexical_score,
                rrf_score=c.rrf_score,
                rerank_score=c.rerank_score,
                final_rank=c.final_rank,
                passed_relevance_gate=c.passed_relevance_gate,
                retrieval_sources=sorted(list(c.retrieval_sources)),
            )
            for c in final_candidates
        ]

        return RetrievalResponse(
            workspace_id=workspace_id,
            query=query_str,
            results=results,
            total_results=len(results),
            has_sufficient_evidence=has_evidence,
            relevance_threshold=threshold,
            routing_path=routing_decision.path,
            timings=timings,
            diagnostics=diagnostics if diagnostics else None,
        )
