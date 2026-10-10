"""RAG Service coordinating Phase 7 retrieval, relevance gating, prompt assembly,

LLM token streaming, citation extraction, and message persistence.
"""
import json
import asyncio
import logging
import time
import uuid
from typing import AsyncIterator, Dict, List, Optional, Set, Any
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.observability import request_id_context
from app.models.document import Document
from app.models.message import Message
from app.schemas.message import MessageCreate
from app.schemas.retrieval import RetrievalRequest, RetrievalResponse, RetrievedChunk
from app.services.conversation_service import ConversationService
from app.services.retrieval_service import RetrievalService
from app.services.document_service import DocumentService, _IN_MEMORY_DOCUMENTS
from app.llm import get_llm_provider, BaseLLMProvider, LLMError
from app.rag.prompt_builder import PromptBuilder, ChatMode, SourceEvidence
from app.rag.citation_service import CitationService, Citation, WebCitation
from app.web_search import get_web_search_provider, WebSearchProvider
from app.web_search.base import WebSearchResult
from app.web_search.exceptions import WebSearchError
from app.schemas.math_solver import MathSolverResult, SolverMetadata
from app.tools.math_solver import MathSolver
from app.tools.math_task_parser import MathTaskParser
from app.tools.solver_router import SolverRouter
from app.tools.math_presentation import generate_guidance, verified_final

logger = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I couldn't find enough information in the uploaded documents or web sources to answer that question confidently. "
    "Please ensure the relevant course materials or notes are uploaded and indexed, or try rephrasing your query."
)

# System instruction used when answering from Tavily web evidence (no document context).
WEB_FALLBACK_SYSTEM_INSTRUCTION = (
    "You are AI Tutor Assistant, an expert learning assistant. "
    "The user's question could not be answered from their uploaded documents. "
    "You have been provided with web search results as <WEB_EVIDENCE> context. "
    "Answer the question clearly and thoroughly using ONLY the information inside <WEB_EVIDENCE>. "
    "Do NOT fabricate facts, URLs, or source titles that are not present in <WEB_EVIDENCE>. "
    "Do NOT include [Web Source X] markers in your final answer — write natural prose only. "
    "Structure your response with headings and bullet points where helpful."
)


class RAGService:
    """Orchestrates document-grounded answer generation with strict evidence gating."""

    @staticmethod
    def _emit_stage_metrics(
        retrieval_response: RetrievalResponse,
        *,
        evidence_gate_decision: str,
        evidence_gate_ms: float,
        web_search_ms: float,
        llm_ttft_ms: float | None,
        generation_ms: float,
        used_web_fallback: bool,
        provider: Optional[BaseLLMProvider] = None,
        total_request_latency_ms: float,
    ) -> dict[str, Any]:
        timings = retrieval_response.timings
        metrics = {
            "request_id": request_id_context.get(),
            "retrieval_mode": retrieval_response.routing_path,
            "evidence_gate_decision": evidence_gate_decision,
            "used_web_fallback": used_web_fallback,
            "model_provider": type(provider).__name__ if provider else None,
            "model_selected": getattr(provider, "model_name", None) if provider else None,
            "query_embedding_ms": timings.query_embedding_ms,
            "dense_retrieval_ms": timings.dense_retrieval_ms,
            "lexical_retrieval_ms": timings.lexical_retrieval_ms,
            "rrf_ms": timings.rrf_ms,
            "rerank_ms": timings.rerank_ms,
            "evidence_gate_ms": round(evidence_gate_ms, 2),
            "web_search_ms": round(web_search_ms, 2),
            "llm_ttft_ms": round(llm_ttft_ms, 2) if llm_ttft_ms is not None else None,
            "generation_ms": round(generation_ms, 2),
            "total_retrieval_ms": timings.total_retrieval_ms,
            "total_request_latency_ms": round(total_request_latency_ms, 2),
        }
        logger.info("chat_stage_metrics %s", json.dumps(metrics, separators=(",", ":")))
        return metrics

    @classmethod
    async def resolve_document_names(
        cls,
        db: Optional[AsyncSession],
        document_ids: Set[uuid.UUID],
        user_id: uuid.UUID,
    ) -> Dict[uuid.UUID, str]:
        """Resolves document IDs to their original filenames with tenant isolation."""
        if not document_ids:
            return {}

        names: Dict[uuid.UUID, str] = {}
        if db is not None:
            stmt = select(Document.id, Document.original_filename).where(
                Document.id.in_(document_ids),
                Document.user_id == user_id,
            )
            try:
                result = await db.execute(stmt)
                for row in result.all():
                    names[row.id] = row.original_filename
            except Exception as e:
                logger.warning(
                    "resolve_document_names query failed on session: %s. Rolling back and retrying.",
                    e,
                )
                try:
                    await db.rollback()
                    result = await db.execute(stmt)
                    for row in result.all():
                        names[row.id] = row.original_filename
                except Exception as retry_err:
                    logger.error("resolve_document_names failed after rollback: %s", retry_err)
        else:
            for doc_id in document_ids:
                doc = _IN_MEMORY_DOCUMENTS.get(doc_id)
                if doc and doc.user_id == user_id:
                    names[doc.id] = doc.original_filename

        return names

    @classmethod
    async def stream_chat(
        cls,
        db: Optional[AsyncSession],
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        query: str,
        chat_mode_str: str = "Detailed Guidance",
        workspace_id: Optional[uuid.UUID] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
    ) -> AsyncIterator[str]:
        """Executes full RAG workflow and yields SSE events.

        Events yielded:
        - `event: status` -> status updates
        - `event: citations` -> retrieved evidence & gating diagnostics
        - `event: token` -> incremental text chunks
        - `event: done` -> final assembled message metadata
        - `event: error` -> failure information
        """
        t_start = time.perf_counter()

        # 1. Validate chat mode
        try:
            chat_mode = ChatMode.from_str(chat_mode_str)
        except ValueError as ve:
            yield f"event: error\ndata: {json.dumps({'error': str(ve)})}\n\n"
            return

        # 2. Authorize and verify conversation ownership
        conversation = await ConversationService.get_conversation(db, conversation_id, user_id)
        if not conversation:
            yield f"event: error\ndata: {json.dumps({'error': 'Conversation not found or unauthorized.'})}\n\n"
            return

        conv_workspace_id = conversation.workspace_id
        if workspace_id and workspace_id != conv_workspace_id:
            yield f"event: error\ndata: {json.dumps({'error': 'Specified workspace does not match conversation.'})}\n\n"
            return

        logger.info(
            "RAG generation started: conv=%s, user=%s, mode=%s, workspace=%s",
            conversation_id,
            user_id,
            chat_mode.value,
            conv_workspace_id,
        )

        # 3. Persist incoming user message
        user_msg = await ConversationService.create_message(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
            data=MessageCreate(role="user", content=query),
        )

        yield f"event: status\ndata: {json.dumps({'status': 'retrieving', 'message': 'Searching uploaded documents...'})}\n\n"

        # 4. Fetch recent conversation history (bounded to 6 turns before the current question)
        history_records = await ConversationService.list_messages(db, conversation_id, user_id) or []
        history_list = [
            {"role": m.role, "content": m.content}
            for m in history_records
            if m.id != user_msg.id  # exclude current message
        ][-6:]

        # 5. Execute Phase 7 hybrid retrieval and reranking
        t_retrieval_start = time.perf_counter()
        try:
            retrieval_req = RetrievalRequest(query=query)
            retrieval_response: RetrievalResponse = await RetrievalService.retrieve(
                db=db,
                workspace_id=conv_workspace_id,
                user_id=user_id,
                request=retrieval_req,
            )
        except Exception as e:
            logger.error("Retrieval failed during chat for conversation %s: %s", conversation_id, e)
            yield f"event: error\ndata: {json.dumps({'error': 'Failed to retrieve document evidence.'})}\n\n"
            return

        retrieval_duration_ms = round((time.perf_counter() - t_retrieval_start) * 1000, 2)
        logger.info(
            "Retrieval completed in %.2fms: %d candidates, route=%s, has_evidence=%s",
            retrieval_duration_ms,
            len(retrieval_response.results),
            retrieval_response.routing_path,
            retrieval_response.has_sufficient_evidence,
        )

        # 6. Strict Evidence Gating — Phase 9 Web Search Fallback
        #
        # Two-tier gate (both must pass to use the document path):
        #   Tier 1: has_sufficient_evidence — at least one chunk scored ≥ RELEVANCE_THRESHOLD (0.35).
        #           Indicates the corpus has topically adjacent material.
        #   Tier 2: top-1 Cross-Encoder score ≥ MIN_ANSWERABLE_RERANK_SCORE (0.55).
        #           Indicates the best retrieved chunk can actually answer the query,
        #           not merely share vocabulary with it.
        #           NOTE: Tier 2 only applies on the QUALITY path where rerank_score comes
        #           from the Cross-Encoder. On the FAST path, `rerank_score` is set to
        #           `rrf_score` (≈0.016–0.033), which is not comparable. FAST-path
        #           dual-consensus selection is already a strong answerability signal.
        # Chunks passing Tier 1 but failing Tier 2 (QUALITY path only) are "related but
        # not answerable" — web fallback is the correct route. This decision is made from
        # retrieval scores alone, before any LLM call.
        gate_started = time.perf_counter()
        is_quality_path = retrieval_response.routing_path == "QUALITY"
        top_rerank_score: float = max(
            (c.rerank_score for c in retrieval_response.results if c.rerank_score is not None),
            default=0.0,
        )
        tier2_passes: bool = (
            # FAST path: dual-consensus passing already implies answerability
            not is_quality_path
            # QUALITY path: require cross-encoder top-1 ≥ MIN_ANSWERABLE_RERANK_SCORE
            or top_rerank_score >= settings.MIN_ANSWERABLE_RERANK_SCORE
        )
        evidence_is_answerable: bool = (
            retrieval_response.has_sufficient_evidence
            and bool(retrieval_response.results)
            and tier2_passes
        )
        evidence_gate_ms = (time.perf_counter() - gate_started) * 1000
        web_search_ms = 0.0

        # Phase 9.1: retrieval and existing evidence gates always precede tools.
        # Document references can only use relevant uploaded-document statements.
        solver_result = None
        solver_metadata = None
        solver_used = False
        intent = SolverRouter.route(query)
        if intent == "math_problem":
            solver_started = time.perf_counter()
            task = None
            source = "retrieved_document" if SolverRouter.is_document_reference(query) else "user_question"
            try:
                if source == "retrieved_document":
                    if not evidence_is_answerable:
                        raise ValueError("No answerable exercise evidence")
                    task = await asyncio.to_thread(MathTaskParser.from_retrieved, query, retrieval_response.results)
                else:
                    task = await asyncio.to_thread(MathTaskParser.parse, query)
                solver_used = True
                solver_result = await MathSolver.execute(task, settings.MATH_SOLVER_TIMEOUT_SECONDS)
            except Exception as exc:
                solver_result = MathSolverResult(
                    success=False, problem_type=task.operation if task else None,
                    error="Calculation verification was unavailable. Please provide the complete exercise and variable.",
                    error_type=type(exc).__name__,
                )
            solver_metadata = SolverMetadata(used=solver_used, operation=solver_result.problem_type, verified=solver_result.success)
            logger.info("solver_execution %s", json.dumps({
                "solver_intent": intent, "solver_operation": solver_result.problem_type,
                "solver_used": solver_used, "solver_success": solver_result.success,
                "solver_duration_ms": round((time.perf_counter() - solver_started) * 1000, 2),
                "solver_source": source, "solver_error_type": solver_result.error_type,
            }, separators=(",", ":")))

        if not evidence_is_answerable and solver_result is None:
            logger.info(
                "Insufficient/non-answerable document evidence for query in conv %s "
                "(route=%s, has_evidence=%s, top_rerank=%.4f, answerable_threshold=%.4f, "
                "tier2_passes=%s). Checking web fallback (enabled=%s).",
                conversation_id,
                retrieval_response.routing_path,
                retrieval_response.has_sufficient_evidence,
                top_rerank_score,
                settings.MIN_ANSWERABLE_RERANK_SCORE,
                tier2_passes,
                settings.WEB_SEARCH_FALLBACK_ENABLED,
            )



            # --- Phase 9: Web Search Fallback (Tavily) ---
            if settings.WEB_SEARCH_FALLBACK_ENABLED:
                logger.info("[Phase 9] Web fallback ENABLED — entering Tavily path.")
                yield f"event: status\ndata: {json.dumps({'status': 'insufficient_evidence', 'message': 'Not enough information in your documents. Searching the web...'})}\n\n"
                yield f"event: status\ndata: {json.dumps({'status': 'web_search', 'message': 'Searching web...'})}\n\n"

                # 1. Fetch web results from Tavily
                web_search_started = time.perf_counter()
                try:
                    web_search_provider = get_web_search_provider()
                    tavily_results: List[WebSearchResult] = await web_search_provider.search(
                        query=query,
                        max_results=settings.TAVILY_MAX_RESULTS,
                    )
                    web_search_ms = (time.perf_counter() - web_search_started) * 1000
                    logger.info(
                        "[Phase 9] Tavily returned %d results for conv=%s",
                        len(tavily_results),
                        conversation_id,
                    )
                except WebSearchError as wse:
                    logger.error(
                        "[Phase 9] Tavily search FAILED (%s): %s",
                        type(wse).__name__,
                        wse.message,
                    )
                    yield f"event: error\ndata: {json.dumps({'error': f'Web search failed: {wse.message}'})}\n\n"
                    return
                except Exception as e:
                    logger.error("[Phase 9] Unexpected Tavily error: %s", e, exc_info=True)
                    yield f"event: error\ndata: {json.dumps({'error': 'An unexpected error occurred during web search.'})}\n\n"
                    return

                if not tavily_results:
                    logger.warning("[Phase 9] Tavily returned zero results — cannot answer from web.")
                    yield f"event: error\ndata: {json.dumps({'error': INSUFFICIENT_EVIDENCE_MESSAGE})}\n\n"
                    return

                # 2. Build bounded web context for Gemini (capped at WEB_SEARCH_MAX_SOURCES)
                capped_results = tavily_results[: settings.WEB_SEARCH_MAX_SOURCES]
                web_context_sections: List[str] = []
                for i, result in enumerate(capped_results, start=1):
                    web_context_sections.append(
                        f"[Web Source {i}]\n"
                        f"Title: {result.title}\n"
                        f"URL: {result.url}\n"
                        f"Content:\n{result.content}"
                    )
                web_context_str = "\n\n".join(web_context_sections)

                # 3. Assemble prompt that grounds Gemini on the Tavily evidence
                history_str = "\n".join(
                    f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')}"
                    for m in history_list
                ) or "None"
                web_prompt = (
                    "<WEB_EVIDENCE>\n"
                    f"{web_context_str}\n"
                    "</WEB_EVIDENCE>\n\n"
                    "<CONVERSATION_HISTORY>\n"
                    f"{history_str}\n"
                    "</CONVERSATION_HISTORY>\n\n"
                    "<USER_QUERY>\n"
                    f"{query.strip()}\n"
                    "</USER_QUERY>\n\n"
                    "Answer the user's question using ONLY the information provided in <WEB_EVIDENCE> above."
                )

                yield f"event: status\ndata: {json.dumps({'status': 'generating', 'message': 'Formulating answer...'})}\n\n"

                # 4. Stream tokens from existing generate_stream() — NO native grounding
                provider = llm_provider or get_llm_provider()
                accumulated_tokens: List[str] = []
                generation_started = time.perf_counter()
                llm_ttft_ms = None
                try:
                    async for token in provider.generate_stream(
                        prompt=web_prompt,
                        system_instruction=WEB_FALLBACK_SYSTEM_INSTRUCTION,
                    ):
                        if token:
                            if llm_ttft_ms is None:
                                llm_ttft_ms = (time.perf_counter() - generation_started) * 1000
                            accumulated_tokens.append(token)
                            yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
                    logger.info(
                        "[Phase 9] Tavily-grounded stream completed. tokens=%d",
                        len(accumulated_tokens),
                    )
                except LLMError as le:
                    logger.error(
                        "[Phase 9] LLM generation FAILED after Tavily search (LLMError=%s): %s",
                        type(le).__name__,
                        le.message,
                    )
                    yield f"event: error\ndata: {json.dumps({'error': f'Web search failed: {le.message}'})}\n\n"
                    return
                except Exception as e:
                    logger.error("[Phase 9] Unexpected error in web-grounded LLM streaming: %s", e, exc_info=True)
                    yield f"event: error\ndata: {json.dumps({'error': 'An unexpected error occurred during web-grounded response generation.'})}\n\n"
                    return
                generation_ms = (time.perf_counter() - generation_started) * 1000

                full_raw_text = "".join(accumulated_tokens).strip()
                if not full_raw_text:
                    full_raw_text = INSUFFICIENT_EVIDENCE_MESSAGE

                # 5. Build structured WebCitation list from Tavily results
                seen_web_urls: set = set()
                web_citations: List[WebCitation] = []
                for result in capped_results:
                    if result.url and result.url not in seen_web_urls:
                        seen_web_urls.add(result.url)
                        web_citations.append(WebCitation.from_raw(result.to_dict()))
                web_citations_payload = [wc.to_dict() for wc in web_citations]

                # 6. Persist assistant message
                asst_msg = await ConversationService.create_message(
                    db=db,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    data=MessageCreate(
                        role="assistant",
                        content=full_raw_text,
                        citations=[],
                    ),
                )

                total_elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
                stage_metrics = cls._emit_stage_metrics(
                    retrieval_response,
                    evidence_gate_decision="insufficient",
                    evidence_gate_ms=evidence_gate_ms,
                    web_search_ms=web_search_ms,
                    llm_ttft_ms=llm_ttft_ms,
                    generation_ms=generation_ms,
                    used_web_fallback=True,
                    provider=provider,
                    total_request_latency_ms=total_elapsed_ms,
                )
                logger.info(
                    "[Phase 9] Tavily-grounded generation completed in %.2fms: %d web sources, msg=%s",
                    total_elapsed_ms,
                    len(web_citations),
                    asst_msg.id,
                )

                done_payload = {
                    "message_id": str(asst_msg.id),
                    "content": full_raw_text,
                    "citations": [],
                    "web_sources": web_citations_payload,
                    "has_sufficient_evidence": False,
                    "used_web_fallback": True,
                    "routing_path": retrieval_response.routing_path,
                    "total_elapsed_ms": total_elapsed_ms,
                    "stage_metrics": stage_metrics,
                }
                yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"
                return

            # --- No web fallback (disabled): emit deterministic message ---
            yield f"event: status\ndata: {json.dumps({'status': 'insufficient_evidence', 'message': 'No sufficient evidence found in documents.'})}\n\n"

            fallback_text = INSUFFICIENT_EVIDENCE_MESSAGE
            yield f"event: token\ndata: {json.dumps({'token': fallback_text})}\n\n"

            asst_msg = await ConversationService.create_message(
                db=db,
                conversation_id=conversation_id,
                user_id=user_id,
                data=MessageCreate(role="assistant", content=fallback_text, citations=[]),
            )

            total_elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
            stage_metrics = cls._emit_stage_metrics(
                retrieval_response,
                evidence_gate_decision="insufficient",
                evidence_gate_ms=evidence_gate_ms,
                web_search_ms=web_search_ms,
                llm_ttft_ms=None,
                generation_ms=0,
                used_web_fallback=False,
                total_request_latency_ms=total_elapsed_ms,
            )
            done_payload = {
                "message_id": str(asst_msg.id),
                "content": fallback_text,
                "citations": [],
                "web_sources": [],
                "has_sufficient_evidence": False,
                "routing_path": retrieval_response.routing_path,
                "total_elapsed_ms": total_elapsed_ms,
                "stage_metrics": stage_metrics,
            }
            yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"
            return

        # 7. Resolve real document names for chunks that passed relevance gate
        passed_chunks = [c for c in retrieval_response.results if c.passed_relevance_gate] if evidence_is_answerable else []
        if not passed_chunks and evidence_is_answerable:
            # Fallback to top ranked chunk if none passed explicitly
            passed_chunks = retrieval_response.results[:settings.RERANK_TOP_K]

        if solver_result is not None and solver_result.success and task is not None and task.source_chunk_ids:
            passed_chunks = [chunk for chunk in passed_chunks if chunk.chunk_id in task.source_chunk_ids]
        elif solver_result is not None and task is None and source == "retrieved_document":
            # Do not let an adjacent exercise stand in for the requested one.
            passed_chunks = []

        doc_ids = {c.document_id for c in passed_chunks}
        doc_names = await cls.resolve_document_names(db, doc_ids, user_id)

        # 8. Assemble grounded prompt
        assembled = PromptBuilder.assemble(
            query=query,
            evidence_chunks=passed_chunks,
            document_names=doc_names,
            chat_mode=chat_mode,
            conversation_history=history_list,
            solver_result=solver_result,
            solver_used=solver_used,
            solver_operation=solver_result.problem_type if solver_result else None,
            solver_verified=solver_result.success if solver_result else False,
        )

        yield f"event: status\ndata: {json.dumps({'status': 'generating', 'message': 'Formulating grounded answer...'})}\n\n"

        # 9. Invoke LLM streaming
        provider = llm_provider or get_llm_provider()
        accumulated_tokens: List[str] = []
        generation_started = time.perf_counter()
        llm_ttft_ms = None

        async def response_tokens():
            if solver_result is not None and solver_result.success and chat_mode != ChatMode.FULL_SOLUTION:
                # Hint-only output is enforced before any token reaches the client.
                yield await generate_guidance(provider, assembled, solver_result, chat_mode, query)
                return
            if solver_result is not None and not solver_result.success:
                yield "Calculation verification was unavailable. "
            async for token in provider.generate_stream(
                prompt=assembled.prompt, system_instruction=assembled.system_instruction,
            ):
                yield token
            if solver_result is not None and solver_result.success:
                yield "\n\n" + verified_final(solver_result)
                if solver_result.notes:
                    yield "\n" + " ".join(solver_result.notes)

        try:
            async for token in response_tokens():
                if token:
                    if llm_ttft_ms is None:
                        llm_ttft_ms = (time.perf_counter() - generation_started) * 1000
                    accumulated_tokens.append(token)
                    yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
        except LLMError as le:
            logger.error("LLM generation provider error: %s", le.message)
            yield f"event: error\ndata: {json.dumps({'error': le.message})}\n\n"
            return
        except Exception as e:
            logger.error("Unexpected error during LLM streaming: %s", e, exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': 'An error occurred during response generation.'})}\n\n"
            return
        generation_ms = (time.perf_counter() - generation_started) * 1000

        full_raw_text = "".join(accumulated_tokens).strip()

        # 10. Extract and validate citations
        citations, normalized_content = CitationService.extract_and_validate_citations(
            text=full_raw_text,
            sources=assembled.sources,
        )
        citations_payload = [c.to_dict() for c in citations]

        # 11. Persist assistant message in PostgreSQL
        asst_msg = await ConversationService.create_message(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
            data=MessageCreate(
                role="assistant",
                content=normalized_content,
                citations=citations_payload,
            ),
        )

        total_elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
        stage_metrics = cls._emit_stage_metrics(
            retrieval_response,
            evidence_gate_decision="sufficient" if evidence_is_answerable else "insufficient",
            evidence_gate_ms=evidence_gate_ms,
            web_search_ms=web_search_ms,
            llm_ttft_ms=llm_ttft_ms,
            generation_ms=generation_ms,
            used_web_fallback=False,
            provider=provider,
            total_request_latency_ms=total_elapsed_ms,
        )
        logger.info(
            "RAG generation completed successfully in %.2fms: %d citations, msg=%s",
            total_elapsed_ms,
            len(citations),
            asst_msg.id,
        )

        # 12. Emit terminal done event
        done_payload = {
            "message_id": str(asst_msg.id),
            "content": normalized_content,
            "citations": citations_payload,
            "has_sufficient_evidence": evidence_is_answerable,
            "routing_path": retrieval_response.routing_path,
            "total_elapsed_ms": total_elapsed_ms,
            "stage_metrics": stage_metrics,
        }
        if solver_metadata is not None:
            done_payload["solver"] = solver_metadata.model_dump()
        yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"

    @classmethod
    async def generate_chat(
        cls,
        db: Optional[AsyncSession],
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        query: str,
        chat_mode_str: str = "Detailed Guidance",
        workspace_id: Optional[uuid.UUID] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
    ) -> Dict[str, Any]:
        """Non-streaming generation helper for tests and automated benchmarks."""
        full_text = ""
        citations: List[Dict[str, Any]] = []
        message_id: Optional[str] = None
        has_evidence = False
        solver_metadata = None

        async for event_chunk in cls.stream_chat(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
            query=query,
            chat_mode_str=chat_mode_str,
            workspace_id=workspace_id,
            llm_provider=llm_provider,
        ):
            for line in event_chunk.split("\n"):
                line = line.strip()
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    try:
                        data = json.loads(data_str)
                        if "message_id" in data:
                            message_id = data["message_id"]
                            full_text = data.get("content", full_text)
                            citations = data.get("citations", [])
                            has_evidence = data.get("has_sufficient_evidence", False)
                            solver_metadata = data.get("solver")
                        elif "error" in data:
                            raise RuntimeError(data["error"])
                    except json.JSONDecodeError:
                        pass

        return {
            "message_id": message_id,
            "content": full_text,
            "citations": citations,
            "has_sufficient_evidence": has_evidence,
            "solver": solver_metadata,
        }
