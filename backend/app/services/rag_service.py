"""RAG Service coordinating Phase 7 retrieval, relevance gating, prompt assembly,

LLM token streaming, citation extraction, and message persistence.
"""
import json
import logging
import time
import uuid
from typing import AsyncIterator, Dict, List, Optional, Set, Any
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.document import Document
from app.models.message import Message
from app.schemas.message import MessageCreate
from app.schemas.retrieval import RetrievalRequest, RetrievalResponse, RetrievedChunk
from app.services.conversation_service import ConversationService
from app.services.retrieval_service import RetrievalService
from app.services.document_service import DocumentService, _IN_MEMORY_DOCUMENTS
from app.llm import get_llm_provider, BaseLLMProvider, LLMError
from app.rag.prompt_builder import PromptBuilder, ChatMode, SourceEvidence
from app.rag.citation_service import CitationService, Citation

logger = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I couldn't find enough information in the uploaded documents to answer that confidently. "
    "Please ensure the relevant course materials or notes are uploaded and indexed."
)


class RAGService:
    """Orchestrates document-grounded answer generation with strict evidence gating."""

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

        # 6. Strict Evidence Gating
        if not retrieval_response.has_sufficient_evidence or not retrieval_response.results:
            logger.info(
                "Insufficient document evidence for query in conv %s. Yielding deterministic fallback.",
                conversation_id,
            )
            # Emit status indicating lack of evidence
            yield f"event: status\ndata: {json.dumps({'status': 'insufficient_evidence', 'message': 'No sufficient evidence found in documents.'})}\n\n"

            # Stream deterministic answer token by token (or chunked)
            fallback_text = INSUFFICIENT_EVIDENCE_MESSAGE
            yield f"event: token\ndata: {json.dumps({'token': fallback_text})}\n\n"

            # Persist assistant message with empty citations
            asst_msg = await ConversationService.create_message(
                db=db,
                conversation_id=conversation_id,
                user_id=user_id,
                data=MessageCreate(role="assistant", content=fallback_text, citations=[]),
            )

            done_payload = {
                "message_id": str(asst_msg.id),
                "content": fallback_text,
                "citations": [],
                "has_sufficient_evidence": False,
                "routing_path": retrieval_response.routing_path,
                "total_elapsed_ms": round((time.perf_counter() - t_start) * 1000, 2),
            }
            yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"
            return

        # 7. Resolve real document names for chunks that passed relevance gate
        passed_chunks = [c for c in retrieval_response.results if c.passed_relevance_gate]
        if not passed_chunks:
            # Fallback to top ranked chunk if none passed explicitly
            passed_chunks = retrieval_response.results[:settings.RERANK_TOP_K]

        doc_ids = {c.document_id for c in passed_chunks}
        doc_names = await cls.resolve_document_names(db, doc_ids, user_id)

        # 8. Assemble grounded prompt
        assembled = PromptBuilder.assemble(
            query=query,
            evidence_chunks=passed_chunks,
            document_names=doc_names,
            chat_mode=chat_mode,
            conversation_history=history_list,
        )

        yield f"event: status\ndata: {json.dumps({'status': 'generating', 'message': 'Formulating grounded answer...'})}\n\n"

        # 9. Invoke LLM streaming
        provider = llm_provider or get_llm_provider()
        accumulated_tokens: List[str] = []

        try:
            async for token in provider.generate_stream(
                prompt=assembled.prompt,
                system_instruction=assembled.system_instruction,
            ):
                if token:
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
            "has_sufficient_evidence": True,
            "routing_path": retrieval_response.routing_path,
            "total_elapsed_ms": total_elapsed_ms,
        }
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
                        elif "error" in data:
                            raise RuntimeError(data["error"])
                    except json.JSONDecodeError:
                        pass

        return {
            "message_id": message_id,
            "content": full_text,
            "citations": citations,
            "has_sufficient_evidence": has_evidence,
        }
