"""Canonical, bounded chat-generation context snapshots.

This module deliberately contains no database reads or RAG decisions. The RAG
preparation layer supplies the exact resolved inputs; this service validates,
serializes, hashes, and atomically accepts those inputs through the C2 RPC.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.services.ingestion.chunker import StructureAwareChunker


SNAPSHOT_VERSION = 1
# Canonical compact JSON is capped at 3.5 MB. The DB separately caps its
# whitespace-bearing JSONB rendering at 4 MiB.
MAX_CANONICAL_SNAPSHOT_BYTES = 3_500_000
DATABASE_SNAPSHOT_BYTES = 4_194_304
MAX_HISTORY_MESSAGES = 6  # PromptBuilder's current default history window.
MAX_EVIDENCE_ITEMS = 50  # RetrievalRequest's maximum rerank_top_k.


class SnapshotHistoryItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant", "system"]
    content: str


class SnapshotPromptCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: str = Field(min_length=1, max_length=48)
    prompt: str | None = None
    max_output_tokens: int = Field(ge=1)
    source_indices: list[int] = Field(default_factory=list)
    group_index: int | None = Field(default=None, ge=1)
    estimated_input_tokens: int | None = Field(default=None, ge=0)


class SnapshotEvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_index: int = Field(ge=1)
    document_id: uuid.UUID
    document_name: str
    chunk_id: uuid.UUID
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    content: str
    token_count: int | None = Field(default=None, ge=0)
    token_count_adjustment: int = Field(default=0, ge=0)
    retrieval_rank: int | None = Field(default=None, ge=1)
    dense_score: float | None = None
    lexical_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


class SnapshotCitationSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_index: int = Field(ge=1)
    document_id: uuid.UUID
    document_name: str = Field(min_length=1)
    chunk_id: uuid.UUID
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)


class GenerationContextSnapshot(BaseModel):
    """All mutable inputs needed to reproduce a generation without rereads."""

    model_config = ConfigDict(extra="forbid")
    original_request: str = Field(min_length=1, max_length=4000)
    normalized_mode: Literal["Light Guidance", "Detailed Guidance", "Full Solution"]
    workspace_id: uuid.UUID | None
    selected_document_ids: list[uuid.UUID] = Field(max_length=50)
    history: list[SnapshotHistoryItem] = Field(max_length=MAX_HISTORY_MESSAGES)
    resolved_document_references: dict[str, Any]
    generation_path: Literal["document", "web_fallback", "whole_document_summary"]
    system_instruction: str
    prompt_plan: list[SnapshotPromptCall] = Field(min_length=1)
    evidence: list[SnapshotEvidenceItem] = Field(max_length=MAX_EVIDENCE_ITEMS)
    routing_decisions: dict[str, Any]
    citation_sources: list[SnapshotCitationSource]
    web_results: list[dict[str, Any]]
    solver_output: dict[str, Any] | None = None
    whole_document_source_chunks: list[SnapshotEvidenceItem]
    whole_document_summary_plan: dict[str, Any] | None


class PreparedSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)
    version: int
    payload: dict[str, Any]
    context_hash: str
    canonical_bytes: int


class SnapshotPreparationError(ValueError):
    """Context is malformed or exceeds an explicit Phase 1 snapshot limit."""


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for hashing and stable acceptance retries."""
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise SnapshotPreparationError("Generation context is not canonical JSON") from exc


def build_whole_document_summary_snapshot_fields(
    chunks: list[Any], document: Any, query: str, provider: Any,
) -> dict[str, Any]:
    """Build a source-once summary plan using DocumentSummaryService's planner."""
    from app.services.document_summary_service import DocumentSummaryService

    ordered = sorted(chunks, key=lambda item: item.chunk_index)
    plan = DocumentSummaryService.build_execution_plan(
        ordered, document, query, provider, include_reduction_estimate=True,
    )
    source_by_index = {
        index: {
            "source_index": index,
            "document_id": str(document.id),
            "document_name": document.original_filename,
            "chunk_id": str(chunk.id),
            "page_start": chunk.page_number_start,
            "page_end": chunk.page_number_end,
            "content": chunk.content.strip(),
            "token_count": chunk.token_count,
            "token_count_adjustment": DocumentSummaryService._persisted_token_adjustment(
                chunk.content, int(chunk.token_count),
            ),
        }
        for index, chunk in enumerate(ordered, start=1)
    }
    groups = [list(indices) for indices in plan.map_source_indices]
    direct = len(plan.map_groups) == 1
    prompt_plan = []
    for index, indices in enumerate(groups, start=1):
        prompt_plan.append({
            "stage": "direct" if direct else f"map_{index}",
            # Source text is reconstructed from source_indices at execution.
            "prompt": None,
            "max_output_tokens": plan.final_output_tokens if direct else plan.map_output_tokens,
            "source_indices": indices,
            "group_index": index,
            "estimated_input_tokens": plan.map_input_token_estimates[index - 1],
        })
    summary_plan: dict[str, Any] = {
        "plan_version": DocumentSummaryService.PLAN_VERSION,
        "prompt_template_version": DocumentSummaryService.PROMPT_TEMPLATE_VERSION,
        "reduction_strategy_version": DocumentSummaryService.REDUCTION_STRATEGY_VERSION,
        "tokenizer_version": DocumentSummaryService.tokenizer_version(),
        "document_id": str(document.id),
        "document_name": document.original_filename,
        "query": query,
        "model_configuration": {
            "model_name": plan.model_name,
            "thinking_level": plan.thinking_level,
            "generation_config_version": "gemini-summary-config-v1",
        },
        "input_budget_tokens": plan.input_budget_tokens,
        "map_output_tokens": plan.map_output_tokens,
        "final_output_tokens": plan.final_output_tokens,
        "map_groups": groups,
        "reduction": {
            "strategy": DocumentSummaryService.REDUCTION_STRATEGY_VERSION,
            "input_budget_tokens": plan.input_budget_tokens,
            "intermediate_output_tokens": plan.map_output_tokens,
            "final_output_tokens": plan.final_output_tokens,
            "output_dependent_grouping": True,
        },
        "estimates": {
            "source_chunk_count": len(ordered),
            "persisted_source_tokens": sum(int(chunk.token_count) for chunk in ordered),
            "map_calls": 0 if direct else len(groups),
            "reduction_calls_at_max_output": plan.estimated_reduce_calls,
            "total_calls_at_max_output": plan.estimated_total_calls,
            "sequential_timeout_seconds_at_estimated_calls": plan.estimated_sequential_timeout_seconds,
            "per_call_timeout_seconds": settings.LLM_STREAMING_TIMEOUT_SECONDS,
        },
    }
    summary_plan["plan_hash"] = hashlib.sha256(canonical_json_bytes(summary_plan)).hexdigest()
    citation_sources = [
        {
            "source_index": index,
            "document_id": item["document_id"],
            "document_name": item["document_name"],
            "chunk_id": item["chunk_id"],
            "page_start": item["page_start"],
            "page_end": item["page_end"],
        }
        for index, item in source_by_index.items()
    ]
    return {
        "system_instruction": plan.system_instruction,
        "prompt_plan": prompt_plan,
        "evidence": [],
        "citation_sources": citation_sources,
        "whole_document_source_chunks": list(source_by_index.values()),
        "whole_document_summary_plan": summary_plan,
    }


def prepare_snapshot(payload: GenerationContextSnapshot | dict[str, Any]) -> PreparedSnapshot:
    """Validate a complete prepared context and compute its immutable identity."""
    try:
        snapshot = payload if isinstance(payload, GenerationContextSnapshot) else GenerationContextSnapshot.model_validate(payload)
    except ValidationError as exc:
        raise SnapshotPreparationError("Generation context is incomplete or exceeds a context limit") from exc
    _validate_snapshot_source_mappings(snapshot)
    counter = StructureAwareChunker()
    max_web_results = min(settings.TAVILY_MAX_RESULTS, settings.WEB_SEARCH_MAX_SOURCES)
    if len(snapshot.web_results) > max_web_results:
        raise SnapshotPreparationError(
            f"Web context exceeds the configured {max_web_results}-result limit"
        )
    for call in snapshot.prompt_plan:
        if snapshot.generation_path == "whole_document_summary":
            stage = call.stage.lower()
            output_budget = (
                min(settings.DOCUMENT_SUMMARY_MAP_OUTPUT_TOKENS, settings.LLM_MAX_OUTPUT_TOKENS)
                if stage.startswith("map") or stage == "reduce_intermediate"
                else settings.LLM_MAX_OUTPUT_TOKENS
            )
        else:
            # Regular document and web chat have no separate configured input
            # token ceiling; do not impose the summary-only budget on them.
            output_budget = settings.CHAT_MAX_OUTPUT_TOKENS
        if call.max_output_tokens > output_budget:
            raise SnapshotPreparationError(
                f"Prompt call '{call.stage}' exceeds its configured {output_budget}-token output budget"
            )
        if snapshot.generation_path != "whole_document_summary" and call.prompt is None:
            raise SnapshotPreparationError("Non-summary prompt calls must store their exact prompt")
    if snapshot.generation_path == "whole_document_summary":
        _validate_summary_plan(snapshot)
    encoded = canonical_json_bytes(snapshot.model_dump(mode="json"))
    if len(encoded) > MAX_CANONICAL_SNAPSHOT_BYTES:
        raise SnapshotPreparationError(
            f"Generation context exceeds the {MAX_CANONICAL_SNAPSHOT_BYTES}-byte snapshot limit"
        )
    return PreparedSnapshot(
        version=SNAPSHOT_VERSION,
        payload=snapshot.model_dump(mode="json"),
        context_hash=hashlib.sha256(encoded).hexdigest(),
        canonical_bytes=len(encoded),
    )


def _validate_summary_plan(snapshot: GenerationContextSnapshot) -> None:
    from app.services.document_summary_service import DocumentSummaryError, DocumentSummaryService

    plan = snapshot.whole_document_summary_plan
    if not isinstance(plan, dict):
        raise SnapshotPreparationError("Whole-document summary plan is missing")
    supplied_hash = plan.get("plan_hash")
    unhashed_plan = {key: value for key, value in plan.items() if key != "plan_hash"}
    if not isinstance(supplied_hash, str) or hashlib.sha256(canonical_json_bytes(unhashed_plan)).hexdigest() != supplied_hash:
        raise SnapshotPreparationError("Whole-document summary plan hash mismatch")
    if plan.get("plan_version") != DocumentSummaryService.PLAN_VERSION:
        raise SnapshotPreparationError("Unsupported whole-document summary plan version")
    if plan.get("prompt_template_version") != DocumentSummaryService.PROMPT_TEMPLATE_VERSION:
        raise SnapshotPreparationError("Unsupported whole-document summary prompt version")
    if plan.get("reduction_strategy_version") != DocumentSummaryService.REDUCTION_STRATEGY_VERSION:
        raise SnapshotPreparationError("Unsupported whole-document reduction strategy")
    if plan.get("tokenizer_version") != DocumentSummaryService.tokenizer_version():
        raise SnapshotPreparationError("Summary plan tokenizer differs from the prepared runtime")
    source_items = [item.model_dump(mode="json") for item in snapshot.whole_document_source_chunks]
    expected_indices = list(range(1, len(source_items) + 1))
    if [item.get("source_index") for item in source_items] != expected_indices:
        raise SnapshotPreparationError("Summary source chunks must be in deterministic source order")
    groups = plan.get("map_groups")
    if not isinstance(groups, list) or not groups:
        raise SnapshotPreparationError("Summary map groups are missing")
    flattened = [source_index for group in groups if isinstance(group, list) for source_index in group]
    if flattened != expected_indices:
        raise SnapshotPreparationError("Summary groups must reference each source chunk once in order")
    if len(snapshot.prompt_plan) != len(groups) or snapshot.evidence:
        raise SnapshotPreparationError("Summary prompts must be source references, not duplicated evidence")
    if any(call.prompt is not None for call in snapshot.prompt_plan):
        raise SnapshotPreparationError("Whole-document summary prompts cannot embed source text")
    if [call.source_indices for call in snapshot.prompt_plan] != groups:
        raise SnapshotPreparationError("Summary prompt references do not match the deterministic groups")
    if plan.get("query") != snapshot.original_request:
        raise SnapshotPreparationError("Summary plan query differs from the accepted request")
    if plan.get("input_budget_tokens") != max(1, int(settings.DOCUMENT_SUMMARY_INPUT_TOKEN_BUDGET * 0.75)):
        raise SnapshotPreparationError("Summary plan input budget differs from the prepared runtime configuration")
    expected_map_output = min(settings.DOCUMENT_SUMMARY_MAP_OUTPUT_TOKENS, settings.LLM_MAX_OUTPUT_TOKENS)
    if (plan.get("map_output_tokens") != expected_map_output
            or plan.get("final_output_tokens") != settings.LLM_MAX_OUTPUT_TOKENS):
        raise SnapshotPreparationError("Summary output budgets differ from the prepared runtime configuration")
    selected_ids = {str(item) for item in snapshot.selected_document_ids}
    if str(plan.get("document_id")) not in selected_ids:
        raise SnapshotPreparationError("Summary document differs from the accepted document selection")
    if any(item.get("document_id") != str(plan.get("document_id"))
           or item.get("document_name") != plan.get("document_name") for item in source_items):
        raise SnapshotPreparationError("Summary source identity differs from the planned document")
    estimates = plan.get("estimates")
    if not isinstance(estimates, dict):
        raise SnapshotPreparationError("Summary execution estimates are missing")
    direct = len(groups) == 1
    expected_map_calls = 0 if direct else len(groups)
    if (estimates.get("source_chunk_count") != len(source_items)
            or estimates.get("map_calls") != expected_map_calls
            or estimates.get("persisted_source_tokens") != sum(int(item.get("token_count") or 0) for item in source_items)):
        raise SnapshotPreparationError("Summary execution estimates do not match the immutable sources")
    reduction_calls = estimates.get("reduction_calls_at_max_output")
    expected_total = (
        None if reduction_calls is None
        else 1 if direct
        else expected_map_calls + int(reduction_calls)
    )
    if (estimates.get("total_calls_at_max_output") != expected_total
            or estimates.get("per_call_timeout_seconds") != settings.LLM_STREAMING_TIMEOUT_SECONDS
            or estimates.get("sequential_timeout_seconds_at_estimated_calls") != (
                None if expected_total is None else expected_total * settings.LLM_STREAMING_TIMEOUT_SECONDS
            )):
        raise SnapshotPreparationError("Summary call/runtime estimates are inconsistent")
    counter = DocumentSummaryService
    for call_index, call in enumerate(snapshot.prompt_plan, start=1):
        final = call.stage == "direct"
        expected_stage = "direct" if len(groups) == 1 else f"map_{call_index}"
        expected_output = (
            settings.LLM_MAX_OUTPUT_TOKENS if final
            else min(settings.DOCUMENT_SUMMARY_MAP_OUTPUT_TOKENS, settings.LLM_MAX_OUTPUT_TOKENS)
        )
        if call.stage != expected_stage or call.max_output_tokens != expected_output or call.group_index != call_index:
            raise SnapshotPreparationError("Summary call configuration differs from the prepared runtime plan")
        try:
            prompt = counter.reconstruct_document_prompt(
                snapshot.original_request, str(plan["document_name"]), source_items,
                call.source_indices, int(call.group_index or 1), final=final,
            )
        except (DocumentSummaryError, KeyError, TypeError, ValueError) as exc:
            raise SnapshotPreparationError("Summary prompt cannot be reconstructed from its source references") from exc
        adjusted_count = counter._prompt_token_count_with_adjustments(
            prompt, snapshot.system_instruction,
            [int(item.get("token_count_adjustment") or 0)
             for item in source_items if int(item["source_index"]) in call.source_indices],
        )
        if call.estimated_input_tokens != adjusted_count:
            raise SnapshotPreparationError(
                f"Summary input-token estimate differs from shared live accounting "
                f"({call.estimated_input_tokens} != {adjusted_count})"
            )
        if adjusted_count > int(plan["input_budget_tokens"]):
            raise SnapshotPreparationError("Summary map input exceeds its captured token budget")


def _validate_snapshot_source_mappings(snapshot: GenerationContextSnapshot) -> None:
    if snapshot.generation_path == "whole_document_summary":
        if snapshot.whole_document_summary_plan is None:
            raise SnapshotPreparationError("Whole-document summary plan is missing")
    elif snapshot.whole_document_source_chunks or snapshot.whole_document_summary_plan is not None:
        raise SnapshotPreparationError("Non-summary snapshots cannot contain whole-document summary inputs")

    sources = [*snapshot.evidence, *snapshot.whole_document_source_chunks]
    source_indexes = [item.source_index for item in sources]
    if len(source_indexes) != len(set(source_indexes)):
        raise SnapshotPreparationError("Evidence source indexes must be unique")
    source_by_index = {item.source_index: item for item in sources}
    citation_indexes: set[int] = set()
    for citation in snapshot.citation_sources:
        source = source_by_index.get(citation.source_index)
        if source is None or citation.source_index in citation_indexes:
            raise SnapshotPreparationError("Citation source does not map uniquely to snapshot evidence")
        citation_indexes.add(citation.source_index)
        if (
            citation.document_id != source.document_id
            or citation.document_name != source.document_name
            or citation.chunk_id != source.chunk_id
            or citation.page_start != source.page_start
            or citation.page_end != source.page_end
        ):
            raise SnapshotPreparationError("Citation metadata differs from its snapshot evidence source")


def verify_snapshot(payload: dict[str, Any], expected_hash: str) -> None:
    """Raise before use if a fetched snapshot differs from its accepted hash."""
    try:
        snapshot = GenerationContextSnapshot.model_validate(payload)
    except ValidationError as exc:
        raise SnapshotPreparationError("Generation context snapshot is invalid") from exc
    _validate_snapshot_source_mappings(snapshot)
    encoded = canonical_json_bytes(snapshot.model_dump(mode="json"))
    if len(encoded) > MAX_CANONICAL_SNAPSHOT_BYTES:
        raise SnapshotPreparationError("Generation context snapshot exceeds the byte limit")
    actual_hash = hashlib.sha256(encoded).hexdigest()
    if not isinstance(expected_hash, str) or actual_hash != expected_hash:
        raise SnapshotPreparationError("Generation context snapshot hash mismatch")
    if snapshot.generation_path == "whole_document_summary":
        _validate_summary_plan(snapshot)


def canonical_request(
    *, content: str, mode: str, workspace_id: uuid.UUID | None,
    document_id: uuid.UUID | None, active_document_id: uuid.UUID | None,
) -> dict[str, Any]:
    """Return the complete request identity bound by the acceptance RPC."""
    return {
        "content": content,
        "mode": mode,
        "workspace_id": str(workspace_id) if workspace_id else None,
        "document_id": str(document_id) if document_id else None,
        "active_document_id": str(active_document_id) if active_document_id else None,
    }


def request_fingerprint(
    *, content: str, mode: str, workspace_id: uuid.UUID | None,
    document_id: uuid.UUID | None, active_document_id: uuid.UUID | None,
) -> str:
    identity = canonical_request(
        content=content, mode=mode, workspace_id=workspace_id,
        document_id=document_id, active_document_id=active_document_id,
    )
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


async def accept_generation_snapshot(
    db: AsyncSession,
    *,
    generation_id: uuid.UUID,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
    expected_conversation_version: int,
    fingerprint: str,
    content: str,
    normalized_mode: str,
    workspace_id: uuid.UUID | None,
    document_id: uuid.UUID | None,
    active_document_id: uuid.UUID | None,
    prepared: PreparedSnapshot,
) -> dict[str, Any]:
    """Create message, run, initial checkpoint, and snapshot in one DB transaction."""
    canonical_snapshot = canonical_json_bytes(prepared.payload).decode("utf-8")
    if hashlib.sha256(canonical_snapshot.encode("utf-8")).hexdigest() != prepared.context_hash:
        raise SnapshotPreparationError("Prepared snapshot hash does not match its canonical JSON")
    expected_fingerprint = request_fingerprint(
        content=content, mode=normalized_mode, workspace_id=workspace_id,
        document_id=document_id, active_document_id=active_document_id,
    )
    if fingerprint != expected_fingerprint:
        raise SnapshotPreparationError("Request fingerprint does not match the accepted request")
    canonical_request_text = canonical_json_bytes(canonical_request(
        content=content, mode=normalized_mode, workspace_id=workspace_id,
        document_id=document_id, active_document_id=active_document_id,
    )).decode("utf-8")
    selected_ids = list(dict.fromkeys(
        str(value) for value in (document_id, active_document_id) if value is not None
    ))
    if (
        prepared.payload.get("original_request") != content
        or prepared.payload.get("normalized_mode") != normalized_mode
        or prepared.payload.get("workspace_id") != (str(workspace_id) if workspace_id else None)
        or prepared.payload.get("selected_document_ids") != selected_ids
    ):
        raise SnapshotPreparationError(
            "Prepared context does not match the accepted request or document selection"
        )
    try:
        row = (await db.execute(text("""
            SELECT * FROM public.create_chat_generation_with_snapshot(
                :generation_id, :conversation_id, :user_id, :expected_version,
                :fingerprint, :content, :mode, :workspace_id, :document_id,
                :active_document_id, :snapshot_version, CAST(:snapshot AS jsonb),
                :context_hash, :canonical_snapshot, :canonical_request
            )
        """), {
            "generation_id": generation_id,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "expected_version": expected_conversation_version,
            "fingerprint": fingerprint,
            "content": content,
            "mode": normalized_mode,
            "workspace_id": workspace_id,
            "document_id": document_id,
            "active_document_id": active_document_id,
            "snapshot_version": prepared.version,
            "snapshot": canonical_snapshot,
            "context_hash": prepared.context_hash,
            "canonical_snapshot": canonical_snapshot,
            "canonical_request": canonical_request_text,
        })).mappings().one()
        await db.commit()
        return dict(row)
    except Exception:
        await db.rollback()
        raise
