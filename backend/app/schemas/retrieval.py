"""Schemas for Phase 7 Hybrid Retrieval + Cross-Encoder Reranking."""
import uuid
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator


class RetrievalRoutingMode(str, Enum):
    ALWAYS_FAST = "always_fast"
    ALWAYS_QUALITY = "always_quality"
    ADAPTIVE = "adaptive"



class RetrievalRequest(BaseModel):
    """Request payload for hybrid retrieval and reranking search."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User search query to retrieve relevant document chunks for.",
    )
    dense_top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Maximum candidates retrieved via dense vector similarity search.",
    )
    lexical_top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Maximum candidates retrieved via PostgreSQL full-text search.",
    )
    rrf_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=200,
        description="Reciprocal Rank Fusion smoothing constant (default: 60).",
    )
    candidate_pool_size: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Maximum number of candidates preserved after RRF before reranking.",
    )
    rerank_top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Final number of top reranked chunks to return.",
    )
    relevance_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum Cross-Encoder score required to pass relevance gate.",
    )
    routing_mode: Optional[str] = Field(
        default=None,
        description="Routing policy override ('always_fast', 'always_quality', 'adaptive').",
    )

    @field_validator("query")
    @classmethod
    def validate_non_whitespace_query(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query cannot be empty or contain only whitespace.")
        return stripped


class RetrievedChunk(BaseModel):
    """Ranked evidence document chunk with provenance and retrieval channel scores."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    content: str
    page_number_start: int
    page_number_end: int
    chunk_index: int
    dense_score: Optional[float] = None
    lexical_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    final_rank: int
    passed_relevance_gate: bool
    retrieval_sources: List[str] = Field(
        default_factory=list,
        description="Retrieval channels from which this chunk was retrieved (['dense'], ['lexical'], or both).",
    )

    model_config = ConfigDict(from_attributes=True)


class RetrievalTimingMetrics(BaseModel):
    """Fine-grained latency instrumentation for each stage of the retrieval pipeline."""

    query_embedding_ms: float = Field(..., description="Time taken to compute BGE-M3 query embedding.")
    dense_retrieval_ms: float = Field(..., description="Time taken for pgvector cosine similarity search.")
    lexical_retrieval_ms: float = Field(..., description="Time taken for PostgreSQL tsvector BM25 search.")
    rrf_ms: float = Field(..., description="Time taken for Reciprocal Rank Fusion.")
    rerank_ms: float = Field(..., description="Time taken for Cross-Encoder inference.")
    total_retrieval_ms: float = Field(..., description="End-to-end retrieval latency in milliseconds.")


class RetrievalResponse(BaseModel):
    """Response payload containing ranked, relevance-gated document chunks and diagnostics."""

    workspace_id: uuid.UUID
    query: str
    results: List[RetrievedChunk]
    total_results: int
    has_sufficient_evidence: bool = Field(
        ...,
        description="True if at least one candidate chunk passed the calibrated relevance threshold.",
    )
    relevance_threshold: float
    routing_path: Optional[str] = Field(
        default=None,
        description="Selected retrieval path ('FAST' or 'QUALITY').",
    )
    timings: RetrievalTimingMetrics
    diagnostics: Optional[Dict[str, Any]] = None
