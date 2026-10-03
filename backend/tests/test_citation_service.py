"""Unit tests for Phase 8 citation extraction and validation service."""
import uuid
import pytest
from app.rag.prompt_builder import SourceEvidence
from app.rag.citation_service import CitationService, Citation


def test_citation_extraction_and_validation():
    doc1_id = uuid.uuid4()
    chunk1_id = uuid.uuid4()
    doc2_id = uuid.uuid4()
    chunk2_id = uuid.uuid4()

    sources = [
        SourceEvidence(
            source_index=1,
            document_id=doc1_id,
            document_name="mechanics.pdf",
            chunk_id=chunk1_id,
            page_start=15,
            page_end=15,
            content="Newton's second law relates force to mass and acceleration: F = ma.",
        ),
        SourceEvidence(
            source_index=2,
            document_id=doc2_id,
            document_name="thermodynamics.pdf",
            chunk_id=chunk2_id,
            page_start=3,
            page_end=4,
            content="Entropy of an isolated system always increases over time.",
        ),
    ]

    # LLM text with valid source tags and an invalid hallucinated tag [Source 99]
    generated_text = (
        "Force is defined as F = ma [Source 1]. "
        "Entropy always increases [Source 2]. "
        "Also some fabricated fact [Source 99]."
    )

    citations, normalized = CitationService.extract_and_validate_citations(
        text=generated_text,
        sources=sources,
    )

    # Valid citations extracted
    assert len(citations) == 2
    assert citations[0].document_name == "mechanics.pdf"
    assert citations[0].chunk_id == chunk1_id
    assert citations[0].page_start == 15
    assert citations[0].page_end == 15

    assert citations[1].document_name == "thermodynamics.pdf"
    assert citations[1].chunk_id == chunk2_id
    assert citations[1].page_start == 3
    assert citations[1].page_end == 4

    # Normalized text replaced [Source 1] and [Source 2] with [Doc: ...] and stripped [Source 99]
    assert "[Doc: mechanics.pdf, p. 15]" in normalized
    assert "[Doc: thermodynamics.pdf, pp. 3-4]" in normalized
    assert "[Source 99]" not in normalized
    assert "[Doc: 99" not in normalized


def test_citation_deduplication():
    doc_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    sources = [
        SourceEvidence(
            source_index=1,
            document_id=doc_id,
            document_name="notes.pdf",
            chunk_id=chunk_id,
            page_start=1,
            page_end=1,
            content="Content here.",
        )
    ]

    text = "First claim [Source 1]. Second claim [Source 1]."
    citations, normalized = CitationService.extract_and_validate_citations(text, sources)

    # Only 1 unique structured citation emitted
    assert len(citations) == 1
    assert citations[0].document_name == "notes.pdf"


def test_citation_direct_doc_marker_parsing():
    doc_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    sources = [
        SourceEvidence(
            source_index=1,
            document_id=doc_id,
            document_name="calculus.pdf",
            chunk_id=chunk_id,
            page_start=8,
            page_end=9,
            content="Derivatives represent instantaneous rates of change.",
        )
    ]

    text = "Derivatives measure rates of change [Doc: calculus.pdf, p. 8]."
    citations, _ = CitationService.extract_and_validate_citations(text, sources)

    assert len(citations) == 1
    assert citations[0].chunk_id == chunk_id
    assert citations[0].document_name == "calculus.pdf"
