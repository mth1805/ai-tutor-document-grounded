"""Tests for Multilingual (Vietnamese + English) Lexical Search and Cover-Density Ranking."""
import uuid
import pytest
from datetime import datetime, timezone

from app.models.chunk import DocumentChunk
from app.services.ingestion_service import _IN_MEMORY_CHUNKS
from app.services.retrieval_service import RetrievalService


@pytest.fixture(autouse=True)
def clean_in_memory_chunks():
    _IN_MEMORY_CHUNKS.clear()
    yield
    _IN_MEMORY_CHUNKS.clear()


def _add_chunk(workspace_id: uuid.UUID, user_id: uuid.UUID, content: str, chunk_index: int = 0) -> DocumentChunk:
    doc_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    chunk = DocumentChunk(
        id=uuid.uuid4(),
        document_id=doc_id,
        workspace_id=workspace_id,
        user_id=user_id,
        chunk_index=chunk_index,
        content=content,
        page_number_start=1,
        page_number_end=1,
        token_count=len(content.split()),
        created_at=now,
        updated_at=now,
    )
    _IN_MEMORY_CHUNKS.setdefault(doc_id, []).append(chunk)
    return chunk


@pytest.mark.asyncio
async def test_english_query_and_document():
    """Verify pure English document and query lexical retrieval."""
    ws_id = uuid.uuid4()
    user_id = uuid.uuid4()

    target = _add_chunk(
        ws_id,
        user_id,
        "Deep neural networks are composed of multiple layers of computational units.",
    )
    _add_chunk(
        ws_id,
        user_id,
        "Photosynthesis allows green plants to convert solar radiation into sugar.",
    )

    results = await RetrievalService.lexical_retrieve(
        db=None,
        workspace_id=ws_id,
        user_id=user_id,
        query="neural networks layers",
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].chunk_id == target.id
    assert results[0].lexical_score is not None
    assert results[0].lexical_score > 0.0


@pytest.mark.asyncio
async def test_vietnamese_query_and_document():
    """Verify pure Vietnamese document and query with accents and diacritics."""
    ws_id = uuid.uuid4()
    user_id = uuid.uuid4()

    target = _add_chunk(
        ws_id,
        user_id,
        "Trí tuệ nhân tạo và học máy đang được nghiên cứu sâu rộng tại các trường đại học.",
    )
    _add_chunk(
        ws_id,
        user_id,
        "Giải tích toán học và đại số tuyến tính là nền tảng của khoa học dữ liệu.",
    )

    results = await RetrievalService.lexical_retrieve(
        db=None,
        workspace_id=ws_id,
        user_id=user_id,
        query="trí tuệ nhân tạo nghiên cứu",
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].chunk_id == target.id
    assert results[0].lexical_score is not None
    assert results[0].lexical_score > 0.0


@pytest.mark.asyncio
async def test_vietnamese_query_with_potential_english_stopwords():
    """Verify Vietnamese words that overlap with English stopwords (e.g. 'in', 'an') are preserved."""
    ws_id = uuid.uuid4()
    user_id = uuid.uuid4()

    # 'in ấn' means printing, 'an ninh' means security
    target = _add_chunk(
        ws_id,
        user_id,
        "Quy trình in ấn giáo trình và đảm bảo an ninh mạng trong nhà trường.",
    )
    _add_chunk(
        ws_id,
        user_id,
        "Lịch sử địa lý thế giới và sự hình thành các nền văn minh cổ đại.",
    )

    results = await RetrievalService.lexical_retrieve(
        db=None,
        workspace_id=ws_id,
        user_id=user_id,
        query="in ấn an ninh",
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].chunk_id == target.id
    assert results[0].lexical_score is not None
    assert results[0].lexical_score > 0.0


@pytest.mark.asyncio
async def test_mixed_vietnamese_and_english():
    """Verify queries containing mixed English and Vietnamese terms match accurately."""
    ws_id = uuid.uuid4()
    user_id = uuid.uuid4()

    target = _add_chunk(
        ws_id,
        user_id,
        "Mô hình Machine Learning và Deep Learning được ứng dụng trong giáo dục thông minh.",
    )
    _add_chunk(
        ws_id,
        user_id,
        "Vật lý lượng tử giải thích các hạt vi mô và tương tác photon.",
    )

    results = await RetrievalService.lexical_retrieve(
        db=None,
        workspace_id=ws_id,
        user_id=user_id,
        query="Machine Learning trong giáo dục",
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].chunk_id == target.id
    assert results[0].lexical_score is not None


@pytest.mark.asyncio
async def test_cover_density_proximity_ranking():
    """Verify cover-density ranking gives higher score when query words are adjacent vs separated."""
    ws_id = uuid.uuid4()
    user_id = uuid.uuid4()

    # Chunk A: adjacent words ("trí tuệ")
    chunk_dense = _add_chunk(
        ws_id,
        user_id,
        "Hệ thống trí tuệ nhân tạo hiện đại giải quyết bài toán phức tạp nhanh chóng.",
        chunk_index=0,
    )
    # Chunk B: separated words ("trí ... tuệ")
    chunk_separated = _add_chunk(
        ws_id,
        user_id,
        "Trí nhớ của con người rất phong phú nhưng tuệ giác cần trải nghiệm để tích lũy dần theo năm tháng.",
        chunk_index=1,
    )

    results = await RetrievalService.lexical_retrieve(
        db=None,
        workspace_id=ws_id,
        user_id=user_id,
        query="trí tuệ",
        top_k=5,
    )

    assert len(results) == 2
    # The chunk with "trí tuệ" adjacent should receive higher cover-density score
    assert results[0].chunk_id == chunk_dense.id
    assert results[0].lexical_score > results[1].lexical_score


@pytest.mark.asyncio
async def test_no_match_behavior():
    """Verify lexical retrieval gracefully returns an empty list when no terms match."""
    ws_id = uuid.uuid4()
    user_id = uuid.uuid4()

    _add_chunk(
        ws_id,
        user_id,
        "Đại số tuyến tính và ma trận nghịch đảo.",
    )

    results = await RetrievalService.lexical_retrieve(
        db=None,
        workspace_id=ws_id,
        user_id=user_id,
        query="khí hậu toàn cầu sa mạc",
        top_k=5,
    )

    assert results == []
