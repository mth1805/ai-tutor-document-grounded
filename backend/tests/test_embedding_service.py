"""Unit and service tests for EmbeddingService."""
import uuid
import pytest
from datetime import datetime, timezone

from app.models.document import Document
from app.models.chunk import DocumentChunk
from app.services.embedding_service import EmbeddingService
from app.services.document_service import _IN_MEMORY_DOCUMENTS
from app.services.ingestion_service import _IN_MEMORY_CHUNKS
from app.ml.mock_provider import MockEmbeddingProvider


@pytest.fixture(autouse=True)
def clean_in_memory_state():
    """Clear in-memory storage before and after each test."""
    _IN_MEMORY_DOCUMENTS.clear()
    _IN_MEMORY_CHUNKS.clear()
    yield
    _IN_MEMORY_DOCUMENTS.clear()
    _IN_MEMORY_CHUNKS.clear()


@pytest.mark.asyncio
async def test_embed_document_happy_path():
    """Verify that a processed document embeds chunks and transitions status to 'completed'."""
    doc_id = uuid.uuid4()
    ws_id = uuid.uuid4()
    user_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    # 1. Create a processed document in memory
    doc = Document(
        id=doc_id,
        workspace_id=ws_id,
        user_id=user_id,
        original_filename="notes.pdf",
        storage_path=f"{user_id}/{ws_id}/{doc_id}.pdf",
        mime_type="application/pdf",
        file_size=1024,
        status="processed",
        embedding_status="pending",
        created_at=now,
        updated_at=now,
        processed_at=now,
    )
    _IN_MEMORY_DOCUMENTS[doc_id] = doc

    # 2. Add chunks
    chunks = [
        DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc_id,
            workspace_id=ws_id,
            user_id=user_id,
            chunk_index=0,
            content="First chunk content for embedding.",
            page_number_start=1,
            page_number_end=1,
            token_count=10,
            created_at=now,
            updated_at=now,
        ),
        DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc_id,
            workspace_id=ws_id,
            user_id=user_id,
            chunk_index=1,
            content="Second chunk content for embedding.",
            page_number_start=1,
            page_number_end=2,
            token_count=12,
            created_at=now,
            updated_at=now,
        ),
    ]
    _IN_MEMORY_CHUNKS[doc_id] = chunks

    # 3. Execute embedding
    provider = MockEmbeddingProvider()
    success = await EmbeddingService.embed_document(
        db=None,
        document_id=doc_id,
        user_id=user_id,
        provider=provider,
    )

    assert success is True
    assert doc.embedding_status == "completed"
    assert doc.embedded_at is not None
    assert doc.embedding_error is None

    # Verify chunks have embeddings
    persisted_chunks = _IN_MEMORY_CHUNKS[doc_id]
    assert len(persisted_chunks) == 2
    for c in persisted_chunks:
        assert c.embedding is not None
        assert len(c.embedding) == 1024
        assert c.embedding_model == "BAAI/bge-m3"
        assert c.embedded_at is not None


@pytest.mark.asyncio
async def test_embed_document_requires_processed_status():
    """Verify that attempting to embed an unprocessed or failed document raises ValueError."""
    doc_id = uuid.uuid4()
    user_id = uuid.uuid4()
    ws_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    # Document in 'uploaded' state
    doc = Document(
        id=doc_id,
        workspace_id=ws_id,
        user_id=user_id,
        original_filename="raw.pdf",
        storage_path="path",
        mime_type="application/pdf",
        file_size=500,
        status="uploaded",
        embedding_status="pending",
        created_at=now,
        updated_at=now,
    )
    _IN_MEMORY_DOCUMENTS[doc_id] = doc

    with pytest.raises(ValueError, match="must be in 'processed' state"):
        await EmbeddingService.embed_document(
            db=None,
            document_id=doc_id,
            user_id=user_id,
        )

    # Document in 'failed' state
    doc.status = "failed"
    with pytest.raises(ValueError, match="must be in 'processed' state"):
        await EmbeddingService.embed_document(
            db=None,
            document_id=doc_id,
            user_id=user_id,
        )


@pytest.mark.asyncio
async def test_idempotent_reembedding():
    """Verify that re-embedding a document updates canonical chunks in place without creating duplicate records."""
    doc_id = uuid.uuid4()
    user_id = uuid.uuid4()
    ws_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    doc = Document(
        id=doc_id,
        workspace_id=ws_id,
        user_id=user_id,
        original_filename="idem.pdf",
        storage_path="path",
        mime_type="application/pdf",
        file_size=500,
        status="processed",
        embedding_status="completed",
        created_at=now,
        updated_at=now,
        processed_at=now,
    )
    _IN_MEMORY_DOCUMENTS[doc_id] = doc

    chunk_id = uuid.uuid4()
    _IN_MEMORY_CHUNKS[doc_id] = [
        DocumentChunk(
            id=chunk_id,
            document_id=doc_id,
            workspace_id=ws_id,
            user_id=user_id,
            chunk_index=0,
            content="Idempotency test content.",
            page_number_start=1,
            page_number_end=1,
            token_count=5,
            created_at=now,
            updated_at=now,
        )
    ]

    provider = MockEmbeddingProvider()

    # Pass 1
    await EmbeddingService.embed_document(None, doc_id, user_id, provider=provider)
    first_vec = _IN_MEMORY_CHUNKS[doc_id][0].embedding
    assert len(_IN_MEMORY_CHUNKS[doc_id]) == 1
    assert _IN_MEMORY_CHUNKS[doc_id][0].id == chunk_id

    # Pass 2
    await EmbeddingService.embed_document(None, doc_id, user_id, provider=provider)
    second_vec = _IN_MEMORY_CHUNKS[doc_id][0].embedding
    assert len(_IN_MEMORY_CHUNKS[doc_id]) == 1
    assert _IN_MEMORY_CHUNKS[doc_id][0].id == chunk_id
    assert first_vec == second_vec


@pytest.mark.asyncio
async def test_embedding_failure_transitions_to_failed_state():
    """Verify that provider failure marks embedding_status as 'failed' and records error message."""
    doc_id = uuid.uuid4()
    user_id = uuid.uuid4()
    ws_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    doc = Document(
        id=doc_id,
        workspace_id=ws_id,
        user_id=user_id,
        original_filename="fail.pdf",
        storage_path="path",
        mime_type="application/pdf",
        file_size=500,
        status="processed",
        embedding_status="pending",
        created_at=now,
        updated_at=now,
        processed_at=now,
    )
    _IN_MEMORY_DOCUMENTS[doc_id] = doc

    _IN_MEMORY_CHUNKS[doc_id] = [
        DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc_id,
            workspace_id=ws_id,
            user_id=user_id,
            chunk_index=0,
            content="Failing chunk.",
            page_number_start=1,
            page_number_end=1,
            token_count=5,
            created_at=now,
            updated_at=now,
        )
    ]

    # Create a broken provider that raises RuntimeError
    class BrokenProvider(MockEmbeddingProvider):
        def encode_batch(self, *args, **kwargs):
            raise RuntimeError("CUDA OOM or GPU driver crash")

    success = await EmbeddingService.embed_document(
        None, doc_id, user_id, provider=BrokenProvider()
    )

    assert success is False
    assert doc.embedding_status == "failed"
    assert "CUDA OOM" in (doc.embedding_error or "")


@pytest.mark.asyncio
async def test_cross_user_embedding_denied():
    """Verify that User B cannot embed User A's document."""
    user_a = uuid.uuid4()
    user_b = uuid.uuid4()
    doc_id = uuid.uuid4()
    ws_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    doc = Document(
        id=doc_id,
        workspace_id=ws_id,
        user_id=user_a,
        original_filename="secret.pdf",
        storage_path="path",
        mime_type="application/pdf",
        file_size=500,
        status="processed",
        embedding_status="pending",
        created_at=now,
        updated_at=now,
        processed_at=now,
    )
    _IN_MEMORY_DOCUMENTS[doc_id] = doc

    success = await EmbeddingService.embed_document(
        None, doc_id, user_b, provider=MockEmbeddingProvider()
    )
    assert success is False
    assert doc.embedding_status == "pending"


@pytest.mark.asyncio
async def test_get_embedding_status():
    """Verify get_embedding_status returns chunk counts and accurate status."""
    doc_id = uuid.uuid4()
    user_id = uuid.uuid4()
    ws_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    doc = Document(
        id=doc_id,
        workspace_id=ws_id,
        user_id=user_id,
        original_filename="status.pdf",
        storage_path="path",
        mime_type="application/pdf",
        file_size=500,
        status="processed",
        embedding_status="completed",
        created_at=now,
        updated_at=now,
        processed_at=now,
        embedded_at=now,
    )
    _IN_MEMORY_DOCUMENTS[doc_id] = doc

    _IN_MEMORY_CHUNKS[doc_id] = [
        DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc_id,
            workspace_id=ws_id,
            user_id=user_id,
            chunk_index=0,
            content="Sample text",
            page_number_start=1,
            page_number_end=1,
            token_count=5,
            embedding=[0.1] * 1024,
            embedding_model="BAAI/bge-m3",
            embedded_at=now,
            created_at=now,
            updated_at=now,
        )
    ]

    info = await EmbeddingService.get_embedding_status(None, doc_id, user_id)
    assert info is not None
    assert info["document_id"] == doc_id
    assert info["embedding_status"] == "completed"
    assert info["total_chunks"] == 1
    assert info["embedded_chunks"] == 1
    assert info["embedding_model"] == "BAAI/bge-m3"
