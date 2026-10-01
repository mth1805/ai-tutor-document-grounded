"""Tests for Phase 6 pgvector migration syntax, dimension, HNSW index, and permissions."""
from pathlib import Path

MIGRATION_FILE = (
    Path(__file__).resolve().parents[2]
    / "supabase"
    / "migrations"
    / "20261001060000_bge_m3_pgvector_embeddings.sql"
)


def test_migration_file_exists():
    """Verify the Phase 6 migration file exists in the migrations directory."""
    assert MIGRATION_FILE.is_file(), f"Migration file missing: {MIGRATION_FILE}"


def test_pgvector_extension_enabled():
    """Verify safe pgvector extension activation."""
    content = MIGRATION_FILE.read_text(encoding="utf-8")
    assert "CREATE EXTENSION IF NOT EXISTS vector;" in content


def test_embedding_vector_dimension():
    """Verify BGE-M3 exact vector dimension 1024 is specified on document_chunks."""
    content = MIGRATION_FILE.read_text(encoding="utf-8")
    assert "vector(1024)" in content
    assert "embedding vector(1024)" in content
    assert "embedding_model TEXT" in content
    assert "embedding_version TEXT" in content
    assert "embedded_at TIMESTAMPTZ" in content


def test_hnsw_cosine_index():
    """Verify HNSW index with cosine distance operator class is defined with standard parameters."""
    content = MIGRATION_FILE.read_text(encoding="utf-8")
    assert "idx_chunks_embedding" in content
    assert "USING hnsw (embedding vector_cosine_ops)" in content
    assert "m = 16" in content
    assert "ef_construction = 64" in content


def test_embedding_lifecycle_columns_and_constraints():
    """Verify documents table contains embedding status lifecycle columns and check constraint."""
    content = MIGRATION_FILE.read_text(encoding="utf-8")
    assert "embedding_status" in content
    assert "embedding_started_at TIMESTAMPTZ" in content
    assert "embedded_at TIMESTAMPTZ" in content
    assert "embedding_error TEXT" in content
    assert "CHECK (embedding_status IN ('pending', 'processing', 'completed', 'failed'))" in content
    assert "idx_documents_embedding_status" in content


def test_authenticated_grants_preserved():
    """Verify permissions for authenticated role are explicitly granted."""
    content = MIGRATION_FILE.read_text(encoding="utf-8")
    assert "GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.document_chunks TO authenticated;" in content
