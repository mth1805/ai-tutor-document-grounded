"""Tests for Phase 7 full-text search migration syntax, GIN index, and permissions."""
from pathlib import Path

MIGRATION_FILE_BASELINE = (
    Path(__file__).resolve().parents[2]
    / "supabase"
    / "migrations"
    / "20261001070000_document_chunks_fts_bm25.sql"
)

MIGRATION_FILE_MULTILINGUAL = (
    Path(__file__).resolve().parents[2]
    / "supabase"
    / "migrations"
    / "20261001080000_document_chunks_multilingual_fts.sql"
)


def test_fts_migration_files_exist():
    """Verify both Phase 7 baseline and multilingual migration files exist."""
    assert MIGRATION_FILE_BASELINE.is_file(), f"Migration file missing: {MIGRATION_FILE_BASELINE}"
    assert MIGRATION_FILE_MULTILINGUAL.is_file(), f"Migration file missing: {MIGRATION_FILE_MULTILINGUAL}"


def test_multilingual_tsvector_column_definition():
    """Verify multilingual tsvector combining 'simple' and 'english' is defined."""
    content = MIGRATION_FILE_MULTILINGUAL.read_text(encoding="utf-8")
    assert "tsv tsvector" in content
    assert "to_tsvector('simple', coalesce(content, ''))" in content
    assert "to_tsvector('english', coalesce(content, ''))" in content
    assert "idx_chunks_tsv" in content
    assert "USING gin(tsv)" in content


def test_authenticated_grants_reaffirmed():
    """Verify permissions for authenticated role are explicitly granted in migrations."""
    for mig in [MIGRATION_FILE_BASELINE, MIGRATION_FILE_MULTILINGUAL]:
        content = mig.read_text(encoding="utf-8")
        assert (
            "GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.document_chunks TO authenticated;"
            in content
        )
