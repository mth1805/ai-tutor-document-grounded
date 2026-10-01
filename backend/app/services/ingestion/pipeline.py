"""Ingestion Pipeline orchestrating format parsing, text cleaning, and structure-aware chunking."""
import logging
from typing import List, Optional

from app.services.ingestion.models import DocumentPage, ProcessedChunk
from app.services.ingestion.cleaner import clean_text
from app.services.ingestion.chunker import StructureAwareChunker
from app.services.ingestion.parsers.factory import get_parser_for_mime_or_filename
from app.services.ingestion.ocr.base import OCRProvider
from app.services.ingestion.ocr.tesseract import TesseractOCRProvider

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """End-to-end pipeline transforming raw document bytes into cleaned, page-anchored chunks."""

    def __init__(
        self,
        ocr_provider: Optional[OCRProvider] = None,
        target_chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_size: int = 80,
        ocr_min_chars: int = 50,
    ):
        self.ocr_provider = ocr_provider or TesseractOCRProvider()
        self.chunker = StructureAwareChunker(
            target_chunk_size=target_chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
        )
        self.ocr_min_chars = ocr_min_chars

    def process(
        self,
        content: bytes,
        filename: str,
        mime_type: Optional[str] = None,
    ) -> List[ProcessedChunk]:
        """Executes parsing, page cleaning, and chunking on raw file bytes.

        Args:
            content: Raw binary content of the file.
            filename: Original file name.
            mime_type: Optional MIME type string.

        Returns:
            List of ProcessedChunk objects.
        """
        # 1. Resolve parser
        parser = get_parser_for_mime_or_filename(
            mime_type=mime_type,
            filename=filename,
            ocr_provider=self.ocr_provider,
            min_text_chars=self.ocr_min_chars,
        )

        # 2. Parse file into pages
        pages = parser.parse(content=content, filename=filename)

        # 3. Clean page text
        cleaned_pages: List[DocumentPage] = []
        for p in pages:
            cleaned_text = clean_text(p.text)
            if cleaned_text:
                cleaned_pages.append(
                    DocumentPage(
                        page_number=p.page_number,
                        text=cleaned_text,
                        source_type=p.source_type,
                        used_ocr=p.used_ocr,
                    )
                )

        # 4. Chunk cleaned pages
        chunks = self.chunker.chunk_pages(cleaned_pages)
        return chunks
