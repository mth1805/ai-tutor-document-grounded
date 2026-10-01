"""Ingestion package exports."""
from app.services.ingestion.models import DocumentPage, ProcessedChunk
from app.services.ingestion.cleaner import clean_text
from app.services.ingestion.chunker import StructureAwareChunker
from app.services.ingestion.pipeline import IngestionPipeline
from app.services.ingestion.ocr.base import OCRProvider, OCRUnavailableError, OCRError
from app.services.ingestion.ocr.tesseract import TesseractOCRProvider
from app.services.ingestion.ocr.mock import MockOCRProvider
from app.services.ingestion.parsers.base import (
    BaseParser,
    EmptyDocumentError,
    UnsupportedDocumentError,
    DocumentCorruptError,
    ParserError,
)

__all__ = [
    "DocumentPage",
    "ProcessedChunk",
    "clean_text",
    "StructureAwareChunker",
    "IngestionPipeline",
    "OCRProvider",
    "OCRUnavailableError",
    "OCRError",
    "TesseractOCRProvider",
    "MockOCRProvider",
    "BaseParser",
    "EmptyDocumentError",
    "UnsupportedDocumentError",
    "DocumentCorruptError",
    "ParserError",
]
