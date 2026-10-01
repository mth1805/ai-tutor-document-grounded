"""Parsers package."""
from app.services.ingestion.parsers.base import (
    BaseParser,
    ParserError,
    EmptyDocumentError,
    UnsupportedDocumentError,
    DocumentCorruptError,
)
from app.services.ingestion.parsers.pdf import PDFParser
from app.services.ingestion.parsers.docx import DocxParser
from app.services.ingestion.parsers.txt import TxtParser
from app.services.ingestion.parsers.image import ImageParser
from app.services.ingestion.parsers.factory import get_parser_for_mime_or_filename

__all__ = [
    "BaseParser",
    "ParserError",
    "EmptyDocumentError",
    "UnsupportedDocumentError",
    "DocumentCorruptError",
    "PDFParser",
    "DocxParser",
    "TxtParser",
    "ImageParser",
    "get_parser_for_mime_or_filename",
]
