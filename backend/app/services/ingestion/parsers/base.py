"""Base class and common exceptions for document parsers."""
from abc import ABC, abstractmethod
from typing import List
from app.services.ingestion.models import DocumentPage


class ParserError(Exception):
    """Base exception for parsing errors."""
    pass


class EmptyDocumentError(ParserError):
    """Raised when an uploaded document contains no readable or extractable text."""
    pass


class UnsupportedDocumentError(ParserError):
    """Raised when a document type is unsupported by the parser pipeline."""
    pass


class DocumentCorruptError(ParserError):
    """Raised when a document binary is malformed or corrupted."""
    pass


class BaseParser(ABC):
    """Abstract base class for all file format parsers."""

    @abstractmethod
    def parse(self, content: bytes, filename: str) -> List[DocumentPage]:
        """Parses raw document bytes into a sequence of DocumentPage objects.

        Args:
            content: Raw binary bytes of the uploaded file.
            filename: Original file name.

        Returns:
            List of DocumentPage objects preserving 1-indexed page numbers.

        Raises:
            EmptyDocumentError: If document contains no readable text.
            DocumentCorruptError: If document is unreadable or malformed.
            ParserError: If parsing fails.
        """
        pass
