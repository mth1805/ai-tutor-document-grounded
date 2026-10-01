"""Plain text parser with multi-encoding fallback and structure preservation."""
import logging
from typing import List

from app.services.ingestion.models import DocumentPage
from app.services.ingestion.cleaner import clean_text
from app.services.ingestion.parsers.base import (
    BaseParser,
    EmptyDocumentError,
    DocumentCorruptError,
)

logger = logging.getLogger(__name__)

# Encodings to attempt in order
ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]


class TxtParser(BaseParser):
    """Parses plain text files with robust multi-encoding detection."""

    def parse(self, content: bytes, filename: str) -> List[DocumentPage]:
        decoded_text: str = ""
        success = False

        for enc in ENCODINGS:
            try:
                decoded_text = content.decode(enc)
                success = True
                break
            except UnicodeDecodeError:
                continue

        if not success:
            raise DocumentCorruptError(
                f"Text document '{filename}' could not be decoded with any standard character encoding."
            )

        cleaned = clean_text(decoded_text)
        if not cleaned:
            raise EmptyDocumentError(
                f"Text file '{filename}' contains no readable text."
            )

        return [
            DocumentPage(
                page_number=1,
                text=cleaned,
                source_type="txt",
                used_ocr=False,
            )
        ]
