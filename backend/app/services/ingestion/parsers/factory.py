"""Parser factory for resolving format-specific parsers."""
import os
from typing import Optional
from app.services.ingestion.parsers.base import BaseParser, UnsupportedDocumentError
from app.services.ingestion.parsers.pdf import PDFParser
from app.services.ingestion.parsers.docx import DocxParser
from app.services.ingestion.parsers.doc import LegacyDocParser
from app.services.ingestion.parsers.txt import TxtParser
from app.services.ingestion.parsers.image import ImageParser
from app.services.ingestion.ocr.base import OCRProvider


def get_parser_for_mime_or_filename(
    mime_type: Optional[str],
    filename: str,
    ocr_provider: OCRProvider,
    min_text_chars: int = 50,
) -> BaseParser:
    """Returns the appropriate parser instance based on MIME type and file extension."""
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    mime = (mime_type or "").lower()

    # 1. PDF
    if "application/pdf" in mime or ext == "pdf":
        return PDFParser(ocr_provider=ocr_provider, min_text_chars=min_text_chars)

    # 2. Legacy DOC must be converted before the existing DOCX parser can read it.
    if ext == "doc":
        return LegacyDocParser()

    # 3. DOCX
    if "wordprocessingml" in mime or ext == "docx":
        return DocxParser()

    # 4. Plain Text
    if "text/plain" in mime or ext in ("txt", "text"):
        return TxtParser()

    # 5. Images
    if (
        mime.startswith("image/")
        or ext in ("png", "jpg", "jpeg", "webp")
    ):
        return ImageParser(ocr_provider=ocr_provider)

    raise UnsupportedDocumentError(
        f"Unsupported document format for '{filename}' (MIME: {mime_type}, extension: .{ext})."
    )
