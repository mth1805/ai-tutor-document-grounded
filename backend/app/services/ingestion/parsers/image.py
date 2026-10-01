"""Image parser invoking OCR for standalone image files (PNG, JPEG, WEBP)."""
import io
import logging
from typing import List
from PIL import Image

from app.services.ingestion.models import DocumentPage
from app.services.ingestion.cleaner import clean_text
from app.services.ingestion.parsers.base import (
    BaseParser,
    EmptyDocumentError,
    DocumentCorruptError,
)
from app.services.ingestion.ocr.base import OCRProvider, OCRUnavailableError

logger = logging.getLogger(__name__)


class ImageParser(BaseParser):
    """Parses raster images by extracting embedded text via OCR."""

    def __init__(self, ocr_provider: OCRProvider):
        self.ocr_provider = ocr_provider

    def parse(self, content: bytes, filename: str) -> List[DocumentPage]:
        try:
            image = Image.open(io.BytesIO(content))
            image.load()  # Verify image integrity
        except Exception as e:
            logger.error("Failed to load image %s: %s", filename, e)
            raise DocumentCorruptError(
                f"Image file '{filename}' is corrupted or not a valid image: {e}"
            ) from e

        if not self.ocr_provider.is_available():
            raise OCRUnavailableError(
                f"Processing image '{filename}' requires OCR, but no OCR engine (such as Tesseract) "
                "is available on the server."
            )

        ocr_text = self.ocr_provider.extract_text_from_image(image)
        cleaned = clean_text(ocr_text)

        if not cleaned:
            raise EmptyDocumentError(
                f"No readable text could be recognized in image '{filename}' via OCR."
            )

        return [
            DocumentPage(
                page_number=1,
                text=cleaned,
                source_type="image",
                used_ocr=True,
            )
        ]
