"""Mock OCR Provider for testing without external binaries."""
from typing import Any, Optional
from app.services.ingestion.ocr.base import OCRProvider, OCRUnavailableError


class MockOCRProvider(OCRProvider):
    """Configurable mock OCR provider for unit tests."""

    def __init__(self, available: bool = True, return_text: str = "Extracted text from scanned document via OCR"):
        self._available = available
        self._return_text = return_text

    def is_available(self) -> bool:
        return self._available

    def set_available(self, available: bool) -> None:
        self._available = available

    def set_return_text(self, text: str) -> None:
        self._return_text = text

    def extract_text_from_image(self, image: Any) -> str:
        if not self._available:
            raise OCRUnavailableError("Mock OCR engine is unavailable")
        return self._return_text
