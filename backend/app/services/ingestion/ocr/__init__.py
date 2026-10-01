"""OCR Provider package."""
from app.services.ingestion.ocr.base import (
    OCRProvider,
    OCRError,
    OCRUnavailableError,
)
from app.services.ingestion.ocr.tesseract import TesseractOCRProvider
from app.services.ingestion.ocr.mock import MockOCRProvider

__all__ = [
    "OCRProvider",
    "OCRError",
    "OCRUnavailableError",
    "TesseractOCRProvider",
    "MockOCRProvider",
]
