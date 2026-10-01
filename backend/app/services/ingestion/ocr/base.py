"""OCR Provider abstract base class and common exceptions."""
from abc import ABC, abstractmethod
from typing import Any


class OCRError(Exception):
    """Base exception for OCR processing errors."""
    pass


class OCRUnavailableError(OCRError):
    """Raised when OCR is required to process a document or scanned page but no OCR engine is available."""
    pass


class OCRProvider(ABC):
    """Abstract base class for Optical Character Recognition providers."""

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if the OCR engine and its underlying binaries are available on the system."""
        pass

    @abstractmethod
    def extract_text_from_image(self, image: Any) -> str:
        """Extracts text from a PIL Image or image buffer.

        Args:
            image: PIL.Image.Image or compatible image object.

        Returns:
            Extracted text string.

        Raises:
            OCRUnavailableError: If OCR is not configured/installed.
            OCRError: If OCR extraction encounters an error.
        """
        pass
