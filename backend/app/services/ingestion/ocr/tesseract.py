"""Tesseract OCR Provider implementation with local binary discovery and caching."""
import shutil
import logging
import os
from functools import cached_property
from typing import Any, Optional
from app.core.config import settings
from app.services.ingestion.ocr.base import OCRProvider, OCRUnavailableError, OCRError

logger = logging.getLogger(__name__)


class TesseractOCRProvider(OCRProvider):
    """Tesseract OCR provider using pytesseract with cached executable detection."""

    def __init__(self, tesseract_cmd: Optional[str] = None, languages: Optional[str] = None):
        self._tesseract_cmd = tesseract_cmd or settings.TESSERACT_CMD
        self.languages = languages or settings.TESSERACT_LANGUAGES
        self._pytesseract_module = None
        self._diagnostic: Optional[str] = None

    @cached_property
    def diagnostic(self) -> Optional[str]:
        """Cached explanation of missing executable, Python wrapper, or traineddata."""
        try:
            import pytesseract
            self._pytesseract_module = pytesseract
        except ImportError:
            return "Python package pytesseract is unavailable; install backend requirements."

        executable = self._tesseract_cmd or shutil.which("tesseract")
        if not executable:
            return "Tesseract executable is unavailable; install Tesseract or configure TESSERACT_CMD."
        if self._tesseract_cmd and not shutil.which(self._tesseract_cmd) and not os.path.isfile(self._tesseract_cmd):
            return f"Configured Tesseract executable '{self._tesseract_cmd}' was not found."
        pytesseract.pytesseract.tesseract_cmd = executable
        try:
            installed = set(pytesseract.get_languages(config=""))
        except Exception as exc:
            return f"Tesseract executable '{executable}' could not be queried: {exc}"
        required = {part.strip() for part in self.languages.split("+") if part.strip()}
        missing = sorted(required - installed)
        if missing:
            return (
                f"Tesseract language data unavailable for: {', '.join(missing)}. "
                f"Install the corresponding traineddata files; configured languages are '{self.languages}'."
            )
        return None

    def is_available(self) -> bool:
        """Cached check for Tesseract OCR availability."""
        return self.diagnostic is None

    def extract_text_from_image(self, image: Any) -> str:
        """Extracts text from an image using pytesseract."""
        if not self.is_available():
            raise OCRUnavailableError(f"OCR is required, but {self.diagnostic}")

        try:
            text = self._pytesseract_module.image_to_string(image, lang=self.languages)
            if not text or not text.strip():
                raise OCRError("Tesseract completed but returned no recognized text.")
            return text
        except Exception as e:
            logger.error("Tesseract OCR extraction failed: %s", e)
            raise OCRError(f"Tesseract OCR extraction failed: {e}") from e
