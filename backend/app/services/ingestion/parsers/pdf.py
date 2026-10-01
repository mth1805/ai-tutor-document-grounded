"""PDF document parser with native text extraction and selective page-level OCR."""
import io
import logging
from typing import List, Optional
from PIL import Image

from app.services.ingestion.models import DocumentPage
from app.services.ingestion.cleaner import clean_text
from app.services.ingestion.parsers.base import (
    BaseParser,
    EmptyDocumentError,
    DocumentCorruptError,
    ParserError,
)
from app.services.ingestion.ocr.base import OCRProvider, OCRUnavailableError

logger = logging.getLogger(__name__)


class PDFParser(BaseParser):
    """Parses PDF documents by prioritizing native digital text and selectively invoking OCR only for scanned/image pages."""

    def __init__(
        self,
        ocr_provider: OCRProvider,
        min_text_chars: int = 50,
        render_dpi: int = 150,
    ):
        """
        Args:
            ocr_provider: OCR provider instance to use for scanned pages.
            min_text_chars: Minimum number of characters of native text for a page to be considered sufficient without OCR.
            render_dpi: DPI resolution for rendering scanned PDF pages for OCR.
        """
        self.ocr_provider = ocr_provider
        self.min_text_chars = min_text_chars
        self.render_dpi = render_dpi

    def parse(self, content: bytes, filename: str) -> List[DocumentPage]:
        """Extracts text page by page from PDF bytes."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ParserError("PyMuPDF (fitz) is required for PDF parsing.")

        try:
            doc = fitz.open(stream=content, filetype="pdf")
        except Exception as e:
            logger.error("Failed to open PDF %s: %s", filename, e)
            raise DocumentCorruptError(f"PDF file '{filename}' could not be opened or is corrupted: {e}") from e

        if doc.page_count == 0:
            doc.close()
            raise EmptyDocumentError(f"PDF '{filename}' has 0 pages.")

        pages: List[DocumentPage] = []
        total_extracted_chars = 0

        try:
            for page_idx in range(doc.page_count):
                page_num = page_idx + 1
                page = doc[page_idx]

                # STEP 1: Attempt native text extraction
                native_text = page.get_text() or ""
                cleaned_native = clean_text(native_text)

                # STEP 2: Evaluate whether native text is sufficient
                # If page contains adequate text, use native text (no OCR needed)
                if len(cleaned_native) >= self.min_text_chars:
                    pages.append(
                        DocumentPage(
                            page_number=page_num,
                            text=cleaned_native,
                            source_type="pdf",
                            used_ocr=False,
                        )
                    )
                    total_extracted_chars += len(cleaned_native)
                    continue

                # STEP 3: Page text is insufficient; check if page has images or is scanned
                image_list = page.get_images()
                is_scanned_candidate = len(image_list) > 0 or len(cleaned_native) < 10

                if is_scanned_candidate:
                    # Page likely requires OCR
                    if not self.ocr_provider.is_available():
                        # If OCR is unavailable, determine whether to fail:
                        # If there's some native text, we might retain it, but if it has images and minimal text, fail clearly
                        if len(cleaned_native) < 15:
                            raise OCRUnavailableError(
                                f"Page {page_num} of PDF '{filename}' contains scanned or image content requiring OCR, "
                                "but no OCR engine (such as Tesseract) is installed or available on the server."
                            )
                        # Partial text fallback
                        pages.append(
                            DocumentPage(
                                page_number=page_num,
                                text=cleaned_native,
                                source_type="pdf",
                                used_ocr=False,
                            )
                        )
                        total_extracted_chars += len(cleaned_native)
                        continue

                    # Render page to raster image for OCR
                    try:
                        pix = page.get_pixmap(dpi=self.render_dpi)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        ocr_text = self.ocr_provider.extract_text_from_image(img)
                        cleaned_ocr = clean_text(ocr_text)

                        # Prefer OCR text if richer than sparse native text
                        chosen_text = cleaned_ocr if len(cleaned_ocr) > len(cleaned_native) else cleaned_native
                        pages.append(
                            DocumentPage(
                                page_number=page_num,
                                text=chosen_text,
                                source_type="pdf",
                                used_ocr=True,
                            )
                        )
                        total_extracted_chars += len(chosen_text)
                    except OCRUnavailableError:
                        raise
                    except Exception as ocr_err:
                        logger.warning("OCR failed on page %d of %s: %s", page_num, filename, ocr_err)
                        if len(cleaned_native) > 0:
                            pages.append(
                                DocumentPage(
                                    page_number=page_num,
                                    text=cleaned_native,
                                    source_type="pdf",
                                    used_ocr=False,
                                )
                            )
                            total_extracted_chars += len(cleaned_native)
                        else:
                            raise ParserError(f"OCR processing failed for page {page_num}: {ocr_err}") from ocr_err
                else:
                    # Page is naturally short (e.g. title page or blank page)
                    pages.append(
                        DocumentPage(
                            page_number=page_num,
                            text=cleaned_native,
                            source_type="pdf",
                            used_ocr=False,
                        )
                    )
                    total_extracted_chars += len(cleaned_native)
        finally:
            doc.close()

        if total_extracted_chars == 0:
            raise EmptyDocumentError(
                f"PDF '{filename}' contained no readable text after parsing and OCR evaluation."
            )

        return pages
