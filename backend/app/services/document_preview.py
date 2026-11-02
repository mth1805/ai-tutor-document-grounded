"""Private, temporary Word previews. No preview artifacts are stored persistently."""
import logging
import io
import re
import shutil
import subprocess
import tempfile
import uuid
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Optional
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings

logger = logging.getLogger(__name__)


async def get_preview_content(
    db: Optional[AsyncSession], document_id: uuid.UUID, user_id: uuid.UUID,
) -> Optional[tuple[bytes, str, str]]:
    from app.services.document_service import DocumentService
    from app.services.ingestion_service import IngestionService

    doc, content = await DocumentService.get_document_file(db, document_id, user_id)
    if not doc or content is None:
        return None
    mime, name = doc.mime_type, doc.original_filename
    if Path(name).suffix.lower() in {".doc", ".docx"}:
        try:
            preview = await run_in_threadpool(convert_word_to_verified_pdf, content, name)
            content = preview
            mime, name = "application/pdf", str(Path(name).with_suffix(".pdf"))
        except Exception as exc:
            logger.warning("word_preview_conversion_failed document_id=%s category=%s", document_id, type(exc).__name__)
            # DOCX fallback must read the original: older persisted chunks may
            # have omitted equations. Legacy DOC can still use scoped chunks
            # when LibreOffice is unavailable.
            text = ""
            if Path(doc.original_filename).suffix.lower() == ".docx":
                try:
                    text = await run_in_threadpool(extract_word_text, content, doc.original_filename)
                except Exception:
                    logger.warning("word_preview_text_failed document_id=%s", document_id)
            else:
                chunks = await IngestionService.list_document_chunks(db, document_id, user_id)
                text = "\n\n".join(chunk.content for chunk in (chunks or []))
            if not text:
                try:
                    text = await run_in_threadpool(extract_word_text, content, doc.original_filename)
                except Exception:
                    raise HTTPException(422, "Preview is unavailable. Please download the original document.")
            content = text.encode("utf-8")
            mime, name = "text/plain", str(Path(name).with_suffix(".txt"))
    return content, mime, name


def convert_word_to_verified_pdf(content: bytes, filename: str) -> bytes:
    """Try native conversion; require readable equation evidence for DOCX math.

    PDF text cannot reliably prove stacked mathematical layout. A conservative
    failed check uses the existing readable text preview, never a blank formula.
    """
    pdf = convert_word_to_pdf(content, filename)
    if Path(filename).suffix.lower() != ".docx":
        return pdf
    import docx
    from app.services.ingestion.parsers.docx_math import document_equations

    equations = document_equations(docx.Document(io.BytesIO(content)).element.body)
    if not equations:
        return pdf
    if any(not equation.supported for equation in equations):
        raise RuntimeError("PREVIEW_MATH_UNVERIFIABLE")
    import fitz

    def normalized(text: str) -> str:
        return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).translate(
            str.maketrans({"−": "-", "×": "*", "÷": "/"})
        )

    with fitz.open(stream=pdf, filetype="pdf") as document:
        rendered = normalized("\n".join(page.get_text() for page in document))
    expected = Counter(normalized(equation.text) for equation in equations)
    if any(rendered.count(expression) < count for expression, count in expected.items()):
        raise RuntimeError("PREVIEW_MATH_UNVERIFIABLE")
    return pdf


def convert_word_to_pdf(content: bytes, filename: str) -> bytes:
    executable = shutil.which(settings.LIBREOFFICE_CMD)
    if not executable:
        raise RuntimeError("PREVIEW_CONVERTER_UNAVAILABLE")
    with tempfile.TemporaryDirectory(prefix="ai-tutor-preview-") as directory:
        root = Path(directory)
        source = root / ("source" + Path(filename).suffix.lower())
        source.write_bytes(content)
        # A separate profile prevents concurrent LibreOffice processes attaching
        # to another request's instance. No shell or user-controlled command args.
        result = subprocess.run(
            [executable, f"-env:UserInstallation={(root / 'profile').as_uri()}",
             "--headless", "--convert-to", "pdf:writer_pdf_Export",
             "--outdir", directory, str(source)],
            capture_output=True, timeout=settings.LIBREOFFICE_TIMEOUT_SECONDS,
            check=False,
        )
        output = root / "source.pdf"
        if result.returncode != 0 or not output.is_file():
            raise RuntimeError("PREVIEW_CONVERSION_FAILED")
        pdf = output.read_bytes()
        if not pdf.startswith(b"%PDF-"):
            raise RuntimeError("PREVIEW_INVALID_PDF")
        return pdf


def extract_word_text(content: bytes, filename: str) -> str:
    from app.services.ingestion.parsers.factory import get_parser_for_mime_or_filename
    from app.services.ingestion.ocr.tesseract import TesseractOCRProvider
    parser = get_parser_for_mime_or_filename(mime_type=None, filename=filename, ocr_provider=TesseractOCRProvider())
    return "\n\n".join(page.text for page in parser.parse(content, filename))
