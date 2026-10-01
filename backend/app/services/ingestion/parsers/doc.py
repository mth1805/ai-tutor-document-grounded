"""Safe temporary LibreOffice conversion for legacy binary Word documents."""
import logging
import shutil
import subprocess
import tempfile
from functools import cached_property
from pathlib import Path
from typing import List, Optional

from app.core.config import settings
from app.services.ingestion.models import DocumentPage
from app.services.ingestion.parsers.base import BaseParser, DocumentCorruptError, ParserError
from app.services.ingestion.parsers.docx import DocxParser

logger = logging.getLogger(__name__)


class LegacyDocParser(BaseParser):
    """Converts a legacy .doc in an isolated temp directory, then uses DocxParser."""

    def __init__(
        self,
        libreoffice_cmd: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        docx_parser: Optional[DocxParser] = None,
    ):
        self.libreoffice_cmd = libreoffice_cmd or settings.LIBREOFFICE_CMD
        self.timeout_seconds = timeout_seconds or settings.LIBREOFFICE_TIMEOUT_SECONDS
        self.docx_parser = docx_parser or DocxParser()

    @cached_property
    def executable(self) -> Optional[str]:
        return shutil.which(self.libreoffice_cmd)

    def parse(self, content: bytes, filename: str) -> List[DocumentPage]:
        if self.executable is None:
            raise ParserError(
                "Legacy .doc processing requires LibreOffice; executable "
                f"'{self.libreoffice_cmd}' was not found. Install LibreOffice or set LIBREOFFICE_CMD."
            )
        try:
            with tempfile.TemporaryDirectory(prefix="ai-tutor-doc-") as temp_dir:
                source = Path(temp_dir) / "source.doc"
                source.write_bytes(content)
                result = subprocess.run(
                    [self.executable, "--headless", "--convert-to", "docx", "--outdir", temp_dir, str(source)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
                converted = Path(temp_dir) / "source.docx"
                if result.returncode != 0 or not converted.is_file() or converted.stat().st_size == 0:
                    details = (result.stderr or result.stdout or "no conversion details").strip()
                    raise DocumentCorruptError(f"Legacy Word document '{filename}' could not be converted: {details}")
                pages = self.docx_parser.parse(converted.read_bytes(), filename)
                return [
                    DocumentPage(p.page_number, p.text, "doc", p.used_ocr)
                    for p in pages
                ]
        except subprocess.TimeoutExpired as exc:
            raise ParserError(
                f"LibreOffice conversion of '{filename}' exceeded the {self.timeout_seconds}-second timeout."
            ) from exc
        except DocumentCorruptError:
            raise
        except ParserError:
            raise
        except Exception as exc:
            logger.exception("Legacy DOC conversion failed for %s", filename)
            raise DocumentCorruptError(f"Legacy Word document '{filename}' could not be converted or parsed: {exc}") from exc
