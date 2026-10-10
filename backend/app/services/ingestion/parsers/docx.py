"""DOCX document parser preserving structural elements (headings, lists, tables)."""
import io
import logging
from typing import List

from app.services.ingestion.models import DocumentPage
from app.services.ingestion.cleaner import clean_text
from app.services.ingestion.parsers.docx_math import ordered_text
from app.services.ingestion.parsers.base import (
    BaseParser,
    EmptyDocumentError,
    DocumentCorruptError,
    ParserError,
)

logger = logging.getLogger(__name__)


class DocxParser(BaseParser):
    """Parses DOCX files using python-docx while preserving document hierarchy and structure."""

    def parse(self, content: bytes, filename: str) -> List[DocumentPage]:
        try:
            import docx
        except ImportError:
            raise ParserError("python-docx library is required for DOCX parsing.")

        try:
            doc = docx.Document(io.BytesIO(content))
        except Exception as e:
            logger.error("Failed to parse DOCX file %s: %s", filename, e)
            raise DocumentCorruptError(
                f"Word document '{filename}' is corrupted or not a valid DOCX file: {e}"
            ) from e

        elements: List[str] = []

        # Iterate over body elements (paragraphs and tables) in document order
        # doc.element.body contains elements in document order
        for child in doc.element.body:
            tag = child.tag.split("}")[-1]  # Get XML tag without namespace
            if tag == "p":
                # Paragraph element
                p = docx.text.paragraph.Paragraph(child, doc)
                text = ordered_text(child).strip()
                if not text:
                    continue

                style_name = p.style.name.lower() if p.style and p.style.name else ""
                if "heading 1" in style_name:
                    elements.append(f"# {text}")
                elif "heading 2" in style_name:
                    elements.append(f"## {text}")
                elif "heading 3" in style_name:
                    elements.append(f"### {text}")
                elif "list" in style_name or "bullet" in style_name:
                    elements.append(f"- {text}")
                else:
                    elements.append(text)

            elif tag == "tbl":
                # Table element
                tbl = docx.table.Table(child, doc)
                table_lines = []
                for row in tbl.rows:
                    row_cells = [
                        " ".join(ordered_text(p).strip() for p in cell._tc.iterchildren()
                                 if p.tag.split("}")[-1] == "p").replace("\n", " ")
                        for cell in row.cells
                    ]
                    # Only add if row has non-empty cells
                    if any(row_cells):
                        table_lines.append("| " + " | ".join(row_cells) + " |")
                if table_lines:
                    elements.append("\n".join(table_lines))

        raw_document_text = "\n\n".join(elements)
        cleaned_text = clean_text(raw_document_text)

        if not cleaned_text:
            raise EmptyDocumentError(
                f"Word document '{filename}' contains no extractable text."
            )

        # DOCX format has dynamic layout without fixed physical page breaks;
        # represent as a structured logical page (page_number = 1)
        return [
            DocumentPage(
                page_number=1,
                text=cleaned_text,
                source_type="docx",
                used_ocr=False,
            )
        ]
