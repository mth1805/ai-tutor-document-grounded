"""Citation extraction, verification, and transformation service."""
import re
import uuid
from typing import List, Dict, Any, Optional, Set, Tuple
from pydantic import BaseModel, Field

from app.rag.prompt_builder import SourceEvidence


class Citation(BaseModel):
    """Structured citation payload anchoring generated claims to real document chunks."""

    document_id: uuid.UUID
    document_name: str
    chunk_id: uuid.UUID
    page_start: int
    page_end: int
    snippet: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": str(self.document_id),
            "document_name": self.document_name,
            "chunk_id": str(self.chunk_id),
            "page_start": self.page_start,
            "page_end": self.page_end,
            "snippet": self.snippet,
        }


class CitationService:
    """Extracts, validates, and transforms citation markers against verified evidence candidates."""

    SOURCE_PATTERN = re.compile(
        r"\[((?:Source|Doc)\s*[:\-]?\s*\d+(?:\s*,\s*(?:(?:Source|Doc)\s*[:\-]?\s*)?\d+)*)\]",
        re.IGNORECASE,
    )
    SOURCE_INDEX_PATTERN = re.compile(r"(?:Source|Doc)?\s*[:\-]?\s*(\d+)", re.IGNORECASE)
    DOC_PAGE_PATTERN = re.compile(
        r"\[Doc:\s*([^,\]]+),\s*p\.?\s*(\d+)(?:-(\d+))?\]", re.IGNORECASE
    )

    @classmethod
    def extract_and_validate_citations(
        cls,
        text: str,
        sources: List[SourceEvidence],
    ) -> Tuple[List[Citation], str]:
        """Parses citation markers in generated text, validates against evidence sources,

        and optionally normalizes markers to `[Doc: filename, p. X]`.

        Args:
            text: Raw generated output from the LLM.
            sources: Verified SourceEvidence items supplied in the prompt.

        Returns:
            Tuple of (validated_citations_list, normalized_text).
        """
        source_by_index: Dict[int, SourceEvidence] = {s.source_index: s for s in sources}
        source_by_name: Dict[str, SourceEvidence] = {s.document_name.lower(): s for s in sources}

        cited_indices: Set[int] = set()
        matched_citations: List[Citation] = []

        # 1. Parse [Source X] references
        for match in cls.SOURCE_PATTERN.finditer(text):
            try:
                for index_match in cls.SOURCE_INDEX_PATTERN.finditer(match.group(1)):
                    idx = int(index_match.group(1))
                    if idx in source_by_index:
                        cited_indices.add(idx)
            except ValueError:
                continue

        # 2. Parse direct [Doc: filename, p. X] references if the LLM emitted them directly
        for match in cls.DOC_PAGE_PATTERN.finditer(text):
            filename = match.group(1).strip().lower()
            if filename in source_by_name:
                cited_indices.add(source_by_name[filename].source_index)

        # 3. If the model didn't emit explicit tags but the context had top evidence,
        # fallback: if only 1 source was provided, bind to it; otherwise only validate explicit citations.
        if not cited_indices and len(sources) == 1:
            cited_indices.add(sources[0].source_index)

        # 4. Build deduplicated structured citations preserving source order
        for idx in sorted(list(cited_indices)):
            src = source_by_index[idx]
            snippet = src.content[:200].replace("\n", " ").strip()
            citation = Citation(
                document_id=src.document_id,
                document_name=src.document_name,
                chunk_id=src.chunk_id,
                page_start=src.page_start,
                page_end=src.page_end,
                snippet=snippet,
            )
            matched_citations.append(citation)

        # 5. Transform [Source X] in text to readable [Doc: filename, p. X]
        def _replace_source_tag(m: re.Match) -> str:
            doc_markers: List[str] = []
            for index_match in cls.SOURCE_INDEX_PATTERN.finditer(m.group(1)):
                src = source_by_index.get(int(index_match.group(1)))
                if src is None:
                    continue
                page_str = (
                    f"p. {src.page_start}"
                    if src.page_start == src.page_end
                    else f"pp. {src.page_start}-{src.page_end}"
                )
                doc_markers.append(f"[Doc: {src.document_name}, {page_str}]")
            return " ".join(doc_markers)

        normalized_text = cls.SOURCE_PATTERN.sub(_replace_source_tag, text)

        # Clean up any duplicate spacing created by stripped tags
        normalized_text = re.sub(r" +", " ", normalized_text)

        return matched_citations, normalized_text
