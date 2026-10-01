"""Structure-aware document chunking with page provenance tracking and configurable token limits."""
import re
import logging
from dataclasses import dataclass
from typing import List, Optional

from app.services.ingestion.models import DocumentPage, ProcessedChunk

logger = logging.getLogger(__name__)

# Sentence splitting regex matching end of sentence followed by whitespace
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class _TextUnit:
    text: str
    page_number: int
    token_count: int


class StructureAwareChunker:
    """Partitions extracted document pages into structure-aware text chunks."""

    def __init__(
        self,
        target_chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_size: int = 80,
    ):
        """
        Args:
            target_chunk_size: Target token count per chunk (standard target 400-600).
            chunk_overlap: Number of tokens of context to carry over between consecutive chunks.
            min_chunk_size: Minimum token threshold for standalone chunks (prevents tiny stub chunks).
        """
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # Initialize tiktoken encoder with fallback
        self._encoder = None
        try:
            import tiktoken
            self._encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning("tiktoken encoder initialization fallback to character heuristic: %s", e)

    def count_tokens(self, text: str) -> int:
        """Calculates token count using tiktoken cl100k_base with character-length fallback."""
        if not text:
            return 0
        if self._encoder is not None:
            try:
                return len(self._encoder.encode(text))
            except Exception:
                pass
        # Fallback approximation: ~4 characters per token
        return max(1, len(text) // 4)

    def _split_oversized_unit(self, text: str, page_number: int) -> List[_TextUnit]:
        """Sub-splits a paragraph that exceeds target_chunk_size into sentences or word chunks."""
        sentences = _SENTENCE_SPLIT_RE.split(text)
        units: List[_TextUnit] = []

        for sent in sentences:
            sent_str = sent.strip()
            if not sent_str:
                continue

            tok_count = self.count_tokens(sent_str)
            if tok_count <= self.target_chunk_size:
                units.append(_TextUnit(text=sent_str, page_number=page_number, token_count=tok_count))
            else:
                # Fallback sub-split long sentence by words
                words = sent_str.split(" ")
                cur_words = []
                cur_tokens = 0
                for w in words:
                    w_tok = self.count_tokens(w + " ")
                    if cur_tokens + w_tok > self.target_chunk_size and cur_words:
                        sub_text = " ".join(cur_words)
                        units.append(_TextUnit(text=sub_text, page_number=page_number, token_count=self.count_tokens(sub_text)))
                        cur_words = [w]
                        cur_tokens = w_tok
                    else:
                        cur_words.append(w)
                        cur_tokens += w_tok
                if cur_words:
                    sub_text = " ".join(cur_words)
                    units.append(_TextUnit(text=sub_text, page_number=page_number, token_count=self.count_tokens(sub_text)))

        return units

    def chunk_pages(self, pages: List[DocumentPage]) -> List[ProcessedChunk]:
        """Chunks a sequence of DocumentPages while preserving page-level provenance.

        Returns:
            List of ProcessedChunk objects with 0-indexed sequential chunk_index.
        """
        # 1. Break pages into structural atomic units (paragraphs / headings)
        atomic_units: List[_TextUnit] = []

        for page in pages:
            if not page.text or not page.text.strip():
                continue

            paragraphs = page.text.split("\n\n")
            for para in paragraphs:
                para_str = para.strip()
                if not para_str:
                    continue

                tok_count = self.count_tokens(para_str)
                if tok_count <= self.target_chunk_size:
                    atomic_units.append(
                        _TextUnit(text=para_str, page_number=page.page_number, token_count=tok_count)
                    )
                else:
                    # Paragraph is longer than target size; sub-split preserving structure
                    sub_units = self._split_oversized_unit(para_str, page.page_number)
                    atomic_units.extend(sub_units)

        if not atomic_units:
            return []

        # 2. Accumulate atomic units into chunks with overlap
        chunks: List[ProcessedChunk] = []
        chunk_idx = 0
        i = 0
        n = len(atomic_units)

        while i < n:
            current_elements: List[_TextUnit] = []
            current_tokens = 0
            start_idx = i

            while i < n:
                elem = atomic_units[i]
                # If adding elem exceeds target and we already have sufficient content
                if current_tokens + elem.token_count > self.target_chunk_size and current_tokens >= self.min_chunk_size:
                    break
                current_elements.append(elem)
                current_tokens += elem.token_count
                i += 1

            if not current_elements:
                # Safeguard: ensure forward progress even if a single element is large
                current_elements.append(atomic_units[i])
                current_tokens += atomic_units[i].token_count
                i += 1

            # Build chunk text
            chunk_content = "\n\n".join(elem.text for elem in current_elements).strip()
            if chunk_content:
                page_start = min(elem.page_number for elem in current_elements)
                page_end = max(elem.page_number for elem in current_elements)
                final_token_count = self.count_tokens(chunk_content)

                chunks.append(
                    ProcessedChunk(
                        chunk_index=chunk_idx,
                        content=chunk_content,
                        page_number_start=page_start,
                        page_number_end=page_end,
                        token_count=final_token_count,
                    )
                )
                chunk_idx += 1

            # Calculate overlap for next iteration
            if i < n and self.chunk_overlap > 0:
                overlap_tokens = 0
                step_back = 0
                # Look backward through current_elements to find overlap boundary
                for back_elem in reversed(current_elements):
                    if overlap_tokens + back_elem.token_count <= self.chunk_overlap and (i - (step_back + 1)) > start_idx:
                        overlap_tokens += back_elem.token_count
                        step_back += 1
                    else:
                        break
                i -= step_back

        return chunks
