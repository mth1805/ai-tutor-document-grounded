"""Mock Cross-Encoder reranker provider for test environments without downloading large weights."""
import hashlib
import re
from typing import List, Optional, Tuple

from app.ml.reranker_base import BaseRerankerProvider


class MockRerankerProvider(BaseRerankerProvider):
    """Deterministic, zero-download reranker for fast unit and integration testing."""

    def __init__(
        self,
        model_name: str = "mock-reranker-bge-reranker-v2-m3",
        device: str = "cpu",
        version: str = "mock-rerank-1.0.0",
    ):
        self._model_name = model_name
        self._device = device
        self._version = version

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def version(self) -> str:
        return self._version

    @property
    def device(self) -> str:
        return self._device

    def predict(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: Optional[int] = None,
    ) -> List[float]:
        """Calculates deterministic relevance scores based on token overlap and content hash."""
        if not pairs:
            return []

        scores: List[float] = []
        word_pattern = re.compile(r"\w+")

        for query, text in pairs:
            q_words = set(word_pattern.findall(query.lower()))
            t_words = set(word_pattern.findall(text.lower()))

            # Token overlap score
            if q_words:
                common = q_words & t_words
                overlap_ratio = len(common) / len(q_words)
            else:
                overlap_ratio = 0.0

            # Deterministic hash delta [0.0, 0.05] for stable ranking between equal overlaps
            pair_hash = int(hashlib.md5(f"{query}::{text}".encode("utf-8")).hexdigest()[:6], 16)
            hash_delta = (pair_hash % 1000) / 20000.0  # 0.0 to 0.05

            if overlap_ratio > 0:
                # Highly relevant: 0.50 to 0.95
                score = 0.50 + (0.40 * overlap_ratio) + hash_delta
            else:
                # Irrelevant: 0.05 to 0.20 (below standard relevance threshold)
                score = 0.10 + hash_delta

            scores.append(round(min(max(score, 0.0), 1.0), 4))

        return scores
