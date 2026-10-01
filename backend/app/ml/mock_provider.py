"""Mock embedding provider for test environments without downloading large weights."""
import hashlib
import math
import struct
from typing import List, Optional

from app.ml.base import BaseEmbeddingProvider


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic, zero-download embedding provider for fast unit/integration testing."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        dimension: int = 1024,
        device: str = "cpu",
        version: str = "mock-1.0.0",
    ):
        self._model_name = model_name
        self._dimension = dimension
        self._device = device
        self._version = version

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def version(self) -> str:
        return self._version

    @property
    def device(self) -> str:
        return self._device

    def _generate_vector(self, text: str, normalize: bool) -> List[float]:
        """Generates a deterministic 1024-dim float vector seeded from text content."""
        # Use SHA-256 digest to seed pseudo-random values
        hasher = hashlib.sha256(text.encode("utf-8"))
        seed_bytes = hasher.digest()

        vec: List[float] = []
        # Expand 32 bytes into 1024 float dimensions deterministically
        for i in range(self._dimension):
            idx = (i * 4) % len(seed_bytes)
            chunk = seed_bytes[idx : idx + 4]
            if len(chunk) < 4:
                chunk = (chunk + seed_bytes)[:4]
            raw_int = struct.unpack(">I", chunk)[0]
            val = ((raw_int + i * 1337) % 20000 - 10000) / 10000.0
            vec.append(val)

        if normalize:
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 1e-9:
                vec = [x / norm for x in vec]

        return vec

    def encode_batch(
        self,
        texts: List[str],
        normalize: bool = True,
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """Encodes texts into deterministic mock vectors."""
        if not texts:
            return []
        return [self._generate_vector(t, normalize) for t in texts]
