"""BGE-M3 embedding provider with local caching, batch encoding, and device fallback."""
import logging
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

from app.core.config import settings
from app.ml.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class BGEEmbeddingProvider(BaseEmbeddingProvider):
    """Centralized BAAI/bge-m3 model provider loaded once per process."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        self._model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self._dimension = settings.EMBEDDING_DIMENSION  # 1024 for BGE-M3
        self._version = "bge-m3-dense-v1"
        self._device = self._resolve_device(device or settings.EMBEDDING_DEVICE)
        self._cache_dir = cache_dir or settings.EMBEDDING_MODEL_CACHE_DIR

        logger.info(
            "Initializing BGEEmbeddingProvider: model=%s, device=%s, cache_dir=%s",
            self._model_name,
            self._device,
            self._cache_dir,
        )

        import torch
        from sentence_transformers import SentenceTransformer

        # Load model with configured cache folder
        cache_folder_str = str(self._cache_dir) if self._cache_dir else None
        self._model = SentenceTransformer(
            self._model_name,
            cache_folder=cache_folder_str,
            device=self._device,
        )
        # Ensure evaluation mode to disable dropouts
        self._model.eval()
        if settings.LOCAL_SHARED_MODELS:
            self._model.max_seq_length = settings.LOCAL_MODEL_MAX_TOKENS
            self._version = "bge-m3-dense-local-windowed-v1"
            self._local_cache = OrderedDict()
        logger.info("BGEEmbeddingProvider successfully initialized.")

    @staticmethod
    def _resolve_device(requested_device: str) -> str:
        """Resolves target compute device with safe CPU fallback when CUDA is unavailable."""
        import torch

        cuda_available = torch.cuda.is_available()
        req = requested_device.lower().strip()

        if req == "auto":
            if not cuda_available:
                logger.warning("embedding_cpu_fallback requested_device=auto cuda_available=false")
            return "cuda" if cuda_available else "cpu"
        elif req.startswith("cuda"):
            if not cuda_available:
                logger.warning(
                    "CUDA device '%s' requested but not available. Safely falling back to CPU.",
                    requested_device,
                )
                return "cpu"
            return req
        return "cpu"

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

    def encode_batch(
        self,
        texts: List[str],
        normalize: bool = True,
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """Encodes texts in micro-batches with torch.no_grad() and L2 normalization."""
        if not texts:
            return []

        import torch

        bs = batch_size or settings.EMBEDDING_BATCH_SIZE

        if settings.LOCAL_SHARED_MODELS:
            import numpy as np
            from app.ml.text_windows import text_windows
            result = []
            budget = settings.LOCAL_MODEL_MAX_TOKENS - 2
            with torch.no_grad():
                for text in texts:
                    key = (text, normalize)
                    if key in self._local_cache:
                        self._local_cache.move_to_end(key)
                        result.append(list(self._local_cache[key]))
                        continue
                    windows = text_windows(self._model.tokenizer, text, budget)
                    vectors = self._model.encode(windows, batch_size=1,
                        normalize_embeddings=normalize, show_progress_bar=False, convert_to_numpy=True)
                    # All windows contribute to the canonical chunk's single vector.
                    weights = [max(1, len(self._model.tokenizer(w, add_special_tokens=False)["input_ids"])) for w in windows]
                    vector = np.average(vectors, axis=0, weights=weights)
                    if normalize:
                        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
                    value = vector.tolist()
                    result.append(value)
                    if len(text) <= 4096:
                        self._local_cache[key] = tuple(value)
                        if len(self._local_cache) > 16:
                            self._local_cache.popitem(last=False)
            return result

        with torch.no_grad():
            embeddings = self._model.encode(
                inputs=texts,
                batch_size=bs,
                normalize_embeddings=normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        # Convert numpy ndarray to Python float lists
        return embeddings.tolist()
