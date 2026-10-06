"""Cross-Encoder reranker provider with local caching, batch scoring, and device fallback."""
import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple

from app.core.config import settings
from app.ml.reranker_base import BaseRerankerProvider

logger = logging.getLogger(__name__)


class CrossEncoderRerankerProvider(BaseRerankerProvider):
    """Centralized CrossEncoder reranker loaded once per process.

    Defaults to multilingual BAAI/bge-reranker-v2-m3 (supporting Vietnamese and English).
    Reuses persistent local weight cache to prevent re-downloads on server restarts.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        self._model_name = model_name or settings.RERANKER_MODEL_NAME
        self._version = "cross-encoder-bge-reranker-v2-m3-v1"
        self._device = self._resolve_device(device or settings.RERANKER_DEVICE)
        self._cache_dir = (
            cache_dir
            or settings.RERANKER_MODEL_CACHE_DIR
            or settings.EMBEDDING_MODEL_CACHE_DIR
        )

        logger.info(
            "Initializing CrossEncoderRerankerProvider: model=%s, device=%s, cache_dir=%s",
            self._model_name,
            self._device,
            self._cache_dir,
        )

        import torch
        from sentence_transformers.cross_encoder import CrossEncoder

        cache_folder_str = str(self._cache_dir) if self._cache_dir else None
        self._model = CrossEncoder(
            self._model_name,
            device=self._device,
            cache_folder=cache_folder_str,
        )

        # Disable dropout layers for deterministic inference
        if hasattr(self._model, "model") and hasattr(self._model.model, "eval"):
            self._model.model.eval()

        logger.info("CrossEncoderRerankerProvider successfully initialized.")

    @staticmethod
    def _resolve_device(requested_device: str) -> str:
        """Resolves target compute device with safe CPU fallback when CUDA is unavailable."""
        import torch

        cuda_available = torch.cuda.is_available()
        req = requested_device.lower().strip()

        if req == "auto":
            return "cuda" if cuda_available else "cpu"
        elif req.startswith("cuda"):
            if not cuda_available:
                logger.warning(
                    "CUDA device '%s' requested for reranker but not available. Falling back to CPU.",
                    requested_device,
                )
                return "cpu"
            return req
        return "cpu"

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
        """Computes sigmoid-normalized relevance scores in [0.0, 1.0] for query-chunk pairs."""
        if not pairs:
            return []

        import torch
        import numpy as np

        bs = batch_size or settings.RERANKER_BATCH_SIZE

        # CrossEncoder expects list of [query, text] lists/tuples
        formatted_pairs = [[query, text] for query, text in pairs]

        with torch.no_grad():
            raw_scores = self._model.predict(
                inputs=formatted_pairs,
                batch_size=bs,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        # Ensure raw_scores is a 1D numpy array
        scores_arr = np.asarray(raw_scores, dtype=float)
        if scores_arr.ndim == 0:
            scores_arr = np.array([float(scores_arr)])
        elif scores_arr.ndim > 1:
            scores_arr = scores_arr.flatten()

        # Apply sigmoid normalization: 1 / (1 + exp(-x)) to map unbounded logits to [0.0, 1.0]
        # For numerical stability:
        normalized_scores = 1.0 / (1.0 + np.exp(-np.clip(scores_arr, -50.0, 50.0)))
        return [float(s) for s in normalized_scores.tolist()]
