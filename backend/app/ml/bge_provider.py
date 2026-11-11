"""BGE-M3 embedding provider with local caching, batch encoding, and device fallback."""
import logging
import time
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
        from app.ml.device_runtime import selected_mode, should_call_modal
        self._requested_mode = selected_mode(device or settings.EMBEDDING_DEVICE)
        self._remote_only = should_call_modal(device or settings.EMBEDDING_DEVICE)
        self._device = "modal_gpu" if self._remote_only else self._resolve_device(
            "cuda" if self._requested_mode == "cuda" else self._requested_mode
        )
        self._cache_dir = cache_dir or settings.EMBEDDING_MODEL_CACHE_DIR

        logger.info(
            "Initializing BGEEmbeddingProvider: model=%s, device=%s, cache_dir=%s",
            self._model_name,
            self._device,
            self._cache_dir,
        )

        self._model = None
        self._cpu_model = None
        self._model_load()
        if settings.LOCAL_SHARED_MODELS:
            if self._model is not None:
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
                if not settings.GPU_FALLBACK_ENABLED:
                    from app.ml.device_runtime import InferenceUnavailable
                    raise InferenceUnavailable("CUDA was explicitly requested but is unavailable and fallback is disabled")
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

    def _model_load(self, device: Optional[str] = None):
        import torch
        from sentence_transformers import SentenceTransformer
        selected = device or self._device
        if selected == "modal_gpu":
            return
        if (selected == "cpu" and self._cpu_model is not None) or (selected != "cpu" and self._model is not None):
            return
        started = time.perf_counter()
        try:
            model = SentenceTransformer(
                self._model_name,
                cache_folder=str(self._cache_dir) if self._cache_dir else None,
                device=selected,
            )
            model.eval()
            if selected.startswith("cuda"):
                # A real allocation and kernel operation catches driver/runtime failures
                # that is_available() alone does not identify.
                torch.zeros(1, device=selected).sum().item()
                gpu = torch.cuda.get_device_name(0)
            else:
                gpu = None
            if selected == "cpu":
                self._cpu_model = model
            else:
                self._model = model
            if settings.LOCAL_SHARED_MODELS:
                model.max_seq_length = settings.LOCAL_MODEL_MAX_TOKENS
            logger.info("inference_device_selected device=%s gpu=%s model_load_ms=%.2f requested=%s",
                        selected, gpu or "none", (time.perf_counter() - started) * 1000, self._requested_mode)
        except Exception as exc:
            if selected.startswith("cuda") and settings.GPU_FALLBACK_ENABLED:
                logger.warning("inference_device_fallback requested=%s selected=cpu reason=cuda_initialization_failed category=%s",
                               self._requested_mode, type(exc).__name__)
                self._device = "cpu"
                self._model_load("cpu")
            else:
                raise

    def encode_batch(
        self,
        texts: List[str],
        normalize: bool = True,
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """Encodes texts in micro-batches with torch.no_grad() and L2 normalization."""
        if not texts:
            return []

        bs = batch_size or settings.EMBEDDING_BATCH_SIZE

        from app.ml.device_runtime import InferenceUnavailable, cpu_fallback_enabled, infer_remote, run_cpu_fallback, should_call_modal
        started = time.perf_counter()
        if self._remote_only or should_call_modal(self._requested_mode):
            try:
                raw = infer_remote("embed", texts, normalize=normalize, batch_size=bs)
                vectors = self._validate_vectors(raw, len(texts))
                logger.info("inference_complete operation=embed device=modal_gpu input_count=%d batch_size=%d total_ms=%.2f",
                            len(texts), bs, (time.perf_counter() - started) * 1000)
                return vectors
            except Exception as exc:
                if not cpu_fallback_enabled():
                    raise InferenceUnavailable("GPU embedding failed and fallback is disabled") from exc
                logger.warning("inference_gpu_fallback operation=embed reason=%s", type(exc).__name__)
                return run_cpu_fallback("embed", self._encode_cpu, texts, normalize, bs,
                                        reason=type(exc).__name__)
        if self._requested_mode == "cuda" and self._device == "cpu" and not cpu_fallback_enabled():
            raise InferenceUnavailable("CUDA is unavailable and CPU fallback is disabled")
        try:
            result = self._encode_local(texts, normalize, bs)
            self._log_inference("embed", len(texts), bs, started)
            return self._validate_vectors(result, len(texts))
        except Exception as exc:
            if not self._device.startswith("cuda") or not cpu_fallback_enabled():
                raise
            if self._is_oom(exc):
                retry_bs = max(1, bs // 2)
                if retry_bs < bs:
                    try:
                        import torch
                        torch.cuda.empty_cache()
                        logger.warning("inference_gpu_oom_retry operation=embed original_batch=%d retry_batch=%d", bs, retry_bs)
                        result = self._encode_local(texts, normalize, retry_bs)
                        self._log_inference("embed", len(texts), retry_bs, started)
                        return result
                    except Exception as retry_exc:
                        exc = retry_exc
            return run_cpu_fallback("embed", self._encode_cpu, texts, normalize, bs,
                                    reason=type(exc).__name__)

    @staticmethod
    def _is_oom(exc: Exception) -> bool:
        return "out of memory" in str(exc).lower() or "cuda oom" in str(exc).lower()

    def _log_inference(self, operation: str, count: int, batch_size: int, started: float) -> None:
        import torch
        memory = torch.cuda.memory_allocated() if self._device.startswith("cuda") and torch.cuda.is_available() else 0
        logger.info("inference_complete operation=%s requested=%s selected=%s gpu=%s input_count=%d batch_size=%d cuda_memory_bytes=%d total_ms=%.2f",
                    operation, self._requested_mode, self._device,
                    torch.cuda.get_device_name(0) if memory else "none", count, batch_size,
                    memory, (time.perf_counter() - started) * 1000)

    def _validate_vectors(self, output, count: int) -> List[List[float]]:
        import numpy as np
        array = np.asarray(output, dtype=float)
        if array.shape != (count, self._dimension) or not np.isfinite(array).all():
            raise ValueError("Embedding inference returned invalid shape or non-finite values")
        return array.tolist()

    def _encode_cpu(self, texts, normalize, batch_size):
        self._model_load("cpu")
        result = self._encode_local(texts, normalize, batch_size, model=self._cpu_model)
        return self._validate_vectors(result, len(texts))

    def _encode_local(self, texts, normalize, batch_size, model=None):
        import torch
        model = model or self._model
        if model is None:
            if self._device == "cpu":
                self._model_load("cpu")
                model = self._cpu_model
            else:
                self._model_load()
                model = self._model
        bs = batch_size

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
                    windows = text_windows(model.tokenizer, text, budget)
                    vectors = model.encode(windows, batch_size=1,
                        normalize_embeddings=normalize, show_progress_bar=False, convert_to_numpy=True)
                    # All windows contribute to the canonical chunk's single vector.
                    weights = [max(1, len(model.tokenizer(w, add_special_tokens=False)["input_ids"])) for w in windows]
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
            embeddings = model.encode(
                inputs=texts,
                batch_size=bs,
                normalize_embeddings=normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        # Convert numpy ndarray to Python float lists
        result = self._validate_vectors(embeddings, len(texts))
        return result
