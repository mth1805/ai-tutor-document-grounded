"""Cross-Encoder reranker provider with local caching, batch scoring, and device fallback."""
import logging
import time
from collections import OrderedDict
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
        from app.ml.device_runtime import selected_mode, should_call_modal
        self._requested_mode = selected_mode(device or settings.RERANKER_DEVICE)
        self._remote_only = should_call_modal(device or settings.RERANKER_DEVICE)
        self._device = "modal_gpu" if self._remote_only else self._resolve_device(
            "cuda" if self._requested_mode == "cuda" else self._requested_mode
        )
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

        self._model = None
        self._cpu_model = None
        self._load_model()
        if settings.LOCAL_SHARED_MODELS:
            if self._model is not None:
                self._model.max_length = settings.LOCAL_MODEL_MAX_TOKENS
            self._version = "mmarco-minilm-local-windowed-v1"
            self._local_cache = OrderedDict()

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
                if not settings.GPU_FALLBACK_ENABLED:
                    from app.ml.device_runtime import InferenceUnavailable
                    raise InferenceUnavailable("CUDA was explicitly requested but is unavailable and fallback is disabled")
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

        bs = batch_size or settings.RERANKER_BATCH_SIZE

        from app.ml.device_runtime import InferenceUnavailable, cpu_fallback_enabled, infer_remote, run_cpu_fallback, should_call_modal
        started = time.perf_counter()
        if self._remote_only or should_call_modal(self._requested_mode):
            try:
                raw = infer_remote("rerank", pairs, batch_size=bs)
                scores = self._validate_scores(raw, len(pairs))
                logger.info("inference_complete operation=rerank device=modal_gpu input_count=%d batch_size=%d total_ms=%.2f",
                            len(pairs), bs, (time.perf_counter() - started) * 1000)
                return scores
            except Exception as exc:
                if not cpu_fallback_enabled():
                    raise InferenceUnavailable("GPU reranking failed and fallback is disabled") from exc
                logger.warning("inference_gpu_fallback operation=rerank reason=%s", type(exc).__name__)
                return run_cpu_fallback("rerank", self._predict_cpu, pairs, bs, reason=type(exc).__name__)
        if self._requested_mode == "cuda" and self._device == "cpu" and not cpu_fallback_enabled():
            raise InferenceUnavailable("CUDA is unavailable and CPU fallback is disabled")
        try:
            result = self._predict_local(pairs, bs)
            self._log_inference(len(pairs), bs, started)
            return self._validate_scores(result, len(pairs))
        except Exception as exc:
            if not self._device.startswith("cuda") or not cpu_fallback_enabled():
                raise
            if "out of memory" in str(exc).lower() or "cuda oom" in str(exc).lower():
                retry_bs = max(1, bs // 2)
                if retry_bs < bs:
                    try:
                        import torch
                        torch.cuda.empty_cache()
                        logger.warning("inference_gpu_oom_retry operation=rerank original_batch=%d retry_batch=%d", bs, retry_bs)
                        result = self._predict_local(pairs, retry_bs)
                        self._log_inference(len(pairs), retry_bs, started)
                        return result
                    except Exception as retry_exc:
                        exc = retry_exc
            return run_cpu_fallback("rerank", self._predict_cpu, pairs, bs, reason=type(exc).__name__)

    def _log_inference(self, count, batch_size, started):
        import torch
        memory = torch.cuda.memory_allocated() if self._device.startswith("cuda") and torch.cuda.is_available() else 0
        logger.info("inference_complete operation=rerank requested=%s selected=%s gpu=%s input_count=%d batch_size=%d cuda_memory_bytes=%d total_ms=%.2f",
                    self._requested_mode, self._device,
                    torch.cuda.get_device_name(0) if memory else "none", count, batch_size,
                    memory, (time.perf_counter() - started) * 1000)

    def _load_model(self, device=None):
        import torch
        from sentence_transformers.cross_encoder import CrossEncoder
        selected = device or self._device
        if selected == "modal_gpu":
            return
        if (selected == "cpu" and self._cpu_model is not None) or (selected != "cpu" and self._model is not None):
            return
        started = time.perf_counter()
        try:
            model = CrossEncoder(self._model_name, device=selected,
                                 cache_folder=str(self._cache_dir) if self._cache_dir else None)
            if hasattr(model, "model") and hasattr(model.model, "eval"):
                model.model.eval()
            if selected.startswith("cuda"):
                torch.zeros(1, device=selected).sum().item()
                gpu = torch.cuda.get_device_name(0)
            else:
                gpu = None
            if selected == "cpu":
                self._cpu_model = model
            else:
                self._model = model
            if settings.LOCAL_SHARED_MODELS:
                model.max_length = settings.LOCAL_MODEL_MAX_TOKENS
            logger.info("inference_device_selected device=%s gpu=%s model_load_ms=%.2f requested=%s",
                        selected, gpu or "none", (time.perf_counter() - started) * 1000, self._requested_mode)
        except Exception as exc:
            if selected.startswith("cuda") and settings.GPU_FALLBACK_ENABLED:
                logger.warning("inference_device_fallback requested=%s selected=cpu reason=cuda_initialization_failed category=%s",
                               self._requested_mode, type(exc).__name__)
                self._device = "cpu"
                self._load_model("cpu")
            else:
                raise

    def _predict_cpu(self, pairs, batch_size):
        self._load_model("cpu")
        result = self._predict_local(pairs, batch_size, model=self._cpu_model)
        return self._validate_scores(result, len(pairs))

    @staticmethod
    def _validate_scores(output, count):
        import numpy as np
        scores = np.asarray(output, dtype=float).reshape(-1)
        if len(scores) != count or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError("Reranker inference returned invalid scores")
        return [float(score) for score in scores.tolist()]

    def _predict_local(self, pairs, batch_size, model=None):
        import torch
        import numpy as np
        model = model or self._model
        if model is None:
            if self._device == "cpu":
                self._load_model("cpu")
                model = self._cpu_model
            else:
                self._load_model()
                model = self._model
        bs = batch_size

        # CrossEncoder expects list of [query, text] lists/tuples
        formatted_pairs = [[query, text] for query, text in pairs]

        if settings.LOCAL_SHARED_MODELS:
            from app.ml.text_windows import text_windows
            result = []
            tokenizer = model.tokenizer
            with torch.no_grad():
                for query, passage in pairs:
                    key = (query, passage)
                    if key in self._local_cache:
                        self._local_cache.move_to_end(key)
                        result.append(self._local_cache[key])
                        continue
                    query_windows = text_windows(tokenizer, query, settings.LOCAL_MODEL_MAX_TOKENS // 2)
                    logits = []
                    for query_window in query_windows:
                        query_length = len(tokenizer(query_window, add_special_tokens=False)["input_ids"])
                        windows = text_windows(tokenizer, passage, settings.LOCAL_MODEL_MAX_TOKENS - query_length - 4)
                        raw = model.predict([[query_window, window] for window in windows],
                            batch_size=1, show_progress_bar=False, convert_to_numpy=True,
                            activation_fn=torch.nn.Identity())
                        logits.extend(np.asarray(raw, dtype=float).reshape(-1).tolist())
                    score = float(1.0 / (1.0 + np.exp(-np.clip(max(logits), -50, 50))))
                    result.append(score)
                    if len(query) + len(passage) <= 8192:
                        self._local_cache[key] = score
                        if len(self._local_cache) > 32:
                            self._local_cache.popitem(last=False)
            return result

        with torch.no_grad():
            raw_scores = model.predict(
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
        return self._validate_scores(normalized_scores, len(pairs))
