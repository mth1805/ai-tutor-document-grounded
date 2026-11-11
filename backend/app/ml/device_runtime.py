"""Bounded Modal inference calls and common device policy helpers."""
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
import hmac
import logging
import os
import threading
import time
from typing import Any, Callable
from urllib.parse import urlsplit

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# Match the default Modal inference container count. Admission is process-local;
# the Modal function still enforces its global max_containers limit.
_GPU_MAX_OUTSTANDING = 2
_CPU_MAX_OUTSTANDING = 4
_GPU_LATE_WORK_QUARANTINE_SECONDS = 60.0
_CPU_EXECUTOR = ThreadPoolExecutor(max_workers=_CPU_MAX_OUTSTANDING, thread_name_prefix="ml-cpu-fallback")
_GPU_EXECUTOR = ThreadPoolExecutor(max_workers=_GPU_MAX_OUTSTANDING, thread_name_prefix="modal-gpu-request")
_GPU_ADMISSION = threading.BoundedSemaphore(_GPU_MAX_OUTSTANDING)
_CPU_ADMISSION = threading.BoundedSemaphore(_CPU_MAX_OUTSTANDING)


class InferenceUnavailable(RuntimeError):
    """Raised when the requested device and any configured fallback fail."""


class InvalidModalInferenceURL(ValueError):
    """Raised when the configured private inference endpoint is not trusted."""


def validate_modal_inference_url(value: str | None) -> str:
    """Validate a generated Modal HTTPS endpoint without echoing its value."""
    if not isinstance(value, str) or not value.strip():
        raise InvalidModalInferenceURL("MODAL_INFERENCE_URL must be a non-empty HTTPS Modal URL")
    candidate = value.strip()
    if any(character.isspace() or ord(character) < 0x21 for character in candidate):
        raise InvalidModalInferenceURL("MODAL_INFERENCE_URL is malformed")
    try:
        parsed = urlsplit(candidate)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise InvalidModalInferenceURL("MODAL_INFERENCE_URL is malformed") from exc

    if parsed.scheme.lower() != "https":
        raise InvalidModalInferenceURL("MODAL_INFERENCE_URL must use HTTPS")
    if not hostname:
        raise InvalidModalInferenceURL("MODAL_INFERENCE_URL must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise InvalidModalInferenceURL("MODAL_INFERENCE_URL must not contain credentials")
    normalized_host = hostname.rstrip(".").lower()
    if not (normalized_host.endswith(".modal.run") or normalized_host.endswith(".modal.com")):
        raise InvalidModalInferenceURL("MODAL_INFERENCE_URL must use a Modal endpoint hostname")
    if port not in (None, 443):
        raise InvalidModalInferenceURL("MODAL_INFERENCE_URL must use the standard HTTPS port")
    if parsed.query or parsed.fragment:
        raise InvalidModalInferenceURL("MODAL_INFERENCE_URL must not contain a query or fragment")
    normalized = parsed._replace(scheme="https").geturl()
    return normalized.rstrip("/")


def is_valid_bearer_token(expected: str, supplied: str) -> bool:
    """Constant-time endpoint token check; callers must not log either value."""
    return bool(expected) and hmac.compare_digest(supplied, f"Bearer {expected}")


def selected_mode(legacy_device: str = "cpu") -> str:
    mode = settings.DEVICE_MODE.strip().lower()
    if mode not in {"auto", "cuda", "cpu"}:
        raise ValueError("DEVICE_MODE must be auto, cuda, or cpu")
    # Retain explicit legacy per-model cuda/auto overrides during migration.
    if mode == "auto" and legacy_device.strip().lower().startswith("cuda"):
        return "cuda"
    return mode


def local_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def should_call_modal(legacy_device: str = "cpu") -> bool:
    # The explicit Modal role takes precedence over CUDA availability so the GPU
    # endpoint cannot recursively call itself when its CUDA runtime is broken.
    worker_role = os.getenv("MODAL_INFERENCE_WORKER_ROLE", "").strip().lower()
    if worker_role == "gpu_inference":
        return False

    # Production API/queue/ingestion processes always route model inference to
    # the dedicated GPU worker. Missing endpoint settings fail at call time.
    # Development retains local CPU behavior when Modal is not configured.
    mode = selected_mode(legacy_device)
    if settings.ENVIRONMENT.lower() in {"production", "prod"}:
        return True
    return mode != "cpu" and not local_cuda_available() and bool(
        settings.MODAL_INFERENCE_URL and settings.MODAL_INFERENCE_TOKEN
    )


def cpu_fallback_enabled() -> bool:
    """Production inference must fail clearly rather than run the large CPU models."""
    return settings.GPU_FALLBACK_ENABLED and settings.ENVIRONMENT.lower() not in {"production", "prod"}


def _release_when_done(
    future: Future,
    admission: threading.BoundedSemaphore,
    *,
    operation: str,
    worker: str,
    started: float,
    caller_timed_out: threading.Event,
) -> None:
    """Keep a slot occupied until underlying work really finishes."""
    def release() -> None:
        admission.release()

    try:
        if caller_timed_out.is_set():
            logger.info(
                "inference_late_work_finished worker=%s operation=%s duration_ms=%.2f",
                worker, operation, (time.perf_counter() - started) * 1000,
            )
        elif future.cancelled():
            logger.info("inference_task_cancelled worker=%s operation=%s", worker, operation)
    finally:
        if worker == "gpu" and caller_timed_out.is_set() and not future.cancelled():
            # Disconnecting the HTTP client cannot prove that Modal stopped an
            # already-accepted inference. Keep this slot quarantined for the
            # endpoint's minimum execution timeout to dampen timeout/retry waves.
            timer = threading.Timer(_GPU_LATE_WORK_QUARANTINE_SECONDS, release)
            timer.daemon = True
            timer.start()
        else:
            release()


def _submit_bounded(executor, admission, fn, *args, operation: str, worker: str, **kwargs) -> tuple[Future, threading.Event, float]:
    if not admission.acquire(blocking=False):
        logger.warning("inference_admission_rejected worker=%s operation=%s", worker, operation)
        raise InferenceUnavailable(f"{worker} inference capacity is saturated")
    started = time.perf_counter()
    caller_timed_out = threading.Event()
    try:
        future = executor.submit(fn, *args, **kwargs)
    except BaseException:
        admission.release()
        raise
    future.add_done_callback(lambda done: _release_when_done(
        done, admission, operation=operation, worker=worker, started=started,
        caller_timed_out=caller_timed_out,
    ))
    return future, caller_timed_out, started


def _validate_remote_response(result: Any) -> Any:
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise InferenceUnavailable("Modal GPU returned an invalid response")
    if result.get("device") != "cuda":
        raise InferenceUnavailable("Modal GPU response did not confirm CUDA execution")
    gpu = result.get("gpu")
    if not isinstance(gpu, str) or not gpu.strip():
        raise InferenceUnavailable("Modal GPU response did not identify a GPU")
    if not isinstance(result.get("outputs"), list):
        raise InferenceUnavailable("Modal GPU response omitted inference outputs")
    return result["outputs"]


def infer_remote(operation: str, values: list, *, normalize: bool = True, batch_size: int = 16) -> Any:
    """Split large input lists into bounded requests and preserve output order."""
    if batch_size < 1:
        raise ValueError("Modal inference batch size must be positive")
    request_size = min(batch_size, 128)
    outputs = []
    for offset in range(0, len(values), request_size):
        outputs.extend(_infer_remote_batch(
            operation, values[offset:offset + request_size], normalize=normalize,
            batch_size=request_size,
        ))
    return outputs


def _infer_remote_batch(operation: str, values: list, *, normalize: bool, batch_size: int) -> Any:
    """Call Modal with bounded admission and an end-to-end caller deadline.

    Timing out cancels a request only if it has not started in the local
    executor. Once the HTTP request is running, Modal may continue inference;
    admission capacity remains occupied until that local request actually ends.
    """
    if not should_call_modal():
        raise InferenceUnavailable("Modal GPU inference is not configured")
    try:
        endpoint = validate_modal_inference_url(settings.MODAL_INFERENCE_URL)
    except InvalidModalInferenceURL as exc:
        logger.error("inference_remote_configuration_invalid reason=%s", str(exc))
        raise
    token = settings.MODAL_INFERENCE_TOKEN
    if not token:
        logger.warning("inference_remote_configuration_missing_token")
        raise InferenceUnavailable("Modal GPU inference credentials are not configured")

    total_timeout = float(settings.GPU_WORKER_TIMEOUT_SECONDS)
    http_timeout = httpx.Timeout(total_timeout, connect=min(5.0, total_timeout))
    future, caller_timed_out, started = _submit_bounded(
        _GPU_EXECUTOR, _GPU_ADMISSION, httpx.post, operation=operation, worker="gpu",
        url=endpoint,
        json={"operation": operation, "values": values, "normalize": normalize, "batch_size": batch_size},
        headers={"Authorization": f"Bearer {token}"},
        timeout=http_timeout,
    )
    try:
        # This deadline includes executor handoff, Modal scheduling/queueing,
        # network time, and inference as observed by this caller.
        response = future.result(timeout=total_timeout)
    except FutureTimeout as exc:
        caller_timed_out.set()
        cancelled_locally = future.cancel()
        logger.warning(
            "inference_remote_timeout operation=%s local_cancelled=%s remote_may_continue=%s",
            operation, cancelled_locally, not cancelled_locally,
        )
        raise InferenceUnavailable("Modal GPU inference request timed out") from exc
    except BaseException:
        # This can stop work that has not started. It cannot terminate a running
        # Python thread or guarantee cancellation of an already-sent Modal call.
        future.cancel()
        raise

    response.raise_for_status()
    response_body = response.json()
    outputs = _validate_remote_response(response_body)
    logger.info(
        "inference_remote_complete operation=%s requested=%s selected=cuda gpu=%s input_count=%d batch_size=%d duration_ms=%.2f",
        operation, selected_mode(), response_body["gpu"].strip(), len(values), batch_size,
        (time.perf_counter() - started) * 1000,
    )
    return outputs


def run_cpu_fallback(operation: str, fn: Callable, *args, reason: str, **kwargs):
    """Run bounded CPU fallback; timed-out threads retain their slot until done."""
    logger.warning(
        "inference_device_fallback requested=%s selected=cpu reason=%s operation=%s",
        selected_mode(), reason, operation,
    )
    future, caller_timed_out, started = _submit_bounded(
        _CPU_EXECUTOR, _CPU_ADMISSION, fn, *args,
        operation=operation, worker="cpu_fallback", **kwargs,
    )
    try:
        result = future.result(timeout=settings.CPU_FALLBACK_TIMEOUT_SECONDS)
    except FutureTimeout as exc:
        caller_timed_out.set()
        cancelled_locally = future.cancel()
        logger.warning(
            "inference_cpu_fallback_timeout operation=%s local_cancelled=%s running_work_continues=%s",
            operation, cancelled_locally, not cancelled_locally,
        )
        raise InferenceUnavailable("CPU inference fallback timed out") from exc
    logger.info("inference_cpu_fallback_complete operation=%s duration_ms=%.2f", operation,
                (time.perf_counter() - started) * 1000)
    return result
