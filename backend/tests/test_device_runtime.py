"""GPU-first device selection, secure routing, and bounded fallback behavior."""
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.core.config import settings
from app.ml import device_runtime


MODAL_URL = "https://workspace--app-gpu-inference.modal.run"


def _remote_settings(monkeypatch):
    monkeypatch.setattr(settings, "DEVICE_MODE", "auto")
    monkeypatch.setattr(settings, "MODAL_INFERENCE_URL", MODAL_URL)
    monkeypatch.setattr(settings, "MODAL_INFERENCE_TOKEN", "never-log-this-token")
    monkeypatch.setattr(settings, "GPU_WORKER_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(device_runtime, "local_cuda_available", lambda: False)


def _response(payload, status=200):
    request = device_runtime.httpx.Request("POST", MODAL_URL)
    return device_runtime.httpx.Response(status, request=request, json=payload)


def test_device_modes_and_remote_gpu_selection(monkeypatch):
    monkeypatch.setattr(settings, "DEVICE_MODE", "cpu")
    monkeypatch.setattr(settings, "MODAL_INFERENCE_URL", MODAL_URL)
    monkeypatch.setattr(settings, "MODAL_INFERENCE_TOKEN", "test-token")
    monkeypatch.setattr(device_runtime, "local_cuda_available", lambda: False)
    assert device_runtime.selected_mode() == "cpu"
    assert not device_runtime.should_call_modal()
    monkeypatch.setattr(settings, "DEVICE_MODE", "auto")
    assert device_runtime.should_call_modal()
    monkeypatch.setattr(settings, "DEVICE_MODE", "cuda")
    assert device_runtime.selected_mode() == "cuda"


def test_remote_request_uses_bounded_timeout_and_server_token(monkeypatch):
    _remote_settings(monkeypatch)
    monkeypatch.setattr(settings, "MODAL_INFERENCE_URL", MODAL_URL + "/")
    monkeypatch.setattr(settings, "GPU_WORKER_TIMEOUT_SECONDS", 3.5)
    captured = {}
    def post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return _response({"ok": True, "outputs": [[0.25]], "device": "cuda", "gpu": "NVIDIA T4"})
    monkeypatch.setattr(device_runtime.httpx, "post", post)
    assert device_runtime.infer_remote("embed", ["x"], batch_size=1) == [[0.25]]
    assert captured["url"] == MODAL_URL
    assert isinstance(captured["timeout"], device_runtime.httpx.Timeout)
    assert captured["timeout"].connect == 3.5
    assert captured["timeout"].read == 3.5
    assert captured["headers"]["Authorization"] == "Bearer never-log-this-token"


def test_remote_embedding_splits_inputs_into_bounded_ordered_requests(monkeypatch):
    _remote_settings(monkeypatch)
    posted = []
    def post(url, **kwargs):
        values = kwargs["json"]["values"]
        posted.extend(values)
        return _response({"ok": True, "outputs": [[value] for value in values], "device": "cuda", "gpu": "NVIDIA T4"})
    monkeypatch.setattr(device_runtime.httpx, "post", post)
    assert device_runtime.infer_remote("embed", [1, 2, 3, 4, 5], batch_size=2) == [[1], [2], [3], [4], [5]]
    assert posted == [1, 2, 3, 4, 5]


def test_remote_scheduling_wait_has_an_outer_deadline(monkeypatch):
    _remote_settings(monkeypatch)
    monkeypatch.setattr(settings, "GPU_WORKER_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(device_runtime, "_GPU_LATE_WORK_QUARANTINE_SECONDS", 0.01)
    monkeypatch.setattr(device_runtime.httpx, "post", lambda *a, **k: time.sleep(0.1))
    with pytest.raises(device_runtime.InferenceUnavailable, match="request timed out"):
        device_runtime.infer_remote("embed", ["x"])
    time.sleep(0.12)  # Let the deliberately late mock request release its slot.


def test_embedding_remote_invalid_output_falls_back_to_cpu(monkeypatch):
    import torch
    from app.ml.bge_provider import BGEEmbeddingProvider
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(settings, "DEVICE_MODE", "auto")
    monkeypatch.setattr(settings, "EMBEDDING_DEVICE", "cpu")
    monkeypatch.setattr(settings, "MODAL_INFERENCE_URL", MODAL_URL)
    monkeypatch.setattr(settings, "MODAL_INFERENCE_TOKEN", "token")
    monkeypatch.setattr(settings, "GPU_FALLBACK_ENABLED", True)
    provider = BGEEmbeddingProvider()
    monkeypatch.setattr(device_runtime, "infer_remote", lambda *a, **k: [[1.0]])
    expected = [[0.0] * settings.EMBEDDING_DIMENSION]
    monkeypatch.setattr(provider, "_encode_cpu", lambda *a, **k: expected)
    assert provider.encode_batch(["safe input"]) == expected


def test_gpu_oom_retries_smaller_batch(monkeypatch):
    import torch
    from app.ml.bge_provider import BGEEmbeddingProvider
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(settings, "DEVICE_MODE", "cuda")
    monkeypatch.setattr(settings, "GPU_FALLBACK_ENABLED", True)
    monkeypatch.setattr(settings, "MODAL_INFERENCE_URL", None)
    provider = BGEEmbeddingProvider.__new__(BGEEmbeddingProvider)
    provider._requested_mode = "cuda"
    provider._remote_only = False
    provider._device = "cuda"
    provider._dimension = settings.EMBEDDING_DIMENSION
    calls = []
    def encode(texts, normalize, size, model=None):
        calls.append(size)
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory")
        return [[0.0] * settings.EMBEDDING_DIMENSION for _ in texts]
    monkeypatch.setattr(provider, "_encode_local", encode)
    monkeypatch.setattr(provider, "_log_inference", lambda *a, **k: None)
    assert len(provider.encode_batch(["a", "b"], batch_size=8)) == 2
    assert calls == [8, 4]


def test_reranker_gpu_oom_retries_smaller_batch(monkeypatch):
    from app.ml.reranker_provider import CrossEncoderRerankerProvider
    monkeypatch.setattr(settings, "DEVICE_MODE", "cuda")
    monkeypatch.setattr(settings, "GPU_FALLBACK_ENABLED", True)
    monkeypatch.setattr(settings, "MODAL_INFERENCE_URL", None)
    provider = CrossEncoderRerankerProvider.__new__(CrossEncoderRerankerProvider)
    provider._requested_mode = "cuda"
    provider._remote_only = False
    provider._device = "cuda"
    calls = []
    def predict(pairs, size, model=None):
        calls.append(size)
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory")
        return [0.8 for _ in pairs]
    monkeypatch.setattr(provider, "_predict_local", predict)
    monkeypatch.setattr(provider, "_log_inference", lambda *a, **k: None)
    assert provider.predict([("q", "a"), ("q", "b")], batch_size=8) == [0.8, 0.8]
    assert calls == [8, 4]


def test_cpu_fallback_success_and_failure(monkeypatch):
    monkeypatch.setattr(settings, "CPU_FALLBACK_TIMEOUT_SECONDS", 2)
    assert device_runtime.run_cpu_fallback("embed", lambda a: a + 1, 4, reason="gpu_timeout") == 5
    def fail():
        raise RuntimeError("cpu failed")
    with pytest.raises(RuntimeError, match="cpu failed"):
        device_runtime.run_cpu_fallback("embed", fail, reason="gpu_timeout")


def test_cpu_fallback_timeout_retains_capacity_until_worker_finishes(monkeypatch):
    monkeypatch.setattr(settings, "CPU_FALLBACK_TIMEOUT_SECONDS", 0.05)
    started = threading.Event()
    finish = threading.Event()
    def late_work():
        started.set()
        finish.wait(1)
        return "late"
    try:
        with pytest.raises(device_runtime.InferenceUnavailable, match="timed out"):
            device_runtime.run_cpu_fallback("embed", late_work, reason="gpu_timeout")
        assert started.is_set()
        held = 0
        while device_runtime._CPU_ADMISSION.acquire(blocking=False):
            held += 1
        assert held == device_runtime._CPU_MAX_OUTSTANDING - 1
        with pytest.raises(device_runtime.InferenceUnavailable, match="saturated"):
            device_runtime.run_cpu_fallback("embed", lambda: None, reason="gpu_timeout")
        for _ in range(held):
            device_runtime._CPU_ADMISSION.release()
    finally:
        finish.set()
    deadline = time.monotonic() + 1
    recovered = False
    while time.monotonic() < deadline:
        recovered = device_runtime._CPU_ADMISSION.acquire(blocking=False)
        if recovered:
            break
        time.sleep(0.005)
    assert recovered
    device_runtime._CPU_ADMISSION.release()


def test_repeated_cpu_timeouts_cannot_build_an_unbounded_work_queue(monkeypatch):
    monkeypatch.setattr(settings, "CPU_FALLBACK_TIMEOUT_SECONDS", 0.02)
    finishes = [threading.Event() for _ in range(device_runtime._CPU_MAX_OUTSTANDING)]
    starts = [threading.Event() for _ in finishes]
    try:
        for started, finish in zip(starts, finishes):
            def delayed(started=started, finish=finish):
                started.set()
                finish.wait(1)
            with pytest.raises(device_runtime.InferenceUnavailable, match="timed out"):
                device_runtime.run_cpu_fallback("embed", delayed, reason="gpu_timeout")
            assert started.is_set()
        with pytest.raises(device_runtime.InferenceUnavailable, match="saturated"):
            device_runtime.run_cpu_fallback("embed", lambda: None, reason="gpu_timeout")
    finally:
        for finish in finishes:
            finish.set()
    deadline = time.monotonic() + 1
    recovered = 0
    while time.monotonic() < deadline and recovered < device_runtime._CPU_MAX_OUTSTANDING:
        if device_runtime._CPU_ADMISSION.acquire(blocking=False):
            recovered += 1
        else:
            time.sleep(0.005)
    assert recovered == device_runtime._CPU_MAX_OUTSTANDING
    for _ in range(recovered):
        device_runtime._CPU_ADMISSION.release()


def test_cpu_fallback_saturation_rejects_without_queueing(monkeypatch):
    acquired = [device_runtime._CPU_ADMISSION.acquire(blocking=False)
                for _ in range(device_runtime._CPU_MAX_OUTSTANDING)]
    assert all(acquired)
    called = []
    try:
        with pytest.raises(device_runtime.InferenceUnavailable, match="saturated"):
            device_runtime.run_cpu_fallback("embed", lambda: called.append(True), reason="gpu_timeout")
        assert called == []
    finally:
        for _ in acquired:
            device_runtime._CPU_ADMISSION.release()


def test_explicit_cuda_without_enabled_fallback_fails(monkeypatch):
    import torch
    from app.ml.bge_provider import BGEEmbeddingProvider
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(settings, "GPU_FALLBACK_ENABLED", False)
    with pytest.raises(RuntimeError, match="fallback is disabled"):
        BGEEmbeddingProvider._resolve_device("cuda")


def test_cuda_model_initialization_failure_falls_back_once(monkeypatch):
    import sys
    import torch
    from types import SimpleNamespace
    from app.ml.bge_provider import BGEEmbeddingProvider
    calls = []
    class Model:
        def __init__(self, name, **kwargs):
            calls.append(kwargs["device"])
            if kwargs["device"] == "cuda":
                raise RuntimeError("driver initialization failed")
        def eval(self):
            pass
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(settings, "DEVICE_MODE", "auto")
    monkeypatch.setattr(settings, "EMBEDDING_DEVICE", "cuda")
    monkeypatch.setattr(settings, "GPU_FALLBACK_ENABLED", True)
    monkeypatch.setattr(settings, "MODAL_INFERENCE_URL", None)
    monkeypatch.setattr(settings, "LOCAL_SHARED_MODELS", False)
    provider = BGEEmbeddingProvider()
    assert provider.device == "cpu"
    provider._model_load("cpu")
    assert calls == ["cuda", "cpu"]


def test_reranker_remote_failure_falls_back_and_validates(monkeypatch):
    import torch
    from app.ml.reranker_provider import CrossEncoderRerankerProvider
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(settings, "DEVICE_MODE", "auto")
    monkeypatch.setattr(settings, "RERANKER_DEVICE", "cpu")
    monkeypatch.setattr(settings, "MODAL_INFERENCE_URL", MODAL_URL)
    monkeypatch.setattr(settings, "MODAL_INFERENCE_TOKEN", "token")
    monkeypatch.setattr(settings, "GPU_FALLBACK_ENABLED", True)
    provider = CrossEncoderRerankerProvider()
    monkeypatch.setattr(device_runtime, "infer_remote", lambda *a, **k: [2.0])
    monkeypatch.setattr(provider, "_predict_cpu", lambda *a, **k: [0.75])
    assert provider.predict([("query", "passage")]) == [0.75]


def test_modal_url_accepts_https_modal_endpoint_and_normalizes_trailing_slash():
    assert device_runtime.validate_modal_inference_url(
        " HTTPS://workspace--app-gpu-inference.modal.run/ "
    ) == "https://workspace--app-gpu-inference.modal.run"


@pytest.mark.parametrize("url", [
    "http://workspace--app.modal.run", "https://", "https:///missing-host",
    "https://user:password@workspace--app.modal.run", "ftp://workspace--app.modal.run",
    "https://workspace--app.modal.run:444", "https://workspace--app.modal.run/?token=secret",
    "https://not-modal.example", "https://[malformed",
])
def test_invalid_modal_urls_are_rejected_without_echoing_value(url):
    with pytest.raises(device_runtime.InvalidModalInferenceURL) as exc:
        device_runtime.validate_modal_inference_url(url)
    assert "secret" not in str(exc.value).lower()
    assert "password" not in str(exc.value).lower()


@pytest.mark.parametrize("url", [
    "http://workspace--app.modal.run", "https://", "https://user:password@workspace--app.modal.run",
    "ftp://workspace--app.modal.run",
])
def test_invalid_url_never_sends_request_or_token(monkeypatch, caplog, url):
    _remote_settings(monkeypatch)
    monkeypatch.setattr(settings, "MODAL_INFERENCE_URL", url)
    requests = []
    monkeypatch.setattr(device_runtime.httpx, "post", lambda *a, **k: requests.append((a, k)))
    with caplog.at_level(logging.ERROR):
        with pytest.raises(device_runtime.InvalidModalInferenceURL):
            device_runtime.infer_remote("embed", ["private text"])
    assert requests == []
    assert "never-log-this-token" not in caplog.text


def test_cpu_mode_does_not_require_or_validate_modal_url_or_token(monkeypatch):
    monkeypatch.setattr(settings, "DEVICE_MODE", "cpu")
    monkeypatch.setattr(settings, "MODAL_INFERENCE_URL", "http://invalid")
    monkeypatch.setattr(settings, "MODAL_INFERENCE_TOKEN", None)
    monkeypatch.setattr(device_runtime, "local_cuda_available", lambda: False)
    assert not device_runtime.should_call_modal()


@pytest.mark.parametrize("provider_name,operation,input_value", [
    ("BGEEmbeddingProvider", "embed", "x"),
    ("CrossEncoderRerankerProvider", "rerank", ("q", "p")),
])
def test_production_requires_remote_gpu_and_never_falls_back_without_credentials(
    monkeypatch, provider_name, operation, input_value,
):
    import torch
    from app.ml.bge_provider import BGEEmbeddingProvider
    from app.ml.reranker_provider import CrossEncoderRerankerProvider

    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "DEVICE_MODE", "auto")
    monkeypatch.setattr(settings, "EMBEDDING_DEVICE", "auto")
    monkeypatch.setattr(settings, "RERANKER_DEVICE", "auto")
    monkeypatch.setattr(settings, "MODAL_INFERENCE_URL", None)
    monkeypatch.setattr(settings, "MODAL_INFERENCE_TOKEN", None)
    monkeypatch.setattr(settings, "GPU_FALLBACK_ENABLED", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    provider_type = BGEEmbeddingProvider if provider_name == "BGEEmbeddingProvider" else CrossEncoderRerankerProvider
    provider = provider_type()
    local_calls = []
    if operation == "embed":
        monkeypatch.setattr(provider, "_encode_cpu", lambda *args, **kwargs: local_calls.append(True))
        with pytest.raises(device_runtime.InferenceUnavailable, match="fallback is disabled"):
            provider.encode_batch([input_value])
    else:
        monkeypatch.setattr(provider, "_predict_cpu", lambda *args, **kwargs: local_calls.append(True))
        with pytest.raises(device_runtime.InferenceUnavailable, match="fallback is disabled"):
            provider.predict([input_value])
    assert provider.device == "modal_gpu"
    assert local_calls == []


@pytest.mark.parametrize("mode", ["auto", "cpu"])
def test_production_cpu_orchestrator_always_routes_remotely(monkeypatch, mode):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "DEVICE_MODE", mode)
    monkeypatch.setenv("MODAL_INFERENCE_WORKER_ROLE", "cpu_orchestrator")
    monkeypatch.setattr(device_runtime, "local_cuda_available", lambda: False)
    assert device_runtime.should_call_modal()


def test_gpu_worker_with_cuda_available_uses_local_inference(monkeypatch):
    import torch
    from app.ml.bge_provider import BGEEmbeddingProvider

    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "DEVICE_MODE", "cuda")
    monkeypatch.setattr(settings, "EMBEDDING_DEVICE", "cuda")
    monkeypatch.setenv("MODAL_INFERENCE_WORKER_ROLE", "gpu_inference")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    provider = BGEEmbeddingProvider.__new__(BGEEmbeddingProvider)
    provider._requested_mode = "cuda"
    provider._remote_only = device_runtime.should_call_modal("cuda")
    provider._device = "cuda"
    provider._dimension = settings.EMBEDDING_DIMENSION
    local_result = [[0.0] * settings.EMBEDDING_DIMENSION]
    local_calls = []
    monkeypatch.setattr(provider, "_encode_local", lambda *args, **kwargs: local_calls.append(True) or local_result)
    monkeypatch.setattr(provider, "_log_inference", lambda *args, **kwargs: None)
    monkeypatch.setattr(device_runtime, "infer_remote", lambda *args, **kwargs: pytest.fail("GPU worker recursed to Modal"))

    assert provider.encode_batch(["text"]) == local_result
    assert local_calls == [True]


@pytest.mark.parametrize("provider_name", ["BGEEmbeddingProvider", "CrossEncoderRerankerProvider"])
def test_gpu_worker_without_cuda_fails_without_remote_recursion(monkeypatch, provider_name):
    import torch
    from app.ml.bge_provider import BGEEmbeddingProvider
    from app.ml.reranker_provider import CrossEncoderRerankerProvider

    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "DEVICE_MODE", "cuda")
    monkeypatch.setattr(settings, "EMBEDDING_DEVICE", "cuda")
    monkeypatch.setattr(settings, "RERANKER_DEVICE", "cuda")
    monkeypatch.setattr(settings, "GPU_FALLBACK_ENABLED", False)
    monkeypatch.setenv("MODAL_INFERENCE_WORKER_ROLE", "gpu_inference")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    remote_calls = []
    monkeypatch.setattr(device_runtime, "infer_remote", lambda *args, **kwargs: remote_calls.append(True))
    provider_type = BGEEmbeddingProvider if provider_name == "BGEEmbeddingProvider" else CrossEncoderRerankerProvider

    with pytest.raises(device_runtime.InferenceUnavailable, match="CUDA was explicitly requested"):
        provider_type()
    assert not device_runtime.should_call_modal("cuda")
    assert remote_calls == []


@pytest.mark.parametrize("provider_name", ["BGEEmbeddingProvider", "CrossEncoderRerankerProvider"])
def test_production_gpu_oom_never_falls_back_to_cpu(monkeypatch, provider_name):
    import torch
    from app.ml.bge_provider import BGEEmbeddingProvider
    from app.ml.reranker_provider import CrossEncoderRerankerProvider

    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "DEVICE_MODE", "cuda")
    monkeypatch.setattr(settings, "GPU_FALLBACK_ENABLED", True)
    monkeypatch.setenv("MODAL_INFERENCE_WORKER_ROLE", "gpu_inference")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(device_runtime, "infer_remote", lambda *args, **kwargs: pytest.fail("GPU worker recursed to Modal"))
    monkeypatch.setattr(device_runtime, "run_cpu_fallback", lambda *args, **kwargs: pytest.fail("Production GPU fell back to CPU"))
    provider_type = BGEEmbeddingProvider if provider_name == "BGEEmbeddingProvider" else CrossEncoderRerankerProvider
    provider = provider_type.__new__(provider_type)
    provider._requested_mode = "cuda"
    provider._remote_only = False
    provider._device = "cuda"
    if provider_name == "BGEEmbeddingProvider":
        provider._dimension = settings.EMBEDDING_DIMENSION
        monkeypatch.setattr(provider, "_encode_local", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")))
        monkeypatch.setattr(provider, "_log_inference", lambda *args, **kwargs: None)
        invoke = lambda: provider.encode_batch(["text"], batch_size=8)
    else:
        monkeypatch.setattr(provider, "_predict_local", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")))
        monkeypatch.setattr(provider, "_log_inference", lambda *args, **kwargs: None)
        invoke = lambda: provider.predict([("query", "text")], batch_size=8)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        invoke()


@pytest.mark.parametrize("provider_name,operation,input_value", [
    ("BGEEmbeddingProvider", "embed", "x"),
    ("CrossEncoderRerankerProvider", "rerank", ("q", "p")),
])
def test_production_remote_auth_failure_does_not_fall_back(
    monkeypatch, provider_name, operation, input_value,
):
    import torch
    from app.ml.bge_provider import BGEEmbeddingProvider
    from app.ml.reranker_provider import CrossEncoderRerankerProvider

    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "DEVICE_MODE", "auto")
    monkeypatch.setattr(settings, "EMBEDDING_DEVICE", "auto")
    monkeypatch.setattr(settings, "RERANKER_DEVICE", "auto")
    monkeypatch.setattr(settings, "MODAL_INFERENCE_URL", MODAL_URL)
    monkeypatch.setattr(settings, "MODAL_INFERENCE_TOKEN", "token")
    monkeypatch.setattr(settings, "GPU_FALLBACK_ENABLED", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    provider_type = BGEEmbeddingProvider if provider_name == "BGEEmbeddingProvider" else CrossEncoderRerankerProvider
    provider = provider_type()
    local_calls = []
    if operation == "embed":
        monkeypatch.setattr(provider, "_encode_cpu", lambda *args, **kwargs: local_calls.append(True))
        invoke = lambda: provider.encode_batch([input_value])
    else:
        monkeypatch.setattr(provider, "_predict_cpu", lambda *args, **kwargs: local_calls.append(True))
        invoke = lambda: provider.predict([input_value])
    monkeypatch.setattr(device_runtime.httpx, "post", lambda *args, **kwargs: _response({}, status=401))
    with pytest.raises(device_runtime.InferenceUnavailable, match="fallback is disabled"):
        invoke()
    assert local_calls == []


def test_local_development_auto_mode_keeps_cpu_inference_without_modal_credentials(monkeypatch):
    import torch
    from app.ml.bge_provider import BGEEmbeddingProvider

    monkeypatch.setattr(settings, "ENVIRONMENT", "development")
    monkeypatch.setattr(settings, "DEVICE_MODE", "auto")
    monkeypatch.setattr(settings, "EMBEDDING_DEVICE", "auto")
    monkeypatch.setattr(settings, "MODAL_INFERENCE_URL", None)
    monkeypatch.setattr(settings, "MODAL_INFERENCE_TOKEN", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    provider = BGEEmbeddingProvider()
    expected = [[0.0] * settings.EMBEDDING_DIMENSION]
    monkeypatch.setattr(provider, "_encode_local", lambda *args, **kwargs: expected)
    assert provider.device == "cpu"
    assert provider.encode_batch(["local input"]) == expected


def test_missing_token_does_not_send_remote_request(monkeypatch):
    monkeypatch.setattr(settings, "DEVICE_MODE", "auto")
    monkeypatch.setattr(settings, "MODAL_INFERENCE_URL", MODAL_URL)
    monkeypatch.setattr(settings, "MODAL_INFERENCE_TOKEN", None)
    monkeypatch.setattr(device_runtime, "local_cuda_available", lambda: False)
    requests = []
    monkeypatch.setattr(device_runtime.httpx, "post", lambda *a, **k: requests.append((a, k)))
    with pytest.raises(device_runtime.InferenceUnavailable, match="not configured"):
        device_runtime.infer_remote("embed", ["x"])
    assert requests == []


def test_bearer_token_validation_rejects_missing_invalid_and_rotated_values():
    check = device_runtime.is_valid_bearer_token
    assert check("expected", "Bearer expected")
    assert not check("expected", "")
    assert not check("expected", "Bearer wrong")
    assert not check("", "Bearer expected")
    assert not check("new-value", "Bearer old-value")


def test_valid_remote_response_requires_cuda_gpu_and_outputs():
    assert device_runtime._validate_remote_response(
        {"ok": True, "outputs": [[0.1]], "device": "cuda", "gpu": "NVIDIA T4"}
    ) == [[0.1]]
    with pytest.raises(device_runtime.InferenceUnavailable, match="outputs"):
        device_runtime._validate_remote_response(
            {"ok": True, "outputs": None, "device": "cuda", "gpu": "NVIDIA T4"}
        )


@pytest.mark.parametrize("payload", [
    {"ok": True, "outputs": [[0.1]], "device": "cpu", "gpu": "T4"},
    {"ok": True, "outputs": [[0.1]], "gpu": "T4"},
    {"ok": True, "outputs": [[0.1]], "device": "cuda", "gpu": " "},
    {"ok": True, "device": "cuda", "gpu": "T4"},
])
def test_remote_response_rejects_missing_or_incorrect_gpu_metadata(payload):
    with pytest.raises(device_runtime.InferenceUnavailable):
        device_runtime._validate_remote_response(payload)


@pytest.mark.parametrize("status", [401, 429, 500, 503])
def test_remote_http_failures_are_propagated_for_provider_fallback(monkeypatch, status):
    _remote_settings(monkeypatch)
    monkeypatch.setattr(device_runtime.httpx, "post", lambda *a, **k: _response({}, status=status))
    with pytest.raises(device_runtime.httpx.HTTPStatusError):
        device_runtime.infer_remote("embed", ["x"])


@pytest.mark.parametrize("status", [401, 429, 500, 503])
def test_remote_http_failures_fall_back_to_cpu_in_embedding_provider(monkeypatch, status):
    import torch
    from app.ml.bge_provider import BGEEmbeddingProvider
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(settings, "DEVICE_MODE", "auto")
    monkeypatch.setattr(settings, "EMBEDDING_DEVICE", "cpu")
    monkeypatch.setattr(settings, "MODAL_INFERENCE_URL", MODAL_URL)
    monkeypatch.setattr(settings, "MODAL_INFERENCE_TOKEN", "token")
    monkeypatch.setattr(settings, "GPU_FALLBACK_ENABLED", True)
    provider = BGEEmbeddingProvider()
    monkeypatch.setattr(device_runtime.httpx, "post", lambda *a, **k: _response({}, status=status))
    expected = [[0.0] * settings.EMBEDDING_DIMENSION]
    monkeypatch.setattr(provider, "_encode_cpu", lambda *a, **k: expected)
    assert provider.encode_batch(["x"]) == expected


def test_remote_connection_failure_is_propagated(monkeypatch):
    _remote_settings(monkeypatch)
    def fail(*args, **kwargs):
        request = device_runtime.httpx.Request("POST", MODAL_URL)
        raise device_runtime.httpx.ConnectError("connection failed", request=request)
    monkeypatch.setattr(device_runtime.httpx, "post", fail)
    with pytest.raises(device_runtime.httpx.ConnectError):
        device_runtime.infer_remote("embed", ["x"])


def test_remote_metadata_failure_uses_embedding_cpu_fallback(monkeypatch):
    import torch
    from app.ml.bge_provider import BGEEmbeddingProvider
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(settings, "DEVICE_MODE", "auto")
    monkeypatch.setattr(settings, "EMBEDDING_DEVICE", "cpu")
    monkeypatch.setattr(settings, "MODAL_INFERENCE_URL", MODAL_URL)
    monkeypatch.setattr(settings, "MODAL_INFERENCE_TOKEN", "token")
    monkeypatch.setattr(settings, "GPU_FALLBACK_ENABLED", True)
    provider = BGEEmbeddingProvider()
    monkeypatch.setattr(device_runtime.httpx, "post", lambda *a, **k: _response(
        {"ok": True, "outputs": [[1.0]], "device": "cpu", "gpu": None}
    ))
    expected = [[0.0] * settings.EMBEDDING_DIMENSION]
    monkeypatch.setattr(provider, "_encode_cpu", lambda *a, **k: expected)
    assert provider.encode_batch(["x"]) == expected


def test_embedding_and_reranker_outputs_reject_bad_dimensions_counts_and_non_finite_values():
    import math
    from app.ml.bge_provider import BGEEmbeddingProvider
    from app.ml.reranker_provider import CrossEncoderRerankerProvider
    embedding = BGEEmbeddingProvider.__new__(BGEEmbeddingProvider)
    embedding._dimension = 3
    assert embedding._validate_vectors([[0.1, 0.2, 0.3]], 1) == [[0.1, 0.2, 0.3]]
    for output in ([[0.1, 0.2]], [[0.1, 0.2, math.nan]], [[0.1, 0.2, math.inf]]):
        with pytest.raises(ValueError, match="invalid shape or non-finite"):
            embedding._validate_vectors(output, 1)
    assert CrossEncoderRerankerProvider._validate_scores([0.1, 0.9], 2) == [0.1, 0.9]
    for output, count in (([0.1], 2), ([-0.1], 1), ([1.1], 1), ([math.nan], 1), ([math.inf], 1)):
        with pytest.raises(ValueError, match="invalid scores"):
            CrossEncoderRerankerProvider._validate_scores(output, count)


def test_gpu_admission_rejects_without_submitting_work(monkeypatch):
    _remote_settings(monkeypatch)
    acquired = [device_runtime._GPU_ADMISSION.acquire(blocking=False) for _ in range(2)]
    assert all(acquired)
    posted = []
    monkeypatch.setattr(device_runtime.httpx, "post", lambda *a, **k: posted.append((a, k)))
    try:
        with pytest.raises(device_runtime.InferenceUnavailable, match="saturated"):
            device_runtime.infer_remote("embed", ["x"])
        assert posted == []
    finally:
        for _ in acquired:
            device_runtime._GPU_ADMISSION.release()


def test_remote_cancellation_cancels_work_that_has_not_started(monkeypatch):
    from concurrent.futures import CancelledError, Future
    _remote_settings(monkeypatch)
    class CancelledBeforeStart(Future):
        def result(self, timeout=None):
            raise CancelledError()
    future = CancelledBeforeStart()
    class Executor:
        def submit(self, *args, **kwargs):
            return future
    monkeypatch.setattr(device_runtime, "_GPU_EXECUTOR", Executor())
    with pytest.raises(CancelledError):
        device_runtime.infer_remote("embed", ["cancelled"])
    assert future.cancelled()


def test_multiple_late_gpu_requests_are_bounded_and_capacity_recovers(monkeypatch):
    _remote_settings(monkeypatch)
    monkeypatch.setattr(settings, "GPU_WORKER_TIMEOUT_SECONDS", 0.02)
    monkeypatch.setattr(device_runtime, "_GPU_LATE_WORK_QUARANTINE_SECONDS", 0.02)
    started = threading.Event()
    finish_remote = threading.Event()
    def delayed_post(*args, **kwargs):
        started.set()
        finish_remote.wait(1)
        return _response({"ok": True, "outputs": [[0.1]], "device": "cuda", "gpu": "NVIDIA T4"})
    monkeypatch.setattr(device_runtime.httpx, "post", delayed_post)

    def timed_call():
        with pytest.raises(device_runtime.InferenceUnavailable, match="timed out"):
            device_runtime.infer_remote("embed", ["late"])
    callers = ThreadPoolExecutor(max_workers=2)
    tasks = [callers.submit(timed_call) for _ in range(2)]
    assert started.wait(0.5)
    for task in tasks:
        task.result(timeout=0.5)
    try:
        with pytest.raises(device_runtime.InferenceUnavailable, match="saturated"):
            device_runtime.infer_remote("embed", ["third"])
        assert device_runtime.run_cpu_fallback("embed", lambda: [[0.0]], reason="gpu_timeout") == [[0.0]]
    finally:
        finish_remote.set()
        callers.shutdown(wait=True)

    deadline = time.monotonic() + 1
    recovered = 0
    while time.monotonic() < deadline and recovered < 2:
        if device_runtime._GPU_ADMISSION.acquire(blocking=False):
            recovered += 1
        else:
            time.sleep(0.005)
    assert recovered == 2
    for _ in range(recovered):
        device_runtime._GPU_ADMISSION.release()
