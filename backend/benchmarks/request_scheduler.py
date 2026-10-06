"""Sequential Gemini request pacing and transient-error retries for benchmarks."""
from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 6
    min_request_interval_seconds: float = 15.0
    backoff_base_seconds: float = 5.0
    backoff_max_seconds: float = 60.0

    def __post_init__(self):
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.min_request_interval_seconds < 0:
            raise ValueError("min_request_interval_seconds must be non-negative")
        if self.backoff_base_seconds < 0 or self.backoff_max_seconds < 0:
            raise ValueError("backoff values must be non-negative")
        if self.backoff_max_seconds < self.backoff_base_seconds:
            raise ValueError("backoff_max_seconds must be at least backoff_base_seconds")


class RequestScheduler:
    """Enforce minimum spacing between request starts (benchmark concurrency is one)."""

    def __init__(self, min_interval_seconds: float = 15.0,
                 monotonic: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], object] = asyncio.sleep):
        if min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must be non-negative")
        self.min_interval_seconds = min_interval_seconds
        self._monotonic = monotonic
        self._sleep = sleep
        self._last_request_started: float | None = None

    def remaining_interval(self) -> float:
        if self._last_request_started is None:
            return 0.0
        return max(0.0, self._last_request_started + self.min_interval_seconds - self._monotonic())

    async def begin_request(self) -> None:
        wait_seconds = self.remaining_interval()
        if wait_seconds > 0:
            await self._sleep(wait_seconds)
        self._last_request_started = self._monotonic()

    async def wait_before_retry(self, retry_delay_seconds: float) -> float:
        """Wait once for the larger of backoff and rate spacing; return actual wait."""
        wait_seconds = max(max(0.0, retry_delay_seconds), self.remaining_interval())
        if wait_seconds > 0:
            await self._sleep(wait_seconds)
        return wait_seconds


def _diagnostic(exc: Exception, api_key: str | None = None) -> dict:
    details = getattr(exc, "details", {}) or {}
    cause = exc.__cause__ or exc
    fields = ("exception_class", "http_status", "provider_status", "provider_message",
              "classification", "retryable", "retry_delay_seconds", "quota_scope")
    result = {field: details[field] for field in fields if field in details}
    result.setdefault("exception_class", type(cause).__name__)
    result.setdefault("http_status", getattr(cause, "code", None))
    result.setdefault("provider_status", getattr(cause, "status", None))
    result.setdefault("provider_message", getattr(cause, "message", None) or str(cause))
    status = str(result.get("provider_status") or "").upper()
    message = str(result.get("provider_message") or "")
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    result["provider_message"] = message
    if "http_status" not in result:
        result["http_status"] = None
    if "classification" not in result:
        result["classification"] = "unknown"
    if "retryable" not in result:
        result["retryable"] = False
    if status and not result.get("provider_status"):
        result["provider_status"] = status
    return result


def is_transient_error(diagnostic: dict) -> bool:
    if diagnostic.get("quota_scope") == "daily" or diagnostic.get("classification") == "daily_quota":
        return False
    status = diagnostic.get("http_status")
    provider_status = str(diagnostic.get("provider_status") or "").upper()
    if isinstance(status, int) and 400 <= status < 500 and status != 429:
        return False
    if status == 429 or provider_status == "RESOURCE_EXHAUSTED":
        return True
    if status == 503 or provider_status == "UNAVAILABLE":
        return True
    if status in {500, 502, 504}:
        return diagnostic.get("classification") in {"server_error", "timeout"} or bool(diagnostic.get("retryable"))
    return False


def _find_retry_delay(value) -> float | None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower().replace("_", "").replace("-", "")
            if normalized in {"retrydelay", "retryafter", "retrydelayseconds", "retryafterseconds"}:
                parsed = _parse_delay(item)
                if parsed is not None:
                    return parsed
            nested = _find_retry_delay(item)
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for item in value:
            nested = _find_retry_delay(item)
            if nested is not None:
                return nested
    return None


def _parse_delay(value) -> float | None:
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(ms|s|sec|secs|seconds?)?\s*", value, re.IGNORECASE)
        if match:
            number = float(match.group(1))
            return number / 1000 if (match.group(2) or "s").lower() == "ms" else number
    return None


def explicit_retry_delay_seconds(exc: Exception, diagnostic: dict) -> float | None:
    if diagnostic.get("retry_delay_seconds") is not None:
        return _parse_delay(diagnostic["retry_delay_seconds"])
    cause = exc.__cause__ or exc
    response = getattr(cause, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        retry_after = headers.get("Retry-After") or headers.get("retry-after")
        parsed = _parse_delay(retry_after)
        if parsed is not None:
            return parsed
    for value in (getattr(cause, "details", None), getattr(exc, "details", None)):
        parsed = _find_retry_delay(value)
        if parsed is not None:
            return parsed
    match = re.search(r"retry\s+(?:in|after)\s+(\d+(?:\.\d+)?)\s*(ms|s|sec|secs|seconds?)?",
                      str(diagnostic.get("provider_message") or ""), re.IGNORECASE)
    if match:
        return _parse_delay(match.group(1) + (match.group(2) or "s"))
    return None


async def generate_stream_with_retries(provider, prompt: str, system_instruction: str | None,
                                       scheduler: RequestScheduler, policy: RetryPolicy,
                                       api_key: str | None = None,
                                       monotonic: Callable[[], float] = time.monotonic,
                                       uniform: Callable[[float, float], float] = random.uniform) -> dict:
    """Stream one answer with bounded retries; return exactly one final outcome."""
    retry_delays_ms: list[float] = []
    transient_error_codes: list[int] = []
    failed_call_seconds = 0.0

    for retry_index in range(policy.max_retries + 1):
        await scheduler.begin_request()
        started = monotonic()
        first_token_ms = None
        fragments: list[str] = []
        try:
            async for fragment in provider.generate_stream(prompt=prompt, system_instruction=system_instruction):
                if fragment:
                    if first_token_ms is None:
                        first_token_ms = (monotonic() - started) * 1000
                    fragments.append(fragment)
            return {"success": True, "answer": "".join(fragments).strip(),
                    "ttft_ms": first_token_ms,
                    "generation_latency_ms": (monotonic() - started) * 1000,
                    "failure_latency_ms": None, "partial_generation_text": None,
                    "generation_error": None, "retry_count": len(retry_delays_ms),
                    "retry_delays_ms": retry_delays_ms,
                    "transient_error_codes": transient_error_codes,
                    "final_failure_classification": None,
                    "attempt_count": retry_index + 1}
        except Exception as exc:
            call_seconds = monotonic() - started
            failed_call_seconds += call_seconds
            diagnostic = _diagnostic(exc, api_key)
            transient = is_transient_error(diagnostic)
            if transient and isinstance(diagnostic.get("http_status"), int):
                transient_error_codes.append(diagnostic["http_status"])
            if diagnostic.get("classification") == "daily_quota" or diagnostic.get("quota_scope") == "daily":
                logger.warning("Gemini benchmark blocked by daily quota; no retry will be attempted.")
            if not transient or retry_index >= policy.max_retries:
                return {"success": False, "answer": "", "ttft_ms": None,
                        "generation_latency_ms": None,
                        "failure_latency_ms": failed_call_seconds * 1000,
                        "partial_generation_text": "".join(fragments).strip() or None,
                        "generation_error": diagnostic, "retry_count": len(retry_delays_ms),
                        "retry_delays_ms": retry_delays_ms,
                        "transient_error_codes": transient_error_codes,
                        "final_failure_classification": diagnostic.get("classification", "unknown"),
                        "attempt_count": retry_index + 1}

            explicit_delay = explicit_retry_delay_seconds(exc, diagnostic)
            if explicit_delay is None:
                raw_delay = min(policy.backoff_base_seconds * (2 ** retry_index),
                                policy.backoff_max_seconds)
                jitter_ceiling = min(raw_delay * 0.1, max(0.0, policy.backoff_max_seconds - raw_delay))
                requested_delay = min(policy.backoff_max_seconds,
                                      raw_delay + uniform(0.0, jitter_ceiling))
            else:
                requested_delay = explicit_delay
            actual_wait = await scheduler.wait_before_retry(requested_delay)
            retry_delays_ms.append(actual_wait * 1000)
            logger.warning(
                "Transient Gemini benchmark error (HTTP %s, %s); retry %d/%d in %.3fs. "
                "Waiting does not guarantee quota availability or success.",
                diagnostic.get("http_status"), diagnostic.get("provider_status"),
                retry_index + 1, policy.max_retries, actual_wait,
            )

    raise AssertionError("retry loop exited without returning an outcome")
