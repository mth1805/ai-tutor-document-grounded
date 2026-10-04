"""Tests for benchmark-only Gemini request pacing and bounded retries."""
from collections import deque

import pytest

from app.llm.exceptions import LLMProviderError
from benchmarks.request_scheduler import RequestScheduler, RetryPolicy, generate_stream_with_retries


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    async def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class ScriptedProvider:
    def __init__(self, outcomes, clock=None, duration=0.0):
        self.outcomes = deque(outcomes)
        self.clock = clock
        self.duration = duration
        self.active = 0
        self.max_active = 0
        self.started_at = []

    async def generate_stream(self, prompt, system_instruction=None):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.clock:
            self.started_at.append(self.clock.monotonic())
            if self.duration:
                await self.clock.sleep(self.duration)
        try:
            outcome = self.outcomes.popleft()
            if isinstance(outcome, Exception):
                raise outcome
            yield outcome
        finally:
            self.active -= 1


def sdk_error(status, provider_status, classification, message="temporary failure", **more):
    details = {"exception_class": "ServerError" if status >= 500 else "ClientError",
               "http_status": status, "provider_status": provider_status,
               "provider_message": message, "classification": classification,
               "retryable": classification in {"quota", "server_error"}, **more}
    return LLMProviderError(f"Gemini API error ({status}): {message}", details=details)


def make_scheduler(clock, interval):
    return RequestScheduler(interval, monotonic=clock.monotonic, sleep=clock.sleep)


def no_jitter(low, high):
    return low


@pytest.mark.asyncio
async def test_sequential_requests_and_minimum_spacing():
    clock = FakeClock()
    provider = ScriptedProvider(["one", "two"], clock=clock, duration=0.1)
    scheduler = make_scheduler(clock, interval=15)
    policy = RetryPolicy(max_retries=0, min_request_interval_seconds=15,
                         backoff_base_seconds=1, backoff_max_seconds=5)
    results = []
    for question in ("first", "second"):
        results.append(await generate_stream_with_retries(
            provider, question, None, scheduler, policy, monotonic=clock.monotonic,
            uniform=no_jitter,
        ))
    assert provider.max_active == 1
    assert provider.started_at[1] - provider.started_at[0] >= 15
    assert [result["answer"] for result in results] == ["one", "two"]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [
    sdk_error(429, "RESOURCE_EXHAUSTED", "quota"),
    sdk_error(503, "UNAVAILABLE", "server_error"),
])
async def test_429_or_503_then_success(error):
    clock = FakeClock()
    provider = ScriptedProvider([error, "recovered"], clock)
    policy = RetryPolicy(2, 0, 5, 60)
    result = await generate_stream_with_retries(provider, "Q", None, make_scheduler(clock, 0),
                                                policy, monotonic=clock.monotonic, uniform=no_jitter)
    assert result["success"] is True
    assert result["answer"] == "recovered"
    assert result["retry_count"] == 1
    assert result["transient_error_codes"] == [error.details["http_status"]]


@pytest.mark.asyncio
async def test_explicit_retry_delay_honored_without_extra_interval_sleep():
    clock = FakeClock()
    error = sdk_error(429, "RESOURCE_EXHAUSTED", "quota", retry_delay_seconds=20)
    provider = ScriptedProvider([error, "ok"], clock, duration=1)
    policy = RetryPolicy(1, 15, 5, 60)
    result = await generate_stream_with_retries(provider, "Q", None, make_scheduler(clock, 15),
                                                policy, monotonic=clock.monotonic, uniform=no_jitter)
    assert result["success"] is True
    assert result["retry_delays_ms"] == [20000]
    assert clock.sleeps == [1, 20, 1]


@pytest.mark.asyncio
async def test_multiple_transient_retries_use_exponential_backoff():
    clock = FakeClock()
    provider = ScriptedProvider([
        sdk_error(429, "RESOURCE_EXHAUSTED", "quota"),
        sdk_error(503, "UNAVAILABLE", "server_error"),
        sdk_error(500, "INTERNAL", "server_error"),
        "eventual success",
    ], clock)
    policy = RetryPolicy(3, 0, 2, 30)
    result = await generate_stream_with_retries(provider, "Q", None, make_scheduler(clock, 0),
                                                policy, monotonic=clock.monotonic, uniform=no_jitter)
    assert result["success"] is True
    assert result["retry_count"] == 3
    assert result["retry_delays_ms"] == [2000, 4000, 8000]
    assert result["transient_error_codes"] == [429, 503, 500]


@pytest.mark.asyncio
async def test_exhausted_retries_return_one_failed_outcome_with_diagnostics():
    clock = FakeClock()
    provider = ScriptedProvider([sdk_error(503, "UNAVAILABLE", "server_error") for _ in range(4)], clock)
    policy = RetryPolicy(3, 0, 1, 5)
    result = await generate_stream_with_retries(provider, "Q", None, make_scheduler(clock, 0),
                                                policy, monotonic=clock.monotonic, uniform=no_jitter)
    assert result["success"] is False
    assert result["answer"] == ""
    assert result["attempt_count"] == 4
    assert result["retry_count"] == 3
    assert result["final_failure_classification"] == "server_error"
    assert result["generation_error"]["http_status"] == 503


@pytest.mark.asyncio
async def test_daily_quota_stops_without_retry():
    clock = FakeClock()
    error = sdk_error(429, "RESOURCE_EXHAUSTED", "daily_quota", quota_scope="daily")
    provider = ScriptedProvider([error, "must not be consumed"], clock)
    result = await generate_stream_with_retries(provider, "Q", None, make_scheduler(clock, 0),
                                                RetryPolicy(10, 0, 1, 60),
                                                monotonic=clock.monotonic, uniform=no_jitter)
    assert result["success"] is False
    assert result["attempt_count"] == 1
    assert result["retry_count"] == 0
    assert result["final_failure_classification"] == "daily_quota"


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [
    sdk_error(400, "INVALID_ARGUMENT", "client_error"),
    sdk_error(403, "PERMISSION_DENIED", "authentication"),
])
async def test_non_transient_400_and_403_are_not_retried(error):
    clock = FakeClock()
    provider = ScriptedProvider([error, "must not be consumed"], clock)
    result = await generate_stream_with_retries(provider, "Q", None, make_scheduler(clock, 0),
                                                RetryPolicy(6, 0, 1, 60),
                                                monotonic=clock.monotonic, uniform=no_jitter)
    assert result["success"] is False
    assert result["attempt_count"] == 1
    assert result["retry_count"] == 0


@pytest.mark.asyncio
async def test_explicit_retry_delay_can_be_read_from_google_error_details():
    clock = FakeClock()
    original = RuntimeError("raw SDK exception")
    original.details = {"error": {"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo",
                                                "retryDelay": "12.5s"}]}}
    error = LLMProviderError("wrapped", details={"http_status": 503,
                                                "provider_status": "UNAVAILABLE",
                                                "provider_message": "retry later",
                                                "classification": "server_error"})
    error.__cause__ = original
    provider = ScriptedProvider([error, "ok"], clock)
    result = await generate_stream_with_retries(provider, "Q", None, make_scheduler(clock, 0),
                                                RetryPolicy(1, 0, 5, 60),
                                                monotonic=clock.monotonic, uniform=no_jitter)
    assert result["retry_delays_ms"] == [12500]

