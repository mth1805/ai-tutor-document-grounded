"""Optional LLM-as-a-judge scoring for prepared answers; never runs in pytest."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from pathlib import Path
import time
from typing import NamedTuple


GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_RAGAS_EMBEDDING_MODEL = "gemini-embedding-2-preview"
DEFAULT_MIN_REQUEST_INTERVAL = 5.0
DEFAULT_MAX_RETRIES = 6
DEFAULT_BACKOFF_BASE = 5.0
DEFAULT_BACKOFF_MAX = 60.0


class EvaluatorConfig(NamedTuple):
    provider: str
    model: str
    embedding_model: str
    api_key: str | None


def resolve_evaluator_config(model: str | None = None) -> EvaluatorConfig:
    """Resolve benchmark-only Gemini judge settings without consulting OpenAI credentials."""
    from app.core.config import settings

    provider = os.environ.get("RAGAS_LLM_PROVIDER", "gemini").strip().lower()
    if provider != "gemini":
        raise ValueError("RAGAS_LLM_PROVIDER must be 'gemini'; only Gemini judging is configured.")
    resolved_model = (model or os.environ.get("RAGAS_LLM_MODEL")
                      or os.environ.get("RAGAS_EVALUATOR_MODEL") or settings.GEMINI_MODEL)
    embedding_model = os.environ.get("RAGAS_EMBEDDING_MODEL", DEFAULT_RAGAS_EMBEDDING_MODEL)
    return EvaluatorConfig(provider, resolved_model, embedding_model, settings.GEMINI_API_KEY)


def create_gemini_compatible_client(api_key: str | None):
    """Create an async Gemini-compatible client for judging and embedding requests."""
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required for Ragas evaluation; OPENAI_API_KEY is not used.")
    from openai import AsyncOpenAI

    # Disable OpenAI SDK retries so the benchmark scheduler is the sole retry layer.
    return AsyncOpenAI(api_key=api_key, base_url=GEMINI_OPENAI_BASE_URL, max_retries=0)


def _error_details(exc: BaseException) -> tuple[int | None, str | None, str, str]:
    """Extract status/code/message and classify Gemini quota or transient errors."""
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    code = getattr(exc, "code", None) or getattr(exc, "status", None)
    message = str(exc)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            code = code or error.get("status") or error.get("code")
            message = str(error.get("message") or message)
    combined = f"{code or ''} {message}".lower()
    if "perday" in combined or "per_day" in combined or "daily" in combined or "requestsperday" in combined:
        quota_classification = "daily_quota"
    elif status == 429 or (status not in {400, 401, 403} and
                           ("resource_exhausted" in combined or "quota" in combined)):
        quota_classification = "rpm_quota"
    else:
        quota_classification = "not_quota"
    transient = (status in {429, 500, 502, 503, 504}
                 or (status not in {400, 401, 403}
                     and ("resource_exhausted" in combined or "unavailable" in combined)))
    if status == 429 or "resource_exhausted" in combined:
        error_code = str(code or "RESOURCE_EXHAUSTED")
    elif status is not None:
        error_code = str(status)
    else:
        error_code = str(code or type(exc).__name__)
    classification = ("daily_quota" if quota_classification == "daily_quota" else
                      "rpm_quota" if quota_classification == "rpm_quota" else
                      "transient_server_error" if transient else "non_transient")
    return (int(status) if status is not None else None, error_code, message, classification)


def _explicit_retry_delay(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    for name in ("retry-after", "Retry-After"):
        try:
            value = headers.get(name)
            if value is not None:
                return max(0.0, float(value))
        except (TypeError, ValueError):
            pass
    for name in ("retry-after-ms", "Retry-After-Ms"):
        try:
            value = headers.get(name)
            if value is not None:
                return max(0.0, float(value) / 1000.0)
        except (TypeError, ValueError):
            pass
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            details = error.get("details", [])
            for detail in details if isinstance(details, list) else []:
                if isinstance(detail, dict) and detail.get("retryDelay"):
                    match = re.fullmatch(r"\s*([0-9.]+)s\s*", str(detail["retryDelay"]))
                    if match:
                        return float(match.group(1))
    match = re.search(r"retryDelay\s*[:=]\s*['\"]?([0-9.]+)s", str(exc), re.IGNORECASE)
    return float(match.group(1)) if match else None


class GeminiRequestScheduler:
    """Sequential, bounded benchmark scheduler around each actual chat request."""

    def __init__(self, min_request_interval: float = DEFAULT_MIN_REQUEST_INTERVAL,
                 max_retries: int = DEFAULT_MAX_RETRIES, backoff_base: float = DEFAULT_BACKOFF_BASE,
                 backoff_max: float = DEFAULT_BACKOFF_MAX, *, sleep=asyncio.sleep,
                 monotonic=time.monotonic, jitter=random.uniform):
        self.min_request_interval = max(0.0, min_request_interval)
        self.max_retries = max(0, max_retries)
        self.backoff_base = max(0.0, backoff_base)
        self.backoff_max = max(0.0, backoff_max)
        self.sleep = sleep
        self.monotonic = monotonic
        self.jitter = jitter
        self._lock = asyncio.Lock()
        self._last_request_at: float | None = None
        self._request_index = 0
        self.request_count = 0
        self.retry_count = 0
        self.total_wait_seconds = 0.0
        self.last_error_code: str | None = None
        self.quota_classification: str | None = None
        self.requests: list[dict] = []

    async def _wait(self, seconds: float) -> None:
        if seconds > 0:
            await self.sleep(seconds)
            self.total_wait_seconds += seconds

    async def _space_request(self) -> None:
        if self._last_request_at is None:
            return
        remaining = self.min_request_interval - (self.monotonic() - self._last_request_at)
        if remaining > 0:
            await self._wait(remaining)

    def wrap(self, request):
        async def scheduled_request(*args, **kwargs):
            async with self._lock:
                self._request_index += 1
                logical_request = {"request_id": self._request_index, "request_count": 0,
                                   "retry_count": 0, "total_wait_ms": 0,
                                   "last_error_code": None, "quota_classification": None}
                self.requests.append(logical_request)
                wait_at_start = self.total_wait_seconds
                try:
                    for attempt in range(self.max_retries + 1):
                        await self._space_request()
                        self.request_count += 1
                        logical_request["request_count"] += 1
                        self._last_request_at = self.monotonic()
                        try:
                            return await request(*args, **kwargs)
                        except Exception as exc:
                            status, error_code, _, classification = _error_details(exc)
                            self.last_error_code = logical_request["last_error_code"] = error_code
                            self.quota_classification = logical_request["quota_classification"] = classification
                            if classification == "daily_quota":
                                raise RuntimeError("Gemini daily quota exhausted; stopping Ragas evaluation.") from exc
                            retryable = classification in {"rpm_quota", "transient_server_error"}
                            if not retryable or attempt >= self.max_retries:
                                raise
                            self.retry_count += 1
                            logical_request["retry_count"] += 1
                            explicit_delay = _explicit_retry_delay(exc)
                            delay = (explicit_delay if explicit_delay is not None else
                                     min(self.backoff_base * (2 ** attempt), self.backoff_max))
                            delay += self.jitter(0.0, min(1.0, max(0.0, delay * 0.1)))
                            await self._wait(delay)
                    raise AssertionError("unreachable retry loop exit")
                finally:
                    logical_request["total_wait_ms"] = round(
                        (self.total_wait_seconds - wait_at_start) * 1000
                    )
        return scheduled_request

    def summary(self) -> dict:
        return {"concurrency": 1, "request_count": self.request_count,
                "retry_count": self.retry_count, "total_wait_ms": round(self.total_wait_seconds * 1000),
                "last_error_code": self.last_error_code,
                "quota_classification": self.quota_classification,
                "requests": self.requests}


def prepare_samples(records: list[dict]) -> list[dict]:
    """Map legacy and captured Answer Evaluation rows into the Ragas input contract."""
    mapped = []
    seen_eval_ids = set()
    for index, record in enumerate(records):
        if "final_generated_answer" in record:
            eval_id = record.get("eval_id")
            if eval_id is not None:
                if eval_id in seen_eval_ids:
                    raise ValueError(f"duplicate eval_id would score the same answer twice: {eval_id}")
                seen_eval_ids.add(eval_id)
            if record.get("generation_status") not in {"generated", "success"} \
                    or not record.get("final_generated_answer", "").strip():
                continue
            evidence = record.get("retrieved_evidence", [])
            record = {
                "question": record.get("question"),
                "answer": record.get("final_generated_answer"),
                "contexts": record.get("ragas_contexts") or [item["content"] for item in evidence if item.get("content")],
                "reference": record.get("reference_answer"),
            }
        required = {"question", "answer", "contexts", "reference"}
        missing = required - record.keys()
        if missing:
            raise ValueError(f"record {index} is missing: {', '.join(sorted(missing))}")
        if not isinstance(record["contexts"], list):
            raise ValueError(f"record {index}: contexts must be a list of strings")
        if not all(isinstance(context, str) for context in record["contexts"]):
            raise ValueError(f"record {index}: contexts must contain strings")
        mapped.append(record)
    return mapped


def metric_arguments(record: dict) -> dict[str, dict[str, str | list[str]]]:
    """Build only the inputs accepted by each Ragas 0.4.3 metric's ascore()."""
    user_input = record["question"]
    response = record["answer"]
    contexts = record["contexts"]
    reference = record["reference"]
    return {
        "faithfulness": {
            "user_input": user_input,
            "response": response,
            "retrieved_contexts": contexts,
        },
        "answer_relevance": {
            "user_input": user_input,
            "response": response,
        },
        "context_precision": {
            "user_input": user_input,
            "reference": reference,
            "retrieved_contexts": contexts,
        },
        "context_recall": {
            "user_input": user_input,
            "retrieved_contexts": contexts,
            "reference": reference,
        },
    }


async def score_records(records: list[dict], model: str, api_key: str | None,
                        embedding_model: str = DEFAULT_RAGAS_EMBEDDING_MODEL,
                        scheduler: GeminiRequestScheduler | None = None) -> dict:
    """Use the current Ragas v0.4 collections API; all metrics call the evaluator LLM."""
    try:
        from ragas.llms import llm_factory
        from ragas.embeddings.base import embedding_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError("Install optional dependencies: pip install -r benchmarks/requirements-ragas.txt") from exc
    # Construct inside asyncio.run()'s loop. Ragas 0.4.3 detects whether
    # embeddings.create is async when building OpenAIEmbeddings and uses
    # aembed_text() from AnswerRelevancy.ascore().
    client = create_gemini_compatible_client(api_key)
    scheduler = scheduler or GeminiRequestScheduler()
    try:
        client.chat.completions.create = scheduler.wrap(client.chat.completions.create)
        llm = llm_factory(model, client=client)
        embeddings = embedding_factory("openai", model=embedding_model, client=client)
        metrics = {"faithfulness": Faithfulness(llm=llm),
                   "answer_relevance": AnswerRelevancy(llm=llm, embeddings=embeddings),
                   "context_precision": ContextPrecision(llm=llm),
                   "context_recall": ContextRecall(llm=llm)}
        scores = {name: [] for name in metrics}
        for record in records:
            arguments = metric_arguments(record)
            for name, metric in metrics.items():
                result = await metric.ascore(**arguments[name])
                scores[name].append(float(result.value))
        return {"metrics": {name: {"mean": sum(values) / len(values) if values else None,
                                    "scores": values} for name, values in scores.items()},
                "request_observability": scheduler.summary()}
    except Exception as exc:
        setattr(exc, "ragas_request_observability", scheduler.summary())
        raise
    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, help="JSON array of saved question/answer/context/reference records")
    parser.add_argument("--model", help="Gemini evaluator model (defaults to RAGAS_LLM_MODEL or GEMINI_MODEL)")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/latest_ragas_results.json"))
    parser.add_argument("--min-request-interval", type=float, default=DEFAULT_MIN_REQUEST_INTERVAL,
                        help="Minimum seconds between Gemini LLM requests (default: 5)")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                        help="Maximum scheduler retries per Gemini request (default: 6)")
    parser.add_argument("--backoff-base", type=float, default=DEFAULT_BACKOFF_BASE,
                        help="Exponential retry backoff base in seconds (default: 5)")
    parser.add_argument("--backoff-max", type=float, default=DEFAULT_BACKOFF_MAX,
                        help="Maximum exponential retry backoff in seconds (default: 60)")
    args = parser.parse_args()
    try:
        config = resolve_evaluator_config(args.model)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        import importlib.metadata
        ragas_version = importlib.metadata.version("ragas")
    except importlib.metadata.PackageNotFoundError:
        ragas_version = None
    print(f"Ragas version: {ragas_version or 'not installed'}")
    print(f"Evaluator provider: {config.provider}")
    print(f"Evaluator model: {config.model}")
    print(f"Evaluator embedding model: {config.embedding_model}")
    print(f"GEMINI_API_KEY configured: {'yes' if config.api_key else 'no'}")
    if not config.api_key:
        raise SystemExit("GEMINI_API_KEY is required for Ragas evaluation; OPENAI_API_KEY is not used.")
    raw_records = json.loads(args.records.read_text(encoding="utf-8"))
    if not isinstance(raw_records, list) or not all(isinstance(record, dict) for record in raw_records):
        raise SystemExit("Answer records must be a JSON array of objects.")
    eligible_record_count = sum(
        1 for record in raw_records
        if "final_generated_answer" not in record
        or (record.get("generation_status") in {"generated", "success"}
            and bool(record.get("final_generated_answer", "").strip()))
    )
    records = prepare_samples(raw_records)
    scheduler = GeminiRequestScheduler(args.min_request_interval, args.max_retries,
                                       args.backoff_base, args.backoff_max)
    if records:
        try:
            scored = asyncio.run(score_records(records, config.model, config.api_key,
                                               config.embedding_model, scheduler))
        except Exception as exc:
            print(json.dumps({"request_observability": getattr(
                exc, "ragas_request_observability", scheduler.summary())}, indent=2))
            raise
        report = scored["metrics"]
        request_observability = scored["request_observability"]
    else:
        report = {name: {"mean": None, "scores": []} for name in
                  ("faithfulness", "answer_relevance", "context_precision", "context_recall")}
        request_observability = scheduler.summary()
    output = {"sample_count": len(records), "eligible_record_count": eligible_record_count,
              "skipped_count": len(raw_records) - eligible_record_count,
              "request_observability": request_observability,
              "evaluator_provider": config.provider,
              "evaluator_model": config.model,
              "evaluator_embedding_model": config.embedding_model,
              "evaluator_base_url": GEMINI_OPENAI_BASE_URL,
              "ragas_version": ragas_version,
              "external_llm_evaluator_used": True, "metrics": report}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    if raw_records and "final_generated_answer" in raw_records[0]:
        from benchmarks.answer_eval import answer_report, load_captured_answers
        captured = load_captured_answers(args.records)
        report, markdown = answer_report(captured, output)
        results_path = args.records.parent / "latest_answer_results.json"
        report_path = args.records.parent / "latest_answer_report.md"
        results_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report_path.write_text(markdown, encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
