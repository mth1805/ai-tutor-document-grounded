"""Correlated JSON stage logs without document content or provider error text."""
import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter

logger = logging.getLogger(__name__)
context = ContextVar("ingestion_telemetry", default=None)


def event(stage: str, **fields):
    state = context.get()
    if state is not None and "error_code" in fields:
        state["metrics"].update({key: fields[key] for key in ("error_code", "error_category") if key in fields})
    identifiers = {key: state[key] for key in ("job_id", "document_id", "attempt")} if state else {}
    logger.info(json.dumps({"event": "ingestion_stage", **identifiers, "stage": stage, **fields}))


def metric(**fields):
    state = context.get()
    if state is not None:
        state["metrics"].update(fields)


@contextmanager
def stage(name: str):
    started = perf_counter()
    event(name, status="started")
    status = "completed"
    try:
        yield
    except Exception as exc:
        status = "failed"
        event(name, status="failed", duration_ms=round((perf_counter() - started) * 1000, 2),
              error_category=type(exc).__name__, error_code=f"{name.upper()}_FAILED")
        raise
    finally:
        elapsed = round((perf_counter() - started) * 1000, 2)
        state = context.get()
        if state is not None:
            key = f"{name}_ms"
            state["metrics"][key] = round(state["metrics"].get(key, 0) + elapsed, 2)
        event(name, duration_ms=elapsed, status=status)


class TimedOCR:
    def __init__(self, provider):
        self.provider = provider

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def extract_text_from_image(self, image):
        with stage("OCR"):
            return self.provider.extract_text_from_image(image)
