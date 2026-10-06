"""Development-only, single-owner model lifecycle and bounded inference queue."""
import asyncio
import itertools
import logging
from functools import partial
from fastapi.concurrency import run_in_threadpool
from app.core.config import settings

logger = logging.getLogger(__name__)
ready = asyncio.Event()
_queue = asyncio.PriorityQueue()
_sequence = itertools.count()


async def serve_models():
    from app.ml.loader import warmup_embedding_model, warmup_reranker_model
    try:
        await run_in_threadpool(warmup_embedding_model)
        await run_in_threadpool(warmup_reranker_model)
        ready.set()
        logger.info("local_models_ready owner=backend concurrency=1")
        while True:
            _, _, future, operation = await _queue.get()
            try:
                if not future.cancelled():
                    result = await run_in_threadpool(operation)
                    if not future.done():
                        future.set_result(result)
            except asyncio.CancelledError:
                if not future.done():
                    future.set_exception(RuntimeError("Local model runtime stopped"))
                raise
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)
            finally:
                _queue.task_done()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("local_model_startup_failed category=%s", type(exc).__name__)
    finally:
        ready.clear()
        while not _queue.empty():
            _, _, future, _ = _queue.get_nowait()
            if not future.done():
                future.set_exception(RuntimeError("Local model runtime stopped"))
            _queue.task_done()


async def infer(function, *args, background=False, **kwargs):
    if not settings.LOCAL_SHARED_MODELS:
        return await run_in_threadpool(function, *args, **kwargs)
    if not ready.is_set():
        raise RuntimeError("Local models are warming up or unavailable")
    future = asyncio.get_running_loop().create_future()
    # Interactive work can overtake up to eight queued operations, while aging
    # prevents a steady stream of queries from starving ingestion indefinitely.
    sequence = next(_sequence)
    await _queue.put((sequence + (8 if background else 0), sequence, future, partial(function, *args, **kwargs)))
    return await future
