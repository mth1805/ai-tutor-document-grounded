"""One-query, no-tools smoke for configured Gemini non-streaming and streaming calls."""
from __future__ import annotations

import asyncio
import json
import sys

from benchmarks.capture_answer_eval import safe_generation_diagnostic


async def run() -> dict:
    from app.core.config import settings

    model = settings.GEMINI_MODEL
    if model != "gemini-3.5-flash-lite":
        raise RuntimeError(
            f"Smoke test requires configured gemini-3.5-flash-lite; configured model is {model!r}."
        )
    try:
        from app.llm.gemini_provider import GeminiProvider
        provider = GeminiProvider()
    except Exception as exc:
        message = getattr(exc, "message", None) or str(exc)
        if settings.GEMINI_API_KEY:
            message = message.replace(settings.GEMINI_API_KEY, "[REDACTED]")
        return {"model": model, "status": "initialization_failed",
                "error": {"exception_class": type(exc).__name__, "provider_message": message}}

    prompt = "Reply with exactly the word OK."
    result = {"model": provider.model_name, "thinking_level": provider.thinking_level,
              "max_output_tokens": provider.max_output_tokens, "tools_enabled": False,
              "prompt": prompt, "non_streaming": None, "streaming": None}

    try:
        response = await provider.generate(prompt=prompt)
        result["non_streaming"] = {"status": "passed", "response_characters": len(response)}
    except Exception as exc:
        result["non_streaming"] = {"status": "failed",
                                   "error": safe_generation_diagnostic(exc, provider.api_key)}

    try:
        fragments = []
        async for fragment in provider.generate_stream(prompt=prompt):
            if fragment:
                fragments.append(fragment)
        result["streaming"] = {"status": "passed", "response_characters": len("".join(fragments))}
    except Exception as exc:
        result["streaming"] = {"status": "failed",
                               "partial_response_characters": len("".join(fragments)),
                               "error": safe_generation_diagnostic(exc, provider.api_key)}
    return result


def main() -> None:
    try:
        result = asyncio.run(run())
    except Exception as exc:
        result = {"status": "initialization_failed", "error": {
            "exception_class": type(exc).__name__, "provider_message": str(exc)}}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    non_stream = result.get("non_streaming", {}).get("status")
    stream = result.get("streaming", {}).get("status")
    if result.get("status") != "initialization_failed" and non_stream == stream == "passed":
        return
    raise SystemExit(1)


if __name__ == "__main__":
    main()
