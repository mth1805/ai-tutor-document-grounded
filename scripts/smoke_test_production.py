"""Small live smoke check for a deployed AI Tutor API; all credentials are env-only."""
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = os.getenv("PRODUCTION_API_BASE_URL", "http://localhost:8000").rstrip("/")
TOKEN = os.getenv("PRODUCTION_BEARER_TOKEN")
CONVERSATION_ID = os.getenv("PRODUCTION_CONVERSATION_ID")
MESSAGE = os.getenv("PRODUCTION_CHAT_MESSAGE", "Give a short summary of the uploaded material.")


def request(path, method="GET", payload=None, token=None, timeout=30):
    headers = {}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(BASE_URL + path, data=data, headers=headers, method=method)
    return urlopen(req, timeout=timeout)


def main() -> int:
    failures = []
    for path in ("/health", "/ready"):
        try:
            with request(path) as response:
                if response.status != 200:
                    failures.append(f"{path}: expected 200, got {response.status}")
                else:
                    print(f"PASS {path}")
        except (HTTPError, URLError, TimeoutError) as exc:
            failures.append(f"{path}: {type(exc).__name__}")

    if not TOKEN or not CONVERSATION_ID:
        try:
            request("/api/v1/workspaces")
            failures.append("unauthenticated protected route unexpectedly succeeded")
        except HTTPError as exc:
            if exc.code == 401:
                print("PASS protected route rejects missing authentication")
            else:
                failures.append(f"protected route: expected 401, got {exc.code}")
        except (URLError, TimeoutError) as exc:
            failures.append(f"protected route: {type(exc).__name__}")
    else:
        try:
            with request(f"/api/v1/conversations/{CONVERSATION_ID}", token=TOKEN) as response:
                if response.status != 200:
                    failures.append(f"conversation access: expected 200, got {response.status}")
                else:
                    print("PASS authenticated conversation access")
            with request(
                f"/api/v1/conversations/{CONVERSATION_ID}/chat",
                method="POST",
                payload={"content": MESSAGE},
                token=TOKEN,
                timeout=130,
            ) as response:
                content_type = response.headers.get("Content-Type", "")
                if "text/event-stream" not in content_type:
                    failures.append(f"chat: expected text/event-stream, got {content_type}")
                else:
                    first_frame = response.readline()
                    if not first_frame:
                        failures.append("chat: stream closed before first event")
                    else:
                        print("PASS authenticated chat SSE headers and initial frame")
        except HTTPError as exc:
            failures.append(f"authenticated request returned HTTP {exc.code}")
        except (URLError, TimeoutError, OSError) as exc:
            failures.append(f"authenticated request: {type(exc).__name__}")

    if failures:
        for failure in failures:
            print(f"FAIL {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
