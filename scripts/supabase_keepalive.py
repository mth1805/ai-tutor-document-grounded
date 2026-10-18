"""One bounded development database read; stdlib only, no .env, app imports or retries."""
import base64
import binascii
import json
import os
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

QUERY = "/rest/v1/keepalive_heartbeat?select=id,project_label&id=eq.1&limit=1"
EXPECTED_ROW = {"id": 1, "project_label": "ai-tutor-dev"}
TIMEOUT_SECONDS = 15
MAX_RESPONSE_BYTES = 1024


class CheckFailed(Exception):
    """Contains only fixed, safe diagnostics for workflow logs."""


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the API key to a redirected destination.
        return None


def configuration(environ):
    """Require confirmed development settings; never fall back to app credentials."""
    names = ("SUPABASE_DEV_URL", "SUPABASE_DEV_PROJECT_REF", "SUPABASE_DEV_ANON_KEY")
    values = [environ.get(name, "").strip() for name in names]
    if not all(values):
        raise CheckFailed("missing required SUPABASE_DEV_* GitHub secrets")
    url, project_ref, key = values
    if not re.fullmatch(r"[a-z0-9]{20}", project_ref):
        raise CheckFailed("invalid development project reference")
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != f"{project_ref}.supabase.co"
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment
            or any(c.isspace() for c in url)):
        raise CheckFailed("development URL must match the confirmed project reference exactly")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", key):
        raise CheckFailed("invalid public API key format")
    if not key.startswith("sb_publishable_"):
        try:
            parts = key.split(".")
            if len(parts) != 3:
                raise ValueError
            claims = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
            if claims.get("role") != "anon" or claims.get("ref") != project_ref:
                raise ValueError
        except (ValueError, TypeError, AttributeError, binascii.Error):
            # Configuration guard only; Supabase verifies/authenticates the actual key.
            raise CheckFailed("use a publishable key or an anon JWT for the confirmed development project") from None
    return url.rstrip("/"), key


def check(environ):
    url, key = configuration(environ)
    request = Request(url + QUERY, method="GET", headers={
        "apikey": key, "Accept": "application/json", "Cache-Control": "no-cache",
    })
    # Never print the URL, key, body, HTTP reason, or exception text.
    with build_opener(NoRedirects()).open(request, timeout=TIMEOUT_SECONDS) as response:
        if response.status != 200:
            raise CheckFailed("database read did not return HTTP 200")
        if response.headers.get_content_type() != "application/json":
            raise CheckFailed("database response is not JSON")
        payload = response.read(MAX_RESPONSE_BYTES + 1)
        if len(payload) > MAX_RESPONSE_BYTES:
            raise CheckFailed("database response exceeded the size limit")
        try:
            rows = json.loads(payload)
        except (ValueError, UnicodeError):
            raise CheckFailed("database response contains invalid JSON") from None
        if rows != [EXPECTED_ROW] or type(rows[0].get("id")) is not int:
            raise CheckFailed("expected development heartbeat row missing or incorrect")


def main(environ=None):
    started = time.monotonic()
    try:
        check(os.environ if environ is None else environ)
    except CheckFailed as exc:
        outcome = f"FAIL: {exc}"
    except HTTPError as exc:
        outcome = f"FAIL: database HTTP status={exc.code}"
    except (URLError, TimeoutError, OSError):
        outcome = "FAIL: database network/TLS/timeout error"
    except Exception:
        outcome = "FAIL: unexpected check failure"
    else:
        outcome = "PASS: database HTTP status=200; rows=1"
    print(f"{outcome}; duration_ms={int((time.monotonic() - started) * 1000)}")
    return 0 if outcome.startswith("PASS:") else 1


if __name__ == "__main__":
    raise SystemExit(main())
