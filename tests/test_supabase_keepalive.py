"""Offline checks: no .env, application imports, paid APIs, or Supabase requests."""
import base64
import importlib.util
import io
import json
from contextlib import redirect_stdout
from email.message import Message
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("keepalive", ROOT / "scripts/supabase_keepalive.py")
keepalive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(keepalive)
REF = "abcdefghijklmnopqrst"
URL = f"https://{REF}.supabase.co"
KEY = "sb_publishable_offline_test_key"


def settings(key=KEY):
    return {"SUPABASE_DEV_URL": URL, "SUPABASE_DEV_PROJECT_REF": REF, "SUPABASE_DEV_ANON_KEY": key}


def jwt(role="anon", ref=REF):
    payload = base64.urlsafe_b64encode(json.dumps({"role": role, "ref": ref}).encode()).decode().rstrip("=")
    return f"eyJhbGciOiJIUzI1NiJ9.{payload}.offline-signature"


class KeepaliveTests(unittest.TestCase):
    def setUp(self):
        self.opener_patch = patch.object(keepalive, "build_opener")
        self.factory = self.opener_patch.start()
        self.addCleanup(self.opener_patch.stop)
        self.open = self.factory.return_value.open
        self.response = self.open.return_value.__enter__.return_value
        self.response.status = 200
        self.response.headers = Message()
        self.response.headers["Content-Type"] = "application/json; charset=utf-8"
        self.response.read.return_value = json.dumps([keepalive.EXPECTED_ROW]).encode()

    def run_check(self, config=None):
        output = io.StringIO()
        with redirect_stdout(output):
            result = keepalive.main(settings() if config is None else config)
        log = output.getvalue()
        self.assertNotIn(KEY, log)
        self.assertNotIn(URL, log)
        self.assertIn("duration_ms=", log)
        return result, log

    def test_success_is_one_get_of_one_small_non_sensitive_row(self):
        result, log = self.run_check()
        self.assertEqual(result, 0)
        self.assertIn("status=200; rows=1", log)
        self.open.assert_called_once()
        request = self.open.call_args.args[0]
        self.assertEqual(request.full_url, URL + keepalive.QUERY)
        self.assertEqual(request.get_method(), "GET")
        self.assertIsNone(request.data)
        self.assertEqual(request.get_header("Apikey"), KEY)
        self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual(self.open.call_args.kwargs["timeout"], 15)
        self.response.read.assert_called_once_with(1025)
        self.assertIsInstance(self.factory.call_args.args[0], keepalive.NoRedirects)

    def test_legacy_anon_key_supported_without_logging(self):
        key = jwt()
        result, log = self.run_check(settings(key))
        self.assertEqual(result, 0)
        self.assertNotIn(key, log)

    def test_missing_config_never_falls_back_to_application_or_production(self):
        for missing in settings():
            with self.subTest(missing=missing):
                config = settings()
                del config[missing]
                config.update(SUPABASE_URL=URL, SUPABASE_ANON_KEY=KEY)
                self.assertEqual(self.run_check(config)[0], 1)
        self.open.assert_not_called()

    def test_rejects_wrong_project_and_untrusted_urls_before_network(self):
        urls = ["http://" + REF + ".supabase.co", "https://otherprojectrefabcde.supabase.co",
                URL + ".attacker.invalid", URL + ":443", URL + "/rest/v1", URL + "?x=y",
                URL + "#fragment", "https://user:password@" + REF + ".supabase.co", URL + "\n/path"]
        for url in urls:
            with self.subTest(url=url):
                config = settings()
                config["SUPABASE_DEV_URL"] = url
                self.assertEqual(self.run_check(config)[0], 1)
        self.open.assert_not_called()

    def test_rejects_privileged_or_wrong_project_keys_before_network(self):
        for key in ["sb_secret_do_not_use", jwt("service_role"), jwt(ref="anotherprojectrefabcd"),
                    "not-a-key", "a.b.c", "sb_publishable_bad\nInjected: value"]:
            with self.subTest(key_type=key.split(".")[0]):
                result, log = self.run_check(settings(key))
                self.assertEqual(result, 1)
                self.assertNotIn(key, log)
        self.open.assert_not_called()

    def test_empty_incorrect_or_malformed_data_fail(self):
        payloads = [b"[]", b"{}", b"null", b"not json", b"\xff",
                    json.dumps([{"id": 1, "project_label": "production"}]).encode(),
                    json.dumps([keepalive.EXPECTED_ROW] * 2).encode(),
                    json.dumps([{"id": True, "project_label": "ai-tutor-dev"}]).encode()]
        for payload in payloads:
            with self.subTest(payload=payload):
                self.response.read.return_value = payload
                self.assertEqual(self.run_check()[0], 1)

    def test_rejects_oversized_and_non_json_responses(self):
        self.response.read.return_value = b" " * 1025
        self.assertEqual(self.run_check()[0], 1)
        self.response.headers.replace_header("Content-Type", "text/html")
        self.assertIn("not JSON", self.run_check()[1])

    def test_non_200_response_fails(self):
        self.response.status = 206
        self.assertEqual(self.run_check()[0], 1)

    def test_http_errors_fail_without_body_or_credential_leaks_or_retries(self):
        for status in [301, 302, 307, 401, 403, 404, 429, 500, 503]:
            with self.subTest(status=status):
                self.open.reset_mock()
                body = MagicMock()
                self.open.side_effect = HTTPError(URL, status, KEY, Message(), body)
                result, log = self.run_check()
                self.assertEqual(result, 1)
                self.assertIn(f"status={status}", log)
                self.open.assert_called_once()
                body.read.assert_not_called()

    def test_network_timeout_and_unexpected_errors_are_sanitized_and_not_retried(self):
        for exc in [URLError(KEY), TimeoutError(KEY), OSError(KEY), RuntimeError(KEY)]:
            with self.subTest(error=type(exc).__name__):
                self.open.reset_mock()
                self.open.side_effect = exc
                self.assertEqual(self.run_check()[0], 1)
                self.open.assert_called_once()

    def test_redirect_handler_never_builds_a_redirect_request(self):
        self.assertIsNone(keepalive.NoRedirects().redirect_request(None, None, 302, KEY, {}, URL))

    def test_workflow_yaml_schedule_permissions_and_isolation(self):
        # PyYAML already exists in the development environment; the workflow installs nothing.
        import yaml
        workflow = yaml.safe_load((ROOT / ".github/workflows/supabase-keepalive.yml").read_text())
        self.assertEqual(workflow["on"], {"schedule": [{"cron": "17 3 * * *"}], "workflow_dispatch": None})
        self.assertEqual(workflow["permissions"], {"contents": "read"})
        job = workflow["jobs"]["database-read"]
        self.assertEqual(job["runs-on"], "ubuntu-latest")
        self.assertEqual(job["timeout-minutes"], 2)
        self.assertIn("default_branch", job["if"])
        self.assertEqual(len(job["steps"]), 2)
        self.assertFalse(job["steps"][0]["with"]["persist-credentials"])
        step = job["steps"][1]
        self.assertEqual(step["run"], "python3 -I scripts/supabase_keepalive.py")
        self.assertEqual(step["env"], {name: "${{ secrets." + name + " }}" for name in settings()})

    def test_migration_limits_access_to_one_read_only_non_sensitive_row(self):
        sql = (ROOT / "supabase/migrations/20261018140000_keepalive_heartbeat.sql").read_text()
        self.assertIn("CREATE TABLE IF NOT EXISTS", sql)
        self.assertIn("PRIMARY KEY CHECK (id = 1)", sql)
        self.assertIn("ENABLE ROW LEVEL SECURITY", sql)
        self.assertIn("FROM PUBLIC, anon, authenticated, service_role", sql)
        self.assertIn("GRANT SELECT ON TABLE public.keepalive_heartbeat TO anon", sql)
        self.assertEqual(sql.count("CREATE POLICY"), 1)
        self.assertIn("FOR SELECT TO anon", sql)
        self.assertIn("ON CONFLICT (id) DO NOTHING", sql)
        for private_table in ["workspaces", "documents", "messages", "document_chunks", "conversations"]:
            self.assertNotIn("public." + private_table, sql)


if __name__ == "__main__":
    unittest.main()
