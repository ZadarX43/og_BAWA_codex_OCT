#!/usr/bin/env python3
"""Live Cloudflare Worker checkout/account smoke test.

This script uses a synthetic short-lived subscriber record to exercise the
deployed Worker without requiring inbox access. It does not read or print
secret values.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKER_DIR = ROOT / "worker"
REPORT_PATH = ROOT / "reports" / "latest" / "LIVE_WORKER_ACCOUNT_SMOKE_REPORT.md"
BASE_URL = "https://odds-genius-worker.hughcwade.workers.dev"
SITE_URL = "https://og-bawa-codex-oct.pages.dev"
PRICE_ID = "price_1TTRvkDoSY9qcu1woHJ5iKSZ"


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


NO_REDIRECT_OPENER = urllib.request.build_opener(NoRedirectHandler)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def wrangler(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["wrangler", *args],
        cwd=WORKER_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(f"wrangler {' '.join(args)} failed:\n{completed.stdout}")
    return completed


def http_json(path: str, *, method: str = "GET", body: Any = None, headers: dict[str, str] | None = None):
    args = ["curl", "-sS", "-i", "-X", method, "-H", "accept: application/json"]
    request_headers = headers or {}
    for key, value in request_headers.items():
        args.extend(["-H", f"{key}: {value}"])
    if body is not None:
        args.extend(["-H", "content-type: application/json", "--data", json.dumps(body)])
    args.append(f"{BASE_URL}{path}")
    completed = subprocess.run(
        args,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    status, response_headers, raw = parse_curl_response(completed.stdout or "")
    try:
        payload = json.loads(raw or "{}")
    except json.JSONDecodeError:
        payload = {"raw": raw}
    return status, response_headers, payload


def http_no_redirect(path: str):
    completed = subprocess.run(
        ["curl", "-sS", "-i", f"{BASE_URL}{path}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return (*parse_curl_response(completed.stdout or ""),)


def parse_curl_response(raw: str):
    head, _, body = raw.partition("\r\n\r\n")
    if not body:
        head, _, body = raw.partition("\n\n")
    lines = head.splitlines()
    status = 0
    headers: dict[str, str] = {}
    if lines:
        match = re.search(r"\s(\d{3})\s", lines[0])
        status = int(match.group(1)) if match else 0
    for line in lines[1:]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        headers[key.strip()] = value.strip()
    return status, headers, body


def cookie_header_from_set_cookie(value: str) -> str:
    cookie = SimpleCookie()
    cookie.load(value)
    morsel = cookie.get("og_premium_session")
    if not morsel or not morsel.value:
        raise RuntimeError("verify route did not return og_premium_session cookie")
    return f"og_premium_session={morsel.value}"


def header_value(headers: dict[str, str], name: str) -> str:
    wanted = name.lower()
    for key, value in headers.items():
        if key.lower() == wanted:
            return value
    return ""


def put_kv(key: str, value: dict[str, Any], ttl: int = 1800) -> None:
    wrangler(
        [
            "kv",
            "key",
            "put",
            key,
            json.dumps(value, separators=(",", ":")),
            "--binding",
            "SUBSCRIBER_STATE",
            "--remote",
            "--preview",
            "false",
            "--ttl",
            str(ttl),
        ]
    )


def delete_kv(key: str) -> None:
    wrangler(
        ["kv", "key", "delete", key, "--binding", "SUBSCRIBER_STATE", "--remote", "--preview", "false"],
        check=False,
    )


def cleanup_d1(email: str) -> None:
    safe_email = email.replace("'", "''")
    command = (
        f"DELETE FROM auth_events WHERE email_normalized = '{safe_email}'; "
        f"DELETE FROM users WHERE email_normalized = '{safe_email}';"
    )
    wrangler(["d1", "execute", "ACCOUNT_DB", "--remote", "--command", command, "--yes"], check=False)


def assert_status(name: str, condition: bool, details: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "ok": bool(condition),
        "details": details,
    }


def write_report(checks: list[dict[str, Any]], facts: list[str]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Live Worker Account Smoke Report",
        "",
        f"Generated: `{utc_now()}`",
        f"Worker: `{BASE_URL}`",
        f"Site: `{SITE_URL}`",
        "",
        "## Summary",
        "",
        f"- Checks: `{len(checks)}`",
        f"- Passed: `{sum(1 for item in checks if item['ok'])}`",
        f"- Failed: `{sum(1 for item in checks if not item['ok'])}`",
        "",
        "## Facts",
        "",
        *[f"- {fact}" for fact in facts],
        "",
        "## Checks",
        "",
        *[f"- {'PASS' if item['ok'] else 'FAIL'} `{item['name']}` - {item['details']}" for item in checks],
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    stamp = str(int(time.time()))
    email = f"codex-cloudflare-smoke+{stamp}@oddsgenius.test"
    customer_id = f"cus_codex_smoke_{stamp}"
    subscription_id = f"sub_codex_smoke_{stamp}"
    token = f"SMK{stamp}"
    record = {
        "customer_id": customer_id,
        "subscription_id": subscription_id,
        "status": "active",
        "price_id": PRICE_ID,
        "current_period_end": "2026-06-30T00:00:00.000Z",
        "updated_at": utc_now(),
        "email": email,
        "source_event_type": "codex_live_worker_smoke",
    }
    magic_record = {
        "email": email,
        "customer_id": customer_id,
        "subscription_id": subscription_id,
        "issued_at": utc_now(),
        "exp": int(time.time()) + 900,
    }
    kv_keys = [
        f"subscription:{subscription_id}",
        f"customer:{customer_id}",
        f"email:{email}",
        f"auth_magic:{token}",
    ]
    checks: list[dict[str, Any]] = []
    facts: list[str] = []
    try:
        put_kv(f"subscription:{subscription_id}", record)
        put_kv(f"customer:{customer_id}", record)
        put_kv(f"email:{email}", record)
        put_kv(f"auth_magic:{token}", magic_record, ttl=900)

        health_status, _, health = http_json("/health")
        checks.append(assert_status("health", health_status == 200 and health.get("ok") is True, f"HTTP {health_status}"))
        env_summary = health.get("env_summary") or {}
        required_env = [
            "has_site_url",
            "has_premium_data_source",
            "has_stripe_secret_key",
            "has_stripe_webhook_secret",
            "has_stripe_price_id",
            "has_subscriber_state_binding",
            "has_premium_token_secret",
            "has_auth_magic_link_secret",
            "has_auth_session_secret",
            "has_resend_api_key",
            "has_auth_email_from",
            "has_account_db",
        ]
        missing_env = [key for key in required_env if not env_summary.get(key)]
        checks.append(assert_status("required_env_summary", not missing_env, f"missing={missing_env or 'none'}"))

        checkout_status, _, checkout = http_json(
            "/api/stripe/checkout",
            method="POST",
            body={"email": email, "reference": f"live-worker-smoke-{stamp}"},
        )
        checkout_url = str(checkout.get("url") or "")
        checks.append(
            assert_status(
                "stripe_test_checkout",
                checkout_status == 200 and checkout.get("ok") is True and "cs_test_" in checkout_url,
                f"HTTP {checkout_status}; session={'cs_test' if 'cs_test_' in checkout_url else 'not_test'}",
            )
        )

        verify_status, verify_headers, _ = http_no_redirect(
            f"/api/auth/magic-link/verify?token={urllib.parse.quote(token)}"
        )
        location = header_value(verify_headers, "location")
        set_cookie = header_value(verify_headers, "set-cookie")
        if not set_cookie:
            raise RuntimeError(
                "verify route did not return set-cookie; "
                f"status={verify_status}; location={location}; headers={sorted(verify_headers.keys())}"
            )
        cookie_header = cookie_header_from_set_cookie(set_cookie)
        checks.append(
            assert_status(
                "post_checkout_verify_redirect",
                verify_status == 303 and location == f"{SITE_URL}/account.html?auth=success",
                f"HTTP {verify_status}; location={location}",
            )
        )

        session_status, _, session = http_json("/api/auth/session", headers={"cookie": cookie_header})
        checks.append(
            assert_status(
                "magic_link_session_restore",
                session_status == 200 and session.get("authenticated") is True and session.get("entitled") is True,
                f"HTTP {session_status}; status={session.get('status') or session.get('subscription_status')}",
            )
        )

        premium_status, _, premium = http_json("/api/premium/predictions", headers={"cookie": cookie_header})
        checks.append(
            assert_status(
                "premium_route_gating_allows_session",
                premium_status == 200 and premium.get("ok") is True and premium.get("count", 0) >= 1,
                f"HTTP {premium_status}; count={premium.get('count')}",
            )
        )
        facts.append(f"Premium access tier: `{premium.get('access_tier', 'unknown')}`.")

        portal_status, _, portal = http_json("/api/stripe/portal", method="POST", headers={"cookie": cookie_header})
        portal_url = str(portal.get("url") or "")
        checks.append(
            assert_status(
                "billing_portal",
                portal_status == 200 and portal.get("ok") is True and "billing.stripe.com" in portal_url,
                f"HTTP {portal_status}; url_host={'billing.stripe.com' if 'billing.stripe.com' in portal_url else 'missing'}",
            )
        )

        inactive_record = {**record, "status": "past_due", "updated_at": utc_now()}
        put_kv(f"subscription:{subscription_id}", inactive_record)
        put_kv(f"customer:{customer_id}", inactive_record)
        put_kv(f"email:{email}", inactive_record)

        inactive_session_status, _, inactive_session = http_json("/api/auth/session", headers={"cookie": cookie_header})
        checks.append(
            assert_status(
                "payment_issue_session_lockout",
                inactive_session_status == 200
                and inactive_session.get("authenticated") is False
                and inactive_session.get("status") == "inactive_subscription",
                f"HTTP {inactive_session_status}; status={inactive_session.get('status')}",
            )
        )

        inactive_premium_status, _, inactive_premium = http_json(
            "/api/premium/predictions", headers={"cookie": cookie_header}
        )
        checks.append(
            assert_status(
                "payment_issue_premium_lockout",
                inactive_premium_status == 401
                and inactive_premium.get("locked") is True
                and inactive_premium.get("status") == "inactive_subscription",
                f"HTTP {inactive_premium_status}; status={inactive_premium.get('status')}",
            )
        )

        facts.append(f"Synthetic subscriber `{subscription_id}` was seeded and flipped to `past_due`.")
    finally:
        for key in kv_keys:
            delete_kv(key)
        cleanup_d1(email)

    write_report(checks, facts)
    payload = {
        "ok": all(item["ok"] for item in checks),
        "checks": len(checks),
        "passed": sum(1 for item in checks if item["ok"]),
        "failed": [item["name"] for item in checks if not item["ok"]],
        "report": str(REPORT_PATH.relative_to(ROOT)),
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
