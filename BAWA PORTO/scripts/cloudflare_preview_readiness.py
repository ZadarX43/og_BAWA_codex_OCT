#!/usr/bin/env python3
"""Cloudflare preview readiness gate for the website launch bundle.

This command is product/publish-only. It does not generate predictions, route
deploy rows, or touch the protected football prediction spine.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKER_DIR = ROOT / "worker"
FRONTEND = ROOT / "frontend"
DATA_ROOT = FRONTEND / "public" / "data"
WRANGLER_TOML = WORKER_DIR / "wrangler.toml"
CONFIG_JS = FRONTEND / "assets" / "config.js"
REPORT_PATH = ROOT / "reports" / "latest" / "CLOUDFLARE_PREVIEW_READINESS_REPORT.md"

DEFAULT_WORKER_URL = "https://odds-genius-worker.hughcwade.workers.dev"
DEFAULT_SITE_URL = "https://og-bawa-codex-oct.pages.dev"

REQUIRED_HEALTH_ENV = [
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
    "has_site_data_db",
]

REQUIRED_WORKER_ROUTES = [
    "GET /health",
    "GET /api/premium/predictions",
    "POST /api/stripe/checkout",
    "POST /api/stripe/portal",
    "POST /api/stripe/webhook",
    "GET /api/site/fixtures/current",
    "GET /api/site/fixtures/:fixture_key/context",
    "GET /api/site/teams/:competition_key/:team_slug/premium",
]

REQUIRED_SITE_PATHS = [
    "/",
    "/results",
    "/pricing",
    "/account",
    "/public/data/publish_summary.json",
    "/public/data/public_predictions.json",
    "/public/data/weekly_results.json",
    "/public/data/results_archive.json",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def run_local(name: str, cmd: list[str], why: str) -> dict[str, Any]:
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "name": name,
        "cmd": " ".join(cmd),
        "why": why,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "details": (completed.stdout or "").strip()[:6000] or "(no output)",
    }


def parse_curl_response(raw: str) -> tuple[int, dict[str, str], str]:
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
        headers[key.strip().lower()] = value.strip()
    return status, headers, body


def curl_json(url: str, *, method: str = "GET", body: Any = None) -> tuple[int, dict[str, str], Any]:
    args = ["curl", "-sS", "-i", "-X", method, "-H", "accept: application/json"]
    if body is not None:
        args.extend(["-H", "content-type: application/json", "--data", json.dumps(body)])
    args.append(url)
    completed = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    status, headers, raw_body = parse_curl_response(completed.stdout or "")
    try:
        payload = json.loads(raw_body or "{}")
    except json.JSONDecodeError:
        payload = {"raw": raw_body[:1000]}
    return status, headers, payload


def curl_status(url: str) -> tuple[int, str]:
    completed = subprocess.run(
        ["curl", "-sS", "-L", "-o", "/dev/null", "-w", "%{http_code}", url],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        status = int((completed.stdout or "0").strip()[-3:])
    except ValueError:
        status = 0
    return status, ""


def data_size_bytes() -> int:
    return sum(path.stat().st_size for path in DATA_ROOT.rglob("*") if path.is_file())


def weekly_proof_signature(payload: dict[str, Any]) -> dict[str, Any]:
    def keep_rollup(items: Any, key: str) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            return []
        out = []
        for item in items:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    key: item.get(key),
                    "total_picks": item.get("total_picks"),
                    "settled_picks": item.get("settled_picks"),
                    "pending_picks": item.get("pending_picks"),
                    "wins": item.get("wins"),
                    "losses": item.get("losses"),
                    "voids": item.get("voids"),
                    "hit_rate": item.get("hit_rate"),
                    "roi": item.get("roi"),
                    "profit_units": item.get("profit_units"),
                }
            )
        return sorted(out, key=lambda row: str(row.get(key) or ""))

    def keep_chart(items: Any) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            return []
        out = []
        for item in items:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "date": item.get("date"),
                    "settled_picks": item.get("settled_picks"),
                    "wins": item.get("wins"),
                    "losses": item.get("losses"),
                    "voids": item.get("voids"),
                    "profit_units": item.get("profit_units"),
                    "cumulative_profit_units": item.get("cumulative_profit_units"),
                    "rolling_hit_rate": item.get("rolling_hit_rate"),
                    "cumulative_roi": item.get("cumulative_roi"),
                    "cumulative_hit_rate": item.get("cumulative_hit_rate"),
                }
            )
        return sorted(out, key=lambda row: str(row.get("date") or ""))

    def keep_settlements(items: Any) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            return []
        out = []
        for item in items:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "settlement_key": item.get("settlement_key"),
                    "market": item.get("market"),
                    "pick": item.get("pick"),
                    "result_status": item.get("result_status"),
                    "actual": item.get("actual"),
                    "profit_units": item.get("profit_units"),
                    "final_home_score": item.get("final_home_score"),
                    "final_away_score": item.get("final_away_score"),
                    "provider_status": item.get("provider_status"),
                }
            )
        return sorted(out, key=lambda row: str(row.get("settlement_key") or ""))

    return {
        "period_start": payload.get("period_start"),
        "period_end": payload.get("period_end"),
        "source_file": payload.get("source_file"),
        "published_run_id": payload.get("published_run_id"),
        "total_picks": payload.get("total_picks"),
        "settled_picks": payload.get("settled_picks"),
        "pending_picks": payload.get("pending_picks"),
        "wins": payload.get("wins"),
        "losses": payload.get("losses"),
        "voids": payload.get("voids"),
        "hit_rate": payload.get("hit_rate"),
        "roi": payload.get("roi"),
        "profit_units": payload.get("profit_units"),
        "by_market": keep_rollup(payload.get("by_market"), "market"),
        "by_tier": keep_rollup(payload.get("by_tier"), "tier"),
        "by_league": keep_rollup(payload.get("by_league"), "league"),
        "by_visibility": keep_rollup(payload.get("by_visibility"), "visibility"),
        "chart_points": keep_chart(payload.get("chart_points")),
        "items": keep_settlements(payload.get("items")),
    }


def weekly_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at": payload.get("generated_at"),
        "total_picks": payload.get("total_picks"),
        "settled_picks": payload.get("settled_picks"),
        "pending_picks": payload.get("pending_picks"),
        "wins": payload.get("wins"),
        "losses": payload.get("losses"),
        "voids": payload.get("voids"),
    }


def parse_worker_vars() -> dict[str, str]:
    text = WRANGLER_TOML.read_text(encoding="utf-8") if WRANGLER_TOML.exists() else ""
    vars_block_match = re.search(r"\[vars\]\s*(.*?)(?:\n\[|\Z)", text, flags=re.S)
    vars_block = vars_block_match.group(1) if vars_block_match else ""
    out: dict[str, str] = {}
    for key, value in re.findall(r"^([A-Z0-9_]+)\s*=\s*\"([^\"]*)\"", vars_block, flags=re.M):
        out[key] = value
    return out


def config_worker_url() -> str:
    if not CONFIG_JS.exists():
        return ""
    text = CONFIG_JS.read_text(encoding="utf-8")
    match = re.search(r'WORKER_API_BASE:\s*"([^"]+)"', text)
    return match.group(1).rstrip("/") if match else ""


def write_report(checks: list[dict[str, Any]], facts: list[str], warnings: list[str]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cloudflare Preview Readiness Report",
        "",
        f"Generated: `{utc_now()}`",
        "",
        "## Summary",
        "",
        f"- Checks: `{len(checks)}`",
        f"- Passed: `{sum(1 for item in checks if item['ok'])}`",
        f"- Failed: `{sum(1 for item in checks if not item['ok'])}`",
        f"- Warnings: `{len(warnings)}`",
        "",
        "## Facts",
        "",
        *([f"- {fact}" for fact in facts] or ["- None"]),
        "",
        "## Checks",
        "",
    ]
    for item in checks:
        status = "PASS" if item["ok"] else "FAIL"
        lines.extend([f"- {status} `{item['name']}` - {item['details']}"])
    lines.extend(["", "## Warnings", "", *([f"- {item}" for item in warnings] or ["- None"]), ""])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def check_local_config(worker_url: str, site_url: str) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    checks: list[dict[str, Any]] = []
    facts: list[str] = []
    warnings: list[str] = []

    vars_map = parse_worker_vars()
    configured_worker = config_worker_url()
    required_vars = ["SITE_URL", "PREMIUM_DATA_SOURCE", "STRIPE_PRICE_ID"]
    missing_vars = [key for key in required_vars if not vars_map.get(key)]
    checks.append(
        {
            "name": "worker_config_vars",
            "ok": not missing_vars,
            "details": f"missing={missing_vars or 'none'}",
        }
    )
    if vars_map.get("SITE_URL") and vars_map["SITE_URL"].rstrip("/") != site_url.rstrip("/"):
        warnings.append(f"`worker/wrangler.toml` SITE_URL points to `{vars_map['SITE_URL']}`, not `{site_url}`.")
    if configured_worker and configured_worker != worker_url.rstrip("/"):
        warnings.append(f"`frontend/assets/config.js` Worker URL points to `{configured_worker}`, not `{worker_url}`.")

    wrangler_text = WRANGLER_TOML.read_text(encoding="utf-8") if WRANGLER_TOML.exists() else ""
    checks.append(
        {
            "name": "worker_bindings_configured",
            "ok": all(token in wrangler_text for token in ["SUBSCRIBER_STATE", "ACCOUNT_DB", "SITE_DATA_DB"]),
            "details": "required bindings present in wrangler.toml",
        }
    )
    total_mb = data_size_bytes() / (1024 * 1024)
    facts.append(f"Website data footprint: `{total_mb:.1f} MB`.")
    if total_mb > 200:
        warnings.append("Website data footprint is above 200 MB; review Cloudflare Pages/bandwidth pressure before promotion.")
    return checks, facts, warnings


def check_live(worker_url: str, site_url: str) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    checks: list[dict[str, Any]] = []
    facts: list[str] = []
    warnings: list[str] = []

    health_status, _headers, health = curl_json(f"{worker_url}/health")
    routes = set(health.get("routes") or []) if isinstance(health, dict) else set()
    env_summary = health.get("env_summary") or {} if isinstance(health, dict) else {}
    missing_env = [key for key in REQUIRED_HEALTH_ENV if not env_summary.get(key)]
    missing_routes = [route for route in REQUIRED_WORKER_ROUTES if route not in routes]
    checks.append(
        {
            "name": "worker_health",
            "ok": health_status == 200 and health.get("ok") is True,
            "details": f"HTTP {health_status}",
        }
    )
    checks.append(
        {
            "name": "worker_required_env",
            "ok": not missing_env,
            "details": f"missing={missing_env or 'none'}",
        }
    )
    checks.append(
        {
            "name": "worker_required_routes",
            "ok": not missing_routes,
            "details": f"missing={missing_routes or 'none'}",
        }
    )

    premium_status, _headers, premium = curl_json(f"{worker_url}/api/premium/predictions")
    checks.append(
        {
            "name": "premium_route_signed_out_lock",
            "ok": premium_status == 401 and premium.get("locked") is True,
            "details": f"HTTP {premium_status}; status={premium.get('status')}",
        }
    )

    portal_status, _headers, portal = curl_json(f"{worker_url}/api/stripe/portal", method="POST")
    checks.append(
        {
            "name": "portal_route_signed_out_lock",
            "ok": portal_status == 401 and portal.get("locked") is True,
            "details": f"HTTP {portal_status}; status={portal.get('status')}",
        }
    )

    checkout_status, _headers, checkout = curl_json(
        f"{worker_url}/api/stripe/checkout",
        method="POST",
        body={"email": "codex.preview.readiness@oddsgenius.test", "reference": "cloudflare-preview-readiness"},
    )
    checkout_url = str(checkout.get("url") or "")
    checks.append(
        {
            "name": "stripe_checkout_test_session_route",
            "ok": checkout_status == 200 and checkout.get("ok") is True and "cs_test_" in checkout_url,
            "details": f"HTTP {checkout_status}; session={'cs_test' if 'cs_test_' in checkout_url else 'missing'}",
        }
    )

    for path in REQUIRED_SITE_PATHS:
        status, _body = curl_status(f"{site_url.rstrip('/')}{path}")
        checks.append(
            {
                "name": f"site_path:{path}",
                "ok": status == 200,
                "details": f"HTTP {status}",
            }
        )

    weekly_status, _headers, weekly = curl_json(f"{site_url.rstrip('/')}/public/data/weekly_results.json")
    if weekly_status == 200 and isinstance(weekly, dict):
        facts.append(
            "Preview weekly proof: "
            f"`{weekly.get('settled_picks', 0)}` settled, "
            f"`{weekly.get('pending_picks', 0)}` pending, "
            f"`{weekly.get('wins', 0)}/{weekly.get('losses', 0)}` W/L."
        )
        local_weekly_path = DATA_ROOT / "weekly_results.json"
        if local_weekly_path.exists():
            with local_weekly_path.open("r", encoding="utf-8") as handle:
                local_weekly = json.load(handle)
            local_signature = weekly_proof_signature(local_weekly)
            preview_signature = weekly_proof_signature(weekly)
            local_brief = weekly_summary(local_weekly)
            preview_brief = weekly_summary(weekly)
            checks.append(
                {
                    "name": "preview_weekly_matches_local_bundle",
                    "ok": preview_signature == local_signature,
                    "details": f"preview={preview_brief}; local={local_brief}; volatile timestamps ignored",
                }
            )
    else:
        warnings.append("Could not read preview weekly proof JSON for summary facts.")

    return checks, facts, warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Cloudflare preview readiness checks.")
    parser.add_argument("--worker-url", default=DEFAULT_WORKER_URL)
    parser.add_argument("--site-url", default=DEFAULT_SITE_URL)
    parser.add_argument("--skip-live", action="store_true", help="Run local publish/config checks only.")
    parser.add_argument("--skip-local-publish", action="store_true", help="Do not rerun local proof/static publish commands.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks: list[dict[str, Any]] = []
    facts: list[str] = []
    warnings: list[str] = []

    if not args.skip_local_publish:
        local_commands = [
            ("frontend_js_syntax", ["node", "--check", "frontend/assets/app.js"], "Frontend bundle parses."),
            ("publish_results_proof", ["python3", "scripts/publish_results_proof.py"], "Results settlement and page proof smoke."),
            ("site_publish_smoke", ["python3", "scripts/site_publish_smoke.py"], "Website-safe JSON smoke."),
            ("frontend_static_smoke", ["python3", "scripts/smoke_frontend_static.py"], "Static HTML/assets/data smoke."),
        ]
        for name, cmd, why in local_commands:
            result = run_local(name, cmd, why)
            checks.append({"name": result["name"], "ok": result["ok"], "details": result["details"].splitlines()[0]})
            if not result["ok"]:
                warnings.append(f"`{result['cmd']}` failed; see command output in terminal or component report.")

    config_checks, config_facts, config_warnings = check_local_config(args.worker_url, args.site_url)
    checks.extend(config_checks)
    facts.extend(config_facts)
    warnings.extend(config_warnings)

    if not args.skip_live:
        live_checks, live_facts, live_warnings = check_live(args.worker_url.rstrip("/"), args.site_url.rstrip("/"))
        checks.extend(live_checks)
        facts.extend(live_facts)
        warnings.extend(live_warnings)
    else:
        warnings.append("Live Cloudflare route checks were skipped.")

    write_report(checks, facts, warnings)
    payload = {
        "ok": all(item["ok"] for item in checks),
        "checks": len(checks),
        "passed": sum(1 for item in checks if item["ok"]),
        "failed": [item["name"] for item in checks if not item["ok"]],
        "warnings": len(warnings),
        "report": rel(REPORT_PATH),
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
