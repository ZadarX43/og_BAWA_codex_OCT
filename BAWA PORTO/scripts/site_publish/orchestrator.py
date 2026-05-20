#!/usr/bin/env python3
"""Coordinate the local website intelligence publish pipeline.

This is the first central "site brain" runner. It does not create predictions
or change deploy routing. It verifies upstream artifacts, settles public proof,
compiles compact publish payloads, audits fixture-page completeness, and writes
one orchestration report for the website handoff.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "frontend" / "public" / "data"
DEFAULT_DB = ROOT / "build" / "site_data" / "odds_genius.sqlite"
DEFAULT_PUBLISH_DIR = ROOT / "build" / "site_publish" / "current"
DEFAULT_REPORT_JSON = ROOT / "reports" / "latest" / "SITE_PUBLISH_ORCHESTRATION_REPORT.json"
DEFAULT_REPORT_MD = ROOT / "reports" / "latest" / "SITE_PUBLISH_ORCHESTRATION_REPORT.md"
DEFAULT_COMPLETENESS_JSON = ROOT / "reports" / "latest" / "UPCOMING_FIXTURE_COMPLETENESS_AUDIT.json"
DEFAULT_COMPLETENESS_CSV = ROOT / "reports" / "latest" / "UPCOMING_FIXTURE_COMPLETENESS_AUDIT.csv"
DEFAULT_COMPLETENESS_MD = ROOT / "reports" / "latest" / "UPCOMING_FIXTURE_COMPLETENESS_AUDIT.md"


@dataclass
class CommandResult:
    name: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local website publish orchestration pipeline.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--publish-dir", type=Path, default=DEFAULT_PUBLISH_DIR)
    parser.add_argument("--from-date", default=date.today().isoformat())
    parser.add_argument("--to-date", default="")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--all-fixtures", action="store_true")
    parser.add_argument("--skip-settlement", action="store_true")
    parser.add_argument("--skip-compiler", action="store_true")
    parser.add_argument("--skip-completeness-audit", action="store_true")
    parser.add_argument("--run-cloudflare-readiness", action="store_true")
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def file_status(path: Path) -> dict[str, Any]:
    exists = path.exists()
    stat = path.stat() if exists else None
    return {
        "path": str(path),
        "exists": exists,
        "bytes": stat.st_size if stat else 0,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        if stat
        else "",
    }


def run_command(name: str, command: list[str]) -> CommandResult:
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    return CommandResult(name=name, command=command, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


def artifact_checks(db_path: Path) -> dict[str, Any]:
    publish_summary_path = DATA_ROOT / "publish_summary.json"
    publish_summary = read_json(publish_summary_path, {})
    selected_source = publish_summary.get("selected_source_csv") if isinstance(publish_summary, dict) else ""
    selected_source_path = ROOT / str(selected_source) if selected_source else None
    checks = {
        "site_db": file_status(db_path),
        "publish_summary": file_status(publish_summary_path),
        "public_predictions": file_status(DATA_ROOT / "public_predictions.json"),
        "premium_predictions": file_status(DATA_ROOT / "premium_predictions.json"),
        "live_results_feed": file_status(DATA_ROOT / "live_results_feed.json"),
        "weekly_results": file_status(DATA_ROOT / "weekly_results.json"),
        "results_archive": file_status(DATA_ROOT / "results_archive.json"),
        "api_logo_manifest": file_status(DATA_ROOT / "api_football_logo_asset_manifest.json"),
        "selected_model_source": file_status(selected_source_path) if selected_source_path else {"exists": False, "path": "", "bytes": 0, "mtime_utc": ""},
    }
    missing = [name for name, status in checks.items() if not status.get("exists")]
    return {
        "checks": checks,
        "missing": missing,
        "publish_summary": publish_summary if isinstance(publish_summary, dict) else {},
    }


def summarize_command(result: CommandResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "ok": result.ok,
        "returncode": result.returncode,
        "command": result.command,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def compiler_summary(publish_dir: Path) -> dict[str, Any]:
    manifest = read_json(publish_dir / "manifest.json", {})
    upload_plan = read_json(publish_dir / "upload_plan.json", {})
    changed_manifest = read_json(publish_dir / "changed_manifest.json", {})
    return {
        "manifest_path": str(publish_dir / "manifest.json"),
        "upload_plan_path": str(publish_dir / "upload_plan.json"),
        "d1_delta_path": str(publish_dir / "d1_changed_index.sql"),
        "summary": manifest.get("summary", {}) if isinstance(manifest, dict) else {},
        "upload_plan": {
            "changed_objects": upload_plan.get("changed_objects", 0) if isinstance(upload_plan, dict) else 0,
            "total_changed_bytes": upload_plan.get("total_changed_bytes", 0) if isinstance(upload_plan, dict) else 0,
        },
        "changed_manifest_counts": {
            "objects": len(changed_manifest.get("objects", [])) if isinstance(changed_manifest, dict) else 0,
            "d1_rows": len(changed_manifest.get("d1_rows", [])) if isinstance(changed_manifest, dict) else 0,
        },
    }


def completeness_summary(path: Path) -> dict[str, Any]:
    payload = read_json(path, {})
    if not isinstance(payload, dict):
        return {}
    return {
        "report_path": str(path),
        "window": payload.get("window", {}),
        "summary": payload.get("summary", {}),
    }


def next_actions(artifact_state: dict[str, Any], completeness: dict[str, Any], compiler: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    if artifact_state.get("missing"):
        actions.append("Resolve missing upstream artifacts before publishing: " + ", ".join(artifact_state["missing"]))
    fixtures_total = (((completeness.get("summary") or {}).get("fixtures_total")) or 0)
    if fixtures_total == 0:
        actions.append("Refresh the active fixture/model/site DB window, then rerun the orchestrator for the upcoming weekend.")
    page_status = (((completeness.get("summary") or {}).get("page_status")) or {})
    if page_status.get("blocked"):
        actions.append("Fix Standard blockers from the completeness CSV before promoting the website payload.")
    check_summary = (((completeness.get("summary") or {}).get("checks")) or {})
    if (check_summary.get("weather") or {}).get("missing"):
        actions.append("Wire weather/stadium context into compact fixture payloads or mark weather as graceful fallback.")
    upload_plan = compiler.get("upload_plan") or {}
    if upload_plan.get("changed_objects", 0):
        actions.append("Upload changed payload objects, apply D1 changed index SQL, then run Cloudflare readiness smoke.")
    else:
        actions.append("No changed compact payload objects detected; Cloudflare upload can wait unless upstream data refreshes.")
    return actions


def write_report_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_report_md(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact_state = payload["artifact_state"]
    compiler = payload.get("compiler", {})
    completeness = payload.get("completeness", {})
    lines = [
        "# Site Publish Orchestration Report",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Window: `{payload['window']['from']}` to `{payload['window']['to']}`",
        f"- Missing artifacts: `{len(artifact_state.get('missing', []))}`",
        "",
        "## Commands",
        "",
    ]
    for command in payload["commands"]:
        status = "PASS" if command["ok"] else "FAIL"
        lines.append(f"- {command['name']}: `{status}`")
    lines.extend(["", "## Publish Compiler", ""])
    summary = compiler.get("summary") or {}
    upload_plan = compiler.get("upload_plan") or {}
    lines.extend(
        [
            f"- Objects total: `{summary.get('objects_total', 0)}`",
            f"- Objects changed: `{summary.get('objects_changed', 0)}`",
            f"- Changed bytes: `{summary.get('objects_changed_bytes', 0)}`",
            f"- D1 rows changed: `{summary.get('d1_rows_changed', 0)}`",
            f"- Upload-plan changed objects: `{upload_plan.get('changed_objects', 0)}`",
        ]
    )
    lines.extend(["", "## Completeness", ""])
    comp_summary = completeness.get("summary") or {}
    page_status = comp_summary.get("page_status") or {}
    lines.extend(
        [
            f"- Fixtures audited: `{comp_summary.get('fixtures_total', 0)}`",
            f"- Launch-ready: `{page_status.get('launch_ready', 0)}`",
            f"- Partial: `{page_status.get('partial', 0) + page_status.get('tier_partial', 0)}`",
            f"- Blocked: `{page_status.get('blocked', 0)}`",
        ]
    )
    lines.extend(["", "## Next Actions", ""])
    for action in payload["next_actions"]:
        lines.append(f"- {action}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    db_path = resolve(args.db)
    publish_dir = resolve(args.publish_dir)
    report_json = resolve(args.report_json)
    report_md = resolve(args.report_md)
    start = date.fromisoformat(args.from_date)
    end = date.fromisoformat(args.to_date) if args.to_date else start + timedelta(days=args.days)
    commands: list[CommandResult] = []

    artifact_state = artifact_checks(db_path)

    if not args.skip_settlement:
        commands.append(run_command("results_settlement", [sys.executable, "scripts/settle_published_results.py"]))

    if not args.skip_compiler:
        commands.append(
            run_command(
                "publish_compiler",
                [
                    sys.executable,
                    "scripts/publish_compiler.py",
                    "--db",
                    str(db_path),
                    "--output-dir",
                    str(publish_dir),
                ],
            )
        )

    if not args.skip_completeness_audit:
        audit_command = [
            sys.executable,
            "scripts/audit_upcoming_fixture_page_completeness.py",
            "--db",
            str(db_path),
            "--from-date",
            start.isoformat(),
            "--to-date",
            end.isoformat(),
            "--json-out",
            str(DEFAULT_COMPLETENESS_JSON),
            "--csv-out",
            str(DEFAULT_COMPLETENESS_CSV),
            "--md-out",
            str(DEFAULT_COMPLETENESS_MD),
        ]
        if args.all_fixtures:
            audit_command.append("--all")
        commands.append(run_command("fixture_completeness_audit", audit_command))

    if args.run_cloudflare_readiness:
        commands.append(run_command("cloudflare_preview_readiness", [sys.executable, "scripts/cloudflare_preview_readiness.py"]))

    compiler = compiler_summary(publish_dir)
    completeness = completeness_summary(DEFAULT_COMPLETENESS_JSON)
    payload = {
        "schema": "site_publish_orchestration_report_v1",
        "generated_at": utc_now(),
        "window": {"from": start.isoformat(), "to": end.isoformat(), "all_fixtures": bool(args.all_fixtures)},
        "artifact_state": artifact_state,
        "commands": [summarize_command(result) for result in commands],
        "compiler": compiler,
        "completeness": completeness,
        "next_actions": next_actions(artifact_state, completeness, compiler),
    }
    write_report_json(report_json, payload)
    write_report_md(report_md, payload)

    ok = not artifact_state["missing"] and all(result.ok for result in commands)
    print(
        json.dumps(
            {
                "ok": ok,
                "report_json": str(report_json),
                "report_md": str(report_md),
                "commands": {result.name: result.ok for result in commands},
                "missing_artifacts": artifact_state["missing"],
                "next_actions": payload["next_actions"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
