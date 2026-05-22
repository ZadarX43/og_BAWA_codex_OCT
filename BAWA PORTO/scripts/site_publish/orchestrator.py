#!/usr/bin/env python3
"""Coordinate the local website intelligence publish pipeline.

This is the central "site brain" runner. It does not create predictions or
change deploy routing. It settles proof, rebuilds the local site SQLite,
recalculates injury market impact, compiles fixture-brain payloads, compiles
compact publish artifacts, audits fixture-page completeness, and writes one
orchestration report for the website handoff.
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
DEFAULT_FIXTURE_BRAIN_DIR = ROOT / "build" / "site_brain" / "current"
DEFAULT_REPORT_JSON = ROOT / "reports" / "latest" / "SITE_PUBLISH_ORCHESTRATION_REPORT.json"
DEFAULT_REPORT_MD = ROOT / "reports" / "latest" / "SITE_PUBLISH_ORCHESTRATION_REPORT.md"
DEFAULT_UPSTREAM_JSON = ROOT / "reports" / "latest" / "SITE_UPSTREAM_INVENTORY.json"
DEFAULT_UPSTREAM_MD = ROOT / "reports" / "latest" / "SITE_UPSTREAM_INVENTORY.md"
DEFAULT_FIXTURE_BRAIN_JSON = ROOT / "reports" / "latest" / "FIXTURE_BRAIN_COMPILER_REPORT.json"
DEFAULT_FIXTURE_BRAIN_MD = ROOT / "reports" / "latest" / "FIXTURE_BRAIN_COMPILER_REPORT.md"
DEFAULT_SUMMARY_DRY_RUN_DIR = ROOT / "reports" / "latest" / "fixture_summary_dry_run"
DEFAULT_R2_UPLOAD_JSON = ROOT / "reports" / "latest" / "SITE_PUBLISH_R2_UPLOAD_REPORT.json"
DEFAULT_COMPLETENESS_JSON = ROOT / "reports" / "latest" / "UPCOMING_FIXTURE_COMPLETENESS_AUDIT.json"
DEFAULT_COMPLETENESS_CSV = ROOT / "reports" / "latest" / "UPCOMING_FIXTURE_COMPLETENESS_AUDIT.csv"
DEFAULT_COMPLETENESS_MD = ROOT / "reports" / "latest" / "UPCOMING_FIXTURE_COMPLETENESS_AUDIT.md"
DEFAULT_INJURY_MARKET_IMPACT_DIR = ROOT / "reports" / "latest" / "injury_shock_market_impact"
DEFAULT_INJURY_MARKET_IMPACT_FIXTURE = DEFAULT_INJURY_MARKET_IMPACT_DIR / "INJURY_SHOCK_MARKET_IMPACT_FIXTURE.csv"
DEFAULT_INJURY_MARKET_IMPACT_PLAYER = DEFAULT_INJURY_MARKET_IMPACT_DIR / "INJURY_SHOCK_MARKET_IMPACT_PLAYER.csv"
DEFAULT_INJURY_MARKET_IMPACT_REPORT = DEFAULT_INJURY_MARKET_IMPACT_DIR / "INJURY_SHOCK_MARKET_IMPACT_REPORT.md"
INJURY_COVERAGE_NAME = "INJURY_SHOCK_ELITE_STANDARD_COVERAGE.csv"
INJURY_PLAYER_RATINGS_NAME = "INJURY_SHOCK_ELITE_STANDARD_PLAYER_IMPACT_WITH_RATINGS.csv"


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
    parser.add_argument("--fixture-brain-dir", type=Path, default=DEFAULT_FIXTURE_BRAIN_DIR)
    parser.add_argument("--from-date", default=date.today().isoformat())
    parser.add_argument("--to-date", default="")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--all-fixtures", action="store_true")
    parser.add_argument("--skip-upstream-inventory", action="store_true")
    parser.add_argument("--skip-settlement", action="store_true")
    parser.add_argument("--skip-results-validation", action="store_true")
    parser.add_argument("--skip-site-db-export", action="store_true")
    parser.add_argument("--include-history", action="store_true", help="Pass through to export_site_sqlite.py.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--normalized-root", type=Path, default=ROOT / "data_sources" / "api_football" / "normalized")
    parser.add_argument("--skip-injury-market-impact", action="store_true")
    parser.add_argument("--injury-coverage-csv", type=Path, default=None)
    parser.add_argument("--injury-player-impact-csv", type=Path, default=None)
    parser.add_argument("--injury-market-impact-dir", type=Path, default=DEFAULT_INJURY_MARKET_IMPACT_DIR)
    parser.add_argument("--skip-summary-dry-run", action="store_true")
    parser.add_argument("--summary-dry-run-dir", type=Path, default=DEFAULT_SUMMARY_DRY_RUN_DIR)
    parser.add_argument("--skip-compiler", action="store_true")
    parser.add_argument("--skip-fixture-brain", action="store_true")
    parser.add_argument("--skip-completeness-audit", action="store_true")
    parser.add_argument("--run-r2-upload", action="store_true", help="Upload changed compact payload objects to R2.")
    parser.add_argument("--r2-all-objects", action="store_true", help="Upload every compact object instead of changed objects only.")
    parser.add_argument("--r2-bucket", default="", help="Override the R2 bucket used by upload_publish_plan_r2.py.")
    parser.add_argument("--d1-database", default="", help="Override the D1 database used by upload_publish_plan_r2.py.")
    parser.add_argument("--r2-start-index", type=int, default=1, help="Start index for resumable R2 uploads.")
    parser.add_argument("--r2-retries", type=int, default=3)
    parser.add_argument("--apply-d1", action="store_true", help="Apply the changed D1 index SQL after R2 upload.")
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


def latest_injury_source(filename: str) -> Path:
    candidates = sorted(
        (ROOT / "reports" / "latest").glob(f"injury_shock_elite_standard_*/{filename}"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return ROOT / "reports" / "latest" / "injury_shock_elite_standard_2026_05_22_to_2026_05_26" / filename


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


def fixture_brain_summary(path: Path) -> dict[str, Any]:
    manifest = read_json(path / "manifest.json", {})
    if not isinstance(manifest, dict):
        return {}
    return {
        "manifest_path": str(path / "manifest.json"),
        "summary": manifest.get("summary", {}),
        "source_summary": manifest.get("source_summary", {}),
    }


def summary_dry_run_summary(path: Path) -> dict[str, Any]:
    payload = read_json(path / "index.json", {})
    if not isinstance(payload, dict):
        return {}
    return {
        "index_path": str(path / "index.json"),
        "report_path": str(path / "FIXTURE_SUMMARY_DRY_RUN_REPORT.md"),
        "fixtures_rendered": payload.get("fixtures_rendered", 0),
        "tiers": payload.get("tiers", []),
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


def r2_upload_summary(path: Path) -> dict[str, Any]:
    payload = read_json(path, {})
    if not isinstance(payload, dict):
        return {}
    return {
        "report_path": str(path),
        "ok": payload.get("ok", False),
        "uploaded_objects": payload.get("uploaded_objects", 0),
        "failed_objects": len(payload.get("errors") or []),
        "uploaded_bytes": payload.get("uploaded_bytes", 0),
        "d1_applied": bool((payload.get("d1") or {}).get("applied")),
    }


def upstream_summary(path: Path) -> dict[str, Any]:
    payload = read_json(path, {})
    if not isinstance(payload, dict):
        return {}
    return {
        "report_path": str(path),
        "window": payload.get("window", {}),
        "readiness": payload.get("readiness", {}),
        "site_db": {
            "fixture_date_min": (payload.get("site_db") or {}).get("fixture_date_min", ""),
            "fixture_date_max": (payload.get("site_db") or {}).get("fixture_date_max", ""),
            "window_fixture_count": (payload.get("site_db") or {}).get("window_fixture_count", 0),
        },
        "prediction_outputs": {
            "window_csv_count": (payload.get("prediction_outputs") or {}).get("window_csv_count", 0),
            "selected_source_rows": (payload.get("prediction_outputs") or {}).get("selected_source_rows", 0),
        },
        "api_football": {
            "fixture_window_rows": (payload.get("api_football") or {}).get("fixture_window_rows", 0),
            "fixture_window_leagues": len((payload.get("api_football") or {}).get("fixture_window_leagues", {})),
        },
    }


def next_actions(
    artifact_state: dict[str, Any],
    upstream: dict[str, Any],
    completeness: dict[str, Any],
    compiler: dict[str, Any],
    fixture_brain: dict[str, Any],
    r2_upload: dict[str, Any],
) -> list[str]:
    actions: list[str] = []
    if artifact_state.get("missing"):
        actions.append("Resolve missing upstream artifacts before publishing: " + ", ".join(artifact_state["missing"]))
    readiness = upstream.get("readiness") or {}
    if readiness.get("blockers"):
        actions.append("Resolve upstream inventory blockers: " + ", ".join(readiness["blockers"]))
    if "site_db_window_empty" in (readiness.get("blockers") or []):
        actions.append("Rebuild compact site DB after the fresh model/API-football run so the target fixture window is present.")
    if completeness.get("summary"):
        fixtures_total = (((completeness.get("summary") or {}).get("fixtures_total")) or 0)
        if fixtures_total == 0:
            actions.append("Refresh the active fixture/model/site DB window, then rerun the orchestrator for the upcoming weekend.")
        page_status = (((completeness.get("summary") or {}).get("page_status")) or {})
        if page_status.get("blocked"):
            actions.append("Fix Standard blockers from the completeness CSV before promoting the website payload.")
        check_summary = (((completeness.get("summary") or {}).get("checks")) or {})
        if (check_summary.get("weather") or {}).get("missing"):
            actions.append("Wire weather/stadium context into compact fixture payloads or mark weather as graceful fallback.")
    if compiler.get("upload_plan"):
        upload_plan = compiler.get("upload_plan") or {}
        if upload_plan.get("changed_objects", 0) and not r2_upload:
            actions.append("Upload changed payload objects, apply D1 changed index SQL, then run Cloudflare readiness smoke.")
        elif r2_upload and not r2_upload.get("ok", False):
            actions.append("R2 upload was attempted but failed; inspect SITE_PUBLISH_R2_UPLOAD_REPORT.json before promotion.")
        elif upload_plan.get("changed_objects", 0) and r2_upload.get("ok", False):
            actions.append("Changed compact payload objects were uploaded; run or review Cloudflare readiness before promotion.")
        else:
            actions.append("No changed compact payload objects detected; Cloudflare upload can wait unless upstream data refreshes.")
    brain_summary = fixture_brain.get("summary") or {}
    if brain_summary and brain_summary.get("fixtures_compiled", 0) == 0:
        actions.append("Fixture-brain compiler is wired, but the active window has no local site DB fixtures yet.")
    return actions


def write_report_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_report_md(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact_state = payload["artifact_state"]
    upstream = payload.get("upstream_inventory", {})
    compiler = payload.get("compiler", {})
    completeness = payload.get("completeness", {})
    lines = [
        "# Site Publish Orchestration Report",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Window: `{payload['window']['from']}` to `{payload['window']['to']}`",
        f"- Pipeline blocked: `{payload.get('pipeline_blocked', False)}`",
        f"- Missing artifacts: `{len(artifact_state.get('missing', []))}`",
        "",
        "## Commands",
        "",
    ]
    for command in payload["commands"]:
        status = "PASS" if command["ok"] else "FAIL"
        lines.append(f"- {command['name']}: `{status}`")
    lines.extend(["", "## Upstream Inventory", ""])
    readiness = upstream.get("readiness") or {}
    site_db = upstream.get("site_db") or {}
    prediction_outputs = upstream.get("prediction_outputs") or {}
    api_football = upstream.get("api_football") or {}
    lines.extend(
        [
            f"- Readiness: `{readiness.get('state', 'unknown')}`",
            f"- Site DB window fixtures: `{site_db.get('window_fixture_count', 0)}`",
            f"- Prediction CSVs in window: `{prediction_outputs.get('window_csv_count', 0)}`",
            f"- API-football fixture rows in window: `{api_football.get('fixture_window_rows', 0)}`",
        ]
    )
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
    lines.extend(["", "## Fixture Brain", ""])
    brain_summary = payload.get("fixture_brain", {}).get("summary", {})
    lines.extend(
        [
            f"- Fixtures compiled: `{brain_summary.get('fixtures_compiled', 0)}`",
            f"- Total bytes: `{brain_summary.get('total_bytes', 0)}`",
        ]
    )
    lines.extend(["", "## Summary Dry Run", ""])
    dry_run = payload.get("summary_dry_run", {})
    if dry_run:
        lines.extend(
            [
                f"- Fixtures rendered: `{dry_run.get('fixtures_rendered', 0)}`",
                f"- Tiers: `{', '.join(dry_run.get('tiers', []))}`",
                f"- Report: `{dry_run.get('report_path', '')}`",
            ]
        )
    else:
        lines.append("- Summary dry run not run in this orchestration pass.")
    r2_upload = payload.get("r2_upload", {})
    lines.extend(["", "## R2 / D1 Publish", ""])
    if r2_upload:
        lines.extend(
            [
                f"- Upload ok: `{r2_upload.get('ok', False)}`",
                f"- Uploaded objects: `{r2_upload.get('uploaded_objects', 0)}`",
                f"- Failed objects: `{r2_upload.get('failed_objects', 0)}`",
                f"- Uploaded bytes: `{r2_upload.get('uploaded_bytes', 0)}`",
                f"- D1 applied: `{r2_upload.get('d1_applied', False)}`",
            ]
        )
    else:
        lines.append("- Upload not run in this orchestration pass.")
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
    fixture_brain_dir = resolve(args.fixture_brain_dir)
    report_json = resolve(args.report_json)
    report_md = resolve(args.report_md)
    start = date.fromisoformat(args.from_date)
    end = date.fromisoformat(args.to_date) if args.to_date else start + timedelta(days=args.days)
    commands: list[CommandResult] = []
    pipeline_blocked = False
    data_root = resolve(args.data_root)
    normalized_root = resolve(args.normalized_root)
    summary_dry_run_dir = resolve(args.summary_dry_run_dir)
    injury_market_impact_dir = resolve(args.injury_market_impact_dir)
    injury_coverage_csv = resolve(args.injury_coverage_csv) if args.injury_coverage_csv else latest_injury_source(INJURY_COVERAGE_NAME)
    injury_player_csv = resolve(args.injury_player_impact_csv) if args.injury_player_impact_csv else latest_injury_source(INJURY_PLAYER_RATINGS_NAME)
    injury_market_fixture_csv = injury_market_impact_dir / "INJURY_SHOCK_MARKET_IMPACT_FIXTURE.csv"
    injury_market_player_csv = injury_market_impact_dir / "INJURY_SHOCK_MARKET_IMPACT_PLAYER.csv"

    def run_step(name: str, command: list[str], required: bool = True) -> None:
        nonlocal pipeline_blocked
        if pipeline_blocked:
            return
        result = run_command(name, command)
        commands.append(result)
        if required and not result.ok:
            pipeline_blocked = True

    if not args.skip_settlement:
        run_step("results_settlement", [sys.executable, "scripts/settle_published_results.py"])

    if not args.skip_results_validation:
        run_step("results_validation", [sys.executable, "validate_weekly_results.py"])

    if not args.skip_site_db_export:
        export_command = [
            sys.executable,
            "scripts/export_site_sqlite.py",
            "--data-root",
            str(data_root),
            "--normalized-root",
            str(normalized_root),
            "--output",
            str(db_path),
        ]
        if args.include_history:
            export_command.append("--include-history")
        run_step("site_sqlite_export", export_command)

    if not args.skip_upstream_inventory:
        run_step(
            "upstream_inventory",
            [
                sys.executable,
                "scripts/site_publish/upstream_inventory.py",
                "--db",
                str(db_path),
                "--from-date",
                start.isoformat(),
                "--to-date",
                end.isoformat(),
                "--json-out",
                str(DEFAULT_UPSTREAM_JSON),
                "--md-out",
                str(DEFAULT_UPSTREAM_MD),
            ],
        )

    if not args.skip_injury_market_impact:
        run_step(
            "injury_market_impact",
            [
                sys.executable,
                "scripts/build_injury_shock_market_impact_sidecar.py",
                "--coverage-csv",
                str(injury_coverage_csv),
                "--player-impact-csv",
                str(injury_player_csv),
                "--outdir",
                str(injury_market_impact_dir),
            ],
        )

    if not args.skip_fixture_brain:
        brain_command = [
            sys.executable,
            "scripts/site_publish/fixture_brain_compiler.py",
            "--db",
            str(db_path),
            "--output-dir",
            str(fixture_brain_dir),
            "--from-date",
            start.isoformat(),
            "--to-date",
            end.isoformat(),
            "--injury-fixture-csv",
            str(injury_coverage_csv),
            "--injury-player-csv",
            str(injury_market_player_csv),
            "--injury-market-impact-csv",
            str(injury_market_fixture_csv),
            "--report-json",
            str(DEFAULT_FIXTURE_BRAIN_JSON),
            "--report-md",
            str(DEFAULT_FIXTURE_BRAIN_MD),
        ]
        if args.all_fixtures:
            brain_command.append("--all-fixtures")
        run_step("fixture_brain_compiler", brain_command)

    if not args.skip_summary_dry_run:
        run_step(
            "fixture_summary_dry_run",
            [
                sys.executable,
                "scripts/build_fixture_summary_dry_run.py",
                "--fixture-brain-dir",
                str(fixture_brain_dir),
                "--outdir",
                str(summary_dry_run_dir),
            ],
        )

    if not args.skip_compiler:
        run_step(
            "publish_compiler",
            [
                sys.executable,
                "scripts/publish_compiler.py",
                "--db",
                str(db_path),
                "--output-dir",
                str(publish_dir),
                "--fixture-brain-dir",
                str(fixture_brain_dir),
            ],
        )

    if not args.skip_completeness_audit:
        audit_command = [
            sys.executable,
            "scripts/audit_upcoming_fixture_page_completeness.py",
            "--db",
            str(db_path),
            "--publish-dir",
            str(publish_dir),
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
        run_step("fixture_completeness_audit", audit_command)

    if args.run_r2_upload:
        upload_command = [
            sys.executable,
            "scripts/site_publish/upload_publish_plan_r2.py",
            "--publish-dir",
            str(publish_dir),
            "--start-index",
            str(args.r2_start_index),
            "--retries",
            str(args.r2_retries),
        ]
        if args.r2_bucket:
            upload_command.extend(["--bucket", args.r2_bucket])
        if args.d1_database:
            upload_command.extend(["--d1-database", args.d1_database])
        if args.r2_all_objects:
            upload_command.append("--all-objects")
        if args.apply_d1:
            upload_command.append("--apply-d1")
        run_step("r2_publish_upload", upload_command)

    if args.run_cloudflare_readiness:
        run_step("cloudflare_preview_readiness", [sys.executable, "scripts/cloudflare_preview_readiness.py"], required=False)

    artifact_state = artifact_checks(db_path)

    compiler = {} if args.skip_compiler else compiler_summary(publish_dir)
    fixture_brain = {} if args.skip_fixture_brain else fixture_brain_summary(fixture_brain_dir)
    summary_dry_run = {} if args.skip_summary_dry_run else summary_dry_run_summary(summary_dry_run_dir)
    upstream = {} if args.skip_upstream_inventory else upstream_summary(DEFAULT_UPSTREAM_JSON)
    completeness = {} if args.skip_completeness_audit else completeness_summary(DEFAULT_COMPLETENESS_JSON)
    r2_upload = r2_upload_summary(DEFAULT_R2_UPLOAD_JSON) if args.run_r2_upload else {}
    payload = {
        "schema": "site_publish_orchestration_report_v1",
        "generated_at": utc_now(),
        "window": {"from": start.isoformat(), "to": end.isoformat(), "all_fixtures": bool(args.all_fixtures)},
        "pipeline_blocked": pipeline_blocked,
        "injury_market_sources": {
            "coverage_csv": str(injury_coverage_csv),
            "player_impact_csv": str(injury_player_csv),
            "market_fixture_csv": str(injury_market_fixture_csv),
            "market_player_csv": str(injury_market_player_csv),
        },
        "artifact_state": artifact_state,
        "commands": [summarize_command(result) for result in commands],
        "upstream_inventory": upstream,
        "compiler": compiler,
        "fixture_brain": fixture_brain,
        "summary_dry_run": summary_dry_run,
        "r2_upload": r2_upload,
        "completeness": completeness,
        "next_actions": next_actions(artifact_state, upstream, completeness, compiler, fixture_brain, r2_upload),
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
