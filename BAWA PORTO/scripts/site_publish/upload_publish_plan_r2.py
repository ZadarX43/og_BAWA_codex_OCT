#!/usr/bin/env python3
"""Upload compact site publish objects from an upload plan to Cloudflare R2.

The publish compiler does the heavy calculation locally and writes:

  build/site_publish/current/upload_plan.json
  build/site_publish/current/d1_changed_index.sql

This uploader applies the remote side of that contract: upload changed objects
only, then optionally apply the D1 delta when it contains statements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PUBLISH_DIR = ROOT / "build" / "site_publish" / "current"
DEFAULT_WORKER_DIR = ROOT / "worker"
DEFAULT_BUCKET = "odds-genius-site-payloads"
DEFAULT_D1_DATABASE = "odds-genius-site-data"
DEFAULT_REPORT = ROOT / "reports" / "latest" / "SITE_PUBLISH_R2_UPLOAD_REPORT.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload changed compact site publish payloads to Cloudflare R2.")
    parser.add_argument("--publish-dir", type=Path, default=DEFAULT_PUBLISH_DIR)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--worker-dir", type=Path, default=DEFAULT_WORKER_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--d1-database", default=DEFAULT_D1_DATABASE)
    parser.add_argument("--apply-d1", action="store_true", help="Apply d1_changed_index.sql when it contains SQL statements.")
    parser.add_argument("--all-objects", action="store_true", help="Upload every object in manifest.json instead of only upload_plan.json changes.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Upload at most N objects, for smoke testing.")
    parser.add_argument("--start-index", type=int, default=1, help="One-based object index to start from, for resuming a failed upload.")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_canonical_json_file(path: Path) -> str:
    return hashlib.sha256(canonical_json(read_json(path)).encode("utf-8")).hexdigest()


def run(command: list[str], cwd: Path, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {
            "command": command,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "dry_run": True,
        }
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "dry_run": False,
    }


def d1_sql_has_statements(path: Path) -> bool:
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("--"):
            return True
    return False


def upload_objects(args: argparse.Namespace, objects: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    uploaded: list[dict[str, Any]] = []
    errors: list[str] = []
    start_offset = max(0, int(args.start_index or 1) - 1)
    selected = objects[start_offset:]
    if args.limit and args.limit > 0:
        selected = selected[: args.limit]
    for offset, item in enumerate(selected, start=start_offset + 1):
        rel_path = str(item.get("relative_path") or "")
        object_key = str(item.get("object_key") or "")
        source_path = args.publish_dir / rel_path
        if not rel_path or not object_key:
            errors.append(f"Object {offset} is missing relative_path or object_key.")
            continue
        if not source_path.exists():
            errors.append(f"{rel_path} does not exist under {args.publish_dir}.")
            continue
        expected_bytes = int(item.get("bytes") or 0)
        actual_bytes = source_path.stat().st_size
        if expected_bytes and actual_bytes != expected_bytes:
            errors.append(f"{rel_path} byte mismatch: expected {expected_bytes}, got {actual_bytes}.")
            continue
        expected_sha = str(item.get("sha256") or "")
        actual_canonical_sha = sha256_canonical_json_file(source_path)
        if expected_sha and actual_canonical_sha != expected_sha:
            errors.append(f"{rel_path} sha256 mismatch: expected {expected_sha}, got {actual_canonical_sha}.")
            continue
        actual_file_sha = sha256_file(source_path)
        command = [
            "npx",
            "wrangler",
            "r2",
            "object",
            "put",
            f"{args.bucket}/{object_key}",
            "--file",
            str(source_path),
            "--content-type",
            "application/json; charset=utf-8",
            "--cache-control",
            "public, max-age=300",
            "--remote",
            "--force",
        ]
        attempts: list[dict[str, Any]] = []
        result = None
        for attempt in range(1, max(1, int(args.retries or 1)) + 1):
            result = run(command, args.worker_dir, args.dry_run)
            attempts.append(result)
            if result["returncode"] == 0:
                break
            if attempt < max(1, int(args.retries or 1)) and not args.dry_run:
                time.sleep(max(0.0, float(args.retry_delay or 0.0)) * attempt)
        result = result or attempts[-1]
        record = {
            "index": offset,
            "relative_path": rel_path,
            "object_key": object_key,
            "bytes": actual_bytes,
            "sha256": actual_canonical_sha,
            "file_sha256": actual_file_sha,
            "ok": result["returncode"] == 0,
            "command": result["command"],
            "stdout_tail": result["stdout"],
            "stderr_tail": result["stderr"],
            "attempts": len(attempts),
        }
        uploaded.append(record)
        if not record["ok"]:
            errors.append(f"Upload failed for {object_key}: {result['stderr'] or result['stdout']}")
            break
    return uploaded, errors


def maybe_apply_d1(args: argparse.Namespace) -> dict[str, Any]:
    sql_path = args.publish_dir / "d1_changed_index.sql"
    has_statements = d1_sql_has_statements(sql_path)
    if not has_statements:
        return {
            "path": str(sql_path),
            "has_statements": False,
            "applied": False,
            "reason": "No D1 index row changes.",
        }
    if not args.apply_d1:
        return {
            "path": str(sql_path),
            "has_statements": True,
            "applied": False,
            "reason": "Pass --apply-d1 to apply this delta.",
        }
    command = [
        "npx",
        "wrangler",
        "d1",
        "execute",
        args.d1_database,
        "--remote",
        "--file",
        str(sql_path),
    ]
    result = run(command, args.worker_dir, args.dry_run)
    return {
        "path": str(sql_path),
        "has_statements": True,
        "applied": result["returncode"] == 0,
        "command": result["command"],
        "stdout_tail": result["stdout"],
        "stderr_tail": result["stderr"],
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.publish_dir = args.publish_dir if args.publish_dir.is_absolute() else ROOT / args.publish_dir
    args.worker_dir = args.worker_dir if args.worker_dir.is_absolute() else ROOT / args.worker_dir
    args.report = args.report if args.report.is_absolute() else ROOT / args.report

    plan_path = args.publish_dir / ("manifest.json" if args.all_objects else "upload_plan.json")
    if not plan_path.exists():
        print(f"Missing upload source: {plan_path}", file=sys.stderr)
        return 2

    plan = read_json(plan_path)
    objects = plan.get("objects", []) if isinstance(plan, dict) else []
    if not isinstance(objects, list):
        print(f"Invalid upload plan objects: {plan_path}", file=sys.stderr)
        return 2

    uploaded, errors = upload_objects(args, objects)
    d1 = maybe_apply_d1(args) if not errors else {"applied": False, "reason": "Skipped because R2 upload failed."}
    report = {
        "schema": "site_publish_r2_upload_report_v1",
        "generated_at": utc_now(),
        "dry_run": bool(args.dry_run),
        "bucket": args.bucket,
        "publish_dir": str(args.publish_dir),
        "plan_path": str(plan_path),
        "upload_mode": "all_objects" if args.all_objects else "changed_objects",
        "planned_changed_objects": len(objects),
        "planned_changed_bytes": (
            sum(int(item.get("bytes") or 0) for item in objects)
            if args.all_objects
            else int(plan.get("total_changed_bytes") or 0)
            if isinstance(plan, dict)
            else 0
        ),
        "uploaded_objects": len([item for item in uploaded if item.get("ok")]),
        "uploaded_bytes": sum(int(item.get("bytes") or 0) for item in uploaded if item.get("ok")),
        "errors": errors,
        "start_index": int(args.start_index or 1),
        "ok": not errors and len(uploaded) == len(objects[max(0, int(args.start_index or 1) - 1):][: args.limit or None]),
        "d1": d1,
        "objects": uploaded,
    }
    write_report(args.report, report)
    print(json.dumps({key: report[key] for key in ("ok", "bucket", "planned_changed_objects", "uploaded_objects", "uploaded_bytes", "errors", "d1")}, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
