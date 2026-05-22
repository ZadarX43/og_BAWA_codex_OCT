from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import NORMALIZED_FILES
from .raw_helpers import to_int
from .schema_contracts import NORMALIZED_SCHEMAS
from .scaffold import build_csv_stub

PURPOSE = "Build normalized sidelined table from API-Football raw sidelined payloads."
TARGET_PATH = NORMALIZED_FILES["sidelined"]


def build_stub() -> pd.DataFrame:
    return build_csv_stub(TARGET_PATH, NORMALIZED_SCHEMAS["sidelined"], PURPOSE, placeholder_row=False)


def _read_payloads(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    payloads: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if line:
                payloads.append(json.loads(line))
    return payloads


def _open_absence(value: Any) -> int:
    text = str(value or "").strip().lower()
    return int(text in {"", "unknown", "none", "ongoing", "null"})


def _first(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return ""


def build_sidelined(sidelined_raw: str | None = None, output_csv: str = str(TARGET_PATH)) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    payloads = _read_payloads(sidelined_raw) if sidelined_raw else []
    for payload in payloads:
        fetch_meta = payload.get("_og_fetch") or {}
        source_scope = fetch_meta.get("source_scope") or ""
        params = fetch_meta.get("params") or {}
        source_params = json.dumps(params, sort_keys=True, ensure_ascii=True)
        fetched_ts = fetch_meta.get("fetched_ts_utc") or ""
        for item in payload.get("response", []) or []:
            player = item.get("player") or {}
            coach = item.get("coach") or {}
            team = item.get("team") or {}
            start_date = _first(item.get("start"), item.get("start_date"), item.get("from"))
            end_date = _first(item.get("end"), item.get("end_date"), item.get("to"))
            rows.append(
                {
                    "player_id": to_int(player.get("id"), to_int(params.get("player"))),
                    "player_name": player.get("name") or "",
                    "coach_id": to_int(coach.get("id"), to_int(params.get("coach"))),
                    "coach_name": coach.get("name") or "",
                    "team_id": to_int(team.get("id")),
                    "team_name": team.get("name") or "",
                    "absence_type": _first(item.get("type"), item.get("absence_type")),
                    "reason": _first(item.get("reason"), item.get("detail")),
                    "start_date": start_date or "",
                    "end_date": end_date or "",
                    "is_open_absence": _open_absence(end_date),
                    "source_scope": source_scope,
                    "source_params": source_params,
                    "fetched_ts_utc": fetched_ts,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=NORMALIZED_SCHEMAS["sidelined"])
    else:
        df = df.drop_duplicates(
            subset=["player_id", "coach_id", "absence_type", "reason", "start_date", "end_date"]
        ).reset_index(drop=True)
        df = df.reindex(columns=NORMALIZED_SCHEMAS["sidelined"])
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=PURPOSE)
    parser.add_argument("--write-stub", action="store_true")
    parser.add_argument("--sidelined-raw", default="")
    parser.add_argument("--output-csv", default=str(TARGET_PATH))
    args = parser.parse_args()
    if args.write_stub:
        df = build_stub()
        print(f"WROTE STUB: {TARGET_PATH} rows={len(df)}")
        return
    df = build_sidelined(args.sidelined_raw or None, args.output_csv)
    print(f"WROTE: {args.output_csv} rows={len(df)}")


if __name__ == "__main__":
    main()
