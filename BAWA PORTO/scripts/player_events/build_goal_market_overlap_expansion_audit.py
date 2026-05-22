from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd


def slugify(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def norm_fixture(date_value: str, home: str, away: str) -> str:
    return f"{str(date_value)[:10]}__{slugify(home)}__{slugify(away)}"


def team_slug_match(left: str, right: str) -> bool:
    left_slug = slugify(left)
    right_slug = slugify(right)
    if not left_slug or not right_slug:
        return False
    if left_slug == right_slug or left_slug in right_slug or right_slug in left_slug:
        return True
    left_tokens = set(left_slug.split("_"))
    right_tokens = set(right_slug.split("_"))
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))
    return overlap >= 0.5


def parse_focus_map(risk_md: str) -> pd.DataFrame:
    path = Path(risk_md)
    rows: list[dict[str, object]] = []
    if not path.exists():
        return pd.DataFrame()
    current_fixture = None
    fixture_label = None
    focuses: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("## "):
            if current_fixture and fixture_label:
                rows.append({"fixture_key": current_fixture, "fixture_label": fixture_label, "prematch_risk_focus": ", ".join(focuses)})
            current_fixture = line.replace("## ", "", 1).strip()
            fixture_label = None
            focuses = []
        elif line.startswith("- ") and " | focus=" in line:
            left, right = line[2:].split("| focus=", 1)
            fixture_label = left.strip()
            focuses = [part.strip() for part in right.split(",") if part.strip()]
    if current_fixture and fixture_label:
        rows.append({"fixture_key": current_fixture, "fixture_label": fixture_label, "prematch_risk_focus": ", ".join(focuses)})
    return pd.DataFrame(rows)


def load_runner_meta(runner_csv: str) -> pd.DataFrame:
    df = pd.read_csv(runner_csv, low_memory=False)
    cols = ["fixture_key", "match_date", "home_team_name", "away_team_name", "league", "competition"]
    meta = df[cols].drop_duplicates("fixture_key").copy()
    meta["fixture_norm_key"] = meta.apply(lambda row: norm_fixture(row["match_date"], row["home_team_name"], row["away_team_name"]), axis=1)
    return meta


def iter_merged_files(glob_patterns: Iterable[str]) -> Iterable[Path]:
    seen: set[Path] = set()
    for pattern in glob_patterns:
        for path in Path().glob(pattern):
            if path not in seen and path.is_file():
                seen.add(path)
                yield path


def load_goal_outcomes(glob_patterns: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    usecols = ["match_date", "home_team_name", "away_team_name", "league", "home_team_goal_count", "away_team_goal_count"]
    for path in iter_merged_files(glob_patterns):
        if "__snapshot_proxy" in path.name or "__backup__" in str(path) or "baselines" in str(path) or "truth" in path.name:
            continue
        try:
            header = pd.read_csv(path, nrows=0).columns.tolist()
        except Exception:
            continue
        cols = [c for c in usecols if c in header]
        if not {"match_date", "home_team_name", "away_team_name", "home_team_goal_count", "away_team_goal_count"}.issubset(cols):
            continue
        try:
            part = pd.read_csv(path, usecols=cols, low_memory=False)
        except Exception:
            continue
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["match_date", "home_team_name", "away_team_name"])
    df["fixture_norm_key"] = df.apply(lambda row: norm_fixture(row["match_date"], row["home_team_name"], row["away_team_name"]), axis=1)
    df["home_slug"] = df["home_team_name"].map(slugify)
    df["away_slug"] = df["away_team_name"].map(slugify)
    return df.drop_duplicates("fixture_norm_key")


def load_ranked_predictions(ranked_roots: list[str]) -> pd.DataFrame:
    paths = []
    for ranked_root in ranked_roots:
        root = Path(ranked_root)
        if not root.exists():
            continue
        paths.extend(list(root.rglob("ranked_board_ftr_*.csv")))
        paths.extend(list(root.rglob("ranked_board_btts_*.csv")))
        paths.extend(list(root.rglob("ranked_board_ou25_*.csv")))
    rows: list[pd.DataFrame] = []
    wanted = ["fixture_key", "home", "away", "match_date", "market", "selection", "source_file"]
    for path in paths:
        try:
            header = pd.read_csv(path, nrows=0).columns.tolist()
        except Exception:
            continue
        cols = [c for c in wanted if c in header]
        if not {"home", "away", "match_date", "market", "selection"}.issubset(cols):
            continue
        part = pd.read_csv(path, usecols=cols, low_memory=False)
        part["source_csv"] = str(path)
        rows.append(part)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["fixture_norm_key"] = df.apply(lambda row: norm_fixture(row["match_date"], row["home"], row["away"]), axis=1)
    df["home_slug"] = df["home"].map(slugify)
    df["away_slug"] = df["away"].map(slugify)
    return df.drop_duplicates(["fixture_norm_key", "market"])


def build(risk_md: str, runner_csv: str, joined_csv: str, ranked_roots: list[str], merged_globs: list[str], output_csv: str, output_md: str) -> pd.DataFrame:
    risk = parse_focus_map(risk_md)
    meta = load_runner_meta(runner_csv)
    joined = pd.read_csv(joined_csv, low_memory=False) if Path(joined_csv).exists() else pd.DataFrame()
    ranked = load_ranked_predictions(ranked_roots)
    actuals = load_goal_outcomes(merged_globs)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    if risk.empty or meta.empty:
        empty = pd.DataFrame()
        empty.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Goal-Market Overlap Expansion Audit\n\nNo structural risk fixtures matched.\n")
        return empty

    base = risk.merge(meta, on="fixture_key", how="left")
    base["joined_flag"] = base["fixture_key"].isin(joined.get("fixture_key", pd.Series(dtype=str)))
    rows: list[dict[str, object]] = []
    for _, row in base.iterrows():
        row_date = str(row.get("match_date", ""))[:10]
        ranked_match = ranked[
            (ranked["match_date"] == row_date)
            & ranked["home"].apply(lambda v: team_slug_match(row.get("home_team_name", ""), v))
            & ranked["away"].apply(lambda v: team_slug_match(row.get("away_team_name", ""), v))
        ] if not ranked.empty else pd.DataFrame()
        actual_match = actuals[
            (actuals["match_date"] == row_date)
            & actuals["home_team_name"].apply(lambda v: team_slug_match(row.get("home_team_name", ""), v))
            & actuals["away_team_name"].apply(lambda v: team_slug_match(row.get("away_team_name", ""), v))
        ] if not actuals.empty else pd.DataFrame()

        has_ranked = not ranked_match.empty
        has_actual = not actual_match.empty
        if row.get("joined_flag"):
            status = "MATCHED_IN_JOIN"
            reason = "Fixture already joined cleanly into the goal-market surprise audit."
        elif has_actual and has_ranked:
            status = "HAS_OVERLAP_NOT_JOINED"
            reason = "Historical ranked-board and actual result both exist, but the fixture did not survive into the joined audit path."
        elif has_actual and not has_ranked:
            status = "HAS_ACTUAL_NO_RANKED_HISTORY"
            reason = "We have the result in canonical merged data, but no historical ranked FTR/BTTS/OU25 board matched this fixture."
        elif has_ranked and not has_actual:
            status = "HAS_RANKED_NO_ACTUAL"
            reason = "Historical ranked board exists, but no canonical actual result row matched cleanly."
        else:
            status = "NO_HISTORICAL_OVERLAP"
            reason = "Neither a ranked-board history row nor a canonical actual result matched this structural-risk fixture."
        rows.append(
            {
                "fixture_key": row.get("fixture_key"),
                "fixture_label": row.get("fixture_label"),
                "prematch_risk_focus": row.get("prematch_risk_focus"),
                "match_date": row.get("match_date"),
                "league": row.get("league"),
                "competition": row.get("competition"),
                "home_team_name": row.get("home_team_name"),
                "away_team_name": row.get("away_team_name"),
                "joined_flag": int(bool(row.get("joined_flag"))),
                "has_ranked_history": int(has_ranked),
                "has_actual_result": int(has_actual),
                "ranked_markets_found": "|".join(sorted(set(ranked_match["market"].astype(str)))) if has_ranked else "",
                "ranked_rows": int(len(ranked_match)),
                "status": status,
                "reason": reason,
            }
        )
    out = pd.DataFrame(rows).sort_values(["status", "match_date", "fixture_key"])
    out.to_csv(output_csv, index=False)

    lines = [
        "# Goal-Market Overlap Expansion Audit",
        "",
        "- Shows which structural-risk fixtures do and do not find historical goal-market overlap.",
        "- This is the audit we need before we over-interpret sparse surprise-join coverage.",
        "",
    ]
    summary = out.groupby("status", dropna=False).agg(rows=("fixture_key", "size")).reset_index()
    lines.append("## Summary")
    for _, s in summary.iterrows():
        lines.append(f"- {s['status']} | rows={int(s['rows'])}")
    lines.append("")
    for status, sub in out.groupby("status", sort=False):
        lines.append(f"## {status}")
        for _, r in sub.iterrows():
            lines.append(
                f"- {r['fixture_key']} | risk={r['prematch_risk_focus']} | ranked={int(r['has_ranked_history'])} | actual={int(r['has_actual_result'])} | markets={r['ranked_markets_found'] or 'none'}"
            )
            lines.append(f"  reason: {r['reason']}")
        lines.append("")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a goal-market overlap expansion audit for structural-risk fixtures.")
    parser.add_argument("--risk-md", required=True)
    parser.add_argument("--runner-csv", required=True)
    parser.add_argument("--joined-csv", required=True)
    parser.add_argument("--ranked-root", action="append", required=True)
    parser.add_argument("--merged-glob", action="append", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build(args.risk_md, args.runner_csv, args.joined_csv, args.ranked_root, args.merged_glob, args.output_csv, args.output_md)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")
