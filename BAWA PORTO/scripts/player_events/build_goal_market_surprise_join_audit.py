from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd

RELEVANT_MARKETS = {
    "missing DM": {"ftr", "btts", "ou25"},
    "missing full-back": {"btts", "ou25"},
    "missing CB duel anchor": {"ftr", "ou25"},
}


def slugify(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def norm_fixture(date_value: str, home: str, away: str) -> str:
    return f"{str(date_value)[:10]}__{slugify(home)}__{slugify(away)}"


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
                rows.append(
                    {
                        "fixture_key": current_fixture,
                        "fixture_label": fixture_label,
                        "prematch_risk_focus": ", ".join(focuses),
                    }
                )
            current_fixture = line.replace("## ", "", 1).strip()
            fixture_label = None
            focuses = []
        elif line.startswith("- ") and " | focus=" in line:
            left, right = line[2:].split("| focus=", 1)
            fixture_label = left.strip()
            focuses = [part.strip() for part in right.split(",") if part.strip()]
    if current_fixture and fixture_label:
        rows.append(
            {
                "fixture_key": current_fixture,
                "fixture_label": fixture_label,
                "prematch_risk_focus": ", ".join(focuses),
            }
        )
    return pd.DataFrame(rows)


def load_runner_meta(runner_csv: str) -> pd.DataFrame:
    df = pd.read_csv(runner_csv, low_memory=False)
    if df.empty:
        return df
    cols = [
        "fixture_key",
        "match_date",
        "home_team_name",
        "away_team_name",
        "league",
        "competition",
    ]
    meta = df[cols].drop_duplicates("fixture_key").copy()
    meta["fixture_norm_key"] = meta.apply(
        lambda row: norm_fixture(row["match_date"], row["home_team_name"], row["away_team_name"]), axis=1
    )
    return meta


def iter_merged_files(glob_patterns: Iterable[str]) -> Iterable[Path]:
    seen: set[Path] = set()
    for pattern in glob_patterns:
        for path in Path().glob(pattern):
            if path not in seen and path.is_file():
                seen.add(path)
                yield path


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


def load_goal_outcomes(glob_patterns: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    usecols = [
        "match_date",
        "home_team_name",
        "away_team_name",
        "league",
        "fixture_key",
        "home_team_goal_count",
        "away_team_goal_count",
    ]
    for path in iter_merged_files(glob_patterns):
        if "__snapshot_proxy" in path.name or "__backup__" in str(path) or "baselines" in str(path) or "truth" in path.name:
            continue
        try:
            header = pd.read_csv(path, nrows=0).columns.tolist()
        except Exception:
            continue
        cols = [c for c in usecols if c in header]
        required = {"match_date", "home_team_name", "away_team_name", "home_team_goal_count", "away_team_goal_count"}
        if not required.issubset(cols):
            continue
        try:
            part = pd.read_csv(path, usecols=cols, low_memory=False)
        except Exception:
            continue
        rows.append(part)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["match_date", "home_team_name", "away_team_name"])
    df["home_goals"] = pd.to_numeric(df["home_team_goal_count"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_team_goal_count"], errors="coerce")
    df = df.dropna(subset=["home_goals", "away_goals"])
    df["fixture_norm_key"] = df.apply(lambda row: norm_fixture(row["match_date"], row["home_team_name"], row["away_team_name"]), axis=1)
    df["actual_scoreline"] = df["home_goals"].astype(int).astype(str) + "-" + df["away_goals"].astype(int).astype(str)
    df["actual_ftr"] = df.apply(
        lambda row: "HOME" if row["home_goals"] > row["away_goals"] else "AWAY" if row["away_goals"] > row["home_goals"] else "DRAW",
        axis=1,
    )
    df["actual_btts"] = df.apply(lambda row: "YES" if row["home_goals"] > 0 and row["away_goals"] > 0 else "NO", axis=1)
    df["actual_ou25"] = df.apply(lambda row: "OVER25" if (row["home_goals"] + row["away_goals"]) > 2 else "UNDER25", axis=1)
    df["home_slug"] = df["home_team_name"].map(slugify)
    df["away_slug"] = df["away_team_name"].map(slugify)
    df = df.sort_values(["fixture_norm_key"]).drop_duplicates("fixture_norm_key", keep="last")
    return df[["fixture_norm_key", "match_date", "home_team_name", "away_team_name", "home_slug", "away_slug", "league", "actual_scoreline", "actual_ftr", "actual_btts", "actual_ou25"]]


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
    wanted = [
        "fixture_key",
        "home",
        "away",
        "match_date",
        "market",
        "selection",
        "deploy_tier",
        "keep",
        "model_p",
        "rank",
        "slip_leg_bucket",
        "review_flag",
        "review_reason",
        "team_intel_overlay_action",
        "team_intel_overlay_slip_caution_flag",
    ]
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
    df["model_p"] = pd.to_numeric(df.get("model_p"), errors="coerce")
    df["rank"] = pd.to_numeric(df.get("rank"), errors="coerce")
    df["keep_bool"] = df.get("keep").astype(str).str.lower().isin(["true", "1", "yes"])
    df = df.sort_values(["fixture_norm_key", "market", "rank", "model_p"], ascending=[True, True, True, False])
    return df.drop_duplicates(["fixture_norm_key", "market"], keep="first")


def classify_surprise(row: pd.Series) -> str:
    if pd.isna(row.get("actual_selection")):
        return "NO_ACTUAL_OUTCOME"
    if pd.isna(row.get("selection")):
        return "NO_GOAL_MARKET_SELECTION"
    if int(row.get("hit_flag", 0)) == 1:
        return "CAUTION_ABSORBED"
    relevant_markets = set()
    for focus in [p.strip() for p in str(row.get("prematch_risk_focus", "")).split(",") if p.strip()]:
        relevant_markets |= RELEVANT_MARKETS.get(focus, set())
    direct = row.get("market") in relevant_markets
    high_conf = pd.to_numeric(row.get("model_p"), errors="coerce") >= 0.60
    if direct and high_conf:
        return "DIRECT_CAUTION_HIGH_CONFIDENCE_MISS"
    if direct:
        return "DIRECT_CAUTION_MISS"
    if high_conf:
        return "OFF_AXIS_HIGH_CONFIDENCE_MISS"
    return "OFF_AXIS_MISS"


def build_audit(risk_md: str, runner_csv: str, ranked_roots: list[str], merged_globs: list[str], output_csv: str, output_md: str) -> pd.DataFrame:
    risk = parse_focus_map(risk_md)
    meta = load_runner_meta(runner_csv)
    preds = load_ranked_predictions(ranked_roots)
    actuals = load_goal_outcomes(merged_globs)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    if risk.empty or meta.empty:
        empty = pd.DataFrame()
        empty.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Goal-Market Surprise Join Audit\n\nNo structural risk fixtures matched.\n")
        return empty

    base = risk.merge(meta, on="fixture_key", how="left")
    base["home_slug"] = base["home_team_name"].map(slugify)
    base["away_slug"] = base["away_team_name"].map(slugify)

    joined_rows: list[dict[str, object]] = []
    for _, row in base.iterrows():
        row_date = str(row.get("match_date", ""))[:10]
        actual_match = actuals[actuals["fixture_norm_key"] == row.get("fixture_norm_key")]
        if actual_match.empty:
            actual_match = actuals[
                (actuals["match_date"] == row_date)
                & actuals["home_team_name"].apply(lambda v: team_slug_match(row.get("home_team_name", ""), v))
                & actuals["away_team_name"].apply(lambda v: team_slug_match(row.get("away_team_name", ""), v))
            ]
        actual_rec = actual_match.iloc[0].to_dict() if not actual_match.empty else {}

        pred_match = preds[preds["fixture_norm_key"] == row.get("fixture_norm_key")]
        if pred_match.empty:
            pred_match = preds[
                (preds["match_date"] == row_date)
                & preds["home"].apply(lambda v: team_slug_match(row.get("home_team_name", ""), v))
                & preds["away"].apply(lambda v: team_slug_match(row.get("away_team_name", ""), v))
            ]
        for _, pred in pred_match.iterrows():
            merged = row.to_dict()
            merged.update({f"actual__{k}": v for k, v in actual_rec.items()})
            merged.update({f"pred__{k}": v for k, v in pred.to_dict().items()})
            joined_rows.append(merged)

    audit = pd.DataFrame(joined_rows)
    if audit.empty:
        audit.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Goal-Market Surprise Join Audit\n\nNo joined goal-market rows matched.\n")
        return audit
    audit["actual_scoreline"] = audit.get("actual__actual_scoreline")
    audit["league"] = audit.get("actual__league").fillna(audit.get("league"))
    audit["market"] = audit.get("pred__market")
    audit["selection"] = audit.get("pred__selection")
    audit["deploy_tier"] = audit.get("pred__deploy_tier")
    audit["model_p"] = audit.get("pred__model_p")
    audit["keep_bool"] = audit.get("pred__keep_bool")
    audit["slip_leg_bucket"] = audit.get("pred__slip_leg_bucket")
    audit["team_intel_overlay_action"] = audit.get("pred__team_intel_overlay_action")
    audit["review_flag"] = audit.get("pred__review_flag")
    audit["review_reason"] = audit.get("pred__review_reason")
    audit["source_csv"] = audit.get("pred__source_csv")
    actual_map = {"ftr": "actual__actual_ftr", "btts": "actual__actual_btts", "ou25": "actual__actual_ou25"}
    audit["actual_selection"] = audit.apply(lambda row: row.get(actual_map.get(str(row["market"]), "")), axis=1)
    audit["hit_flag"] = (audit["selection"].astype(str) == audit["actual_selection"].astype(str)).astype(int)
    audit["surprise_condition"] = audit.apply(classify_surprise, axis=1)
    audit["direct_caution_market_flag"] = audit.apply(
        lambda row: int(
            any(
                row["market"] in RELEVANT_MARKETS.get(focus.strip(), set())
                for focus in str(row.get("prematch_risk_focus", "")).split(",")
                if focus.strip()
            )
        ),
        axis=1,
    )
    keep_cols = [
        "fixture_key",
        "fixture_label",
        "prematch_risk_focus",
        "match_date",
        "league",
        "home_team_name",
        "away_team_name",
        "market",
        "selection",
        "actual_selection",
        "actual_scoreline",
        "hit_flag",
        "deploy_tier",
        "model_p",
        "keep_bool",
        "slip_leg_bucket",
        "team_intel_overlay_action",
        "review_flag",
        "review_reason",
        "direct_caution_market_flag",
        "surprise_condition",
        "source_csv",
    ]
    for col in keep_cols:
        if col not in audit.columns:
            audit[col] = pd.NA
    audit = audit[keep_cols]
    audit = audit.sort_values(["match_date", "fixture_key", "market"])
    audit.to_csv(output_csv, index=False)

    lines = [
        "# Goal-Market Surprise Join Audit",
        "",
        "- Joins player-market structural cautions to historical goal-market walkforward selections and realized FTR / BTTS / OU25 outcomes.",
        "- `DIRECT_CAUTION_HIGH_CONFIDENCE_MISS` = caution touched that market, model confidence was high, and the goal-market pick still missed.",
        "- `CAUTION_ABSORBED` = caution existed but the goal-market pick still landed.",
        "",
    ]
    if audit.empty:
        lines.append("No joined goal-market rows matched.")
    else:
        summary = (
            audit.groupby(["prematch_risk_focus", "market", "surprise_condition"], dropna=False)
            .agg(rows=("fixture_key", "size"), fixtures=("fixture_key", pd.Series.nunique), hit_rate=("hit_flag", "mean"))
            .reset_index()
            .sort_values(["rows", "hit_rate"], ascending=[False, False])
        )
        lines.append("## Summary")
        for _, row in summary.iterrows():
            lines.append(
                f"- risk={row['prematch_risk_focus']} | {row['market']} | {row['surprise_condition']} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | hit_rate={row['hit_rate']:.3f}"
            )
        lines.append("")
        for fixture_key, sub in audit.groupby("fixture_key", sort=False):
            head = sub.iloc[0]
            lines.append(f"## {fixture_key}")
            lines.append(
                f"- {head['home_team_name']} vs {head['away_team_name']} | risk={head['prematch_risk_focus']} | actual_score={head['actual_scoreline']}"
            )
            for _, row in sub.iterrows():
                lines.append(
                    f"- {row['market']} | pick={row['selection']} | actual={row['actual_selection']} | hit={int(row['hit_flag'])} | tier={row['deploy_tier']} | model_p={pd.to_numeric(row['model_p'], errors='coerce'):.3f} | {row['surprise_condition']}"
                )
            lines.append("")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a true goal-market surprise join audit from structural cautions and historical walkforward outputs.")
    parser.add_argument("--risk-md", required=True)
    parser.add_argument("--runner-csv", required=True)
    parser.add_argument("--ranked-root", action="append", required=True)
    parser.add_argument("--merged-glob", action="append", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build_audit(args.risk_md, args.runner_csv, args.ranked_root, args.merged_glob, args.output_csv, args.output_md)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")
