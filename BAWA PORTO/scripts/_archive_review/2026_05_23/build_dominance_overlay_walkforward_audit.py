from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def _split_tokens(value: object) -> set[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    return {token.strip() for token in text.split("|") if token.strip()}


def _pick_first_existing(row: pd.Series, columns: Iterable[str], default: float | None = None) -> float | None:
    for column in columns:
        if column in row and pd.notna(row[column]):
            try:
                return float(row[column])
            except Exception:
                continue
    return default


def _fixture_key(df: pd.DataFrame) -> pd.Series:
    return (
        df.get("match_date", pd.Series("", index=df.index)).astype(str)
        + " | "
        + df.get("home_team_name", pd.Series("", index=df.index)).astype(str)
        + " | "
        + df.get("away_team_name", pd.Series("", index=df.index)).astype(str)
    )


def _load_deploy_frames(patterns: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for pattern in patterns:
        for path in sorted(Path().glob(pattern)):
            if path.is_file():
                frame = pd.read_csv(path, low_memory=False)
                frame["__source_file"] = str(path)
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_actuals(actuals_csv: str) -> pd.DataFrame:
    actuals = pd.read_csv(actuals_csv, low_memory=False)
    actuals["fixture_key_join"] = _fixture_key(actuals)
    deduped = (
        actuals.sort_values(["fixture_key_join"])
        .drop_duplicates("fixture_key_join", keep="first")
        .loc[:, ["fixture_key_join", "home_goals", "away_goals"]]
    )
    return deduped


def _load_actuals_from_merged(merged_root: str) -> pd.DataFrame:
    merged_paths = sorted(Path(merged_root).glob("*__merged.csv"))
    frames: list[pd.DataFrame] = []
    for path in merged_paths:
        try:
            frame = pd.read_csv(
                path,
                usecols=["date_GMT", "home_team_name", "away_team_name", "home_team_goal_count", "away_team_goal_count"],
                low_memory=False,
            )
        except Exception:
            continue
        frame = frame.rename(
            columns={
                "date_GMT": "match_date_raw",
                "home_team_goal_count": "home_goals",
                "away_team_goal_count": "away_goals",
            }
        )
        frame["match_date"] = pd.to_datetime(frame["match_date_raw"], errors="coerce").dt.strftime("%Y-%m-%d")
        frame["fixture_key_join"] = (
            frame["match_date"].astype(str)
            + " | "
            + frame["home_team_name"].astype(str)
            + " | "
            + frame["away_team_name"].astype(str)
        )
        frames.append(frame.loc[:, ["fixture_key_join", "home_goals", "away_goals"]])
    if not frames:
        return pd.DataFrame(columns=["fixture_key_join", "home_goals", "away_goals"])
    actuals = pd.concat(frames, ignore_index=True)
    actuals = actuals.dropna(subset=["fixture_key_join"]).drop_duplicates("fixture_key_join", keep="first")
    return actuals


def _score_original_market(row: pd.Series) -> float | None:
    if pd.isna(row.get("home_goals")) or pd.isna(row.get("away_goals")):
        return None
    home_goals = int(row["home_goals"])
    away_goals = int(row["away_goals"])
    market = str(row.get("market", "")).lower()
    selection = str(row.get("selection", "")).upper()

    if market == "ftr":
        if selection == "HOME":
            return float(home_goals > away_goals)
        if selection == "DRAW":
            return float(home_goals == away_goals)
        if selection == "AWAY":
            return float(away_goals > home_goals)
    if market == "ou25":
        total_goals = home_goals + away_goals
        if selection == "OVER25":
            return float(total_goals >= 3)
        if selection == "UNDER25":
            return float(total_goals <= 2)
    if market == "btts":
        both_score = home_goals >= 1 and away_goals >= 1
        if selection == "YES":
            return float(both_score)
        if selection == "NO":
            return float(not both_score)
    return None


def _score_suggested_market(row: pd.Series, suggested_market: str) -> float | None:
    if pd.isna(row.get("home_goals")) or pd.isna(row.get("away_goals")):
        return None
    home_goals = int(row["home_goals"])
    away_goals = int(row["away_goals"])
    market = str(suggested_market or "").upper()
    if not market or market in {"NONE", "AS_IS", "RESEARCH_ONLY"}:
        return None
    if market == "HOME_TEAM_OVER_1_5":
        return float(home_goals >= 2)
    if market == "AWAY_TEAM_OVER_1_5":
        return float(away_goals >= 2)
    if market == "HOME_TEAM_OVER_2_5":
        return float(home_goals >= 3)
    if market == "AWAY_TEAM_OVER_2_5":
        return float(away_goals >= 3)
    if market == "DOMINANT_TEAM_OVER_1_5":
        return float(home_goals >= 2) if str(row.get("dominant_side")) == "HOME" else float(away_goals >= 2)
    if market == "DOMINANT_TEAM_OVER_2_5":
        return float(home_goals >= 3) if str(row.get("dominant_side")) == "HOME" else float(away_goals >= 3)
    return None


def _apply_ppg_mapping(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ppg_home_pre"] = pd.to_numeric(df.get("ppg_home_pre"), errors="coerce")
    df["ppg_away_pre"] = pd.to_numeric(df.get("ppg_away_pre"), errors="coerce")
    if "ppg_diff_pre" in df.columns:
        df["ppg_gap"] = pd.to_numeric(df["ppg_diff_pre"], errors="coerce").abs()
    else:
        df["ppg_gap"] = (df["ppg_home_pre"] - df["ppg_away_pre"]).abs()
    df["dominant_side"] = df.apply(
        lambda row: "HOME" if pd.to_numeric(row.get("ppg_home_pre"), errors="coerce") >= pd.to_numeric(row.get("ppg_away_pre"), errors="coerce") else "AWAY",
        axis=1,
    )
    df["dominant_ppg"] = df.apply(
        lambda row: row["ppg_home_pre"] if row["dominant_side"] == "HOME" else row["ppg_away_pre"],
        axis=1,
    )
    df["opponent_ppg"] = df.apply(
        lambda row: row["ppg_away_pre"] if row["dominant_side"] == "HOME" else row["ppg_home_pre"],
        axis=1,
    )
    # Research-only mapping from user: PPG 3.0 -> xG proxy 2.5, 0.0 -> 0.0.
    df["dominant_ppg_xg_proxy"] = (df["dominant_ppg"] / 3.0) * 2.5
    df["opponent_ppg_xg_proxy"] = (df["opponent_ppg"] / 3.0) * 2.5
    df["dominant_team_goals_pred"] = df.apply(
        lambda row: _pick_first_existing(
            row,
            ["home_goals_pred", "lambda_home", "pre_match_xg_home"],
            default=row["dominant_ppg_xg_proxy"],
        )
        if row["dominant_side"] == "HOME"
        else _pick_first_existing(
            row,
            ["away_goals_pred", "lambda_away", "pre_match_xg_away"],
            default=row["dominant_ppg_xg_proxy"],
        ),
        axis=1,
    )
    df["dominant_scored_rate"] = df.apply(
        lambda row: _pick_first_existing(row, ["scored_rate_5_home"], default=None)
        if row["dominant_side"] == "HOME"
        else _pick_first_existing(row, ["scored_rate_5_away"], default=None),
        axis=1,
    )
    df["opponent_conceded_rate"] = df.apply(
        lambda row: _pick_first_existing(row, ["conceded_rate_5_away"], default=None)
        if row["dominant_side"] == "HOME"
        else _pick_first_existing(row, ["conceded_rate_5_home"], default=None),
        axis=1,
    )
    return df


def _row_matches_rule(row: pd.Series, rule: pd.Series) -> tuple[bool, dict[str, bool]]:
    row_tokens = _split_tokens(row.get("context_reason_codes"))
    applies_to_markets = {m.strip().lower() for m in str(rule.get("applies_to_markets", "")).split("|") if m.strip()}
    require_any = _split_tokens(rule.get("requires_any_tokens"))
    require_all = _split_tokens(rule.get("requires_all_tokens"))
    redirect_any = _split_tokens(rule.get("redirect_if_any_tokens"))
    block_any = _split_tokens(rule.get("block_if_any_tokens"))

    checks = {
        "require_any_hit": (not require_any) or bool(row_tokens & require_any),
        "require_all_hit": require_all.issubset(row_tokens),
        "redirect_token_hit": bool(row_tokens & redirect_any),
        "blocked": bool(row_tokens & block_any),
        "market_ok": (not applies_to_markets) or (str(row.get("market", "")).lower() in applies_to_markets),
        "ppg_gap_ok": pd.to_numeric(row.get("ppg_gap"), errors="coerce") >= pd.to_numeric(rule.get("min_ppg_gap"), errors="coerce"),
        "dominant_ppg_xg_ok": pd.to_numeric(row.get("dominant_ppg_xg_proxy"), errors="coerce") >= pd.to_numeric(rule.get("min_dominant_ppg_xg_proxy"), errors="coerce"),
        "dominant_team_goals_ok": pd.to_numeric(row.get("dominant_team_goals_pred"), errors="coerce") >= pd.to_numeric(rule.get("min_dominant_team_goals_pred"), errors="coerce"),
        "dominant_scored_rate_ok": pd.to_numeric(row.get("dominant_scored_rate"), errors="coerce") >= pd.to_numeric(rule.get("min_dominant_scored_rate"), errors="coerce"),
        "opponent_conceded_rate_ok": pd.to_numeric(row.get("opponent_conceded_rate"), errors="coerce") >= pd.to_numeric(rule.get("min_opponent_conceded_rate"), errors="coerce"),
    }
    matched = (
        checks["require_any_hit"]
        and checks["require_all_hit"]
        and not checks["blocked"]
        and checks["market_ok"]
        and checks["ppg_gap_ok"]
        and checks["dominant_ppg_xg_ok"]
        and checks["dominant_team_goals_ok"]
        and checks["dominant_scored_rate_ok"]
        and checks["opponent_conceded_rate_ok"]
    )
    return matched, checks


def _collapse_overlay_action(label: str) -> str:
    label = str(label)
    if label == "PROMOTE_TO_TEAM_GOALS" or label == "GOOD_FIXTURE_WRONG_MARKET":
        return "REDIRECT"
    if label == "DOWNGRADE_TO_OBSERVE":
        return "DOWNGRADE"
    if label == "FULL_AVOID":
        return "AVOID"
    return "HOLD"


def build_audit(
    deploy_globs: list[str],
    actuals_csv: str | None,
    merged_root: str | None,
    rules_csv: str,
    outdir: str,
) -> dict[str, Path]:
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    deploy = _load_deploy_frames(deploy_globs)
    if deploy.empty:
        raise SystemExit("No deploy files matched the provided patterns.")

    deploy = deploy.copy()
    if actuals_csv:
        actuals = _load_actuals(actuals_csv)
    elif merged_root:
        actuals = _load_actuals_from_merged(merged_root)
    else:
        raise SystemExit("Provide either --actuals-csv or --merged-root.")
    deploy["fixture_key_join"] = _fixture_key(deploy)
    merged = deploy.merge(actuals, on="fixture_key_join", how="left")
    merged = _apply_ppg_mapping(merged)
    merged["original_hit"] = merged.apply(_score_original_market, axis=1)

    rules = pd.read_csv(rules_csv).sort_values("priority")
    overlay_labels: list[str] = []
    suggested_markets: list[str] = []
    applied_rule_ids: list[str] = []
    redirect_token_hits: list[bool] = []
    blocked_hits: list[bool] = []

    for _, row in merged.iterrows():
        label = "KEEP_STANDARD"
        suggested_market = "AS_IS"
        applied_rule_id = ""
        redirect_hit = False
        blocked = False

        for _, rule in rules.iterrows():
            if not bool(rule.get("enabled_for_research", True)):
                continue
            matched, checks = _row_matches_rule(row, rule)
            redirect_hit = redirect_hit or checks["redirect_token_hit"]
            blocked = blocked or checks["blocked"]
            if matched:
                label = str(rule["candidate_label"])
                suggested_market = str(rule["suggested_market"])
                applied_rule_id = str(rule["rule_id"])
                break

        overlay_labels.append(label)
        suggested_markets.append(suggested_market)
        applied_rule_ids.append(applied_rule_id)
        redirect_token_hits.append(redirect_hit)
        blocked_hits.append(blocked)

    merged["overlay_label"] = overlay_labels
    merged["overlay_suggested_market"] = suggested_markets
    merged["overlay_rule_id"] = applied_rule_ids
    merged["overlay_redirect_token_hit"] = redirect_token_hits
    merged["overlay_block_token_hit"] = blocked_hits
    merged["overlay_action"] = merged["overlay_label"].map(_collapse_overlay_action)
    merged["redirected_hit"] = merged.apply(
        lambda row: _score_suggested_market(row, str(row.get("overlay_suggested_market", ""))),
        axis=1,
    )
    merged["saved_loser"] = (
        merged["original_hit"].eq(0)
        & (
            merged["overlay_action"].isin(["DOWNGRADE", "AVOID"])
            | merged["redirected_hit"].eq(1)
        )
    )
    merged["harmed_winner"] = (
        merged["original_hit"].eq(1)
        & (
            merged["overlay_action"].isin(["DOWNGRADE", "AVOID"])
            | merged["redirected_hit"].eq(0)
        )
    )

    scored_csv = out_path / "dominance_overlay_walkforward_scored.csv"
    summary_md = out_path / "dominance_overlay_walkforward_summary.md"
    redirect_wins_csv = out_path / "dominance_overlay_redirect_wins.csv"
    saved_losers_csv = out_path / "dominance_overlay_saved_losers.csv"
    harmed_winners_csv = out_path / "dominance_overlay_harmed_winners.csv"

    merged.to_csv(scored_csv, index=False)
    merged[merged["redirected_hit"].eq(1)].to_csv(redirect_wins_csv, index=False)
    merged[merged["saved_loser"]].to_csv(saved_losers_csv, index=False)
    merged[merged["harmed_winner"]].to_csv(harmed_winners_csv, index=False)

    scored = merged[merged["original_hit"].notna()].copy()
    lines = [
        "# Dominance Overlay Walkforward Summary",
        "",
        f"- rows_loaded=`{len(merged)}`",
        f"- rows_scored=`{len(scored)}`",
        f"- deploy_globs=`{' ; '.join(deploy_globs)}`",
        f"- actuals_csv=`{actuals_csv}`",
        f"- merged_root=`{merged_root}`",
        f"- rules_csv=`{rules_csv}`",
        "",
        "## Overlay Label Counts",
    ]
    for label, count in scored["overlay_label"].value_counts(dropna=False).items():
        lines.append(f"- `{label}`: `{int(count)}`")

    lines.extend(["", "## Core Metrics"])
    original_hit_rate = scored["original_hit"].mean() if len(scored) else float("nan")
    redirected = scored[scored["overlay_action"].eq("REDIRECT") & scored["redirected_hit"].notna()]
    redirected_hit_rate = redirected["redirected_hit"].mean() if len(redirected) else float("nan")
    lines.append(f"- original_hit_rate=`{original_hit_rate:.4f}`" if pd.notna(original_hit_rate) else "- original_hit_rate=`nan`")
    lines.append(f"- redirected_hit_rate=`{redirected_hit_rate:.4f}`" if pd.notna(redirected_hit_rate) else "- redirected_hit_rate=`nan`")
    lines.append(f"- saved_losers=`{int(scored['saved_loser'].sum())}`")
    lines.append(f"- harmed_winners=`{int(scored['harmed_winner'].sum())}`")

    lines.extend(["", "## By Original Market"])
    for market, group in scored.groupby("market", dropna=False):
        lines.append(
            f"- `{market}` | rows=`{len(group)}` | original_hit=`{group['original_hit'].mean():.4f}` | saved_losers=`{int(group['saved_loser'].sum())}` | harmed_winners=`{int(group['harmed_winner'].sum())}`"
        )

    lines.extend(["", "## Redirect Market Performance"])
    if len(redirected):
        for market, group in redirected.groupby("overlay_suggested_market", dropna=False):
            lines.append(f"- `{market}` | rows=`{len(group)}` | hit_rate=`{group['redirected_hit'].mean():.4f}`")
    else:
        lines.append("- No redirected rows scored yet.")

    summary_md.write_text("\n".join(lines) + "\n")

    return {
        "scored_csv": scored_csv,
        "summary_md": summary_md,
        "redirect_wins_csv": redirect_wins_csv,
        "saved_losers_csv": saved_losers_csv,
        "harmed_winners_csv": harmed_winners_csv,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a research-only dominance overlay walkforward audit from deploy files plus final-score actuals.")
    parser.add_argument(
        "--deploy-glob",
        action="append",
        required=True,
        help="Glob pattern for deploy files. Pass multiple times for ELITE/STANDARD/OBSERVE families.",
    )
    parser.add_argument("--actuals-csv", help="CSV containing match_date, home_team_name, away_team_name, home_goals, away_goals.")
    parser.add_argument("--merged-root", help="Optional canonical merged root to derive actual match scores from instead of an explicit actuals CSV.")
    parser.add_argument("--rules-csv", required=True, help="Machine-readable dominance overlay candidate rules CSV.")
    parser.add_argument("--outdir", required=True, help="Output directory for scored audit artifacts.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    outputs = build_audit(
        deploy_globs=args.deploy_glob,
        actuals_csv=args.actuals_csv,
        merged_root=args.merged_root,
        rules_csv=args.rules_csv,
        outdir=args.outdir,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")
