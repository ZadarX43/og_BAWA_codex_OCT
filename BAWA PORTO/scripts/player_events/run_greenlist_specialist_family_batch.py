from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAYER_EVENTS_DIR = REPO_ROOT / "scripts" / "player_events"
FEATURES_DIR = REPO_ROOT / "data_sources" / "api_football" / "features" / "player_events"
NORMALIZED_DIR = REPO_ROOT / "data_sources" / "api_football" / "normalized"
COMBINED_DIR = REPO_ROOT / "reports" / "player_events" / "combined_boards"
QUALITY_DIR = REPO_ROOT / "reports" / "player_events" / "quality_audits"

BATCHES = {
    "greenlist_batch1": [
        "Spain_La_Liga",
        "Italy_Serie_A",
        "Europa_League",
        "Champions_League",
    ],
    "greenlist_batch2": [
        "England_Premier_League",
        "Germany_Bundesliga",
        "France_Ligue_1",
        "Portugal_Liga",
    ],
    "greenlist_batch3": [
        "Netherlands_Eredivisie",
        "Belgium_Pro",
        "Scotland_Premiership",
        "Norway_Eliteserien",
    ],
    "greenlist_batch4": [
        "Japan_J1",
        "USA_MLS",
        "Brazil_Serie_A",
        "England_Championship",
    ],
    "greenlist_batch5": [
        "England_EFL_League_1",
        "Europa_Conference",
    ],
}

FAMILY_CONFIGS = {
    "4231v442": {
        "title": "4-2-3-1 vs 4-4-2 Weekend Board",
        "forms": "4-2-3-1 vs 4-4-2,4-4-2 vs 4-2-3-1",
        "super": "formation_4231_vs_442__super_elite_board.csv",
        "attacking": "formation_4231_vs_442__attacking_board.csv",
        "contact": "formation_4231_vs_442__contact_board.csv",
        "merged": "formation_4231_vs_442__final_weekend_merge_board.csv",
        "merged_md": "formation_4231_vs_442__final_weekend_merge_board.md",
    },
    "4231v433": {
        "title": "4-2-3-1 vs 4-3-3 Weekend Board",
        "forms": "4-2-3-1 vs 4-3-3,4-3-3 vs 4-2-3-1",
        "super": "formation_4231_vs_433__super_elite_board.csv",
        "attacking": "formation_4231_vs_433__attacking_board.csv",
        "contact": "formation_4231_vs_433__contact_board.csv",
        "merged": "formation_4231_vs_433__final_weekend_merge_board.csv",
        "merged_md": "formation_4231_vs_433__final_weekend_merge_board.md",
    },
    "3421v4231": {
        "title": "3-4-2-1 vs 4-2-3-1 Weekend Board",
        "forms": "3-4-2-1 vs 4-2-3-1,4-2-3-1 vs 3-4-2-1",
        "super": "formation_3421_vs_4231__super_elite_board.csv",
        "attacking": "formation_3421_vs_4231__attacking_board.csv",
        "contact": "formation_3421_vs_4231__contact_board.csv",
        "merged": "formation_3421_vs_4231__final_weekend_merge_board.csv",
        "merged_md": "formation_3421_vs_4231__final_weekend_merge_board.md",
    },
}


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _markdown_table(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return [header, sep] + rows


def _maybe_run(script_name: str, league_tag: str, season: int = 2024) -> bool:
    script = PLAYER_EVENTS_DIR / script_name
    try:
        _run([sys.executable, str(script), "--league-tag", league_tag, "--season", str(season)])
        return True
    except subprocess.CalledProcessError:
        return False


def _merge_inputs(league_tags: list[str], batch_tag: str) -> Path:
    frames = []
    for league in league_tags:
        path = FEATURES_DIR / f"player_events_fixture_input__{league}__2024.csv"
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output = FEATURES_DIR / f"player_events_fixture_input__{batch_tag.upper()}__2024.csv"
    out.to_csv(output, index=False)
    return output


def _batch_family_name(batch_name: str, stem: str) -> str:
    return f"{batch_name}__{stem}"


def _build_batch_super(attacking_csv: Path, contact_csv: Path, output_csv: Path, output_md: Path) -> None:
    frames = []
    if attacking_csv.exists():
        att = pd.read_csv(attacking_csv, low_memory=False)
        if not att.empty:
            att = att[
                pd.to_numeric(att["fixture_quality_score"], errors="coerce").fillna(0.0).ge(0.79)
                & pd.to_numeric(att["market_score"], errors="coerce").fillna(0.0).ge(100.0)
                & pd.to_numeric(att["starting_xi_quality_edge"], errors="coerce").fillna(0.0).ge(5.0)
            ].copy()
            att["super_group"] = "ATTACK"
            frames.append(att)
    if contact_csv.exists():
        con = pd.read_csv(contact_csv, low_memory=False)
        if not con.empty:
            con = con[
                pd.to_numeric(con["fixture_quality_score"], errors="coerce").fillna(0.0).ge(0.78)
                & pd.to_numeric(con["market_score"], errors="coerce").fillna(0.0).ge(95.0)
                & pd.to_numeric(con["starting_xi_quality_edge"], errors="coerce").fillna(0.0).le(-5.0)
            ].copy()
            con["super_group"] = "CONTACT"
            frames.append(con)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out = (
            out.sort_values(["fixture_key", "market_score"], ascending=[True, False])
            .groupby(["fixture_key", "super_group"], group_keys=False)
            .head(2)
            .reset_index(drop=True)
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    lines = ["# Batch Super-Elite Board", ""]
    for _, row in out.iterrows():
        lines.append(
            f"- {row['fixture_key']}: {row['player_name']} ({row['team_name']}) | {row['market']} | score={row['market_score']:.1f} | edge={row['starting_xi_quality_edge']:.1f}"
        )
    output_md.write_text("\n".join(lines) + "\n")


def run_batch(batch_name: str) -> Path:
    leagues = BATCHES[batch_name]
    status_rows: list[dict] = []
    for league in leagues:
        status = {"league_tag": league}
        status["referee_profiles"] = _maybe_run("build_referee_profiles.py", league)
        status["style_overlay"] = _maybe_run("build_fixture_style_overlay.py", league)
        status["og_overlay"] = _maybe_run("build_og_goal_environment_overlay.py", league)
        status["quality_overlay"] = _maybe_run("build_player_form_quality_overlay.py", league)
        try:
            _run([sys.executable, str(PLAYER_EVENTS_DIR / "build_player_events_fixture_input.py"), "--league-tag", league, "--season", "2024"])
            status["fixture_input"] = True
        except subprocess.CalledProcessError:
            status["fixture_input"] = False
        status_rows.append(status)

    successful_leagues = [row["league_tag"] for row in status_rows if row.get("fixture_input")]
    if not successful_leagues:
        raise SystemExit(f"No successful fixture-input leagues for batch {batch_name}.")
    merged_input = _merge_inputs(successful_leagues, batch_name)

    batch_merged_paths: list[str] = []
    for family_key, cfg in FAMILY_CONFIGS.items():
        contact_name = _batch_family_name(batch_name, cfg["contact"])
        attacking_name = _batch_family_name(batch_name, cfg["attacking"])
        super_name = _batch_family_name(batch_name, cfg["super"])
        merged_name = _batch_family_name(batch_name, cfg["merged"])
        merged_md_name = _batch_family_name(batch_name, cfg["merged_md"])

        _run([
            sys.executable,
            str(PLAYER_EVENTS_DIR / "build_4231_vs_442_weekend_board.py"),
            "--input",
            str(merged_input),
            "--output-csv",
            str(COMBINED_DIR / contact_name),
            "--output-md",
            str(COMBINED_DIR / contact_name.replace(".csv", ".md")),
            "--target-formations",
            cfg["forms"],
            "--title",
            cfg["title"],
            "--include-contact",
        ])
        _run([
            sys.executable,
            str(PLAYER_EVENTS_DIR / "build_4231_vs_442_weekend_board.py"),
            "--input",
            str(merged_input),
            "--output-csv",
            str(COMBINED_DIR / attacking_name),
            "--output-md",
            str(COMBINED_DIR / attacking_name.replace(".csv", ".md")),
            "--target-formations",
            cfg["forms"],
            "--title",
            cfg["title"],
            "--include-attacking",
        ])
        _build_batch_super(
            attacking_csv=COMBINED_DIR / attacking_name,
            contact_csv=COMBINED_DIR / contact_name,
            output_csv=COMBINED_DIR / super_name,
            output_md=COMBINED_DIR / super_name.replace(".csv", ".md"),
        )
        _run([
            sys.executable,
            str(PLAYER_EVENTS_DIR / "build_4231_vs_442_final_weekend_merge_board.py"),
            "--super-csv",
            str(COMBINED_DIR / super_name),
            "--attacking-csv",
            str(COMBINED_DIR / attacking_name),
            "--contact-csv",
            str(COMBINED_DIR / contact_name),
            "--output-csv",
            str(COMBINED_DIR / merged_name),
            "--output-md",
            str(COMBINED_DIR / merged_md_name),
            "--title",
            cfg["title"].replace("Weekend Board", "Final Weekend Merge Board"),
        ])
        batch_merged_paths.append(str(COMBINED_DIR / merged_name))

    merged_inputs = ",".join(batch_merged_paths)
    family_tags = ",".join(FAMILY_CONFIGS.keys())
    _run([
        sys.executable,
        str(PLAYER_EVENTS_DIR / "build_master_specialist_board.py"),
        "--inputs",
        merged_inputs,
        "--family-tags",
        family_tags,
        "--output-csv",
        str(COMBINED_DIR / f"{batch_name}__master_specialist_board.csv"),
        "--output-md",
        str(COMBINED_DIR / f"{batch_name}__master_specialist_board.md"),
    ])
    _run([
        sys.executable,
        str(PLAYER_EVENTS_DIR / "build_specialist_final_shortlist_export.py"),
        "--inputs",
        str(COMBINED_DIR / f"{batch_name}__master_specialist_board.csv"),
        "--output-csv",
        str(COMBINED_DIR / f"{batch_name}__specialist_final_shortlist_export.csv"),
        "--output-md",
        str(COMBINED_DIR / f"{batch_name}__specialist_final_shortlist_export.md"),
    ])
    _run([
        sys.executable,
        str(PLAYER_EVENTS_DIR / "build_master_weekend_specialist_sheet.py"),
        "--input-csv",
        str(COMBINED_DIR / f"{batch_name}__master_specialist_board.csv"),
        "--output-csv",
        str(COMBINED_DIR / f"{batch_name}__master_weekend_specialist_sheet.csv"),
        "--output-md",
        str(COMBINED_DIR / f"{batch_name}__master_weekend_specialist_sheet.md"),
        "--max-fixtures",
        "16",
    ])

    fixtures_frames = []
    stats_frames = []
    for league in successful_leagues:
        fx = NORMALIZED_DIR / f"fixtures_master__{league}__2024.csv"
        st = NORMALIZED_DIR / f"match_player_stats__{league}__2024.csv"
        if fx.exists():
            fixtures_frames.append(pd.read_csv(fx, low_memory=False))
        if st.exists():
            stats_frames.append(pd.read_csv(st, low_memory=False))
    merged_fx = pd.concat(fixtures_frames, ignore_index=True) if fixtures_frames else pd.DataFrame()
    merged_st = pd.concat(stats_frames, ignore_index=True) if stats_frames else pd.DataFrame()
    merged_fx_path = NORMALIZED_DIR / f"fixtures_master__{batch_name.upper()}__2024.csv"
    merged_st_path = NORMALIZED_DIR / f"match_player_stats__{batch_name.upper()}__2024.csv"
    merged_fx.to_csv(merged_fx_path, index=False)
    merged_st.to_csv(merged_st_path, index=False)

    _run([
        sys.executable,
        str(PLAYER_EVENTS_DIR / "audit_specialist_family_backtest.py"),
        "--inputs",
        merged_inputs,
        "--family-tags",
        family_tags,
        "--fixtures-csv",
        str(merged_fx_path),
        "--stats-csv",
        str(merged_st_path),
        "--output-prefix",
        str(QUALITY_DIR / f"{batch_name}__specialist_family_backtest"),
    ])
    family_summary_df = pd.read_csv(
        QUALITY_DIR / f"{batch_name}__specialist_family_backtest__family_summary.csv",
        low_memory=False,
    )
    family_market_df = pd.read_csv(
        QUALITY_DIR / f"{batch_name}__specialist_family_backtest__family_market_summary.csv",
        low_memory=False,
    )

    summary_path = COMBINED_DIR / f"{batch_name}__greenlist_batch_summary.md"
    master = pd.read_csv(COMBINED_DIR / f"{batch_name}__master_specialist_board.csv", low_memory=False)
    shortlist = pd.read_csv(COMBINED_DIR / f"{batch_name}__specialist_final_shortlist_export.csv", low_memory=False)
    status_df = pd.DataFrame(status_rows)
    status_df.to_csv(COMBINED_DIR / f"{batch_name}__greenlist_batch_status.csv", index=False)

    lines = [
        f"# {batch_name} Greenlist Specialist Batch Summary",
        "",
        f"- attempted leagues: {', '.join(leagues)}",
        f"- successful fixture-input leagues: {', '.join(successful_leagues)}",
        f"- master specialist rows: {len(master)} | fixtures: {master['fixture_key'].nunique() if not master.empty else 0}",
        f"- final shortlist rows: {len(shortlist)} | fixtures: {shortlist['fixture_key'].nunique() if not shortlist.empty else 0}",
        "",
        "## League Status",
    ]
    lines.extend(_markdown_table(status_df))
    lines.append("")
    lines.append("## Family Hit Rates")
    if family_summary_df.empty:
        lines.append("No family rows matched.")
    else:
        family_summary_print = family_summary_df.copy()
        for col in ["row_hit_rate", "fixture_quality_avg", "market_score_avg"]:
            if col in family_summary_print.columns:
                family_summary_print[col] = pd.to_numeric(family_summary_print[col], errors="coerce").round(3)
        lines.extend(_markdown_table(family_summary_print))
    lines.append("")
    lines.append("## Family x Market Hit Rates")
    if family_market_df.empty:
        lines.append("No family x market rows matched.")
    else:
        family_market_print = family_market_df.copy()
        for col in ["hit_rate", "avg_score", "avg_fixture_quality"]:
            if col in family_market_print.columns:
                family_market_print[col] = pd.to_numeric(family_market_print[col], errors="coerce").round(3)
        lines.extend(_markdown_table(family_market_print))
    lines.append("")
    summary_path.write_text("\n".join(lines) + "\n")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the specialist-family workflow across greenlist leagues in controlled batches.")
    parser.add_argument("--batch-name", default="greenlist_batch1", choices=sorted(BATCHES.keys()))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_batch(args.batch_name)
    print(f"WROTE: {summary}")
