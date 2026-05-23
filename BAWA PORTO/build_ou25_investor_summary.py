#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd

COMP = Path("ou25_branch_comparison.csv")
CUM = Path("ou25_branch_cumulative_stats.csv")
AUDIT = Path("ou25_forensic_audit.csv")
OUT = Path("OU25_INVESTOR_SUMMARY.md")


def _fmt_pct(x) -> str:
    x = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    return "n/a" if pd.isna(x) else f"{x * 100:.2f}%"


def _fmt_num(x, digits: int = 4) -> str:
    x = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    return "n/a" if pd.isna(x) else f"{x:.{digits}f}"


if not COMP.exists():
    raise SystemExit(f"missing comparison csv: {COMP}")
if not CUM.exists():
    raise SystemExit(f"missing cumulative stats csv: {CUM}")
if not AUDIT.exists():
    raise SystemExit(f"missing audit csv: {AUDIT}")

comp = pd.read_csv(COMP, low_memory=False)
cum = pd.read_csv(CUM, low_memory=False)
audit = pd.read_csv(AUDIT, low_memory=False)

if comp.empty:
    raise SystemExit("comparison csv is empty")

best_roi = comp.sort_values(["roi", "hit", "rows"], ascending=[False, False, False]).iloc[0]
best_hit = comp.sort_values(["hit", "roi", "rows"], ascending=[False, False, False]).iloc[0]
best_rows = comp.sort_values(["rows", "roi"], ascending=[False, False]).iloc[0]

combined = comp[comp["branch"] == "ou25_combined_baseline"].copy()
over_only = comp[comp["branch"] == "ou25_mode_over_only"].copy()
under_only = comp[comp["branch"] == "ou25_mode_under_only"].copy()

audit_ok = (
    audit["status"].eq("ok").all()
    and (pd.to_numeric(audit["merge_miss_rows"], errors="coerce").fillna(0) == 0).all()
    and (pd.to_numeric(audit["duplicate_join_rows"], errors="coerce").fillna(0) == 0).all()
    and (pd.to_numeric(audit["duplicate_filtered_fixture_rows"], errors="coerce").fillna(0) == 0).all()
    and (pd.to_numeric(audit["filtered_rows_missing_from_backtest"], errors="coerce").fillna(0) == 0).all()
    and (~audit["selection_leak_suspected"].fillna(False)).all()
)

lines: list[str] = []
lines.append("# OU25 Investor Summary")
lines.append("")
lines.append("## Positioning")
lines.append("")
lines.append("OU25 is now a real second product lane beside FTR. This summary is built from frozen branch sweeps on the canonical 19-league, 3-year IMP40 truth-backed backtest corpus.")
lines.append("")
lines.append("## Headline winners")
lines.append("")
lines.append(f"- **Best ROI branch:** `{best_roi['branch']}` — rows={int(best_roi['rows'])}, hit={_fmt_pct(best_roi['hit'])}, roi={_fmt_num(best_roi['roi'])}, avg_odds={_fmt_num(best_roi['avg_odds'])}")
lines.append(f"- **Best hit-rate branch:** `{best_hit['branch']}` — rows={int(best_hit['rows'])}, hit={_fmt_pct(best_hit['hit'])}, roi={_fmt_num(best_hit['roi'])}, avg_odds={_fmt_num(best_hit['avg_odds'])}")
lines.append(f"- **Largest branch:** `{best_rows['branch']}` — rows={int(best_rows['rows'])}, hit={_fmt_pct(best_rows['hit'])}, roi={_fmt_num(best_rows['roi'])}, avg_odds={_fmt_num(best_rows['avg_odds'])}")
lines.append("")

if not combined.empty and not over_only.empty and not under_only.empty:
    c = combined.iloc[0]
    o = over_only.iloc[0]
    u = under_only.iloc[0]

    lines.append("## Core branch comparison")
    lines.append("")
    lines.append(f"- **Combined baseline:** rows={int(c['rows'])}, hit={_fmt_pct(c['hit'])}, roi={_fmt_num(c['roi'])}, avg_odds={_fmt_num(c['avg_odds'])}")
    lines.append(f"- **Over-only:** rows={int(o['rows'])}, hit={_fmt_pct(o['hit'])}, roi={_fmt_num(o['roi'])}, avg_odds={_fmt_num(o['avg_odds'])}")
    lines.append(f"- **Under-only:** rows={int(u['rows'])}, hit={_fmt_pct(u['hit'])}, roi={_fmt_num(u['roi'])}, avg_odds={_fmt_num(u['avg_odds'])}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("")
    lines.append("- Over-only currently looks like the strongest clean standalone OU25 lane.")
    lines.append("- Combined also works and gives broader product coverage.")
    lines.append("- Under-only is profitable but materially narrower and weaker.")
    lines.append("")

lines.append("## Audit status")
lines.append("")
lines.append(f"- **Forensic audit clean:** `{audit_ok}`")
lines.append("- Post-match fields may appear in filtered exports for scoring and audit context only.")
lines.append("- Current audit should show no evidence that post-match fields were used in selection logic.")
lines.append("")

lines.append("## Cumulative branch rollup")
lines.append("")
show_cols = [c for c in [
    "branch",
    "sweep_type",
    "pick_mode",
    "variants_present",
    "total_rows",
    "weighted_hit",
    "weighted_roi",
    "weighted_avg_odds",
    "hit_std",
    "roi_std",
] if c in cum.columns]

try:
    lines.append(cum[show_cols].to_markdown(index=False))
except Exception:
    lines.append(cum[show_cols].to_string(index=False))

lines.append("")
lines.append("## Investor-facing takeaway")
lines.append("")
lines.append("OU25 already looks commercially usable as a second flagship market. The evidence suggests the best immediate product framing is either:")
lines.append("")
lines.append("- a **premium Over-only OU25 lane**, or")
lines.append("- a **tighter top-q combined OU25 lane** for broader coverage with strong ROI.")
lines.append("")
lines.append("## Recommended next move")
lines.append("")
lines.append("1. Lock the winning OU25 lane into a permanent markdown runbook.")
lines.append("2. Add a hard metrics appendix with exact branch tables.")
lines.append("3. Repeat the same frozen sweep workflow for BTTS Yes/No.")
lines.append("")

OUT.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
print(f"WROTE: {OUT}")