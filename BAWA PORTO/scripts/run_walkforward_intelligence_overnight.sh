#!/usr/bin/env bash
set -euo pipefail

# Research-only overnight wrapper.
# It can either reuse an existing scored walk-forward root or generate a fresh
# walk-forward root first. It never edits deploy gates or production routing.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_LABEL="${RUN_LABEL:-intelligence_overlay_$(date +%Y_%m_%d_%H%M)}"
MANIFEST="${MANIFEST:-walkforward_manifest_3y_thu_wed.csv}"
BENCHMARK_MODE="${BENCHMARK_MODE:-0}"
BENCHMARK_WF_ROOT="${BENCHMARK_WF_ROOT:-predictions_output/hybrid_shadow_walkforward_2026_05_01_parity_rebuild}"
if [[ "$BENCHMARK_MODE" == "1" ]]; then
  WF_ROOT="${WF_ROOT:-$BENCHMARK_WF_ROOT}"
else
  WF_ROOT="${WF_ROOT:-predictions_output/walk_forward_${RUN_LABEL}}"
fi
REPORT_ROOT="${REPORT_ROOT:-reports/latest/walkforward_intelligence_${RUN_LABEL}}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-predictions_output}"
MERGED_DIR="${MERGED_DIR:-Matches/__merged__}"
CALENDAR_FLAGS_DIR="${CALENDAR_FLAGS_DIR:-predictions_output/calendar_flags}"
RUN_FULL_WALKFORWARD="${RUN_FULL_WALKFORWARD:-0}"
ALLOW_LINEUP_SHADOW="${ALLOW_LINEUP_SHADOW:-0}"
SOURCE_IMPLIED_MIN="${SOURCE_IMPLIED_MIN:-20}"
LEAGUES="${LEAGUES:-}"
TG_REFERENCE_DIR="${TG_REFERENCE_DIR:-reports/2026-05-06/team_goal_shadow_market_backtest}"
FEATURES_DIR="${FEATURES_DIR:-data_sources/api_football/features}"
NORMALIZED_DIR="${NORMALIZED_DIR:-data_sources/api_football/normalized}"
CALENDAR_FEATURES_MODE="${CALENDAR_FEATURES_MODE:-0}"
CALENDAR_FEATURES_DIR="${CALENDAR_FEATURES_DIR:-data_sources/api_football/features_calendar_year}"
CALENDAR_NORMALIZED_DIR="${CALENDAR_NORMALIZED_DIR:-data_sources/api_football/normalized_calendar_year}"
TRUSTED_FIXTURE_IDENTITY_MODE="${TRUSTED_FIXTURE_IDENTITY_MODE:-0}"
TRUSTED_FIXTURE_ONLY="${TRUSTED_FIXTURE_ONLY:-0}"
FIXTURE_IDENTITY_MAP="${FIXTURE_IDENTITY_MAP:-}"
BENCHMARK_EXPECTED_SCORED_FILES="${BENCHMARK_EXPECTED_SCORED_FILES:-139}"
BENCHMARK_MIN_COMPETITIONS="${BENCHMARK_MIN_COMPETITIONS:-24}"

mkdir -p "$REPORT_ROOT"

echo "== Odds Genius intelligence walk-forward overnight run =="
echo "run label: $RUN_LABEL"
echo "walk-forward root: $WF_ROOT"
echo "report root: $REPORT_ROOT"
echo "full walk-forward generation: $RUN_FULL_WALKFORWARD"
echo "lineup shadow mode: $ALLOW_LINEUP_SHADOW"
echo "benchmark mode: $BENCHMARK_MODE"
echo "calendar features mode: $CALENDAR_FEATURES_MODE"
echo "trusted fixture identity mode: $TRUSTED_FIXTURE_IDENTITY_MODE"
echo "trusted fixture only: $TRUSTED_FIXTURE_ONLY"

if [[ "$BENCHMARK_MODE" == "1" && "$RUN_FULL_WALKFORWARD" == "1" ]]; then
  echo "ERROR: BENCHMARK_MODE=1 must reuse the benchmark scored root; set RUN_FULL_WALKFORWARD=0." >&2
  exit 2
fi

if [[ "$RUN_FULL_WALKFORWARD" == "1" ]]; then
  BOOKIE_ARGS="--markets ftr,btts,ou25 --ou25-implied-min 0.20 --btts-implied-min 0.20 --strict"
  if [[ -n "$LEAGUES" ]]; then
    BOOKIE_ARGS="--leagues \"$LEAGUES\" $BOOKIE_ARGS"
  fi

  python3 run_walkforward_windows.py \
    --manifest "$MANIFEST" \
    --base-outdir "$WF_ROOT" \
    --predictions-dir "$PREDICTIONS_DIR" \
    --merged-dir "$MERGED_DIR" \
    --calendar-flags-dir "$CALENDAR_FLAGS_DIR" \
    --source-implied-min "$SOURCE_IMPLIED_MIN" \
    --bookie-extra-args "$BOOKIE_ARGS" \
    --deploy-extra-args "--preset V1 --ftr-profile accuracy --ftr-priority-ordering" \
    --score-candidates \
    --write-window-file-manifest
fi

if [[ "$CALENDAR_FEATURES_MODE" == "1" ]]; then
  python3 scripts/build_api_football_calendar_year_bridge.py \
    --scored-root "$WF_ROOT" \
    --features-dir "$FEATURES_DIR" \
    --normalized-dir "$NORMALIZED_DIR" \
    --features-out-dir "$CALENDAR_FEATURES_DIR" \
    --normalized-out-dir "$CALENDAR_NORMALIZED_DIR" \
    --outdir "$REPORT_ROOT/00_calendar_year_bridge"
  FEATURES_DIR="$CALENDAR_FEATURES_DIR"
  NORMALIZED_DIR="$CALENDAR_NORMALIZED_DIR"
fi

if [[ "$TRUSTED_FIXTURE_IDENTITY_MODE" == "1" ]]; then
  if [[ -z "$FIXTURE_IDENTITY_MAP" ]]; then
    python3 scripts/build_fixture_identity_map.py \
      --scored-root "$WF_ROOT" \
      --api-fixtures-dir "$NORMALIZED_DIR" \
      --team-aliases config/team_identity_aliases.csv \
      --competition-scope config/competition_scope_map.csv \
      --outdir "$REPORT_ROOT/00_fixture_identity_map"
    FIXTURE_IDENTITY_MAP="$REPORT_ROOT/00_fixture_identity_map/fixture_identity_map.csv"
  fi
fi

python3 scripts/audit_walkforward_intelligence_estate.py \
  --scored-root "$WF_ROOT" \
  --features-dir "$FEATURES_DIR" \
  --normalized-dir "$NORMALIZED_DIR" \
  --outdir "$REPORT_ROOT/01_estate_audit"

OVERLAY_ARGS=(
  python3 scripts/build_walkforward_intelligence_overlay_backtest.py
  --scored-root "$WF_ROOT"
  --features-dir "$FEATURES_DIR"
  --outdir "$REPORT_ROOT/02_market_overlay_backtest"
)
if [[ "$ALLOW_LINEUP_SHADOW" == "1" ]]; then
  OVERLAY_ARGS+=(--allow-lineup-shadow)
fi
if [[ -n "$FIXTURE_IDENTITY_MAP" ]]; then
  OVERLAY_ARGS+=(--fixture-identity-map "$FIXTURE_IDENTITY_MAP")
fi
if [[ "$TRUSTED_FIXTURE_ONLY" == "1" ]]; then
  OVERLAY_ARGS+=(--trusted-fixture-only)
fi
"${OVERLAY_ARGS[@]}"

python3 scripts/build_team_goal_shadow_market_backtest.py \
  --scored-root "$WF_ROOT" \
  --outdir "$REPORT_ROOT/03_team_goal_15_shadow"

python3 scripts/build_team_goal_threshold_stability_report.py \
  --backtest-dir "$REPORT_ROOT/03_team_goal_15_shadow" \
  --outdir "$REPORT_ROOT/04_team_goal_threshold_stability"

python3 scripts/validate_benchmark_walkforward_estate.py \
  --scored-root "$WF_ROOT" \
  --features-dir "$FEATURES_DIR" \
  --overlay-dir "$REPORT_ROOT/02_market_overlay_backtest" \
  --tg-reference-dir "$TG_REFERENCE_DIR" \
  --outdir "$REPORT_ROOT/00_benchmark_guard" \
  --expected-scored-files "$BENCHMARK_EXPECTED_SCORED_FILES" \
  --min-competitions "$BENCHMARK_MIN_COMPETITIONS"

cat > "$REPORT_ROOT/RUN_COMPLETE.md" <<EOF
# Walk-Forward Intelligence Overnight Run Complete

- run label: \`$RUN_LABEL\`
- walk-forward root: \`$WF_ROOT\`
- calendar features mode: \`$CALENDAR_FEATURES_MODE\`
- trusted fixture identity mode: \`$TRUSTED_FIXTURE_IDENTITY_MODE\`
- trusted fixture only: \`$TRUSTED_FIXTURE_ONLY\`
- fixture identity map: \`${FIXTURE_IDENTITY_MAP:-not used}\`
- estate audit: \`$REPORT_ROOT/01_estate_audit/SUMMARY.md\`
- calendar-year bridge: \`$REPORT_ROOT/00_calendar_year_bridge/SUMMARY.md\` when enabled
- fixture identity report: \`$REPORT_ROOT/00_fixture_identity_map/SUMMARY.md\` when enabled
- benchmark guard: \`$REPORT_ROOT/00_benchmark_guard/SUMMARY.md\`
- market overlay: \`$REPORT_ROOT/02_market_overlay_backtest/SUMMARY.md\`
- TG1.5 shadow: \`$REPORT_ROOT/03_team_goal_15_shadow/team_goal_shadow_market_backtest_summary.md\`
- TG threshold stability: \`$REPORT_ROOT/04_team_goal_threshold_stability/team_goal_threshold_stability_report.md\`

Research-only. No deploy gates changed.
EOF

echo "Complete: $REPORT_ROOT/RUN_COMPLETE.md"
