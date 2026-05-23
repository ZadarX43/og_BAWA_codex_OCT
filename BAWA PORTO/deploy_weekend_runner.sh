#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# OG Weekend Deploy Runner
# ------------------------------------------------------------
# Chains:
#   1) bookie_allmarkets.py
#   2) deploy_rulebook.py --debug | tee log
#   3) tier_audit.py
#   4) summary print (tier counts + promoted rows)
#
# Usage (example):
#   bash deploy_weekend_runner.sh \
#     --date-from 2026-02-20 \
#     --date-to 2026-02-23 \
#     --run-date 2026-02-20
#
# Optional:
#   --out-root predictions_output
#   --bookie-script bookie_allmarkets.py
#   --rulebook-script deploy_rulebook.py
#   --tier-audit-script tier_audit.py
#   --skip-bookie
#   --src-csv <existing allmarkets csv>
#   --extra-bookie-args "--foo 1 --bar 2"
#   --extra-rulebook-args "--preset V1"
# ============================================================

DATE_FROM=""
DATE_TO=""
RUN_DATE=""
OUT_ROOT="predictions_output"

BOOKIE_SCRIPT="bookie_allmarkets.py"
RULEBOOK_SCRIPT="deploy_rulebook.py"
TIER_AUDIT_SCRIPT="tier_audit.py"

SKIP_BOOKIE=0
SRC_CSV=""
EXTRA_BOOKIE_ARGS=""
EXTRA_RULEBOOK_ARGS=""

usage() {
  cat <<'EOF'
Usage:
  bash deploy_weekend_runner.sh --date-from YYYY-MM-DD --date-to YYYY-MM-DD --run-date YYYY-MM-DD [options]

Required:
  --date-from YYYY-MM-DD   Fixture window start for ALLMARKETS build
  --date-to YYYY-MM-DD     Fixture window end for ALLMARKETS build
  --run-date YYYY-MM-DD    Output folder date (e.g., 2026-02-20)

Options:
  --out-root PATH          Root output folder (default: predictions_output)
  --bookie-script PATH     bookie_allmarkets script path (default: bookie_allmarkets.py)
  --rulebook-script PATH   deploy_rulebook script path (default: deploy_rulebook.py)
  --tier-audit-script PATH tier_audit script path (default: tier_audit.py)

  --skip-bookie            Skip step 1 (use existing ALLMARKETS CSV)
  --src-csv PATH           Existing ALLMARKETS CSV (required if --skip-bookie)

  --extra-bookie-args STR  Extra args string passed to bookie_allmarkets.py
  --extra-rulebook-args STR Extra args string passed to deploy_rulebook.py

Examples:
  bash deploy_weekend_runner.sh \
    --date-from 2026-02-20 \
    --date-to 2026-02-23 \
    --run-date 2026-02-20

  bash deploy_weekend_runner.sh \
    --skip-bookie \
    --src-csv "predictions_output/2026-02-20/BOOKIE_IMP62_ALLMARKETS_2026-02-20_to_2026-02-23.csv" \
    --date-from 2026-02-20 \
    --date-to 2026-02-23 \
    --run-date 2026-02-20
EOF
}

# -------------------------
# Parse args
# -------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --date-from)
      DATE_FROM="${2:-}"; shift 2 ;;
    --date-to)
      DATE_TO="${2:-}"; shift 2 ;;
    --run-date)
      RUN_DATE="${2:-}"; shift 2 ;;
    --out-root)
      OUT_ROOT="${2:-}"; shift 2 ;;
    --bookie-script)
      BOOKIE_SCRIPT="${2:-}"; shift 2 ;;
    --rulebook-script)
      RULEBOOK_SCRIPT="${2:-}"; shift 2 ;;
    --tier-audit-script)
      TIER_AUDIT_SCRIPT="${2:-}"; shift 2 ;;
    --skip-bookie)
      SKIP_BOOKIE=1; shift ;;
    --src-csv)
      SRC_CSV="${2:-}"; shift 2 ;;
    --extra-bookie-args)
      EXTRA_BOOKIE_ARGS="${2:-}"; shift 2 ;;
    --extra-rulebook-args)
      EXTRA_RULEBOOK_ARGS="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "❌ Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

# -------------------------
# Validation
# -------------------------
if [[ -z "$DATE_FROM" || -z "$DATE_TO" || -z "$RUN_DATE" ]]; then
  echo "❌ Missing required args." >&2
  usage
  exit 2
fi

if [[ "$SKIP_BOOKIE" -eq 1 && -z "$SRC_CSV" ]]; then
  echo "❌ --skip-bookie requires --src-csv" >&2
  exit 2
fi

if [[ ! -f "$RULEBOOK_SCRIPT" ]]; then
  echo "❌ Rulebook script not found: $RULEBOOK_SCRIPT" >&2
  exit 2
fi

if [[ ! -f "$TIER_AUDIT_SCRIPT" ]]; then
  echo "❌ Tier audit script not found: $TIER_AUDIT_SCRIPT" >&2
  exit 2
fi

if [[ "$SKIP_BOOKIE" -eq 0 && ! -f "$BOOKIE_SCRIPT" ]]; then
  echo "❌ Bookie script not found: $BOOKIE_SCRIPT" >&2
  exit 2
fi

# -------------------------
# Paths
# -------------------------
RUN_OUTDIR="${OUT_ROOT}/${RUN_DATE}"
mkdir -p "$RUN_OUTDIR"

TS="$(date +%Y%m%d_%H%M%S)"
RULEBOOK_LOG="${RUN_OUTDIR}/deploy_rulebook_debug_${DATE_FROM}_to_${DATE_TO}_${TS}.log"
TIER_AUDIT_OUTDIR="${RUN_OUTDIR}/tier_audit_${DATE_FROM}_to_${DATE_TO}_${TS}"

echo "============================================================"
echo "OG Weekend Deploy Runner"
echo "============================================================"
echo "Date window   : ${DATE_FROM} -> ${DATE_TO}"
echo "Run date      : ${RUN_DATE}"
echo "Output dir    : ${RUN_OUTDIR}"
echo "Timestamp     : ${TS}"
echo "Skip bookie   : ${SKIP_BOOKIE}"
echo "============================================================"
echo

# -------------------------
# Step 1) bookie_allmarkets.py (optional)
# -------------------------
if [[ "$SKIP_BOOKIE" -eq 0 ]]; then
  echo "▶ Step 1/4: Building ALLMARKETS via ${BOOKIE_SCRIPT}"
  echo

  # shellcheck disable=SC2086
  python3 "$BOOKIE_SCRIPT" \
    --date-from "$DATE_FROM" \
    --date-to "$DATE_TO" \
    --outdir "$RUN_OUTDIR" \
    ${EXTRA_BOOKIE_ARGS}

  echo
  echo "✅ bookie_allmarkets.py completed"
  echo
fi

# -------------------------
# Discover ALLMARKETS src
# -------------------------
if [[ -z "$SRC_CSV" ]]; then
  # Prefer exact date-range match in run dir
  mapfile -t CANDIDATES < <(ls -1t "${RUN_OUTDIR}"/BOOKIE_IMP*_ALLMARKETS_"${DATE_FROM}"_to_"${DATE_TO}".csv 2>/dev/null || true)

  # Fallback: any ALLMARKETS in run dir
  if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
    mapfile -t CANDIDATES < <(ls -1t "${RUN_OUTDIR}"/BOOKIE_IMP*_ALLMARKETS_*.csv 2>/dev/null || true)
  fi

  if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
    echo "❌ Could not find ALLMARKETS CSV in ${RUN_OUTDIR}" >&2
    exit 2
  fi

  SRC_CSV="${CANDIDATES[0]}"
fi

if [[ ! -f "$SRC_CSV" ]]; then
  echo "❌ ALLMARKETS source CSV not found: $SRC_CSV" >&2
  exit 2
fi

echo "Using ALLMARKETS source:"
echo "  ${SRC_CSV}"
echo

# -------------------------
# Step 2) deploy_rulebook.py --debug
# -------------------------
echo "▶ Step 2/4: Running deploy_rulebook.py --debug"
echo "  Log: ${RULEBOOK_LOG}"
echo

# shellcheck disable=SC2086
python3 "$RULEBOOK_SCRIPT" \
  --src "$SRC_CSV" \
  --outdir "$RUN_OUTDIR" \
  --debug \
  ${EXTRA_RULEBOOK_ARGS} | tee "$RULEBOOK_LOG"

echo
echo "✅ deploy_rulebook.py completed"
echo

# -------------------------
# Locate tier outputs
# -------------------------
find_one_tier_file() {
  local pattern="$1"
  local label="$2"
  local file
  file="$(ls -1t ${pattern} 2>/dev/null | head -n 1 || true)"
  if [[ -z "$file" ]]; then
    echo "❌ Could not find ${label} tier file with pattern: ${pattern}" >&2
    exit 2
  fi
  echo "$file"
}

ELITE_CSV="$(find_one_tier_file "${RUN_OUTDIR}/*__DEPLOY_TIER_ELITE.csv" "ELITE")"
STANDARD_CSV="$(find_one_tier_file "${RUN_OUTDIR}/*__DEPLOY_TIER_STANDARD.csv" "STANDARD")"
OBSERVE_CSV="$(find_one_tier_file "${RUN_OUTDIR}/*__DEPLOY_TIER_OBSERVE.csv" "OBSERVE")"

echo "Resolved tier files:"
echo "  ELITE   : ${ELITE_CSV}"
echo "  STANDARD: ${STANDARD_CSV}"
echo "  OBSERVE : ${OBSERVE_CSV}"
echo

# -------------------------
# Step 3) tier_audit.py
# -------------------------
echo "▶ Step 3/4: Running tier_audit.py"
echo "  Audit outdir: ${TIER_AUDIT_OUTDIR}"
echo

python3 "$TIER_AUDIT_SCRIPT" \
  --elite "$ELITE_CSV" \
  --standard "$STANDARD_CSV" \
  --observe "$OBSERVE_CSV" \
  --outdir "$TIER_AUDIT_OUTDIR"

echo
echo "✅ tier_audit.py completed"
echo

# -------------------------
# Step 4) Small summary print
# -------------------------
echo "▶ Step 4/4: Summary print (tier counts + promoted rows)"
echo

python3 - "$ELITE_CSV" "$STANDARD_CSV" "$OBSERVE_CSV" <<'PY'
import sys
import pandas as pd

elite_csv, std_csv, obs_csv = sys.argv[1:4]

elite = pd.read_csv(elite_csv)
std = pd.read_csv(std_csv)
obs = pd.read_csv(obs_csv)

def vc_market(df):
    if "market" not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df["market"].astype("string").fillna("").value_counts().sort_index().items()}

print("Tier counts:")
print(f"  ELITE   = {len(elite)}")
print(f"  STANDARD= {len(std)}")
print(f"  OBSERVE = {len(obs)}")
print(f"  TOTAL   = {len(elite)+len(std)+len(obs)}")
print()

print("Market counts by tier:")
for name, df in [("ELITE", elite), ("STANDARD", std), ("OBSERVE", obs)]:
    print(f"  {name}: {vc_market(df)}")
print()

if "context_reason_codes" in std.columns:
    s = std["context_reason_codes"].astype("string").fillna("")
    promo_mask = s.str.contains("DEMOTED_ULTRASHORT_STANDARD", regex=False) | s.str.contains("DEMOTED_MARGIN_STANDARD", regex=False)
    promoted = std.loc[promo_mask].copy()
else:
    promoted = std.iloc[0:0].copy()

print(f"Promoted demoted rows in STANDARD: {len(promoted)}")

if not promoted.empty:
    cols_pref = [
        "league", "home_team_name", "away_team_name", "market", "bookie_pick",
        "bookie_od", "ftr_margin", "draw_chaos_score", "context_reason_codes"
    ]
    cols = [c for c in cols_pref if c in promoted.columns]
    print()
    print(promoted[cols].to_string(index=False))
PY

echo
echo "✅ Summary complete"
echo

echo "============================================================"
echo "DONE"
echo "------------------------------------------------------------"
echo "ALLMARKETS src : ${SRC_CSV}"
echo "Rulebook log   : ${RULEBOOK_LOG}"
echo "Tier audit dir : ${TIER_AUDIT_OUTDIR}"
echo "============================================================"
