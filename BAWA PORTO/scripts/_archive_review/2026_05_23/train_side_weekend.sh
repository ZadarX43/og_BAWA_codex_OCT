#!/usr/bin/env bash
set -u

MARKETS="home_fts,away_fts,home_ge2,away_ge2,home_ge3,away_ge3,btts_fh"

LEAGUES=(
"England Championship"
"Spain La Liga"
"Italy Serie A"
"Germany Bundesliga"
"France Ligue 1"
"Portugal Liga"
"USA MLS"
)

ok=0
fail=0

for LEAGUE in "${LEAGUES[@]}"; do
  TAG="${LEAGUE// /_}"
  CSV="Matches/__merged__/${TAG}__merged.csv"
  LOG="logs/train_side_${TAG}.log"

  echo ""
  echo "=== $LEAGUE ==="
  if [ ! -f "$CSV" ]; then
    echo "❌ Missing merged CSV: $CSV"
    fail=$((fail+1))
    continue
  fi

  python train_markets.py \
    --league "$LEAGUE" \
    --matches-csv "$CSV" \
    --outdir ModelStore \
    --markets "$MARKETS" \
    2>&1 | tee "$LOG"

  if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ $LEAGUE"
    ok=$((ok+1))
  else
    echo "❌ $LEAGUE (see $LOG)"
    fail=$((fail+1))
  fi
done

echo ""
echo "DONE | ok=$ok fail=$fail"
