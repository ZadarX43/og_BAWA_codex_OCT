# SYSTEM_MAP

## Purpose
Canonical high-level map of the Odds Genius / BAWA PORTO system.

## Core Spine
1. Ingest: `footystats_drop_ingest.py`
2. Enrichment: `etl_press_intensity.py`
3. Canonical merged build: `build_merged.py`
4. Post-merge patches: streaks, power ratings, synth odds
5. Integrity gate: `pipeline_qa_gate.py`
6. Prediction board: `bookie_allmarkets.py`
7. Deploy routing: `deploy_rulebook.py`
8. Slip/product formatting: `slip_formatter.py`

## Canonical Data Contract
- Training input: `Matches/__merged__/<LEAGUE_TAG>__merged.csv`
- Deploy source: latest validated `BOOKIE_IMP*_ALLMARKETS*.csv`
- OBSERVE is non-deployable

## Protected Rules
- Never train from raw season CSVs
- Never predict before integrity passes
- `deploy_rulebook.py` owns live gates
- `slip_formatter.py` remains thin

## Source Candidates
- `OG Stack Map.md`
- `OG : BAWA Runbook (One-Page).md`
- `MERGED_REBUILD_PIPELINE_README.md`
- `deploy_weekend_runner.md`
