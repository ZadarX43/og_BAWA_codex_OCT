# PIPELINE_RUNBOOK

## Purpose
Canonical operator runbook for refreshing the data stack safely before predictions or retraining.

## Mandatory Order
1. `footystats_drop_ingest.py`
2. `etl_press_intensity.py` per eligible league folder
3. `build_merged.py --all --recursive --rolling-press`
4. `patch_merge_add_streaks.py`
5. `team_ratings.py --league "<League Name>" --mode rolling` per merged league
6. `patch_merge_add_power_ratings.py`
7. `make_fd_odds_enriched_synth.py --emit-ou25-novig`
8. `patch_merge_add_synth_odds.py --root "<repo root>" --overwrite --harmonize-duplicates`
9. `pipeline_qa_gate.py` or integrity spot-check

## Hard Stop
If integrity fails, do not run `bookie_allmarkets.py`.

## Current CLI Notes
- `footystats_drop_ingest.py` has no safe `--help` mode. Running it ingests from `/Users/hughwade/Desktop/FOOTYSTATS_DROP`.
- `etl_press_intensity.py` requires `--match-dir`, `--player-dir`, and `--out`. Run it in a league loop, not as a bare command.
- `build_merged.py` requires a scope flag such as `--all`; `--recursive --rolling-press` alone is invalid.
- `patch_merge_add_synth_odds.py` requires `--root`. Use `--leagues` when you want to avoid old/support folders without synth files.

## Source Candidates
- `DATA_UPDATE_CHECKLIST.md`
- `DATA_UPDATE_CHECKLIST_RUNBOOK.md`
- `MERGED_REBUILD_PIPELINE_README.md`
- `FootyStats_Ingest_README.md`
