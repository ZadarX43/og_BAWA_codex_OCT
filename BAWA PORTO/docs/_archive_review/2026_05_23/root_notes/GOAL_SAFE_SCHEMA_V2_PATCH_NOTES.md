# Goal Safe Schema v2 Patch Notes

## Summary
Expanded `GOAL_REGRESSOR_FEATURES` with a controlled, pre‑match‑only v2 set.
No snapshot fields, identifiers, or text fields were added.

## Additions
Match context:
- Game Week

Team quality / state:
- home_ppg
- away_ppg

Press baseline / intensity:
- home_press_baseline
- away_press_baseline
- home_press_intensity
- away_press_intensity
- pre_match_press_intensity_home
- pre_match_press_intensity_away

Rolling goal environment:
- rolling5_home_gc
- rolling5_away_gc

Team xG form:
- xg_for_avg_5_home
- xg_for_avg_5_away
- xg_for_avg_10_home
- xg_for_avg_10_away
- xg_against_avg_5_home
- xg_against_avg_5_away
- xg_against_avg_10_home
- xg_against_avg_10_away

## Files
- `_baseline_ftr_pipeline.py`
- `GOAL_SAFE_SCHEMA_V2_FEATURES.csv`
