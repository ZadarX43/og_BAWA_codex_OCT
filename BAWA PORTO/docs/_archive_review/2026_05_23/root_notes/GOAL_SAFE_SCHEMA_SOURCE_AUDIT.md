# Goal Safe Schema Source Audit

## Source of GOAL_REGRESSOR_FEATURES
- File: `_baseline_ftr_pipeline.py`
- Symbol: `GOAL_REGRESSOR_FEATURES`

## Current Safe Feature List (count = 29)
See `GOAL_SAFE_SCHEMA_CURRENT_FEATURES.csv`.

## Legacy Manifest Feature Coverage (reference leagues)
            league_tag            league_name  n_features
England_Premier_League England Premier League          92
  Scotland_Premiership   Scotland Premiership          92
         Spain_La_Liga          Spain La Liga          91
               USA_MLS                USA MLS          76

## Candidate Pre‑Match Features (from merged files, not in safe list)
See `GOAL_SAFE_SCHEMA_CANDIDATE_EXPANSION.csv`.

## Notes
- Candidate list is heuristic: post‑match/leaky patterns excluded by regex.
- Next step is to review candidates and approve an expanded safe schema.