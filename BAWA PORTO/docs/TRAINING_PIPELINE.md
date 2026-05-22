# TRAINING_PIPELINE

## Purpose
Canonical training guidance for market models and artifact validation.

## Locked Rules
- Train only from `Matches/__merged__/<LEAGUE_TAG>__merged.csv`
- Use one trainer per run: `train_markets.py` or `train_investor_leagues_v2.py`
- Do not mix artifact naming conventions in a single run
- Verify `ModelStore` outputs before deploy usage
- Do not expand secondary markets until `FTR`, `BTTS`, and `OU25` are green

## Source Candidates
- `TRAINING_PIPELINE__MARKETS_AND_MODELS.md`
- `NEW_LEAGUE_MODEL_ONBOARDING_README.md`
