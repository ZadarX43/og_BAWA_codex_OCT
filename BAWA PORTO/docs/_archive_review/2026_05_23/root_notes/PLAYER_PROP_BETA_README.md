# Player Prop Beta README

## Purpose

This is the working README for the beta player shortlist system ahead of weekend deployment testing.

The goal is not to build a fully priced player prop engine yet.

The goal is to generate **pre-lineup, fixture-aware player shortlists** for manual review and Monday audit.

This system is intended to help identify:

- most likely shot on target
- most likely shot volume
- most likely tackle
- most likely foul committed
- most likely booking risk

## Current Status

The beta builder is:

- `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/build_player_prop_beta_board.py`

This script combines:

- live fixture context from `BOOKIE_IMP20_ALLMARKETS...csv`
- player season stats from `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/Players`
- goal / match environment context from the main model stack

It is currently:

- pre-lineup
- shortlist-based
- context-aware
- beta / experimental

It is not yet:

- a full player line model
- a bookmaker line-pricing engine
- a fully walk-forward backtested player prop system

## Output Products

The script writes:

- `PLAYER_PROP_BETA__COMBINED.csv`
- `PLAYER_SOT_SHORTLIST.csv`
- `PLAYER_SHOTS_SHORTLIST.csv`
- `PLAYER_TACKLES_SHORTLIST.csv`
- `PLAYER_FOULS_SHORTLIST.csv`
- `PLAYER_BOOKING_RISK.csv`
- `PLAYER_PROP_BETA__SUMMARY.md`

## Core Logic

The beta uses:

- player per-90 rates
- starts / minutes filters
- position group
- current club join
- fixture goal environment
- FTR / OU25 / BTTS / CS context
- attacking environment
- defensive workload environment
- chaos environment

### High-level interpretation

- `PLAYER_SOT_SHORTLIST`
  - attackers / creators with strong SOT profiles in favorable attacking fixtures

- `PLAYER_SHOTS_SHORTLIST`
  - attackers with strong shot volume in favorable team scoring environments

- `PLAYER_TACKLES_SHORTLIST`
  - defenders / midfielders with strong tackle profiles in high defensive workload fixtures

- `PLAYER_FOULS_SHORTLIST`
  - combative players in chaotic / pressure-heavy fixtures

- `PLAYER_BOOKING_RISK`
  - players with booking history and foul profile in high-chaos contexts

## Important Beta Rules

This system should be treated as:

- shortlist intelligence
- not a final prediction engine
- not guaranteed to reflect official lineups

Use it to:

- generate candidates
- manually inspect names
- refine before kickoff
- review outcomes on Monday

Do not use it as if it already has:

- lineup certainty
- injury certainty
- exact prop probabilities
- bookmaker line awareness

## Standard Weekend Workflow

### 1. Run the normal live stack

Get the fresh weekend outputs in place first:

- data ingestion
- merged rebuild
- all usual checks
- `bookie_allmarkets.py`
- `deploy_rulebook.py`
- `slip_formatter.py`

### 2. Build the player beta board

Run:

```bash
python3 '/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/build_player_prop_beta_board.py' \
  --source '/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/predictions_output/YYYY-MM-DD/BOOKIE_IMP20_ALLMARKETS_<date_from>_to_<date_to>.csv' \
  --players-root '/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/Players' \
  --outdir '/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/reports/2026-04-24/PLAYER_PROP_BETA__LIVE' \
  --top-n 3
```

Replace:

- `YYYY-MM-DD`
- `<date_from>`
- `<date_to>`

### 3. Review the output files

Focus first on:

- `PLAYER_SOT_SHORTLIST.csv`
- `PLAYER_SHOTS_SHORTLIST.csv`
- `PLAYER_TACKLES_SHORTLIST.csv`
- `PLAYER_FOULS_SHORTLIST.csv`
- `PLAYER_BOOKING_RISK.csv`

### 4. Manually refine

Before kickoff, review:

- obvious non-starters
- low-minute players
- sample-poor players
- bad joins / wrong team naming
- players who do not fit the actual expected match role

### 5. Optional pre-kickoff patching

Reasonable last-mile edits are:

- increasing minimum minutes
- increasing minimum starts
- reducing `top-n`
- excluding suspect leagues
- excluding suspect positions for a specific market

## Fields to Watch

The most useful review columns are:

- `full_name`
- `player_team`
- `position`
- `fixture_key`
- `market_name`
- `signal_score`
- `confidence`
- `support_note`
- `prelineup_start_confidence`
- `sample_ok_flag`
- `risk_flag_low_sample`
- `risk_flag_low_start_conf`
- `risk_flag_sub_risk`

## Suggested Manual Review Heuristics

### For shots / SOT

Prefer:

- attackers / advanced mids
- solid minutes
- solid starts
- stronger team lambda
- stronger OU25 / open-game support

Be careful with:

- low-minute bench attackers
- players with tiny samples
- defenders appearing high only due to noisy data

### For tackles

Prefer:

- defenders / defensive mids
- underdog or pressure-heavy games
- higher defensive workload contexts

Be careful with:

- attacking players with inflated tiny-sample tackle rates

### For fouls / bookings

Prefer:

- midfielders / defenders
- chaotic fixtures
- players with real foul / card history

Be careful with:

- low-minute players
- players with fake-high rates from tiny samples

## Monday Review Plan

After the weekend, review:

### Outcome questions

- did the shortlisted player start?
- did the player hit the event?
- which shortlist type looked strongest?
- which leagues looked strongest?
- did `confidence` actually correlate with success?
- were the best results clustered in certain fixture shapes?

### Failure questions

- were misses caused by lineup issues?
- were misses caused by low samples?
- were there bad team joins?
- were some markets much noisier than others?
- did certain positions pollute the shortlist?

## Known Limitations

Current limitations include:

- no official lineup integration yet
- no expected minutes model
- no injury / suspension layer
- no bookmaker player line ingestion
- no time-safe historical player snapshot reconstruction
- no direct player corners model
- no direct player throw-ins model

## Best Current Use

Use this system as:

- a premium shortlist experiment
- an internal research board
- a pre-kickoff content support layer
- a Monday audit candidate generator

## Safe Product Language

Use wording like:

- `beta shortlist`
- `most likely candidates`
- `fixture-aligned player signals`
- `pre-lineup player board`

Avoid wording like:

- `fully modeled probabilities`
- `true prop odds`
- `priced edge model`

## Fast Patch Ideas Before Kickoff

If needed, the easiest safe refinements are:

1. raise `--min-minutes`
2. raise `--min-starts`
3. cut `--top-n` from `3` to `2`
4. manually restrict noisy leagues
5. manually filter by position after export

## Next Likely Improvements

After the weekend, likely upgrades are:

1. post-lineup reranker
2. GPT explanation layer
3. league allowlist tuning
4. market-specific thresholds
5. better minutes / start confidence logic

## Working Principle

This beta does not need to be perfect to be valuable.

If it gives:

- sensible names
- sensible fixture fit
- a useful shortlist
- strong Monday review evidence

then it has already done its job.
