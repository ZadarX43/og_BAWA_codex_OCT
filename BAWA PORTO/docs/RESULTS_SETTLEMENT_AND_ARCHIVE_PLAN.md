# Results Settlement And Archive Plan

## Purpose

Define the backend and publishing flow that moves live published picks into a settled, auditable results board once matches finish.

This plan is specifically about:
- settled pick grading
- archive persistence
- public proof updates
- premium/public historical transparency

It is not about changing model generation or live deploy logic.

## Core Principle

Predictions stay offline and published.
Results are graded after the fact and then published as a separate proof/archive layer.

Do not mix:
- live prediction generation
- settlement logic
- frontend-only derived proof

The results board should be built from a repeatable grading pipeline, not inferred in the browser.

## Current State

### Already in place
- public board publish pipeline
- premium board publish pipeline
- `publish_summary.json`
- `weekly_results.json` surface exists on the frontend side
- results page already has proof UI

### Missing
- automatic settled-pick grading from finished matches
- archive persistence for all published picks
- rollup generation after settlement
- movement of picks from active board context into settled results/proof context

## Target Product Outcome

After matches finish:
1. picks are matched to final fixture results
2. each pick is graded as won/lost/void/pending
3. profit/loss is computed
4. aggregate metrics update
5. results archive is published
6. results page reflects the latest truth without manual editing

## Canonical Inputs

### Publish-time inputs
- published public board JSON
- published premium board JSON
- publish summary metadata
- source deploy CSV used for the board

### Settlement inputs
- final scores / fixture completion state
- any authoritative results feed or normalized completed fixture data

## Canonical Outputs

### Required publish outputs
- `frontend/public/data/weekly_results.json`
- `frontend/public/data/results_archive.json`
- optional `frontend/public/data/results_archive.csv`

### Optional operator outputs
- `reports/latest/RESULTS_SETTLEMENT_REPORT.md`
- `reports/latest/results_rollup_snapshot.json`

## Data Model For A Settled Pick

Each archived pick should carry:
- `fixture_id`
- `fixture_key`
- `kickoff_time`
- `league`
- `home_team`
- `away_team`
- `market`
- `pick`
- `confidence_tier`
- `premium_tier`
- `bookie_od`
- `model_prob`
- `bookie_implied_prob`
- `value_edge`
- `result_status`
  - `pending`
  - `won`
  - `lost`
  - `void`
- `profit_units`
- `final_home_score`
- `final_away_score`
- `settled_at`
- `published_run_id` or equivalent publish lineage field

## Rollups To Produce

### Overall
- total picks
- settled picks
- pending picks
- wins
- losses
- voids
- hit rate
- ROI
- profit units

### By market
- FTR
- BTTS
- OU25
- later other markets

### By tier
- ELITE
- STANDARD
- optional confidence/proof buckets later

### By league
- useful for proof, diagnostics, and product storytelling

### By window
- weekly
- rolling 30-day
- monthly
- season-to-date

## Recommended Pipeline Shape

### Step 1. Identify the active published board
Use the same publish lineage already written into:
- `publish_summary.json`

This should tell the settlement process what board is being graded.

### Step 2. Resolve fixture completion
Use normalized final results data to identify:
- finished matches
- final scores
- abandoned/void conditions where available

### Step 3. Grade each pick
Market-specific grading rules:
- FTR
- BTTS
- OU25
- future markets later

### Step 4. Compute profit units
Default simple convention:
- won -> `bookie_od - 1`
- lost -> `-1`
- void -> `0`

Keep this explicit and documented.

### Step 5. Write settled archive outputs
Persist both:
- current-week/current-window summary
- appendable archive layer

### Step 6. Publish website-safe JSON
Frontend should consume static JSON, not compute truth from raw prediction lists.

## Archive Strategy

### Recommended approach
- maintain a cumulative `results_archive.json`
- maintain a current-window `weekly_results.json`

This gives:
- fast current proof page rendering
- full historical drilldown later

### Important rule
Do not overwrite or hide old outcomes.
Transparency is the product advantage.

## Frontend Expectations

Results page should be able to show:
- current window settled count
- hit rate
- ROI
- featured recent wins/losses
- chart-ready data series

Future enhancements:
- search/filter by market
- filter by tier
- filter by league
- month/week archives

## Operational Timing

Recommended cadence:
- run settlement late Monday / Tuesday after weekend matches resolve
- optionally run again idempotently for late official updates

## Idempotency

The grading pipeline should be safe to rerun.

That means:
- same finished fixtures -> same graded outcomes
- no duplicate archive inserts
- no double-counted profit

Recommended key:
- `fixture_id + market + pick + tier`

## Current Script Layer

Canonical publish command:

```bash
python3 scripts/publish_results_proof.py
```

This runs:
- `scripts/settle_published_results.py`
- `scripts/smoke_results_page.py`

Primary outputs:
- `frontend/public/data/weekly_results.json`
- `frontend/public/data/results_archive.json`
- `reports/latest/RESULTS_SETTLEMENT_REPORT.md`
- `reports/latest/RESULTS_PAGE_SMOKE_REPORT.md`
- `reports/latest/RESULTS_PUBLISH_RUN_REPORT.md`

## Publish Workflow At Scale

Weekend operating loop:
1. publish board
2. serve board
3. wait for matches to finish
4. run `python3 scripts/publish_results_proof.py`
5. publish results JSON after the smoke report is green
6. Cloudflare Pages updates proof page

## Relationship To Premium/Public

### Public board
- limited current picks
- public proof page

### Premium board
- full board during active cycle
- historical outcomes can still contribute to full proof metrics

The archive/proof layer should not care whether a pick was public or premium-only when computing truth.
But it may still be useful to preserve:
- `visibility = public|premium`

## Risks To Avoid
- grading from browser data only
- manual editing of win/loss board
- hidden archive rewrites
- profit calculation inconsistencies
- mixing pending and settled metrics unclearly

## Recommended Next Implementation Step

Wire the canonical publish command into the scheduler/automation layer after the website-safe prediction export has completed and provider final scores are available.
