# Portugal Liga Recovery Audit

## Purpose

Turn Portugal Liga from the top-ranked weak-league remediation target into a concrete recovery diagnosis.

This audit answers:

- what Portugal Liga currently achieves in the live routing stack
- what fails before or during `ALLMARKETS`
- whether the bottleneck is model quality, source odds intake, routed suppression, or overlay weakness
- what the shortest path is to move Portugal Liga toward stronger coverage

## Current State

From the current live audit window (`2026-05-09` to `2026-05-11`):

- covered fixtures: `9`
- routed fixtures: `2`
- non-routed fixtures: `7`
- deploy fixtures: `1`
- observe fixtures: `1`
- context fixtures: `7`

Market presence:

- `FTR`: present
- `BTTS`: absent
- `OU25`: absent

Overlay support:

- historical overlay: yes
- current-window odds: yes
- current-window injuries: yes
- current-window lineups: yes
- current-window team stats: yes
- current-window player stats: no
- current-window match events: yes

Current classification:

- `partial_coverage`

Current remediation priority:

- `1`

## Most Important Finding

Portugal Liga is **not** primarily a context or overlay failure.

It is primarily a **source odds / routed market availability problem**.

The strongest evidence:

- `PRE_ALLMARKETS_FIXTURE_LOSS_REPORT_2026-05-09_to_2026-05-11.csv`
- `PRE_ALLMARKETS_FIXTURE_LOSS_DETAILS_2026-05-09_to_2026-05-11.csv`

These show:

- upstream fixtures: `9`
- `ALLMARKETS` fixtures at any market: `2`
- source full `1x2`: `0`
- source missing `1x2`: `9`
- `ALLMARKETS FTR` fixtures: `2`
- `ALLMARKETS OU25` fixtures: `0`
- `ALLMARKETS BTTS` fixtures: `0`

The report note is the key signal:

- `FTR source odds missing for 9 fixture(s).`
- `FTR emitted 2 fixture(s) without source paired odds; likely rescue/backfill.`
- `OU25 source odds missing for 9 fixture(s).`
- `BTTS source odds missing for 9 fixture(s).`

That means the league currently survives through a very thin routed lane, and even that thin lane appears partly rescued rather than normally sourced.

## Routed Outcome Breakdown

### What routed

Two fixtures reached the routed family:

1. `2026_05_10_AVS_Porto`
- published as `DEPLOY`
- market: `FTR`
- tier: `STANDARD`

2. `2026_05_10_Alverca_GD_Estoril_Praia`
- published as `OBSERVE`
- market: `FTR`

### What did not route

Seven fixtures fell out of routed output and became `CONTEXT`:

- `Benfica vs Sporting Braga`
- `CD Tondela vs Moreirense FC`
- `Estrela Amadora vs Famalicão`
- `Gil Vicente vs FC Arouca`
- `Rio Ave FC vs Sporting CP`
- `Santa Clara vs CD Nacional`
- `Vitória Guimarães vs Casa Pia`

## Root Cause Reading

## 1. This is not a pure model blind spot

Portugal Liga does show:

- routed `FTR`
- one deployable row
- one observe row
- historical overlay support
- current-window overlay support

So this is not the same category as:

- `Germany Bundesliga 2`
- `Turkey Super Lig`

Those are closer to real blind spots.

Portugal is clearly **alive**, just incomplete.

## 2. The main bottleneck is source odds availability

The loss report strongly suggests:

- `1x2` source odds are missing for all `9` fixtures
- paired `OU25` source odds are missing for all `9` fixtures
- paired `BTTS` source odds are missing for all `9` fixtures

That immediately explains why:

- `BTTS` never appears
- `OU25` never appears
- only thin `FTR` routing survives

This points to:

- bookmaker intake miss
- pairing/join failure
- source harmonisation issue
- or a Portugal-specific feed coverage gap

more than to a late deploy-rulebook suppression problem

## 3. Deploy-rulebook suppression is probably secondary

The routed evidence does show:

- one `FTR` row promoted to `STANDARD`
- one `FTR` row stuck in `OBSERVE`

So there may still be:

- `OBSERVE` suppression
- elite blocking
- side-margin blocking

But those are **secondary** compared with the bigger problem:

- most Portugal fixtures never even arrive with source odds support in the first place

## 4. Overlay/context is already useful, but not the main fix

Portugal `CONTEXT` cards are already better than pure gap shells because:

- historical overlay support exists
- several fixtures carry meaningful historical lineup/form notes

And the live overlay refresh confirms:

- current-window odds
- injuries
- lineups
- team stats
- match events

for at least part of the league

That means overlay/context enhancement is useful, but it should not distract from the bigger upstream routing issue.

## League Diagnosis

### Classification

- current: `partial_coverage`
- target: `full_coverage`

### Primary issue

- `routing_expansion`

### True underlying bottleneck

More specifically:

- **source odds / intake repair**

inside the routing expansion family

### Secondary issue

- `overlay_context_enhancement`

This is valuable for the non-routed tail, but it is not the first root cause to attack.

## What This Means Product-Wise

Portugal Liga is the ideal first remediation target because:

- it is already partially alive
- it already has context support
- it already has follow-value
- the league looks recoverable
- the gap is concrete and diagnosable

This makes it far more attractive than trying to recover a true blind spot first.

## Recommended Recovery Sequence

## Stage 1 — Source Odds Intake Audit

This is the first required move.

Questions:

- why is `source_full_1x2 = 0`?
- why are paired `BTTS` / `OU25` odds missing for all fixtures?
- is Portugal failing:
  - bookmaker ingest
  - bookmaker normalization
  - fixture-to-odds pairing
  - team-name alias matching
  - league-tag mapping

Success condition:

- Portugal fixtures begin to appear with proper source-paired odds in pre-routing audit

## Stage 2 — ALLMARKETS Recovery

Once source odds are present:

- re-run Portugal window audit
- verify more than `2` fixtures reach `ALLMARKETS`
- verify `BTTS` and `OU25` rows can appear at all

Success condition:

- routed fixture count rises
- market-family breadth expands beyond `FTR`

## Stage 3 — Routed Survival Audit

Only after source availability improves should we ask:

- are `BTTS` rows being demoted too aggressively?
- are `OU25` rows being suppressed too early?
- are `OBSERVE` rows getting stuck unnecessarily?

Success condition:

- more Portugal fixtures survive into safe `OBSERVE`
- at least some `BTTS` / `OU25` presence exists in live publish

## Stage 4 — CONTEXT Tail Upgrade

Even after routing improves, some Portugal fixtures will still miss deploy.

For those:

- strengthen `CONTEXT`
- attach richer overlay notes
- improve followed-team relevance
- allow useful Portugal intelligence even without picks

Success condition:

- non-routed Portugal fixtures feel intentionally informative, not accidentally dropped

## Explicit Non-Goals

Do not start Portugal recovery by:

- forcing more deploys
- lowering frontend wording standards
- widening public claims
- treating `CONTEXT` as a substitute for fixing missing source odds

## Best Next Implementation Task

The clean next implementation after this audit is:

**Task 66 — Portugal Liga Source Odds Intake Audit**

That should inspect:

- source bookmaker coverage for Portugal fixtures
- pairing failures
- league / fixture alias mismatches
- why `FTR` emits only via apparent rescue/backfill
- why `BTTS` / `OU25` have zero source-paired presence

## Final Conclusion

Portugal Liga is the right first remediation target because it is:

- not blind
- not broken beyond recovery
- already valuable in context mode
- clearly bottlenecked by an identifiable upstream issue

The audit conclusion is simple:

**Portugal Liga’s first real fix is not “better context.”  
It is restoring reliable source odds intake so more fixtures can enter routed `FTR`, `BTTS`, and `OU25` lanes.**
