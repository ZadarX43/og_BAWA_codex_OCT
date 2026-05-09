# Portugal Liga Source Odds Intake Audit

## Purpose

Inspect the first real Portugal Liga bottleneck in concrete terms:

- bookmaker coverage for Portugal fixtures
- fixture/team alias mismatches
- why `FTR` only survives via apparent rescue/backfill
- why `BTTS` / `OU25` have zero source-paired presence

This audit exists to convert the Portugal recovery diagnosis into the next implementation target.

## Why Portugal Needs This Audit

Portugal Liga is not a blind spot.

It is already:

- partially routed
- context-supported
- historically overlaid
- current-window overlay-supported

But it is not strong enough because:

- only `2` of `9` fixtures reached `ALLMARKETS`
- only `1` deploy row survived
- only `1` observe row survived
- `BTTS` and `OU25` are entirely absent

The coverage problem is therefore not “no Portugal intelligence exists.”

It is:

**source-paired odds do not appear to be attaching cleanly to Portugal fixtures at the pre-routing stage.**

## Confirmed Evidence

## 1. Pre-ALLMARKETS loss report

From:

- `PRE_ALLMARKETS_FIXTURE_LOSS_REPORT_2026-05-09_to_2026-05-11.csv`

Portugal Liga shows:

- upstream fixtures: `9`
- `ALLMARKETS` fixtures any market: `2`
- source full `1x2`: `0`
- source missing `1x2`: `9`
- `ALLMARKETS FTR` fixtures: `2`
- `FTR emitted without source 1x2`: `2`
- source paired `OU25`: `0`
- source missing paired `OU25`: `9`
- source paired `BTTS`: `0`
- source missing paired `BTTS`: `9`

This is the strongest single signal in the whole recovery stack.

## 2. Pre-ALLMARKETS detail rows

From:

- `PRE_ALLMARKETS_FIXTURE_LOSS_DETAILS_2026-05-09_to_2026-05-11.csv`

Every Portugal fixture is flagged with:

- `issue = SOURCE_ODDS_MISSING`
- `odds_status = MISSING_SOURCE_PAIR`

Across:

- `FTR`
- `OU25`
- `BTTS`

And importantly:

- the two routed `FTR` fixtures are still marked `MISSING_SOURCE_PAIR`
- but `allmarkets_status = EMITTED`

That strongly suggests:

- `FTR` is surviving via rescue/backfill behavior
- not through normal source-paired bookmaker intake

## 3. Routed output confirms the thin lane

In the live routed family:

- `DEPLOY_TIER_STANDARD`: `1` Portugal `FTR` row
- `DEPLOY_TIER_OBSERVE`: `1` Portugal `FTR` row
- `ALLMARKETS`: `2` Portugal rows total
- `BTTS`: `0`
- `OU25`: `0`

So:

- `FTR` is barely alive
- `BTTS` is dead upstream
- `OU25` is dead upstream

## Main Hypotheses

This audit should test four possible root causes.

## Hypothesis A — bookmaker coverage is genuinely missing

Portugal fixtures may not actually have the required bookmaker source lines in the intake layer.

That would mean:

- there is no usable source pair to attach
- the loss report is correctly showing a true upstream data gap

Questions:

- does the bookmaker source feed actually include Portugal `1x2`?
- does it include `BTTS`?
- does it include `OU25`?
- is the issue universal or bookmaker-specific?

If true:

- this is a real intake / provider coverage problem
- not just a join bug

## Hypothesis B — bookmaker rows exist but fixture pairing fails

This is one of the most likely failure modes.

Possibilities:

- fixture keys are not aligning
- team aliases do not match
- accented names break joins
- league-tag mapping is inconsistent
- provider naming differs across markets

Portugal is especially vulnerable because current fixtures include names like:

- `Vitória Guimarães`
- `Famalicão`
- `Sporting Braga`
- `GD Estoril Praia`
- `Moreirense FC`

Potential symptoms:

- raw odds exist
- but pre-routing reports them as `MISSING_SOURCE_PAIR`
- only rescue logic lets thin `FTR` output survive

## Hypothesis C — FTR rescue path exists but BTTS / OU25 lack equivalent recovery

The fact that:

- `FTR emitted_without_source_1x2 = 2`

while:

- `OU25 emitted_without_source_pair = 0`
- `BTTS emitted_without_source_pair = 0`

suggests a structural asymmetry:

- `FTR` may have rescue/backfill logic
- `BTTS` / `OU25` may require strict source pairing

If true:

- the first fix is still source-pair recovery
- but the second fix may be to inspect whether `BTTS` / `OU25` are over-dependent on strict source presence

## Hypothesis D — upstream model estate exists, but is blocked by source-odds gating

Portugal historical model estate clearly exists in older outputs.

That means the issue may not be:

- no Portugal model

but instead:

- live weekend routing refuses to advance because source odds are absent

This would be good news, because it makes Portugal much easier to recover than a true blind spot.

## What This Audit Must Inspect

## 1. Raw bookmaker source presence

For Portugal Liga in the active window:

- find raw bookmaker fixtures
- confirm whether `1x2` exists
- confirm whether `BTTS` exists
- confirm whether `OU25` exists
- identify which bookmaker/provider layers are missing

Deliverable:

- fixture-by-fixture source coverage table

## 2. Pairing / alias integrity

For Portugal fixtures:

- compare fixture names before and after source pairing
- inspect accent stripping
- inspect team alias normalization
- inspect league-tag mapping
- inspect whether:
  - `Portugal Liga`
  - `Portugal_Liga`
  - `Primeira Liga`

are diverging anywhere in the odds-pairing path

Deliverable:

- list of concrete alias or mapping mismatches

## 3. FTR rescue/backfill behavior

The two emitted Portugal `FTR` rows need explaining.

Audit:

- why those two survived
- what rescue path allowed them through
- whether they came from:
  - fallback odds
  - synthetic reconstruction
  - secondary bookmaker path
  - exception logic

Deliverable:

- explicit explanation of why `AVS vs Porto` and `Alverca vs GD Estoril Praia` emitted while the other seven did not

## 4. BTTS / OU25 structural absence

Audit why these have:

- zero source-paired rows
- zero emitted rows

Questions:

- are bookmaker markets actually absent?
- are market-name mappings failing?
- are paired price selectors too strict?
- are these markets being dropped before `ALLMARKETS` for Portugal specifically?

Deliverable:

- first exact break point for `BTTS` and `OU25`

## League-Specific Questions To Answer

For Portugal Liga:

1. Do bookmaker source rows exist for all `9` fixtures?
2. If yes, for which markets?
3. If yes, why are they failing source pairing?
4. Why do two `FTR` rows emit without a valid source pair?
5. Why do `BTTS` and `OU25` never emit at all?
6. Is this caused by:
   - missing provider coverage
   - alias mismatch
   - league mapping mismatch
   - market-name mismatch
   - or strict gating after pairing failure

## Expected Outcome Categories

The audit should end by placing Portugal into one of these practical buckets:

### A. Provider gap

Meaning:

- bookmaker rows truly do not exist

Action:

- expand provider coverage or deprioritise those markets

### B. Pairing bug

Meaning:

- bookmaker rows exist
- but fixture/market join is broken

Action:

- fix alias/mapping/pairing logic

### C. Rescue-only FTR path

Meaning:

- `FTR` survives through fallback
- but `BTTS` / `OU25` do not

Action:

- recover proper source pairing first
- then inspect market-family asymmetry

### D. Market-family gating issue

Meaning:

- paired rows exist
- but Portugal `BTTS` / `OU25` are filtered too aggressively

Action:

- inspect post-pairing gates and routed survival

## Recommended Success Condition

This audit is successful when we can say, with evidence:

- whether Portugal bookmaker rows exist
- whether they fail on pairing
- why only two `FTR` rows survive
- where `BTTS` / `OU25` die

At that point, the next implementation will be obvious rather than speculative.

## Best Follow-On Task

After this audit, the most likely next implementation step is:

**Task 67 — Portugal Liga Odds Pairing Repair**

or, if the evidence says provider rows are absent:

**Task 67B — Portugal Liga Source Coverage Fallback Strategy**

## Final Conclusion

Portugal Liga should still be the first recovery target.

But this audit makes the next move much sharper:

**The first real fix is to inspect and recover source odds pairing for Portugal fixtures, not to widen frontend context language or lower deploy standards.**
