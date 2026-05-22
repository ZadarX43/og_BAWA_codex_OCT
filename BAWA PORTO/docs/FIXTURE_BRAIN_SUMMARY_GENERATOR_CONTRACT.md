# Fixture Brain Summary Generator Contract

Updated: `2026-05-21`

## Purpose

The fixture brain is the structured source for website, Telegram, and later OG GPT natural-language reports.

The generator must summarize curated `summary_inputs` only. It must not inspect raw model files, raw API-Football source rows, or hidden premium payloads outside the tier it is generating.

## Fixture Brain Payload

Current contract:

```text
schema: fixture_brain_payload_v2
contract_version: 2
summary_inputs.schema: fixture_summary_inputs_v1
```

Required top-level sections for summary generation:

- `fixture_core`
- `market_cards`
- `h2h`
- `weather`
- `team_context`
- `player_event_cards`
- `injury_context`
- `fixture_stats`
- `freshness`
- `coverage`
- `summary_inputs`

## Tier Inputs

`summary_inputs.tiers.standard`

- fixture identity
- FTR / OU25 / BTTS / team-goals market reads
- model lean, support, state, source status, probabilities where available
- freshness and basic coverage
- public copy rules

`summary_inputs.tiers.premium`

- Standard inputs
- H2H summary
- weather and space-weather context
- team context summaries
- fixture/team market-intelligence rows
- support and contradiction framing

`summary_inputs.tiers.pro`

- Premium inputs
- player-event beta cards
- injury shock market impact
- key player availability watch
- pre-lineup versus confirmed-lineup phase

`summary_inputs.tiers.pro_plus`

- Pro inputs
- coverage/audit flags
- fixture stats counts
- compact source-contract posture
- explainability and contradiction framing

## Dry Run Generator

Local-only contract smoke:

```bash
python3 scripts/build_fixture_summary_dry_run.py --fixture-brain-dir build/site_brain/current
```

Outputs:

```text
reports/latest/fixture_summary_dry_run/index.json
reports/latest/fixture_summary_dry_run/FIXTURE_SUMMARY_DRY_RUN_REPORT.md
reports/latest/fixture_summary_dry_run/fixtures/<fixture_key>.json
```

The dry run does not call GPT. It proves the structured inputs are complete enough to generate tiered summaries.

## Future GPT Nano Boundary

GPT Nano should receive one tier block at a time:

```text
summary_inputs.tiers.<tier>
```

The prompt should enforce:

- no betting advice
- no invented injuries, lineups, odds, or player stats
- explicit missing-data caveats
- beta labels for player-event outputs
- freshness mention from `freshness.last_updated`
- tier visibility discipline

The generated output should write back as a separate payload, not mutate the fixture brain source contract.
