# Premium Payload Allowlist

Updated: `2026-05-18`

This contract defines what each launch tier can receive from website-safe JSON and Worker-protected routes. It is enforced in `worker/src/index.js` by tier-specific filtering before `/api/premium/predictions` returns rows.

## Access Tiers

| Tier | Product meaning | Route boundary |
| --- | --- | --- |
| Free | Public proof and limited board | Static public JSON only |
| Founder | Discounted early access to core premium fixture intelligence | Protected Worker premium route |
| Premium | Standard paid version of Founder core access | Protected Worker premium route |
| Pro | Premium plus player-event and deeper team/player intelligence | Protected Worker premium route plus team premium surfaces |
| Pro+ | Pro plus audit/download workflow | Protected Worker premium route plus advanced export surfaces |

## Field Allowlist

### Free

- `fixture_id`
- `fixture_key`
- `kickoff_time`
- `league`
- `league_logo_url`
- `league_flag_url`
- `home_team`
- `home_team_logo_url`
- `away_team`
- `away_team_logo_url`
- `market`
- `pick`
- `confidence_tier`
- `bookie_od`
- `logo_join_status`

### Founder / Premium

Free fields plus:

- `model_prob`
- `bookie_implied_prob`
- `value_edge`
- `reason_tokens`
- `human_reason`
- `slip_role_hint`
- `safe_for_small_acca_flag`
- `safe_for_large_acca_flag`
- `correct_score_shortlist`
- `premium_tier`

### Pro

Founder / Premium fields plus:

- `player_event_signals`
- `player_events`
- `shots`
- `shots_on_target`
- `tackles`
- `fouls`
- `player_fouled`
- `key_passes`
- `goalkeeper_saves`
- `corners`
- `bookings`
- `team_intelligence`
- `player_intelligence`
- `lineup_intelligence`
- `injury_context`
- `market_combo_signals`
- `combo_signals`
- `h2h_context`
- `weather_context`

### Pro+

Pro fields plus:

- `audit_summary`
- `audit_trail`
- `model_diagnostics`
- `explainability`
- `calibration`
- `downloadable_payload`
- `advanced_filters`
- `settlement_key`
- `proof_trace`
- `data_coverage`
- `freshness`
- `source_refs`

## Explicitly Blocked

These fields must never be exposed through public or premium website payloads:

- raw model paths
- local filesystem paths
- unfiltered training features
- deploy gate internals not approved for website display
- secrets, tokens, API keys, or provider credentials

The Worker harness includes regression checks for Founder, Pro, and Pro+ filtering.
