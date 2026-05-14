# Public Results Feed Build Notes

Generated: 2026-05-14T13:18:53+00:00

## Outputs

- Website feed: `frontend/public/data/live_results_feed.json`

## Windows

### MLS live-system test night

- Period: 2026-05-13 to 2026-05-14
- Deploy: 10/11 (0.9091)
- Observe: 19/26 (0.7308)

### Weekend prediction audit

- Period: 2026-05-09 to 2026-05-11
- Deploy: 18/26 (0.6923)
- Observe: 43/82 (0.5244)

## Guardrails

- This is a public publishing adapter only.
- `OBSERVE` rows remain research/watchlist rows, not deployable picks.
- Player-event rows remain beta/manual-review only.
