# LEAGUE_REMEDIATION_PLAN_REPORT

Generated: `2026-05-09T14:11:03+00:00`
Source run id: `2026-05-09`
Source window: `2026-05-09` to `2026-05-11`

## Summary
- Weak leagues: `15`
- Blind spots: `2`
- Partial coverage: `13`
- `model_expansion`: `2`
- `routing_expansion`: `7`
- `overlay_context_enhancement`: `6`

## Prioritised Weak-League Actions
### 1. Portugal Liga
- Current: `partial_coverage`
- Target: `full_coverage`
- Primary issue: `routing_expansion`
- Secondary issue: `overlay_context_enhancement`
- Market gaps: `FTR=False` | `BTTS=True` | `OU25=True`
- Action: Use Portugal Liga as the first partial-coverage strengthening target.
- Action: Audit pre-ALLMARKETS fixture loss for the non-routed tail.
- Action: Audit why BTTS rows are not surviving routed publish for this league.
- Action: Audit why OU25 rows are not surviving routed publish for this league.
- Action: Prefer widening observe-safe routed coverage before forcing new deploys.

### 2. Germany Bundesliga 2
- Current: `blind_spot`
- Target: `hidden_for_now`
- Primary issue: `model_expansion`
- Secondary issue: `none`
- Market gaps: `FTR=True` | `BTTS=True` | `OU25=True`
- Action: Decide explicitly whether Bundesliga 2 should recover to context-only or stay hidden.
- Action: Audit whether FootyStats-era goal-market models exist for this league.
- Action: Audit whether bookmaker coverage should be feeding ALLMARKETS for this league.
- Action: If no usable model estate exists, keep the league hidden until a minimal observe-safe base exists.

### 3. Turkey Super Lig
- Current: `blind_spot`
- Target: `hidden_for_now`
- Primary issue: `model_expansion`
- Secondary issue: `none`
- Market gaps: `FTR=True` | `BTTS=True` | `OU25=True`
- Action: Decide whether Turkey Super Lig is worth active recovery or deliberate deprioritisation.
- Action: Audit whether FootyStats-era goal-market models exist for this league.
- Action: Audit whether bookmaker coverage should be feeding ALLMARKETS for this league.
- Action: If no usable model estate exists, keep the league hidden until a minimal observe-safe base exists.

### 4. Belgium Pro
- Current: `partial_coverage`
- Target: `full_coverage`
- Primary issue: `routing_expansion`
- Secondary issue: `none`
- Market gaps: `FTR=False` | `BTTS=True` | `OU25=True`
- Action: Audit why BTTS rows are not surviving routed publish for this league.
- Action: Audit why OU25 rows are not surviving routed publish for this league.
- Action: Prefer widening observe-safe routed coverage before forcing new deploys.

### 5. England Premier League
- Current: `partial_coverage`
- Target: `full_coverage`
- Primary issue: `routing_expansion`
- Secondary issue: `none`
- Market gaps: `FTR=False` | `BTTS=False` | `OU25=False`
- Action: Prefer widening observe-safe routed coverage before forcing new deploys.

### 6. Italy Serie A
- Current: `partial_coverage`
- Target: `full_coverage`
- Primary issue: `routing_expansion`
- Secondary issue: `none`
- Market gaps: `FTR=False` | `BTTS=False` | `OU25=True`
- Action: Audit why OU25 rows are not surviving routed publish for this league.
- Action: Prefer widening observe-safe routed coverage before forcing new deploys.

### 7. Scotland Premiership
- Current: `partial_coverage`
- Target: `full_coverage`
- Primary issue: `routing_expansion`
- Secondary issue: `none`
- Market gaps: `FTR=False` | `BTTS=True` | `OU25=False`
- Action: Audit why BTTS rows are not surviving routed publish for this league.
- Action: Prefer widening observe-safe routed coverage before forcing new deploys.

### 8. England Championship
- Current: `partial_coverage`
- Target: `full_coverage`
- Primary issue: `routing_expansion`
- Secondary issue: `none`
- Market gaps: `FTR=False` | `BTTS=True` | `OU25=True`
- Action: Audit why BTTS rows are not surviving routed publish for this league.
- Action: Audit why OU25 rows are not surviving routed publish for this league.
- Action: Prefer widening observe-safe routed coverage before forcing new deploys.

### 9. Australia A-League
- Current: `partial_coverage`
- Target: `full_coverage`
- Primary issue: `routing_expansion`
- Secondary issue: `none`
- Market gaps: `FTR=False` | `BTTS=True` | `OU25=True`
- Action: Audit why BTTS rows are not surviving routed publish for this league.
- Action: Audit why OU25 rows are not surviving routed publish for this league.
- Action: Prefer widening observe-safe routed coverage before forcing new deploys.

### 10. Austria Bundesliga
- Current: `partial_coverage`
- Target: `partial_coverage`
- Primary issue: `overlay_context_enhancement`
- Secondary issue: `none`
- Market gaps: `FTR=True` | `BTTS=False` | `OU25=False`
- Action: Improve CONTEXT note richness for non-routed fixtures in this league.
- Action: Feed current-window prematch odds into the context lane.
- Action: Add current-window injury overlay support for this league.
- Action: Add current-window lineup overlay support for this league.
- Action: Ensure followed-team and followed-fixture users still receive useful non-pick intelligence.

### 11. South Korea K League
- Current: `partial_coverage`
- Target: `partial_coverage`
- Primary issue: `overlay_context_enhancement`
- Secondary issue: `none`
- Market gaps: `FTR=False` | `BTTS=True` | `OU25=False`
- Action: Improve CONTEXT note richness for non-routed fixtures in this league.
- Action: Feed current-window prematch odds into the context lane.
- Action: Add current-window injury overlay support for this league.
- Action: Add current-window lineup overlay support for this league.
- Action: Ensure followed-team and followed-fixture users still receive useful non-pick intelligence.

### 12. Swiss Super League
- Current: `partial_coverage`
- Target: `partial_coverage`
- Primary issue: `overlay_context_enhancement`
- Secondary issue: `none`
- Market gaps: `FTR=False` | `BTTS=False` | `OU25=True`
- Action: Improve CONTEXT note richness for non-routed fixtures in this league.
- Action: Feed current-window prematch odds into the context lane.
- Action: Add current-window injury overlay support for this league.
- Action: Add current-window lineup overlay support for this league.
- Action: Ensure followed-team and followed-fixture users still receive useful non-pick intelligence.

### 13. Denmark Superliga
- Current: `partial_coverage`
- Target: `partial_coverage`
- Primary issue: `overlay_context_enhancement`
- Secondary issue: `none`
- Market gaps: `FTR=False` | `BTTS=False` | `OU25=True`
- Action: Improve CONTEXT note richness for non-routed fixtures in this league.
- Action: Feed current-window prematch odds into the context lane.
- Action: Add current-window injury overlay support for this league.
- Action: Add current-window lineup overlay support for this league.
- Action: Ensure followed-team and followed-fixture users still receive useful non-pick intelligence.

### 14. Norway Eliteserien
- Current: `partial_coverage`
- Target: `partial_coverage`
- Primary issue: `overlay_context_enhancement`
- Secondary issue: `none`
- Market gaps: `FTR=False` | `BTTS=False` | `OU25=True`
- Action: Improve CONTEXT note richness for non-routed fixtures in this league.
- Action: Feed current-window prematch odds into the context lane.
- Action: Add current-window injury overlay support for this league.
- Action: Add current-window lineup overlay support for this league.
- Action: Ensure followed-team and followed-fixture users still receive useful non-pick intelligence.

### 15. England EFL League 1
- Current: `partial_coverage`
- Target: `partial_coverage`
- Primary issue: `overlay_context_enhancement`
- Secondary issue: `none`
- Market gaps: `FTR=True` | `BTTS=False` | `OU25=True`
- Action: Improve CONTEXT note richness for non-routed fixtures in this league.
- Action: Feed current-window prematch odds into the context lane.
- Action: Add current-window injury overlay support for this league.
- Action: Add current-window lineup overlay support for this league.
- Action: Ensure followed-team and followed-fixture users still receive useful non-pick intelligence.
