# AGENTS.md

## Repo Identity
- Repo root: `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO`
- Project: Odds Genius / BAWA PORTO
- Primary objective: protect the production football prediction pipeline while improving orchestration, docs, and auditability.

## Production Spine
Treat these files as the protected production spine:
- `footystats_drop_ingest.py`
- `etl_press_intensity.py`
- `build_merged.py`
- `patch_merge_add_streaks.py`
- `team_ratings.py`
- `patch_merge_add_power_ratings.py`
- `make_fd_odds_enriched_synth.py`
- `patch_merge_add_synth_odds.py`
- `pipeline_qa_gate.py`
- `bookie_allmarkets.py`
- `deploy_rulebook.py`
- `slip_formatter.py`

Do not make casual behavioral changes to these files. Prefer documented, auditable, minimal edits.

## Non-Negotiable Rules
- Never train from raw season CSVs.
- Canonical training input is `Matches/__merged__/<LEAGUE_TAG>__merged.csv`.
- Never run predictions before integrity checks.
- `deploy_rulebook.py` owns routing, tiers, vetoes, and gates.
- `slip_formatter.py` is thin only: no filler, no rescue, no forced completion.
- Value layer is additive only and must never override deploy gates.
- `OBSERVE` rows are not deployable.
- Weather and player props are beta/research only.

## Safe Operating Order
### Data refresh
1. `footystats_drop_ingest.py`
2. `etl_press_intensity.py`
3. `build_merged.py --recursive --rolling-press`
4. `patch_merge_add_streaks.py`
5. `team_ratings.py`
6. `patch_merge_add_power_ratings.py`
7. `make_fd_odds_enriched_synth.py --emit-ou25-novig`
8. `patch_merge_add_synth_odds.py`
9. `pipeline_qa_gate.py` or equivalent integrity spot-check

Hard stop: if integrity fails, do not run `bookie_allmarkets.py`.

### Live deploy
1. Confirm integrity passed.
2. Run `bookie_allmarkets.py`.
3. Run `deploy_rulebook.py`.
4. Run `slip_formatter.py`.
5. Run regression / sanity checks.
6. Summarize output files.

## Training Guardrails
- Train only from merged canonical inputs.
- Pick one trainer per run: `train_markets.py` or `train_investor_leagues_v2.py`.
- Do not mix artifact naming schemes.
- Verify `ModelStore` outputs after training.
- Run `bookie_allmarkets.py --strict` after training validation when appropriate.
- Do not expand secondary markets until `FTR`, `BTTS`, and `OU25` are green.

## Deploy / Product Guardrails
- `deploy_rulebook.py` is the single source of truth for live routing.
- `slip_formatter.py` must stay a formatting/orchestration layer, not a prediction or rescue layer.
- Do not include `OBSERVE` in live deploy products unless explicitly doing research output.
- Do not add filler logic to complete accas or slip families.
- Do not let value overlays override hard deploy vetoes.

## Research vs Production Boundary
### Production-safe
- Data refresh and integrity
- Canonical training
- All-markets generation
- Deploy routing
- Slip formatting
- Correct-score deploy generation

### Research-only / beta
- Weather gates
- Player props
- New betting markets
- Experimental gate relaxations
- Website rebuilds / UI work unrelated to orchestration safety

## Documentation Targets
Preferred canonical docs live under `docs/`.
Legacy markdown should remain in place until merged and explicitly archived with a report.

## Skills Preference
When available, prefer repo-local skills under `.codex/skills/` for:
- data refresh
- live deploy
- training / validation
- deploy audits
- weather research
- player prop beta
- desktop app workflow

## Change Discipline
- Make the smallest safe change.
- Preserve backward compatibility unless explicitly approved.
- Back up high-risk files before patching routing or pipeline behavior.
- Record contradictions and hidden assumptions in docs rather than silently resolving them.
