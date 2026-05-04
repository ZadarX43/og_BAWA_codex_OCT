# LIVE_DEPLOYMENT_AUDIT

Updated: `2026-05-04`
Repo: `/Users/hughwade/Documents/Code/OG_master/BAWA PORTO`
Scope: Task 1 audit only for website overhaul plus GitHub and Cloudflare integration.

## 1. Current Prediction Outputs

### Production spine outputs already present
- `bookie_allmarkets.py` produces the source all-markets board:
  - `predictions_output/.../BOOKIE_IMP*_ALLMARKETS_<FROM>_to_<TO>.csv`
- `deploy_rulebook.py` produces routed deploy outputs:
  - `.../DEPLOY_CANDIDATES_RAW.csv`
  - `.../DEPLOY_CANDIDATES_AFTER_GATES.csv`
  - `.../DEPLOY_COMBINED_<FROM>_to_<TO>.csv`
  - tier files such as `...__DEPLOY_TIER_ELITE__...csv`, `STANDARD`, `OBSERVE`
- `slip_formatter.py` is the downstream formatting layer for ranked boards and slip families.

### Existing public-facing precursor outputs
- `og_public_match_cards.py` already builds a fixture-level reporting dataset:
  - `predictions_output/PUBLIC_MATCH_CARDS/<FROM>_to_<TO>/public_match_cards.csv`
- `score_public_match_cards.py` scores those exports and writes:
  - `predictions_output/PUBLIC_MATCH_CARDS/_SCORES/month_scores.csv`
  - `predictions_output/PUBLIC_MATCH_CARDS/_SCORES/month_league_scores.csv`
  - `predictions_output/PUBLIC_MATCH_CARDS/_SCORES/issues.csv`
  - `predictions_output/PUBLIC_MATCH_CARDS/_SCORES/league_issues.csv`
- `season_public_match_cards_runner.py` builds season-level aggregate CSV outputs from monthly public match cards.

### Audit read
- The repo already has a strong CSV/reporting estate.
- It does not yet have a dedicated frontend-safe JSON publishing layer.
- The most reusable bridge into a website is likely:
  - routed deploy CSVs for live picks
  - `public_match_cards.csv` for broader fixture-level/public reporting

## 2. Current Frontend Structure

### Present state
- No repo-root frontend application scaffold was found.
- No repo-root `frontend/` directory was found.
- No repo-root `.github/` directory was found.
- No repo-root `package.json` was found.
- No repo-root `wrangler.toml` was found.
- No clear Next.js / Vite / React app entrypoint was found near the root.

### Implication
- The website/webapp handoff is currently documentation-led, not implementation-led.
- Frontend build work should be treated as a new product surface rather than a refactor of an existing web app.

## 3. Current Deployment Commands

### Canonical live deployment order from repo docs
1. `footystats_drop_ingest.py`
2. `etl_press_intensity.py`
3. `build_merged.py --recursive --rolling-press`
4. `patch_merge_add_streaks.py`
5. `team_ratings.py`
6. `patch_merge_add_power_ratings.py`
7. `make_fd_odds_enriched_synth.py --emit-ou25-novig`
8. `patch_merge_add_synth_odds.py`
9. `pipeline_qa_gate.py`
10. `bookie_allmarkets.py`
11. `deploy_rulebook.py`
12. `slip_formatter.py`

### Existing orchestration helpers
- `run_weekend_portfolio.py`
  - existing orchestration around windowed weekend portfolio/backtest-style flows
  - not a website publishing pipeline
- `season_public_match_cards_runner.py`
  - orchestration around public card generation and season scoring
  - useful for results/history pages, not live publish

### Gap
- No dedicated `publish_predictions.py`
- No dedicated `validate_public_export.py`
- No dedicated `run_weekend_pipeline.py` matching the target spec
- No dedicated `grade_weekend_results.py`

## 4. Safe Public Fields

### Best current safe-source baseline
The existing `public_match_cards.csv` is the safest current starting point because it is already reporting-oriented and fixture-level rather than raw deployment logic.

### Recommended public-safe live board fields
- `fixture_id`
- `fixture_key`
- `kickoff_time`
- `league`
- `home_team`
- `away_team`
- `market`
- `pick`
- `confidence_tier`
- `display_confidence`
- `bookie_od`
- `model_prob_display`
- `value_edge_display`
- `short_reason`
- `is_free`

### Current repo fields that can likely map into safe public fields
- fixture identity:
  - `fixture_key`
  - `league`
  - `match_date`
  - `home_team_name`
  - `away_team_name`
- deploy/product summary:
  - `deploy_tier`
  - `market`
  - `selection`
  - `bookie_pick`
- display-safe probability/value candidates after formatting:
  - rounded `model_p_for_bookie`
  - rounded `bookie_implied`
  - rounded `gap` or `gap_novig`
- explanation candidates:
  - a redacted, frontend-safe summary derived from reason tokens rather than raw tokens themselves

### Public caution
- The current `public_match_cards.csv` still includes fields that are too rich for direct website publishing.
- A publish step must explicitly whitelist fields, not just drop a few known-dangerous columns.

## 5. Premium Fields

### Target premium-safe fields from the build spec
- `fixture_id`
- `fixture_key`
- `kickoff_time`
- `league`
- `home_team`
- `away_team`
- `market`
- `pick`
- `confidence_tier`
- `model_prob`
- `bookie_implied_prob`
- `value_edge`
- `bookie_od`
- `reason_tokens`
- `human_reason`
- `slip_role_hint`
- `safe_for_small_acca_flag`
- `safe_for_large_acca_flag`
- `correct_score_shortlist`
- `premium_tier`

### Current repo fields that could support premium export
- `model_p_for_bookie`
- `bookie_implied`
- `bookie_implied_novig`
- `gap`
- `gap_novig`
- `deploy_tier`
- `context_reason_codes`
- `standard_reporting_bucket`
- `cs1`, `cs1_p`, `cs2`, `cs2_p`, `cs3`, `cs3_p`
- `pick_side_mass_top3`
- `pick_side_margin_top3`
- `ftr_margin`
- `power_diff`

### Premium caution
- Premium can contain more detail than public.
- Premium still must not expose threshold formulas, model file paths, raw feature surfaces, or deploy gate internals.

## 6. Unsafe Fields

### Never publish directly from current deploy CSVs
The sampled `DEPLOY_CANDIDATES_AFTER_GATES.csv` header shows many fields that are too revealing for public or premium website output if passed through unredacted.

### Unsafe categories
- training/model internals:
  - model bundle references
  - model lane markers
  - raw source probability columns
- raw gate logic:
  - `deterministic_veto_reason`
  - `deploy_veto_reason`
  - `deterministic_adjust_reason`
  - `learned_veto_reason`
  - `context_reason_codes` in raw form
- threshold/proxy surfaces:
  - `meta_gate_pass`
  - `ou25_policy_mode`
  - `ou25_policy_branch`
  - `ou25_policy_state` if used verbatim
  - `draw_risk_score`
  - `draw_chaos_score`
- raw feature columns:
  - rolling form/xG/clean-sheet/PPG/H2H internals
  - `uefa_*` fields
  - `lambda_*`, `p00_est`, `p_home_ge*`, `p_away_ge*`
  - agreement-count and overlay-support columns
- internal routing metadata:
  - `candidate_rank`
  - `score`
  - `support_*`
  - `meta_*`
  - `profit_first_keep` style internal helpers if present
- secrets and environment-linked data:
  - `.env` values
  - API keys
  - model paths
  - private filesystem paths

### Important note on current public card export
Even `public_match_cards.csv` currently contains richer-than-public fields such as:
- `p_draw_spec`
- `p_draw_spec_thr`
- `draw_overlay_mode`
- `deploy_reason_codes`
- `deploy_markets`

That file is useful as a source, but not ready to ship directly to a public frontend.

## 7. Missing Scripts

### Missing against the master build spec
- `publish_predictions.py`
- `validate_public_export.py`
- `run_weekend_pipeline.py`
- `grade_weekend_results.py`

### Missing support docs requested in the spec
- `CODEX_RUNBOOK.md`
- `MODEL_OUTPUT_SCHEMA.md`
- `PUBLIC_EXPORT_POLICY.md`
- `FREE_PREMIUM_RULES.md`
- `DEPLOYMENT_RUNBOOK.md`
- `ERROR_HANDLING_RUNBOOK.md`
- `USER_FEEDBACK_TO_ISSUE.md`

### Missing web/deploy scaffolding
- repo-root frontend app
- frontend data directory such as `frontend/public/data/`
- GitHub Actions workflows
- Cloudflare Pages config
- Stripe integration surface

## 8. Missing Tests

### Current visible test posture
- The repo contains some Python tests, but not the web publishing safety checks needed for this product surface.

### Missing test coverage for the website/export plan
- export schema tests for public JSON
- export schema tests for premium JSON
- redaction leak tests
- NaN/inf validation tests
- fixture identity contract tests
- regression tests for publish field mapping from deploy CSV
- frontend build test
- frontend component smoke test
- CI checks for export safety

## 9. Recommended First Patches

### Patch 1
Create `PUBLIC_EXPORT_POLICY.md`.
- Define exact public-safe and premium-safe field allowlists.
- Record forbidden field classes.
- Make this the contract for `publish_predictions.py`.

### Patch 2
Create `MODEL_OUTPUT_SCHEMA.md`.
- Document how live deploy CSV columns map into:
  - public website JSON
  - premium website JSON
  - results JSON

### Patch 3
Build `publish_predictions.py`.
- Read the latest routed deploy CSV.
- Emit:
  - `frontend/public/data/public_predictions.json`
  - `frontend/public/data/premium_predictions.json`
  - `frontend/public/data/publish_summary.json`
  - `reports/latest/PUBLISH_REPORT.md`
- Use strict column allowlists.

### Patch 4
Build `validate_public_export.py`.
- Fail closed on:
  - forbidden columns
  - key/path leakage
  - NaN/inf
  - missing required fields

### Patch 5
Create a minimal frontend scaffold.
- Start with a static/mobile-friendly prediction board.
- Consume mock JSON first, then real published JSON.
- Keep it isolated from production model logic.

### Patch 6
Add GitHub Actions.
- `test-backend.yml`
- `test-frontend.yml`
- `validate-public-export.yml`
- `deploy-preview.yml`

### Patch 7
Add Cloudflare Pages config after frontend build exists.
- `dev` branch for preview
- `main` branch for production

## 10. GitHub And Cloudflare Audit Notes

### GitHub
- Git is present locally, but repo-root GitHub Actions scaffolding is not.
- The next safe integration step is file-based:
  - create workflows
  - enforce export validation before deploy

### Cloudflare
- No Cloudflare Pages config was found in the repo.
- No `wrangler.toml` was found.
- No frontend app exists yet to deploy.

### Important session note
- GitHub capabilities are available in this session.
- A Cloudflare plugin/tool surface was referenced in the build note, but it is not available in the current tool list.
- That does not block local scaffolding; it only means Cloudflare-specific remote operations should be treated as a later wiring step unless the tool surface changes.

## 11. Overall Read

- The backend prediction engine and routing spine are already substantial.
- The web product layer is still mostly a specification plus reporting artifacts.
- The cleanest path is not to expose current CSVs directly.
- The cleanest path is:
  1. define export policy
  2. build strict publish/validate scripts
  3. add a thin frontend over redacted JSON
  4. wire GitHub Actions
  5. connect Cloudflare Pages last

## 12. Recommended Next Task

Task 2 from the master build spec is the right next move:
- build `publish_predictions.py`
- build `validate_public_export.py`
- do not alter model logic
- use latest routed deploy CSV as the publishing source
