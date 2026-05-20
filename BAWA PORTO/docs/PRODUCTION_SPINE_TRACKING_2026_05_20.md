# Production Spine Tracking Baseline - 2026-05-20

This baseline records the protected football prediction spine that must be tracked before the next fresh data/model run.

Scope: source-control protection only. No behavioral edits were made to the files below as part of this tracking pass.

## Files Added To Tracking

| File | Purpose | SHA-256 |
| --- | --- | --- |
| `footystats_drop_ingest.py` | FootyStats drop ingest entrypoint | `80e60bf4b5a5a2b21a4111c2021a5569ba472d87ead8db7fe6dcc61d1d3ebb6b` |
| `etl_press_intensity.py` | Press/intensity ETL | `2358b1f7add2055aca01714da42baab3c211905d6fcf91cc17b28d7b2e212184` |
| `build_merged.py` | Canonical merged match build | `ccc24b229ed3512c0fdbd94a03b9edc2f66d50964e2b046019793d4f6e8dbc17` |
| `patch_merge_add_streaks.py` | Streak enrichment patch | `5d83e77ffdaa5cfa33a72405990b0dc4f3910b158eddae75ba52508a654cbd40` |
| `team_ratings.py` | Team rating generation | `73ebe6eb5b86d8d74ba31f6cec15b13fb83a52a7b06f513e317bf7bf9bdb0498` |
| `patch_merge_add_power_ratings.py` | Power rating merge patch | `f14a446525c93ed3bdf4a1fe905256180ba61187d29dae862a75deda1098bc3f` |
| `make_fd_odds_enriched_synth.py` | Synth odds enrichment | `c6aa65c1f175f49e5b628ac81929ab195c4b239c57feffc47f15aa742f630c7d` |
| `patch_merge_add_synth_odds.py` | Synth odds merge patch | `b554943782a233be91d18b2f29c9ac73f577222f47efe68349601686aafce12d` |
| `pipeline_qa_gate.py` | Pipeline integrity gate | `c97f0bcc0d37984219f776710724d14f421d956c311b00468c2a6a56c25200d5` |
| `bookie_allmarkets.py` | All-markets prediction generation | `383065dc40e2875eb9d42621e7ce61c35484df1ed64f69b20dbc8776d1ed3061` |
| `deploy_rulebook.py` | Live routing, tiers, vetoes, and gates | `e31b3e12aefcb1cea70aba219902149ed2c5a6eec7ee0c5d9d8a3994634acb82` |
| `slip_formatter.py` | Thin deploy/slip formatter | `00341bf2b4e253d70791a5c738f2487bb9e28ea5a08e2898c549b90c0c12cd41` |

## Guardrail

Future changes to these files should be intentional, reviewed as production-pipeline changes, and summarized separately from website/data-publishing work.

Before live prediction runs, confirm:

1. These files are tracked and clean unless the run explicitly depends on an approved change.
2. Data integrity has passed before running `bookie_allmarkets.py`.
3. `deploy_rulebook.py` remains the single source of truth for routing, tiers, vetoes, and gates.
4. `slip_formatter.py` remains formatting/orchestration only, with no rescue or filler logic.
