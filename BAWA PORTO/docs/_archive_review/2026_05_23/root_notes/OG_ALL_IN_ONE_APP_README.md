# OG All‑In‑One Pipeline App (Desktop)

## Purpose
A semi‑interactive desktop workflow that chains:

1. FootyStats ingest
2. Merged + Under 2.5 dashboard (for coverage/QA review)
3. Picks pipeline (bookie_allmarkets → deploy_rulebook → slip_formatter)
4. Quick navigation to slips / ranked boards / top‑10 files

This replaces manual terminal runs with a guided, clickable flow.

---

## What The App Does (High‑Level)

**Step 1 — Ingest**
- Runs `footystats_drop_ingest.py` against the drop folder
- Moves approved FootyStats CSV files into the correct league folders
- Writes a summary + log

**Step 2 — Dashboard**
- Launches `merged_pipeline_dashboard.py`
- Lets you verify: merged status, Under 2.5, synth, dedupe

**Step 3 — Pipeline (optional)**
- Lets you run picks immediately (or later):
  - `bookie_allmarkets.py`
  - `deploy_rulebook.py`
  - `slip_formatter.py`

**Step 4 — Outputs & Navigation**
- Opens slips, ranked boards, top‑10 EV/odds, or deploy outputs

---

## App Location

`OG All‑In‑One Pipeline.app` (Desktop)

Compiled from:

`apps/OG_All_In_One_Pipeline.applescript`

---

## Repo Paths Used

- Repo root:
  - ` /Users/hughwade/Documents/Code/OG_master/BAWA PORTO `

- Drop folder:
  - ` /Users/hughwade/Desktop/FOOTYSTATS_DROP `

- Processed folder:
  - ` /Users/hughwade/Desktop/FOOTYSTATS_DROP/_processed `

- Ingest summary:
  - ` /Users/hughwade/Desktop/FOOTYSTATS_DROP/_last_run_summary.txt `

- Ingest log:
  - ` /tmp/og_footystats_ingest.log `

- Dashboard log:
  - ` /tmp/og_merged_dashboard.log `

- Pipeline UI log:
  - ` /tmp/og_picks_pipeline_ui.log `

- Predictions output:
  - ` predictions_output/YYYY-MM-DD/ `

---

## Files / Scripts Used

### Ingest
- `footystats_drop_ingest.py`

### Dashboard
- `merged_pipeline_dashboard.py`
- `rebuild_all_merged.sh` (invoked by dashboard)

### Picks pipeline
- `bookie_allmarkets.py`
- `deploy_rulebook.py`
- `slip_formatter.py`

### App launcher script
- `apps/OG_All_In_One_Pipeline.applescript`

---

## Files Created By The App

When pipeline runs:

```
predictions_output/YYYY-MM-DD/
  BOOKIE_IMP20_ALLMARKETS_<date_from>_to_<date_to>.csv
  BOOKIE_IMP20_ALLMARKETS_<date_from>_to_<date_to>__DEPLOY_TIER_ELITE__PRESET_V1__FTR_accuracy.csv
  BOOKIE_IMP20_ALLMARKETS_<date_from>_to_<date_to>__DEPLOY_TIER_STANDARD__PRESET_V1__FTR_accuracy.csv
  slips/
    ranked_board_*.csv
    ranked_board_ftr_*.csv
    ranked_board_btts_*.csv
    ranked_board_ou25_*.csv
    slips_top10_by_ev_*.csv
    slips_top10_by_odds_*.csv
    slips_singles_*.csv
    slips_doubles_*.csv
    slips_trebles_*.csv
    slips_acca*.csv
    slips_acca_mixed_*.csv
    ranked_board_family_summary_*.csv
```

---

## How The App Works (User Flow)

1. **Launch app**
2. **Ingest runs**
3. **Dashboard opens**
4. **Choose next action**:
   - Run picks now
   - Run last date range
   - Open pipeline UI
   - Open last slips / deploy outputs / ranked boards
   - Open view selector

If running picks:
- Select preset (This Weekend / Next 3 Days / Custom)
- App runs pipeline and shows summary
- App prompts to open slips / outputs

---

## View Selector (Slip Outputs)

The View Selector can open the most recent file for:

- Ranked Board (All)
- Ranked Board (FTR / BTTS / OU25)
- Top‑10 by EV
- Top‑10 by Odds
- Singles / Doubles / Trebles
- Acca 6 / 8 / 10 / 12 / 14
- Acca Mixed 8 / 10 / 12 / 14
- Family Summary

---

## Troubleshooting

**App won’t compile**
- Open the `.applescript` directly in Script Editor (don’t paste)
- Ensure format is **AppleScript** (not JavaScript)

**Ingest fails**
- Check: `/tmp/og_footystats_ingest.log`

**Dashboard doesn’t open**
- Check: `/tmp/og_merged_dashboard.log`

**Pipeline doesn’t run**
- Verify `.venv/bin/python` exists
- Ensure `bookie_allmarkets.py` output exists

---

## Expansion: Standalone Desktop App (Roadmap)

To turn this into a true standalone product (no scripts / no Script Editor), you would need:

### 1. UI Layer
- Build with PyQt / Electron / Tauri / Swift UI
- Date pickers, file selectors, pipelines, progress bar

### 2. Embedded Pipeline Engine
- Bundle Python runtime + models
- Package `.venv` or use PyInstaller

### 3. File Management UI
- “Open Files” / drag‑drop ingest
- Preview + validation + warnings

### 4. Outputs UI
- Render ranked boards in‑app
- Filter by market / league / confidence
- Export to CSV / copy to clipboard

### 5. Persistence & Config
- Save last date ranges, output folders, league lists
- Admin config for threshold overrides

### 6. Web/Webapp Integration
- API endpoints for slips + ranked boards
- Auth + user management
- Hosting for dashboard + live picks

---

## Notes

This app is intentionally semi‑interactive so you can:
- Confirm ingest success
- Verify merge health
- Review coverage
- Then run picks confidently

It provides stability now while giving a clean path to a full standalone product later.
