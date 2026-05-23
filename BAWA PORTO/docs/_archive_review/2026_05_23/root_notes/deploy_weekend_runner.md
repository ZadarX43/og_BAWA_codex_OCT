Nice. Here’s a paired runbook you can save as deploy_weekend_runner.md and keep next to the script.

# deploy_weekend_runner.md

## OG Weekend Deploy Runner — Quick Runbook

Use this when tired / stressed / in a rush before a weekend test deploy.

This runbook matches:

- `deploy_weekend_runner.sh`

It chains:

1. `bookie_allmarkets.py`
2. `deploy_rulebook.py --debug` (saved to log)
3. `tier_audit.py`
4. prints summary (tier counts + promoted demoted rows in STANDARD)

---

## 0) One-time setup

Make the runner executable (only needed once):

```bash
chmod +x deploy_weekend_runner.sh


⸻

1) Standard run (build ALLMARKETS + deploy + audit)

Use this for a fresh weekend run.

bash deploy_weekend_runner.sh \
  --date-from 2026-02-20 \
  --date-to 2026-02-23 \
  --run-date 2026-02-20

What it does
	•	builds BOOKIE_IMP*_ALLMARKETS_*.csv
	•	runs rulebook with --debug
	•	writes tier files:
	•	__DEPLOY_TIER_ELITE.csv
	•	__DEPLOY_TIER_STANDARD.csv
	•	__DEPLOY_TIER_OBSERVE.csv
	•	runs tier_audit.py
	•	prints promoted demoted rows (e.g. DEMOTED_ULTRASHORT_STANDARD, DEMOTED_MARGIN_STANDARD)

⸻

2) Re-run deploy + audit only (skip ALLMARKETS rebuild)

Use this if ALLMARKETS already exists and you’re just testing rule changes.

bash deploy_weekend_runner.sh \
  --skip-bookie \
  --src-csv "predictions_output/2026-02-20/BOOKIE_IMP62_ALLMARKETS_2026-02-20_to_2026-02-23.csv" \
  --date-from 2026-02-20 \
  --date-to 2026-02-23 \
  --run-date 2026-02-20


⸻

3) Pass extra rulebook flags (optional)

If you want to force a preset or test a toggle:

bash deploy_weekend_runner.sh \
  --date-from 2026-02-20 \
  --date-to 2026-02-23 \
  --run-date 2026-02-20 \
  --extra-rulebook-args "--preset V1"

You can also pass multiple flags inside the quotes.

⸻

4) Where outputs go

Base folder:
	•	predictions_output/<run-date>/

Example:
	•	predictions_output/2026-02-20/

Key outputs:
	•	ALLMARKETS CSV
	•	deploy outputs (__DEPLOY_PRESET_V1.csv, .md)
	•	tier CSVs (ELITE, STANDARD, OBSERVE)
	•	rulebook debug log (deploy_rulebook_debug_...log)
	•	tier audit folder (tier_audit_.../)

⸻

5) What to check after each run (minimum checks)

A) Rulebook completed successfully

Look for:
	•	✅ deploy_rulebook.py completed
	•	tier files written
	•	no traceback / crash

B) Tier audit passes

You want:
	•	No cross-tier duplicates on key columns
	•	total tier rows = runtime pass rows (or expected final count after tiering)

C) Promoted rows appear in STANDARD (if expected)

Look for rows containing:
	•	DEMOTED_ULTRASHORT_STANDARD
	•	DEMOTED_MARGIN_STANDARD

These are the demoted FTR rows promoted back into STANDARD via:
	•	Lane 1 = ultra-short favourites
	•	Lane 2 = margin + acceptable draw-chaos

⸻

6) Fast sanity expectations (based on recent test pattern)

These are sanity patterns, not hard guarantees:
	•	deploy_rulebook prints market-by-market gate counts
	•	tier split prints something like:
	•	ELITE = ...
	•	STANDARD = ...
	•	OBSERVE = ...
	•	promoted rows in STANDARD may include strong short FTR favourites
	•	no duplicate fixture-market-pick keys across tiers

⸻

7) Common problems (and fixes)

Problem: “Could not find ALLMARKETS CSV”

Cause: bookie_allmarkets.py didn’t write output where expected, or file name differs.

Fixes:
	•	check predictions_output/<run-date>/
	•	re-run with --skip-bookie --src-csv <exact path>
	•	confirm bookie_allmarkets.py date window and outdir

⸻

Problem: tier_audit.py not found

Cause: script not in current folder.

Fix:

bash deploy_weekend_runner.sh ... --tier-audit-script path/to/tier_audit.py


⸻

Problem: wrong Python env / import errors

Cause: not in project venv.

Fix:
	•	activate your venv first
	•	run from repo root (where scripts live)

Example:

source .venv/bin/activate
pwd
ls deploy_rulebook.py


⸻

8) “Half-asleep” copy/paste templates

Fresh weekend run

bash deploy_weekend_runner.sh \
  --date-from YYYY-MM-DD \
  --date-to YYYY-MM-DD \
  --run-date YYYY-MM-DD

Re-run on existing ALLMARKETS

bash deploy_weekend_runner.sh \
  --skip-bookie \
  --src-csv "predictions_output/YYYY-MM-DD/BOOKIE_IMPXX_ALLMARKETS_YYYY-MM-DD_to_YYYY-MM-DD.csv" \
  --date-from YYYY-MM-DD \
  --date-to YYYY-MM-DD \
  --run-date YYYY-MM-DD


⸻

9) Pre-weekend checklist (30 seconds)
	•	venv activated
	•	in repo root
	•	deploy_weekend_runner.sh exists
	•	tier_audit.py exists
	•	enough disk space in predictions_output/
	•	date window correct (Friday→Monday or whatever you want)
	•	coffee / water / breathing / go

⸻

10) Post-run notes to record (recommended)

Paste these into your run log / markdown notes:
	•	Date window run
	•	ALLMARKETS file path used
	•	Tier counts (ELITE / STANDARD / OBSERVE)
	•	Number of promoted demoted rows in STANDARD
	•	Any weird veto spikes in debug log
	•	Any leagues/markets that look suspicious for manual review

This makes weekend comparisons much easier.

