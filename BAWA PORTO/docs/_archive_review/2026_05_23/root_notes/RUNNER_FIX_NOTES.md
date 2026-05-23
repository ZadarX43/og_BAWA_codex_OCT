# Runner Fix Notes

This note captures the root cause and fix for the walk-forward runner artifact handoff issue.

**Status:** Fixed

**Impact:** The runner could combine and score the wrong deploy artifacts (candidate files) instead of the true tiered outputs, creating the appearance of broken model performance.

**Root Cause Summary**

deploy_rulebook.py writes tier files beside the `--src` file using `Path(args.src).with_name(...)`. That means tier outputs land in the source folder (e.g. `01_source/`), not only in `02_deploy/`. The runner was resolving deploy artifacts in a way that could prefer candidate files or look in the wrong place, breaking the handoff.

**What Changed**

1. Source identity is preserved.
2. The exact native source filename produced by bookie_allmarkets.py is resolved and copied into `01_source/` without renaming.
3. deploy_rulebook.py is executed with `--src` pointing to that exact file.
4. Tier file resolution now looks in the correct locations and order.
5. Candidate files are only used as a last resort.

**Correct Artifact Flow**

1. bookie_allmarkets.py produces the native source CSV for the window.
2. The runner resolves the exact native filename and copies it into `01_source/`.
3. deploy_rulebook.py runs against the copied source.
4. deploy_rulebook.py writes tier files beside the source in `01_source/`.
5. The runner resolves ELITE, STANDARD, and OBSERVE tier files first.
6. The runner combines tier files and scores the combined output.

**Resolution Order (Deploy CSVs)**

1. Exact tier files in `02_deploy/`.
2. Exact tier files in `01_source/`.
3. Rewritten tier files without the exact date token.
4. Recursive/native tier fallback.
5. Candidate files only if no tier files exist.

**Why This Matters**

Candidate files (`DEPLOY_CANDIDATES_AFTER_GATES.csv`, `DEPLOY_CANDIDATES_RAW.csv`) are useful for inspection but they are not the authoritative deploy outputs. The tier files are the real signal routing output and must be used for combining and scoring.

**Runner Output Structure (Per Window)**

1. `01_source/` holds the native source CSV and the tier files written by deploy_rulebook.py.
2. `02_deploy/` holds preset outputs, candidate CSVs, and the combined deploy CSV.
3. `03_scored/` holds scored combined outputs.
4. `logs/` holds the source and deploy logs.

**Self-Check Guardrail**

A self-check console report is now printed per window, summarizing:

1. Source file used.
2. Tier files resolved.
3. Combined file composition.
4. Scored rows vs. deploy rows.

This prevents silent regressions in file resolution and handoff logic.
