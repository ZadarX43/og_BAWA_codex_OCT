Use this for DASHBOARD_README.md:

# DASHBOARD_README.md

## Purpose

This document explains the small Mac utility app setup for the OG / BAWA desktop workflow.

It covers:

- the FootyStats ingest app
- the Merged + Under 2.5 dashboard app
- where each app points
- which Python / shell files power them
- where logs and reports are written
- how to re-save the apps if paths change
- how to rename or re-icon the apps later

This is the desktop control layer for the data-refresh and merged rebuild pipeline.

---

## Current utility app workflow

You now have a clickable, utility-style Mac workflow with no Terminal required.

### Current app stack

1. **FootyStats ingest app**
   - moves new FootyStats raw files into the correct league folders
   - archives processed files
   - provides popup feedback / summary

2. **Merged + Under 2.5 dashboard app**
   - runs the canonical merged rebuild pipeline
   - rebuilds all-seasons enriched files where needed
   - fits / refreshes ODDS synth tables
   - applies Under 2.5 synth
   - rebuilds merged files
   - writes QA + merge + dedupe reports
   - shows a native dashboard with league-by-league status

### Result

You now have:

- no Terminal needed
- stable merge / synth / QA / dedupe pipeline
- clickable Mac utility-style workflow
- visual status control for league rebuild health

---

## 1. FootyStats ingest app

### What it does

This app is the first step after downloading new FootyStats raw CSV files.

It is used to:

- read the FootyStats drop folder
- validate approved file naming
- move files into the correct `Matches/`, `Teams/`, and `Players/` league folders
- replace matching raw exports only
- archive processed files
- show a completion summary

### Main Python file

The ingest logic lives in:

```text
/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/footystats_drop_ingest.py

Launcher / app

You created a Mac app / launcher for this earlier.

That launcher points to the repo root and runs:

footystats_drop_ingest.py

Typical use
	1.	Download fresh FootyStats CSV files
	2.	Put them into the FootyStats drop folder
	3.	Click the FootyStats ingest app
	4.	Confirm files were processed
	5.	Then run the merged dashboard app

⸻

2. Merged + Under 2.5 dashboard app

What it does

This is the second app and acts as the merged pipeline control panel.

It:
	•	runs the merged rebuild engine
	•	reads the output reports
	•	displays a dashboard table for all leagues
	•	shows:
	•	merged status
	•	Under 2.5 presence
	•	synth presence
	•	dedupe status
	•	row / column counts
	•	duplicate row counts
	•	gives buttons to open logs, merged folder, merge report, and dedupe report

Main Python dashboard file

The dashboard logic lives in:

/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/merged_pipeline_dashboard.py

Main engine shell script

The dashboard runs this canonical shell script:

/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/rebuild_all_merged.sh

This shell script is the one true engine for the merged + Under 2.5 pipeline.

Saved Mac app

Current app name:

OG Merged + Under 2.5 Dashboard.app

This app was created in Script Editor as an Application.

⸻

3. What the dashboard app points to

Repo root

/Users/hughwade/Documents/Code/OG_master/BAWA PORTO

Preferred Python interpreter

/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/.venv/bin/python

Dashboard Python file

/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/merged_pipeline_dashboard.py

Shell pipeline engine

/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/rebuild_all_merged.sh


⸻

4. AppleScript launcher used for the dashboard app

The working AppleScript app launcher is:

set repoPath to "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO"
set pyPath to repoPath & "/.venv/bin/python"
set scriptPath to repoPath & "/merged_pipeline_dashboard.py"

try
	do shell script "test -f " & quoted form of scriptPath
on error
	display dialog "merged_pipeline_dashboard.py was not found at:" & return & scriptPath buttons {"Close"} default button "Close"
	return
end try

do shell script "cd " & quoted form of repoPath & " && if [ -x " & quoted form of pyPath & " ]; then nohup " & quoted form of pyPath & " " & quoted form of scriptPath & " >/tmp/og_merged_dashboard.log 2>&1 & else nohup python3 " & quoted form of scriptPath & " >/tmp/og_merged_dashboard.log 2>&1 & fi"

What this does
	•	checks the Python dashboard file exists
	•	changes into the repo folder
	•	prefers .venv/bin/python
	•	falls back to python3 if needed
	•	launches the dashboard without Terminal
	•	writes background startup output to:

/tmp/og_merged_dashboard.log


⸻

5. Important log and report locations

Dashboard launch log

/tmp/og_merged_dashboard.log

Use this if the app launches but nothing appears, or if the app fails to start.

Main logs folder

/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/ModelStore/logs

Key merged pipeline logs

Synth log

ModelStore/logs/rebuild_all_merged__synth_log.txt

Merge log

ModelStore/logs/rebuild_all_merged__merge_log.txt

QA report

ModelStore/logs/rebuild_all_merged__qa.csv

Stable merge report

ModelStore/logs/full_merged_baseline__merge_report.csv

Stable dedupe report

ModelStore/logs/full_merged_baseline__dedupe_report.csv

Merged output folder

/Users/hughwade/Documents/Code/OG_master/BAWA PORTO/Matches/__merged__


⸻

6. Canonical operating order

The desktop workflow should now be:

Step 1

Run the FootyStats ingest app

Step 2

Run the Merged + Under 2.5 dashboard app

Step 3

Click Run Pipeline inside the dashboard if you want a fresh rebuild

Step 4

Review the dashboard rows:
	•	Merged
	•	Under 2.5
	•	Synth
	•	Dedupe

Step 5

Use buttons if needed:
	•	Open Logs
	•	Open Merged Folder
	•	Open Merge Report
	•	Open Dedupe Report

⸻

7. How to re-save the dashboard app if paths change

If you move the repo folder, rename the repo, or move the Python file, the app path must be updated.

Steps
	1.	Open Script Editor
	2.	Open the source script used for the app
	3.	Update these variables:

set repoPath to "/new/path/to/BAWA PORTO"
set pyPath to repoPath & "/.venv/bin/python"
set scriptPath to repoPath & "/merged_pipeline_dashboard.py"

	4.	Save again as:
	•	File Format: Application
	5.	Overwrite the existing app

Most common reason for breakage

The app usually fails because one of these changed:
	•	repo folder path
	•	Python virtualenv path
	•	dashboard Python filename

⸻

8. How to rename the app later

To rename the app:
	1.	In Finder, select the .app
	2.	Press Enter
	3.	Rename it like any other Mac app

Example names:
	•	OG Merged Dashboard.app
	•	OG Merge Control.app
	•	OG Data + Merge Dashboard.app

Renaming the .app file does not break the internal launcher, as long as the script paths remain correct.

⸻

9. How to change the app icon later

To change the app icon on macOS:
	1.	Find a PNG or ICNS icon you want
	2.	Open it in Preview
	3.	Copy it
	4.	Right-click the .app
	5.	Click Get Info
	6.	Click the small app icon in the top-left of the Info window
	7.	Paste

Suggested icons

For the merged dashboard app:
	•	football
	•	data grid
	•	dashboard meter
	•	green control panel icon

For the FootyStats ingest app:
	•	football
	•	inbox / import icon
	•	folder with arrow icon

⸻

10. What files power the whole system

Ingest side

footystats_drop_ingest.py

Merge / synth engine

rebuild_all_merged.sh

Merge builder

build_merged.py

Under 2.5 synthesis

odds_synth.py

Dashboard UI

merged_pipeline_dashboard.py


⸻

11. What “healthy” looks like in the dashboard

A clean healthy dashboard should show:
	•	all target leagues present
	•	Merged = ✓
	•	Under 2.5 = ✓
	•	Synth = ✓
	•	Dedupe = ✓
	•	duplicate rows = 0

Top status should show:
	•	GREEN
	•	total leagues correct
	•	Under 2.5 present across target leagues
	•	duplicate leagues = 0

⸻

12. Quick fault-finding guide

App says file not found

Check:
	•	repo path
	•	merged_pipeline_dashboard.py still exists
	•	.venv/bin/python still exists

App opens but no window appears

Check:

cat /tmp/og_merged_dashboard.log

Dashboard opens but reports look stale

Run pipeline again using:
	•	dashboard Run Pipeline button

Or inspect:
	•	ModelStore/logs/full_merged_baseline__merge_report.csv
	•	ModelStore/logs/full_merged_baseline__dedupe_report.csv

Buttons work but pipeline fails

Open:
	•	Logs
	•	synth log
	•	merge log
	•	QA report

⸻

13. Recommended future upgrades

Possible future polish:
	•	wrap the dashboard more formally with py2app or similar
	•	bundle a custom dock icon
	•	add a last-run timestamp
	•	add per-league warning reasons in the table
	•	add auto-refresh after pipeline completion
	•	add clickable row details panel
	•	add a second tab for ingest health
	•	combine ingest + merge into one unified desktop control center

⸻

14. Current stable baseline

This desktop control setup now gives you:
	•	FootyStats ingest app
	•	merged + Under 2.5 dashboard app
	•	no Terminal needed
	•	stable merge, synth, QA, dedupe pipeline
	•	clickable utility-style workflow

This is now the current clean Mac desktop operating layer for OG / BAWA data refresh and merged rebuild control.

