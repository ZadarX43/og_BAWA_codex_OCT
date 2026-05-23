# FootyStats Ingest

This utility was built to remove the repetitive manual task of taking newly downloaded FootyStats CSV exports and placing them into the correct league folders inside the Odds Genius repo.

It supports a drop-folder workflow and a macOS app workflow.

---

## What it does

The system is made of two parts:

1. A Python script:
   - `footystats_drop_ingest.py`

2. A macOS AppleScript app:
   - `FootyStats Ingest`

The workflow is:

- download FootyStats CSV files
- either:
  - drop them into `~/Desktop/FOOTYSTATS_DROP`, then run the app
  - or drag the files directly onto the app
- the script validates the filenames
- it routes each file to the correct repo folder
- it only replaces the matching latest raw FootyStats export
- it leaves merged, enriched, synth, and custom files alone
- it writes a summary file
- the app shows a popup summary

---

## Repo paths used

### Repo root
`/Users/hughwade/Documents/Code/OG_master/BAWA PORTO`

### Drop folder
`/Users/hughwade/Desktop/FOOTYSTATS_DROP`

### Processed folder
`/Users/hughwade/Desktop/FOOTYSTATS_DROP/_processed`

### Summary file
`/Users/hughwade/Desktop/FOOTYSTATS_DROP/_last_run_summary.txt`

---

## Python script

### File
`footystats_drop_ingest.py`

### Purpose
The Python script performs the real ingest logic.

It:

- scans the drop folder for `.csv` files
- validates that filenames match approved FootyStats raw export patterns
- restricts processing to the currently approved league families configured in `LEAGUE_MAP`
- restricts processing to the latest season/year rules
- identifies `matches`, `teams`, or `players`
- finds the correct destination league folder
- finds the newest matching raw export already in that folder
- backs it up into `_replaced`
- replaces only that matching raw export
- archives the source file into `_processed`
- writes a summary file for the app popup

---

## Important safety behavior

This tool is intentionally designed to avoid damaging the repo.

It does **not** replace every CSV in a destination folder.

It only replaces the matching raw FootyStats export for the same:

- league family
- file type (`matches`, `teams`, `players`)

It ignores unrelated files such as:

- `.DS_Store`
- `*_all_seasons.csv`
- `fd_odds_enriched.csv`
- `fd_odds_enriched_synth.csv`
- `fd_ou25_novig.csv`
- `__TRAIN_MULTISEASON_completed.csv`
- `__merged__` files
- `Upcoming Fixtures` files
- `.bak` files
- `.json` files
- custom/manual CSVs

---

## Latest season rules

The current rules are:

### Multi-season leagues
Only accept:

- `2025-to-2026`

### Single-year leagues
Only accept:

- `2026-to-2026`

This means that if a file is older, it is skipped.

### Current single-year leagues
- Japan J1
- Norway Eliteserien
- Brazil Serie A
- USA MLS
- South Korea K League

If these rules need changing later, edit these values in the Python script:

```python
LATEST_MULTI_SEASON_START_YEAR = 2025
LATEST_MULTI_SEASON_END_YEAR = 2026
LATEST_SINGLE_YEAR = 2026
```

---

## Approved leagues

The current approved league slug mapping is:

```python
LEAGUE_MAP = {
    "england-premier-league": "England Premier League",
    "england-championship": "England Championship",
    "england-efl-league-one": "England EFL League 1",
    "england-fa-cup": "England FA Cup",
    "japan-j1-league": "Japan J1",
    "norway-eliteserien": "Norway Eliteserien",
    "netherlands-eredivisie": "Netherlands Eredivisie",
    "belgium-pro-league": "Belgium Pro",
    "scotland-premiership": "Scotland Premiership",
    "brazil-serie-a": "Brazil Serie A",
    "usa-mls": "USA MLS",
    "portugal-liga-nos": "Portugal Liga",
    "spain-la-liga": "Spain La Liga",
    "italy-serie-a": "Italy Serie A",
    "france-ligue-1": "France Ligue 1",
    "germany-bundesliga": "Germany Bundesliga",
    "europe-uefa-europa-conference-league": "Europa Conference",
    "europe-uefa-europa-league": "Europa League",
    "europe-uefa-champions-league": "Champions League",
    "australia-a-league": "Australia A-League",
    "austria-bundesliga": "Austria Bundesliga",
    "denmark-superliga": "Denmark Superliga",
    "switzerland-super-league": "Swiss Super League",
    "sweden-allsvenskan": "Sweden Allsvenskan",
    "germany-2-bundesliga": "Germany Bundesliga 2",
    "czech-republic-first-league": "Czech First League",
    "south-korea-k-league-1": "South Korea K League",
    "saudi-arabia-pro-league": "Saudi Pro League",
    "saudi-arabia-professional-league": "Saudi Pro League",
    "turkey-super-lig": "Turkey Super Lig",
}
```

### Active single-year slug set

```python
APPROVED_SINGLE_YEAR_SLUGS = {
    "japan-j1-league",
    "norway-eliteserien",
    "brazil-serie-a",
    "usa-mls",
    "south-korea-k-league-1",
}
```

### Current note on Sweden
`sweden-allsvenskan` exists in `LEAGUE_MAP`, but it is not currently active for ingest because the current-season single-year files are not yet being used.

If a new league is added later:

1. add the slug to `LEAGUE_MAP`
2. make sure the repo folders exist under:
   - `Matches/<League Folder>`
   - `Teams/<League Folder>`
   - `Players/<League Folder>`
3. decide whether it belongs in `APPROVED_SINGLE_YEAR_SLUGS`
4. make sure the latest-season rule matches the actual FootyStats season format

---

## How matching works

The script only accepts raw FootyStats files matching this pattern:

- `<slug>-matches-YYYY-to-YYYY-stats.csv`
- `<slug>-teams-YYYY-to-YYYY-stats.csv`
- `<slug>-players-YYYY-to-YYYY-stats.csv`

It also accepts duplicate download suffixes like:

- ` (1)`
- ` (2)`
- ` (3)`

Example:

- `england-premier-league-matches-2025-to-2026-stats (2).csv`

That gets cleaned to:

- `england-premier-league-matches-2025-to-2026-stats.csv`

---

## Backups and processed files

### Replaced files

When a matching raw export is replaced, the old file is backed up into:

- `<destination folder>/_replaced/`

### Processed drop-folder files

Incoming files that are successfully handled are moved into:

- `~/Desktop/FOOTYSTATS_DROP/_processed`

---

## AppleScript app

### App name

`FootyStats Ingest`

### What it does

The app can be used in two ways:

1. Double-click the app
   - opens the drop folder
   - runs the ingest
   - shows a summary popup

2. Drag files onto the app
   - copies dropped files into the drop folder
   - runs the ingest
   - shows a summary popup

### Popup buttons

The popup currently has:

- Open Drop Folder
- Open Processed Folder
- Close

---

## Current AppleScript source

The app was built from this AppleScript:

```applescript
on run
	my runIngest({})
end run

on open droppedItems
	my runIngest(droppedItems)
end open

on runIngest(droppedItems)
	set repoPath to "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO"
	set dropFolder to "/Users/hughwade/Desktop/FOOTYSTATS_DROP"
	set processedFolder to "/Users/hughwade/Desktop/FOOTYSTATS_DROP/_processed"
	set summaryFile to "/Users/hughwade/Desktop/FOOTYSTATS_DROP/_last_run_summary.txt"
	
	do shell script "mkdir -p " & quoted form of dropFolder
	do shell script "mkdir -p " & quoted form of processedFolder
	
	if (count of droppedItems) > 0 then
		repeat with anItem in droppedItems
			set itemPath to POSIX path of anItem
			try
				do shell script "cp -f " & quoted form of itemPath & " " & quoted form of dropFolder
			on error
				-- Ignore non-file drops or copy failures.
			end try
		end repeat
	else
		do shell script "open " & quoted form of dropFolder
	end if
	
	set csvCount to do shell script "find " & quoted form of dropFolder & " -maxdepth 1 -type f -name '*.csv' | wc -l | tr -d ' '"
	
	try
		do shell script "cd " & quoted form of repoPath & " && " & ¬
			"if [ -x .venv/bin/python ]; then .venv/bin/python footystats_drop_ingest.py; else python3 footystats_drop_ingest.py; fi"
		set runStatus to "success"
	on error errMsg number errNum
		set runStatus to "error"
		set runErrorText to "Script error " & errNum & ":" & return & errMsg
	end try
	
	try
		set summaryText to do shell script "cat " & quoted form of summaryFile
	on error
		set summaryText to "No summary file was created."
	end try
	
	if runStatus is "error" then
		set summaryText to "CSV files currently in drop folder: " & csvCount & return & return & runErrorText & return & return & summaryText
	else
		set summaryText to "CSV files currently in drop folder: " & csvCount & return & return & summaryText
	end if
	
	set userChoice to button returned of (display dialog summaryText with title "FootyStats Ingest Summary" buttons {"Open Drop Folder", "Open Processed Folder", "Close"} default button "Close")
	
	if userChoice is "Open Drop Folder" then
		do shell script "open " & quoted form of dropFolder
	else if userChoice is "Open Processed Folder" then
		do shell script "open " & quoted form of processedFolder
	end if
end runIngest
```

---

## How to edit the AppleScript app in future

### Best practice

Save both:

- the compiled `.app`
- a source `.scpt` file

For example:

- `FootyStats Ingest.app`
- `FootyStats Ingest.scpt`

That way the source can be reopened and edited later without rebuilding from scratch.

### To edit it

1. open Script Editor
2. open the `.scpt` source file
3. make changes
4. save
5. export or save again as an Application if needed

---

## How to change the app icon

The easiest macOS method:

1. get a square football PNG or image
2. open it in Preview
3. press Cmd + A
4. press Cmd + C
5. select the app in Finder
6. press Cmd + I
7. click the small icon at top-left of the info window
8. press Cmd + V

That pastes the image as the app icon.

---

## How to edit the Python ingest script later

### Main file

- `footystats_drop_ingest.py`

### Common things you may want to change

#### 1. Change season rules

Edit:

```python
LATEST_MULTI_SEASON_START_YEAR = 2025
LATEST_MULTI_SEASON_END_YEAR = 2026
LATEST_SINGLE_YEAR = 2026
```

#### 2. Add a new league

Edit:

```python
LEAGUE_MAP = {...}
```

#### 3. Change repo or Desktop paths

Edit:

```python
REPO_ROOT = Path(...)
DROP_FOLDER = Path(...)
ARCHIVE_FOLDER = DROP_FOLDER / "_processed"
```

#### 4. Turn off opening the drop folder on launch

Edit:

```python
OPEN_DROP_FOLDER_ON_START = True
```

#### 5. Turn off sounds

Edit:

```python
PLAY_SOUND_ON_COMPLETE = True
```

#### 6. Turn off color output

Edit:

```python
ENABLE_COLOR_OUTPUT = True
```

---

## Suggested future improvements

Possible next upgrades:

- ignore non-CSV drops explicitly and report how many were accepted
- add a popup line for:
  - accepted files
  - skipped files
  - errors
- add Open Replaced Folder button
- create a small standalone GUI app
- add logging to a timestamped log file
- add a dry-run mode toggle in the app
- allow per-league latest-year rules for single-year leagues
- add duplicate detection before copying dropped files

---

## Current status

The tool is now functioning as:

- a drop-folder ingester
- a drag-and-drop macOS app
- a safe raw-export replacer for approved leagues only
- a time-saving utility for maintaining the FootyStats portion of the Odds Genius data pipeline