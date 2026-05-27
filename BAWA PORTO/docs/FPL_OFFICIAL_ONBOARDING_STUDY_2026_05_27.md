# FPL Official Onboarding Study

Date: 2026-05-27

Purpose: capture the design and UX lessons from the official Fantasy Premier League onboarding and squad-selection screens, then translate them into the Odds Genius Fantasy Intelligence product.

## Core Read

Official FPL feels calm because every screen has one obvious job.

It does not ask a new user to understand models, rules, chips, transfers, captaincy, benching, and strategy all at once. It breaks the journey into simple surfaces:

- choose interests
- follow clubs
- follow players
- enter Fantasy
- pick squad
- switch pitch/list view
- search/filter
- add players
- finish

Odds Genius should follow the same pacing. The difference is that OG is not just squad management. OG is a decision assistant. So our equivalent loop should be:

1. Import squad.
2. See the recommended move.
3. Review squad on pitch.
4. Search/compare transfer.
5. Save plan.
6. Come back before deadline.

## Screenshot Lessons

### 1. Squad Selection - List View

What works:

- Two-column layout is instantly understandable: player pool on the left, selected squad on the right.
- The user can see the hard constraints immediately: players selected and bank.
- Search is large and obvious.
- Filters are compact and secondary.
- Player rows are dense but readable.
- Add action is a simple plus button.
- Empty squad slots use plain labels: Select Goalkeeper, Select Defender, etc.

OG translation:

- Our Fantasy page should have a dedicated squad-builder mode, separate from the decision dashboard.
- Left side: searchable player pool.
- Right side: current squad or draft.
- Keep visible counters: 0/15 or 15/15, bank, free transfers, projected points, squad health.
- Use a single primary action per row: Add, Transfer in, or Compare.

### 2. Squad Selection - Pitch View

What works:

- The pitch is the emotional anchor.
- Empty slots are clear and inviting.
- Position labels are not buried in copy.
- Pitch/List toggle makes the user feel in control.
- Bottom actions are simple: Auto Pick, Reset, Enter Squad.

OG translation:

- Our pitch view should be visually stronger and simpler.
- It should have two modes: Pitch View and List View.
- Empty or incomplete states must use position placeholders.
- The pitch should not compete with five other panels on first view.
- Primary actions should sit under the pitch: Auto build, Reset draft, Save plan.

### 3. myPL Overlay

What works:

- The page darkens behind the overlay.
- User is shown four choices only.
- Copy is short and confident.
- The action is reassuring: Got it.
- It teaches a concept without making the user read a guide.

OG translation:

- Use a first-run onboarding overlay for Fantasy.
- Four options:
  - My Squad
  - My Transfers
  - My Watchlist
  - My Plans
- Copy should say: "Odds Genius helps you decide what to do before the deadline."
- Keep this optional and dismissible.

### 4. Fantasy Home

What works:

- The logged-in state is visible.
- Three large cards explain the product before the app asks for action.
- Latest content sits below, not above the core Fantasy entry points.
- The page feels like a hub, not a spreadsheet.

OG translation:

- Add a Fantasy hub before the full command centre.
- Three cards:
  - Import My Squad
  - Get This Week's Move
  - Build Transfer Plan
- Supporting content should sit below: scout-style notes, injury alerts, price-watch, deadline explainers.

### 5. Preferences / Interests

What works:

- Split layout: form on left, inspirational panel on right.
- Large readable headings.
- Toggle rows have huge tap targets.
- Finish button is always clear.
- The user is not overloaded with microcopy.

OG translation:

- Strategy setup should use this pattern.
- Ask only a few things:
  - Strategy mode
  - Willing to take hits?
  - Favourite clubs
  - Players to protect
  - Alert preference
- Avoid showing advanced FPL rules during onboarding.

### 6. My Players

What works:

- Search is huge.
- Player list is clean: image, name, club, follow.
- One action per row.
- Right-side panel keeps the purpose visible.
- Next button stays obvious.

OG translation:

- Watchlist management should look like this.
- Player search results should not show backend tokens.
- Each row should show:
  - player image/crest if available
  - name
  - club / position
  - simple OG label
  - one primary button: Watch / Transfer in / Compare

### 7. My Clubs

What works:

- Club logos make the list immediately scannable.
- Follow buttons are consistent.
- Disabled or already-handled rows are visually softened.
- Next button is fixed and wide.

OG translation:

- Club preferences should feed fixture/news/player relevance.
- Favourite clubs can later personalize:
  - Fantasy alerts
  - match feed
  - Telegram notifications
  - player watchlists

## Main OG Product Implications

The current OG Fantasy page is still too much like an intelligence dashboard. It needs a simpler user journey layered on top.

Recommended structure:

1. Fantasy Hub
   - Import My Squad
   - Get This Week's Move
   - Build Transfer Plan

2. First-Run Setup
   - FPL team ID
   - Strategy mode
   - Willingness to take hits
   - Favourite clubs / players

3. My Team
   - Pitch/List toggle
   - captain/vice
   - bench order
   - lineup validity

4. Transfers
   - player pool/search on left
   - squad/transfer plan on right
   - budget and free transfers always visible

5. Watchlist
   - simple follow rows
   - later: alerts and price/news triggers

6. Plans
   - saved drafts
   - safe/aggressive/wildcard routes
   - sync to account

## Design Rules To Apply

- One primary action per section.
- Use large, readable headings.
- Do not show raw feature tokens to users.
- Use tabs and toggles for mode changes.
- Keep FPL rules available, but collapsed.
- Show constraints visually: bank, 15-player count, formation validity, free transfers.
- Use a pitch as the core emotional object.
- Make empty states productive: "Select Defender", "Import squad", "Choose captain".
- Split "working app" from "proof/product marketing".
- Keep mobile in one clear vertical journey.

## Immediate Website Build Order

1. Add a Fantasy hub state above the current tool.
2. Add a first-run onboarding overlay / setup flow.
3. Redesign My Team around Pitch/List toggle.
4. Split Transfers into official-style left player pool and right squad/plan.
5. Turn Watchlist into simple follow rows.
6. Move advanced proof/rules lower or behind collapsed sections.
7. Keep the command decision panel, but make it the output after import/setup, not the first thing a novice has to understand.

## Product Difference From Official FPL

Official FPL helps users build and manage a team.

Odds Genius should help users decide:

- who to start
- who to bench
- who to captain
- who to sell
- who to buy
- whether to roll
- whether to take a hit
- whether to use a chip
- which players/clubs to watch before deadline

The UX should feel as simple as official FPL, but the value should be stronger: OG turns squad data into deadline decisions.
