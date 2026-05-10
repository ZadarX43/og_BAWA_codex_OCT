# Calm Onboarding + Style Presets Plan

## Purpose

Define the first dedicated onboarding and style-preset layer for Odds Genius.

This task exists to turn the current account-and-dashboard setup into a more intentional first-run experience that supports:

- calmer orientation
- better personal relevance
- cleaner alert behavior
- more disciplined user expectations

This is not a plan to increase urgency or stimulate more betting volume.

It is a plan to help users enter the product with:

- structure
- clarity
- selective delivery

## Product Goal

The onboarding flow should help a user move from:

- generic premium member

into:

- a configured football intelligence user

The experience should feel like:

- setting up a personal analyst desk

not:

- joining a hype room
- subscribing to more noise

## Core Outcomes

Task 71 should produce:

1. a dedicated onboarding flow
2. user style presets
3. a fuller decision companion layer
4. a reset / clarity mode for emotionally difficult moments
5. better delivery expectations around Telegram vs website-only

## Onboarding Principles

The onboarding flow should:

- slow the user down slightly
- reduce choice overload
- orient them around structure rather than “winning tips”
- make alert selectivity explicit
- make `no edge` feel normal and respectable

The onboarding flow should not:

- oversell action
- encourage broad alert opt-in by default
- use urgency
- use tipster language

## Dedicated Onboarding Flow

Recommended sequence:

### Step 1 — Welcome / positioning

Headline direction:

- Signal over noise.
- Build edge, not dependency.
- Better decisions, not more bets.

Supporting copy should explain:

- Odds Genius helps users think better under uncertainty
- not every fixture should lead to action
- the product is selective by design

### Step 2 — Choose what you care about

Let the user choose:

- teams
- leagues
- markets
- fixtures

This should feel like:

- building their watch environment

not:

- completing a marketing profile

### Step 3 — Choose your style preset

Preset choice should simplify the product for users who do not want to configure everything manually on day one.

Recommended presets:

#### Analyst

Use when the user wants:

- broad context
- richer website intelligence
- fewer Telegram interruptions
- more reading than acting

Default posture:

- website-first
- more `OBSERVE` and `MONITOR`
- fewer direct Telegram sends

#### Disciplined bettor

Use when the user wants:

- deploy-focused intelligence
- clearer action thresholds
- stronger filtering

Default posture:

- selective Telegram
- more `DEPLOY`
- high bar for `OBSERVE`

#### Tactical reader

Use when the user wants:

- fixture context
- team/news/weather/shape intelligence
- follow-heavy use

Default posture:

- more follow-based website intelligence
- fewer market-only Telegram interruptions

#### Researcher

Use when the user wants:

- wide website visibility
- richer non-deploy intelligence
- minimal interruption

Default posture:

- website-first mode
- deeper `CONTEXT`
- digest-friendly

## Preset System Design

Presets should initially behave as:

- saved preference bundles

not:

- hard product silos

Recommended first preset outputs:

- channel posture
- alert breadth
- dashboard emphasis
- Telegram interrupt threshold
- website-only breadth

### Example mapping

`Analyst`
- Telegram: narrow
- website: broad
- deploy: enabled
- observe: enabled
- context: enabled
- market-only Telegram: mostly suppressed

`Disciplined bettor`
- Telegram: selective but stronger
- website: medium breadth
- deploy: enabled
- observe: selective
- context: limited unless followed team/fixture

`Tactical reader`
- Telegram: mostly team/fixture-led
- website: broad
- deploy: enabled
- context: enabled
- weather/team-news sensitivity: higher

`Researcher`
- Telegram: minimal
- website: widest breadth
- deploy: enabled
- observe/context/monitor: enabled

## Decision Companion Layer

The decision companion should expand from fixture-page prompts into a clearer reusable product layer.

It should appear:

- during onboarding
- on key fixture pages
- near high-priority deploys
- optionally before sending critical Telegram actions later

Core prompts:

- What is the edge here?
- What could weaken it before kickoff?
- Is this structure or impulse?
- Would no action be the cleaner decision?
- Does this still make sense at the current price?

The tone should feel:

- elite
- calm
- disciplined

not:

- therapeutic
- moralising
- patronising

## Reset / Clarity Mode

This is one of the most important emotional UX additions.

It should be designed for moments when:

- a user’s bet loses
- a deploy fails
- a read was right but the result was noisy
- a user feels tempted to chase

## Goal

Defuse emotional escalation and reduce churn or negative reaction loops.

## Product Rule

After loss-related moments, the platform should never imply:

- urgency
- revenge action
- “get it back”
- stronger next-step stimulation

Instead it should reinforce:

- pause
- perspective
- process
- discipline

## Recommended microcopy directions

- That fixture is settled. The next decision does not need to be immediate.
- One result does not define the quality of your process.
- No edge appears faster when attention is rushed.
- Pause before re-entry. Protect your attention first.
- A missed outcome is not a command to chase the next one.

## Recommended reset states

### After a losing settled pick

Show:

- short calming message
- optional `Return to dashboard`
- optional `View tomorrow’s intelligence`
- optional `Switch to website-only mode for now`

### After a followed deploy loses

Show:

- process-first language
- reminder that selective action matters more than emotional sequence

### After multiple alerts in a short window

Show:

- prompt to reduce interruptions
- prompt to switch to digest or website-first mode

## Why Telegram vs Website-Only

The onboarding flow should teach this early.

Users should understand:

- Telegram is for stronger interruptions
- website is for broader intelligence depth

Recommended explanation block:

- Telegram messages are reserved for stronger, more interruption-worthy updates.
- Website-only items still matter. They are kept here when they are useful without needing to break attention.

## Dashboard / Feed Implications

Task 71 should inform the next dashboard scoring pass.

The onboarding output should influence:

- what appears in `Send now`
- what appears in `Watch closely`
- what appears in `Website only`
- what appears in `No edge / monitor`

This means presets should eventually become inputs to:

- feed ranking
- alert suppression
- channel routing

## Data / Preference Requirements

The first implementation likely needs new or expanded saved fields for:

- `user_style_preset`
- `decision_companion_enabled`
- `reset_mode_enabled`
- `calm_onboarding_completed_at`

Optional later:

- `interruption_tolerance`
- `digest_bias`
- `website_depth_bias`

These should remain product-layer fields and must not affect protected model or deploy logic.

## Recommended Implementation Order

### Phase 1

- define onboarding steps
- define preset names and meanings
- define default mappings into existing preferences

### Phase 2

- build onboarding UI
- save preset selections
- apply preset defaults to dashboard/alerts

### Phase 3

- add reset / clarity mode states
- add stronger decision companion entry points
- tune copy based on real usage

## Immediate Deliverables For Task 71

Task 71 should produce:

1. onboarding flow spec
2. preset mapping table
3. copy guidance for calm onboarding
4. reset-mode copy guidance
5. preference-field recommendations
6. dashboard / alert integration notes

## Success Condition

This task is successful when a new user can enter Odds Genius and quickly feel:

- oriented
- calmer
- understood
- selective rather than overwhelmed

and when a losing or failed deploy moment is handled in a way that reduces:

- panic
- churn
- reactive behavior

instead of feeding it.
