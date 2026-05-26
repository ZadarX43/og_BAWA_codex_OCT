# Website User Flow And Tier QA Plan

Date: 2026-05-26

Purpose: provide a repeatable way to review the website from first open through account onboarding, tier-specific navigation, fixture exploration, paper-slip building, and saved-slip review.

## Demo Tier Simulator

Use query params to simulate account tiers without changing Stripe, Worker auth, or production entitlement rules:

- Standard: `account.html?demo=1&tier=standard`
- Founder: `account.html?demo=1&tier=founder`
- Premium: `account.html?demo=1&tier=premium`
- Pro: `account.html?demo=1&tier=pro`
- Pro+: `account.html?demo=1&tier=pro_plus`

The simulator writes only local browser state. It is for navigation and UI QA. Real customer access remains controlled by Stripe, Worker sessions, and server-side tier checks.

## First Visitor Flow

1. Open homepage.
2. Confirm the first viewport answers: what it is, proof, launch offer, and how to join.
3. Click Matches.
4. Confirm feed explains fixture coverage, market buttons, favourites, and paper slip.
5. Click a fixture.
6. Confirm Standard view shows top market cards only and deeper layers are clearly positioned as unlocks.
7. Click pricing / account CTA.
8. Confirm checkout path is understandable before payment.

## Signed-In Tier Flow

For each tier, use the Account simulator and run:

1. Account home renders correct tier label and signed-in state.
2. Onboarding page loads with saved preferences and clear setup steps.
3. Matches feed keeps the same demo tier across navigation.
4. Expanded match post shows only the right unlocked panels.
5. Fixture page tabs respect tier access.
6. Paper slip can add one outcome per market, update stake, save, reload, and remove selections.
7. Saved slips are visible from Account -> Paper slips.

## Tier Expectations

Standard:
Public feed, top market cards, paper slips, favourites, public proof, locked deeper context.

Founder:
Standard plus fixture context: H2H, weather, lineups, team reads, injury notes, freshness.

Premium:
Founder plus deeper market posture, team context, contradiction/support logic, richer saved-slip reasoning later.

Pro:
Premium plus player-event watchlists and pre-lineup player intelligence.

Pro+:
Pro plus audit/explainability panels, data status, source freshness, coverage flags, and downloadable/debug references.

## Launch Tutorial Seeds

The launch tutorial can become a short guided overlay or slide path:

1. Read the homepage proof.
2. Open Matches.
3. Add one market selection to a paper slip.
4. Expand a match read.
5. Favourite a fixture.
6. Save the slip.
7. Open Account -> Paper slips.
8. Set preferences and alert posture.

## Current QA Focus

Highest-priority checks before launch:

- No stale fixture window on Matches.
- Bottom nav does not cover important actions.
- Team/fixture search routes to the expected destination.
- Tier-locked panels feel intentional rather than unfinished.
- Account onboarding is calm and understandable.
- Saved slips have a clear home inside Account.
