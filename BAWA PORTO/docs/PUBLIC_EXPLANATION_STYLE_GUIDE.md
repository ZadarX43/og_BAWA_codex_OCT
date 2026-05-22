# Public Explanation Style Guide

## Purpose
`public_explanation` is the editorial voice of the system.

It should make Odds Genius feel like:
- a judgement system
- a disciplined analyst
- a consistent product voice

It should never feel like:
- a raw stat dump
- sportsbook hype
- an LLM ramble

## Core Rule
The value is not more data.

The value is:

**a clean judgement about what the data means**

## Voice
Use:
- calm
- precise
- disciplined
- premium
- lightly editorial

Avoid:
- hype
- certainty language
- gambling slang
- internal pipeline jargon

## Preferred Tone
Good:
- “Signal shape is present, but support is below the deployment threshold.”
- “The matchup stays live through attacking pressure, but defensive suppression keeps the read mixed.”
- “Control is stronger than chaos here, so the cleaner read is restraint rather than force.”

Avoid:
- “This is a lock.”
- “Guaranteed edge.”
- “Cleared live routing checks.”
- “Massive smash.”
- “Free money.”

## Length Caps
- team object: `20-30` words
- player object: `12-24` words
- lineup object: `20-35` words
- H2H object: `18-30` words
- fixture object: `30-60` words

## Structure
Public explanation should usually do one of these:

### 1. Support + caution
Best for fixtures

Example:
“Arsenal hold a strong structural edge across team power, home profile, and midfield control. The main caution is Tottenham’s goal heat, which keeps away-goal risk live.”

### 2. Profile summary
Best for teams

Example:
“Elite control side built on defensive resistance, first-goal pressure, and strong home leverage.”

### 3. Role summary
Best for players

Example:
“High-usage creator with strong final-third involvement and reliable chance-generation pressure.”

## Vocabulary Preferences
Prefer:
- support
- caution
- discipline
- control
- structure
- pressure
- resistance
- access
- fragility
- deployment threshold
- live read
- matchup shape
- two-way scoring
- suppression

Avoid:
- lock
- certainty
- must-bet
- guaranteed
- routing checks
- pipeline passed
- crushed
- killer
- smashing
- nailed on

## Deterministic, Not LLM-Freeform
`public_explanation` should be generated from:
- ratings
- tags
- state
- selected templates

It should not be open-ended free text generation.

That keeps:
- tone stable
- phrasing coherent
- outputs auditable

## Object-Specific Notes

### Team
Focus on:
- team identity
- strongest structural traits
- one caution

### Player
Focus on:
- role
- strongest driver
- style tag

### Fixture
Focus on:
- primary signal
- strongest support
- strongest caution

### H2H
Focus on:
- support-only context
- never lead with H2H as the main reason

## Final Rule
Every explanation should feel like one careful analyst wrote the whole product.
