# SLIP_ENGINE

## Purpose
Canonical documentation for slip and portfolio construction over routed deploy outputs.

## Authority
`slip_formatter.py` is a thin formatter over deploy outputs. It must not introduce new prediction logic, rescue logic, or filler behavior.

## Core Rules
- ELITE / STANDARD only for live slips unless explicitly doing research
- one leg per fixture unless explicitly testing composite products
- no filler
- no rescue
- no forced completion
- do not degrade board quality just to hit slip size

## Main Inputs
- tiered deploy CSVs from `deploy_rulebook.py`
- optional audit outputs for slip quality review

## Main Outputs
- ranked board
- family summary
- singles, doubles, trebles
- fixed accas by allowed sizes
- market-specific ranked boards and fixed accas

## Source Candidates
- `slip_formatter.py`
- `slip_formatter.md`
- `docs/DEPLOY_RULEBOOK.md`
