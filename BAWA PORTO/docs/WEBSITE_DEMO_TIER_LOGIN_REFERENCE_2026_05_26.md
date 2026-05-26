# Website Demo Tier Login Reference

Date: 2026-05-26

Use the demo tier URLs to test the website product flow without a real Stripe checkout or magic-link login.

These URLs simulate account state in local browser storage only. They are for product QA and navigation review. Real customer access remains controlled by Stripe, Worker sessions, and server-side entitlement checks.

## Demo Accounts

Standard:

```text
account.html?demo=1&tier=standard
```

Founder:

```text
account.html?demo=1&tier=founder
```

Premium:

```text
account.html?demo=1&tier=premium
```

Pro:

```text
account.html?demo=1&tier=pro
```

Pro+:

```text
account.html?demo=1&tier=pro_plus
```

## Fixture Intelligence Testing

Start here:

```text
matches.html?demo=1&tier=pro
```

Then open a fixture from the Matches feed.

Direct fixture example:

```text
fixture.html?demo=1&tier=pro&fixture=2026_05_09_Bochum_Hannover_96
```

## Tier Behaviour

Standard:
Top public market cards, favourites, paper slips, public proof, and locked deeper context.

Founder:
Standard plus H2H, weather, lineup, team context, injury notes, and freshness.

Premium:
Founder plus deeper market posture, support/contradiction logic, and richer paid context.

Pro:
Premium plus player-event intelligence and pre-lineup watchlists.

Pro+:
Pro plus audit, source freshness, coverage flags, and explainability panels.

## Notes

- No password is needed.
- The simulator is not a real login.
- Clearing the demo account can be done from the Account QA simulator panel.
- If a route still looks locked, confirm the URL includes both `demo=1` and the intended `tier`.
