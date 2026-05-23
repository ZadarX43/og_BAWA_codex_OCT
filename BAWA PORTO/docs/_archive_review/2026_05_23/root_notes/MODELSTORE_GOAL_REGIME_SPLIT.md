# ModelStore Goal Regime Split

## Summary
- 25 leagues: safe schema (20–24 features) — created 2026-04-12
- 4 leagues: legacy schema (76–92 features) — created 2026-02-12

This is a mixed-regime estate, which likely caused performance collapse.

## Safe schema leagues (2026-04-12)
- Australia_A-League
- Australia_A_League
- Austria_Bundesliga
- Belgium_Pro
- Brazil_Serie_A
- Champions_League
- Czech_First_League
- Denmark_Superliga
- England_Championship
- England_EFL_League_1
- England_FA_Cup
- Europa_Conference
- Europa_League
- France_Ligue_1
- Germany_Bundesliga
- Germany_Bundesliga_2
- Italy_Serie_A
- Japan_J1
- Netherlands_Eredivisie
- Norway_Eliteserien
- Portugal_Liga
- Saudi_Pro_League
- South_Korea_K_League
- Swiss_Super_League
- Turkey_Super_Lig

## Legacy schema leagues (2026-02-12)
- USA_MLS
- Spain_La_Liga
- England_Premier_League
- Scotland_Premiership

## Recommendation
Unify goal-ensemble schema across all leagues before retraining markets.
Avoid in-place retraining. Use a shadow goal-layer rebuild first.
