# Prediction overlay patch log

The following summarizes each item from the original 14-point checklist and notes the corresponding patches completed in `prediction_overlay copy.py`.

1. **Optional→str safety for prob/od columns** – Column names that may be absent are treated as `Optional[str]`, with presence checks and casts before use in `_prepare_candidates` and `_pick_prob_odds_for_market`.
2. **Safe numeric parsing** – Replaced direct `float`/`int` calls on optional values with `_to_float`/`_to_int` or `pd.to_numeric`, and ensured environment defaults are stringified.
3. **Logger dynamic attribute (`_logged_labels`)** – Added a casted logger reference to allow dynamic `_logged_labels` tracking without mypy attribute errors.
4. **Candidate normalization before composer gates** – Each candidate pool is normalized via `_map_prob_od_for_market`, recomputing EV/edge and setting `od_source` before composer checks.
5. **Fallback candidate dumps** – Fallback CSV dumps now remap canonical columns, fill `p_model/od`, compute EV/edge, drop non-positive odds, and set default `od_source`.
6. **Draw-gate row logic** – `attach_ftr_with_draw_mix` uses helperized numeric coercion for thresholds and guarded mask access for robust gating counts and diagnostics.
7. **ROI Monte-Carlo sizing/export** – Simulation sizes are normalized to ints and RNG choice shapes corrected to handle multiple acca sizes safely.
8. **Stake normalization for walk-forward** – Walk-forward and ROI sections normalize stake once per scope with unique variable naming to avoid mypy duplicate definitions.
9. **Mask truthiness checks** – Replaced `.any()` truthiness on masks with `bool(np.any(...))` to eliminate `Any|bool` warnings.
10. **Duplicate helper guards** – Legacy helper duplicates are wrapped in `if "<name>" not in globals()` guards to avoid redefinition warnings.
11. **Bet slip output polish** – Ensured `od_source` is present in markdown tables after `od`, and combined odds/EV remain non-zero in output.
12. **Imports & line-length hygiene** – Removed unused imports and adjusted select long lines where appropriate to satisfy pyflakes (tolerating necessary E501s).
13. **Canonical `_prepare_candidates` path** – Composer and helpers route through the guarded `_prepare_candidates` version while keeping the legacy variant unused.
14. **Completion status** – All checklist items above are now patched in the current codebase; no outstanding tasks remain.

## Micro-fix confirmations

* Sorting with optional probability columns now guards against missing keys by building `sort_cols` dynamically before ordering candidate DataFrames.
* Composer respects the `ALLOW_SYNTH_IN_SLIPS` toggle by dropping synthesized odds rows after canonical mapping when the flag is disabled.
* Dynamic logger label tracking uses the casted `_log_any` reference for both initialization and subsequent access to `_logged_labels`.
