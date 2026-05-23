# OU25 One-Pager
## Odds Genius — Over/Under 2.5 Goals Product

### Status
**Frozen discovery complete**

### Universe tested
- 19 leagues
- 3 years
- canonical truth-backed scored backtest corpus
- 19,923 OU25 rows

### What was built
- OU25 standalone frozen gate mode
- combined / over-only / under-only branch support
- odds-band sweeps
- top-q sweep support
- branch comparison outputs
- cumulative stats outputs
- forensic audit outputs

### What passed
- smoke tests
- full frozen sweep batch
- branch comparison build
- cumulative stats build
- forensic audit

### Current best branches

| Branch | Rows | Hit | ROI | Avg Odds |
|---|---:|---:|---:|---:|
| `ou25_combined_topq_080` | 2863 | 82.08% | 0.4435 | 1.7682 |
| `ou25_mode_over_only` | 2946 | 82.89% | 0.4286 | 1.7361 |
| `ou25_band2_178_195` | 4956 | 80.97% | 0.4168 | 1.7610 |
| `ou25_band1_124_176` | 4866 | 81.71% | 0.4153 | 1.7430 |

### Strongest current readings

**Premium quality winner**
- `ou25_combined_topq_080`

**Best directional lane**
- `ou25_mode_over_only`

**Best scale branch**
- `ou25_band2_178_195`

**Best benchmark branch**
- `ou25_band1_124_176`

### Key product insight

OVER-only currently outperforms UNDER-only.

That means OU25 should not be treated as one flat product.  
It likely supports at least two commercial shapes:

- combined OU25 product
- OVER25 specialist product

### Audit conclusion
- row lineage back to canonical backtest: clean
- no merge misses in final audit state
- no duplicate join rows
- no evidence of post-match leakage into filtering or ranking

### Caveat
This is a **frozen discovery result**, not yet the final forward-validated deployment winner.

### Decision taken
OU25 is now officially established as a real Odds Genius product lane.

### Next step
Run **OU25 walk-forward validation** and lock the final deploy branch.