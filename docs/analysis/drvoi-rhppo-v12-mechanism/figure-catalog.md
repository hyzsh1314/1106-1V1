# Figure catalog

## Figure 01 — `figures/figure-01-performance.pdf`

- Purpose: compare frozen-policy win and timeout profiles across all three opponents.
- Data source: each method's `eval-{hold,pursuit,evasive}.json`, 192 episodes per cell.
- Caption requirements: circles are win proportions; vertical bars are Wilson 95% intervals for a fixed checkpoint; orange crosses are timeout proportions; intervals do not represent training-seed variance.
- Key observation: the methods trade simple-opponent retention against Evasive recovery; no current V12 candidate dominates V11.2/V11.3.
- Interpretation checklist: inspect all three panels; do not select on macro average alone; do not read overlapping episode intervals as a seed-level test.

## Figure 02 — `figures/figure-02-evasive-mechanism.pdf`

- Purpose: test whether LOS usage and entry timing explain Evasive outcomes.
- Data source: 192 episode JSONL records and option-segment JSONL records per V12 diagnostic.
- Caption requirements: boxes show episode distributions (median and quartiles; whiskers 1.5 IQR, fliers hidden); left is LOS expert fraction, right is first LOS decision step.
- Key observation: high LOS occupancy alone is insufficient; successful variants also differ in when a persistent LOS option begins.
- Interpretation checklist: distributions are episodes from one frozen seed, not independent training runs; association with outcome does not by itself prove causality.
