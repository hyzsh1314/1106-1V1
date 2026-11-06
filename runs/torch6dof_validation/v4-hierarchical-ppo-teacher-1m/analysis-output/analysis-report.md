# Teacher-assisted hierarchical PPO: strict three-seed analysis

## Comparison question

Does the revised high-level gate avoid option collapse and generalize across three training seeds under a fixed-mix opponent curriculum? The independent unit is the training seed (85, 185, 285; n=3). The 192 evaluation episodes per opponent are nested within a seed and are not treated as independent runs.

## Descriptive results

| Opponent | Win rate | Timeout rate | Legal-shot episode rate | Hit / launch |
|---|---:|---:|---:|---:|
| Hold | 87.7% ± 0.3% | 0.0% ± 0.0% | 97.4% ± 0.0% | 55.3% ± 0.6% |
| Pursuit | 92.4% ± 0.3% | 0.0% ± 0.0% | 99.1% ± 0.3% | 56.8% ± 0.6% |
| Evasive | 36.3% ± 1.7% | 61.6% ± 0.3% | 56.8% ± 2.4% | 35.9% ± 0.7% |

## Evidence-led interpretation

Gate collapse is no longer observed: final teacher-target accuracy is 95.8% ± 1.4%, and all three seeds select the intercept option in more than 92% of evasive decisions while retaining the base option in more than 88% of hold and 96% of pursuit decisions.

Easy-opponent performance is stable (hold 87.7% ± 0.3%; pursuit 92.4% ± 0.3%). Evasive performance is also seed-stable at 36.3% ± 1.7%, but the mean timeout rate remains 61.6% ± 0.3%. This shifts the principal bottleneck from high-level option selection to the capability and firing geometry of the frozen intercept skill.

The 192-episode seed85 gate ablation gives an oracle evasive win rate of 46.4% versus 34.4% for the learned gate. Always-base obtains 0%, while always-intercept matches the oracle. The remaining gap is real, but oracle uses the hidden scripted opponent class at the first decision. A purely geometric gate performs worse on all three opponents, so replacing the learned recurrent gate with a hand threshold is not supported.

The result is a teacher-assisted baseline, not evidence that pure hierarchical PPO learns the same decomposition unaided. The current teacher coefficient remains non-zero at the final update, so the next algorithmic check must anneal teacher supervision to zero and reserve a terminal pure-PPO consolidation phase.

## Claim Candidates

- Claim:
  - Source evidence: Three fixed-mix training seeds and 3 × 192 fixed-opponent episodes per seed.
  - Allowed wording: Teacher-assisted option gating removes the observed seed-level gate collapse and yields stable specialization across the three scripted opponent classes.
  - Forbidden stronger wording: Pure hierarchical PPO is proven superior to standard PPO.
  - Uncertainty: n=3; opponent-class targets are privileged during training; skills are pretrained and frozen.
  - Next check: zero-teacher consolidation and matched fixed-mix standard PPO.
  - Decision: keep with boundary
