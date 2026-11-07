# V12 B6–B8 mechanism analysis

## Analysis question

Can online-distribution return supervision and temporal gate aggregation recover Evasive performance without reproducing the Hold/Pursuit over-intervention failure? The primary unit for algorithm-level inference is the training seed; only seed 85 is available, so all method comparisons below are development evidence rather than algorithm-level significance claims.

## Key findings

- A6/A7 fixed the observed deployment-coverage failure: B6 Evasive gate allowance reached 77.20%, but B6 win rate remained 8.33%. This separates model coverage from control effectiveness.
- B7 gate-latching raised Evasive win rate to 21.88%, but its macro win rate was 44.44%; long skill persistence is necessary but indiscriminate latching harms simple opponents.
- B8 SPRT-latching reached Evasive win rate 14.58% with macro win rate 49.83%.
- Weapon legality remained 100% for every included V12 evaluation; the negative results are not explained by rejected launches or crashes.

## Exact macro summary

| method   |   macro_win |   macro_timeout |   total_launches |   total_hits |
|:---------|------------:|----------------:|-----------------:|-------------:|
| V9       |      0.5000 |          0.3507 |              596 |          291 |
| V11.2    |      0.5903 |          0.2309 |              775 |          344 |
| V11.3    |      0.5781 |          0.2535 |              759 |          336 |
| B1       |      0.4740 |          0.2604 |              808 |          283 |
| B4       |      0.4375 |          0.2552 |              819 |          268 |
| B5       |      0.4653 |          0.2934 |              675 |          275 |
| B6       |      0.4670 |          0.2899 |              715 |          277 |
| B7       |      0.4444 |          0.2431 |              835 |          271 |
| B8       |      0.4983 |          0.2691 |              739 |          296 |

## Evasive mechanism breakdown

| method   | outcome   |   episodes |   mean_los_fraction |   median_first_los |   mean_max_los_dwell |
|:---------|:----------|-----------:|--------------------:|-------------------:|---------------------:|
| B1       | loss      |          9 |              0.6707 |            30.0000 |              51.7778 |
| B1       | win       |         33 |              0.9891 |             0.0000 |             150.0000 |
| B1       | timeout   |        150 |              0.7625 |            44.5000 |             132.5800 |
| B4       | loss      |          9 |              0.6707 |            30.0000 |              51.7778 |
| B4       | win       |         36 |              0.9860 |             0.0000 |             128.2778 |
| B4       | timeout   |        147 |              0.8488 |            13.0000 |             118.0952 |
| B5       | loss      |          9 |              0.5523 |            32.0000 |              51.7778 |
| B5       | win       |         14 |              0.9139 |            53.0000 |             117.6429 |
| B5       | timeout   |        169 |              0.6388 |           138.0000 |             121.3432 |
| B6       | loss      |          9 |              0.6411 |            30.0000 |              43.6667 |
| B6       | win       |         16 |              0.9192 |            61.0000 |             150.0000 |
| B6       | timeout   |        167 |              0.8328 |            19.0000 |             150.0000 |
| B7       | loss      |         10 |              0.8010 |             0.0000 |              53.6000 |
| B7       | win       |         42 |              0.9866 |             0.0000 |             150.0000 |
| B7       | timeout   |        140 |              0.9866 |             0.0000 |             150.0000 |
| B8       | loss      |          9 |              0.5438 |            32.0000 |              51.0000 |
| B8       | win       |         28 |              0.8826 |           115.5000 |             150.0000 |
| B8       | timeout   |        155 |              0.8807 |           150.0000 |             150.0000 |

## Claim Candidates

- Claim:
  - Source evidence: A5 versus A7 offline behavior-stratified reports and B5 versus B6 online gate fractions.
  - Allowed wording: On seed-85 development data, on-policy aggregation corrected the measured gate-coverage shift but did not by itself restore Evasive wins.
  - Forbidden stronger wording: On-policy aggregation is ineffective in general.
  - Uncertainty: One training seed and one reused development evaluation seed.
  - Next check: multi-seed development only after a structure passes the current Pareto screen.
  - Decision: keep.
- Claim:
  - Source evidence: B6 versus B7 under the same B6 checkpoint and A7 predictor.
  - Allowed wording: The frozen-policy diagnostic supports option persistence as a causal mechanism candidate for Evasive recovery, while showing a simple-opponent cost.
  - Forbidden stronger wording: Episode latching is the optimal air-combat strategy.
  - Uncertainty: fixed scripted opponents and development initial-state distribution.
  - Next check: temporal evidence aggregation and randomized opponent behaviors.
  - Decision: keep with boundary.

## Decision

Do not expand B4–B8 to one million steps. Retain A6/A7 as valid coverage infrastructure. The current return-gated branch is a negative result; V11.2/V11.3 remain the strongest balanced development structures until a new candidate improves their Hold/Pursuit/Evasive Pareto front.
