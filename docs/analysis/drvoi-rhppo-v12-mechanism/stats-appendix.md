# Statistical appendix

## Units and validity

- Training-seed unit: n=1 (seed 85). No t-test, ANOVA, seed-level standard deviation, or algorithm-level p-value is valid.
- Frozen-policy evaluation unit: 192 paired initial conditions per opponent (96 red, 96 blue) under development seed 10085.
- Wilson 95% intervals describe episode sampling uncertainty for a fixed checkpoint only; they do not quantify training instability.
- Exact paired McNemar/binomial tests below compare deterministic frozen policies on matched initial states. Holm adjustment covers the listed B6/B7/B8 Evasive contrasts. These tests cannot support algorithm-level superiority.

## Frozen-policy paired contrasts

| contrast   |   paired_episodes |   left_only_wins |   right_only_wins |   paired_win_difference |   mcnemar_exact_p |   holm_adjusted_p |
|:-----------|------------------:|-----------------:|------------------:|------------------------:|------------------:|------------------:|
| B7 - B6    |               189 |                6 |                30 |                  0.1270 |            0.0001 |            0.0002 |
| B8 - B6    |               191 |                2 |                14 |                  0.0628 |            0.0042 |            0.0084 |
| B8 - B7    |               190 |               24 |                11 |                 -0.0684 |            0.0410 |            0.0410 |

## Exact performance table

| method   | opponent   |   episodes |   win_rate |   win_ci_low |   win_ci_high |   timeout_rate |   legal_request_rate |   hit_per_launch |   intercept_fraction |   gate_allowed_fraction |
|:---------|:-----------|-----------:|-----------:|-------------:|--------------:|---------------:|---------------------:|-----------------:|---------------------:|------------------------:|
| V9       | hold       |        192 |     0.6094 |       0.5389 |        0.6756 |         0.0729 |               1.0000 |           0.4466 |               0.0000 |                nan      |
| V9       | pursuit    |        192 |     0.8906 |       0.8386 |        0.9273 |         0.0000 |               1.0000 |           0.5421 |               0.0000 |                nan      |
| V9       | evasive    |        192 |     0.0000 |       0.0000 |        0.0196 |         0.9792 |               1.0000 |           0.0000 |               0.0000 |                nan      |
| V11.2    | hold       |        192 |     0.7135 |       0.6459 |        0.7728 |         0.0000 |               1.0000 |           0.4395 |               0.0000 |                nan      |
| V11.2    | pursuit    |        192 |     0.7969 |       0.7344 |        0.8477 |         0.0000 |               1.0000 |           0.4830 |               0.0000 |                nan      |
| V11.2    | evasive    |        192 |     0.2604 |       0.2035 |        0.3268 |         0.6927 |               1.0000 |           0.3623 |               0.0000 |                nan      |
| V11.3    | hold       |        192 |     0.6927 |       0.6242 |        0.7537 |         0.0000 |               1.0000 |           0.4304 |               0.0000 |                nan      |
| V11.3    | pursuit    |        192 |     0.8281 |       0.7685 |        0.8749 |         0.0000 |               1.0000 |           0.5047 |               0.0000 |                nan      |
| V11.3    | evasive    |        192 |     0.2135 |       0.1615 |        0.2768 |         0.7604 |               1.0000 |           0.3178 |               0.0000 |                nan      |
| B1       | hold       |        192 |     0.4688 |       0.3995 |        0.5393 |         0.0000 |               1.0000 |           0.2757 |               0.4105 |                nan      |
| B1       | pursuit    |        192 |     0.7812 |       0.7176 |        0.8339 |         0.0000 |               1.0000 |           0.4419 |               0.4302 |                nan      |
| B1       | evasive    |        192 |     0.1719 |       0.1251 |        0.2315 |         0.7812 |               1.0000 |           0.2895 |               0.7450 |                nan      |
| B4       | hold       |        192 |     0.4375 |       0.3692 |        0.5082 |         0.0000 |               1.0000 |           0.2514 |               0.5281 |                  0.9214 |
| B4       | pursuit    |        192 |     0.6875 |       0.6188 |        0.7488 |         0.0000 |               1.0000 |           0.4131 |               0.5832 |                  0.8616 |
| B4       | evasive    |        192 |     0.1875 |       0.1386 |        0.2486 |         0.7656 |               1.0000 |           0.2951 |               0.8125 |                  0.9564 |
| B5       | hold       |        192 |     0.5104 |       0.4402 |        0.5802 |         0.0000 |               1.0000 |           0.3322 |               0.1730 |                  0.4461 |
| B5       | pursuit    |        192 |     0.8125 |       0.7514 |        0.8614 |         0.0000 |               1.0000 |           0.5110 |               0.1568 |                  0.2986 |
| B5       | evasive    |        192 |     0.0729 |       0.0439 |        0.1187 |         0.8802 |               1.0000 |           0.2295 |               0.6222 |                  0.3823 |
| B6       | hold       |        192 |     0.5365 |       0.4659 |        0.6056 |         0.0000 |               1.0000 |           0.3409 |               0.3393 |                  0.4641 |
| B6       | pursuit    |        192 |     0.7812 |       0.7176 |        0.8339 |         0.0000 |               1.0000 |           0.4727 |               0.2824 |                  0.3682 |
| B6       | evasive    |        192 |     0.0833 |       0.0519 |        0.1311 |         0.8698 |               1.0000 |           0.2078 |               0.7959 |                  0.7720 |
| B7       | hold       |        192 |     0.4375 |       0.3692 |        0.5082 |         0.0000 |               1.0000 |           0.2500 |               0.5398 |                  0.4514 |
| B7       | pursuit    |        192 |     0.6771 |       0.6080 |        0.7392 |         0.0000 |               1.0000 |           0.4269 |               0.4597 |                  0.3438 |
| B7       | evasive    |        192 |     0.2188 |       0.1661 |        0.2824 |         0.7292 |               1.0000 |           0.2692 |               0.9092 |                  0.6936 |
| B8       | hold       |        192 |     0.5312 |       0.4607 |        0.6005 |         0.0000 |               1.0000 |           0.3301 |               0.3034 |                  0.4484 |
| B8       | pursuit    |        192 |     0.8177 |       0.7570 |        0.8659 |         0.0000 |               1.0000 |           0.5205 |               0.1529 |                  0.3716 |
| B8       | evasive    |        192 |     0.1458 |       0.1029 |        0.2027 |         0.8073 |               1.0000 |           0.2545 |               0.8278 |                  0.6959 |

## Blockers

1. Only one training seed; no estimate of seed variance.
2. Development evaluation seed 10085 has been repeatedly reused for structural decisions.
3. Fixed Hold/Pursuit/Evasive scripts are not opponent-family generalization.
4. A7 empirical LCB coverage at z=1.64 was 81.32%, below a 90% calibration target; it is not a formal confidence guarantee.
