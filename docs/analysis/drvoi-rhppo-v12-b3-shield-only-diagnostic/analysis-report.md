# DRVOI-RHPPO V12 MRAT B3-SHIELD-ONLY-DIAGNOSTIC 审计报告

> B3-SHIELD-ONLY-DIAGNOSTIC 为开发实验，不构成正式多种子算法优越性证据。

## 固定对手结果

| 方法 | 对手 | 胜率 | 超时率 | 发射 | 命中 | LOS占比 | 强制占比 |
|---|---|---:|---:|---:|---:|---:|---:|
| V9 COACH-RHPPO | hold | 0.609 | 0.073 | 262 | 117 | 0.000 | 0.000 |
| V9 COACH-RHPPO | pursuit | 0.891 | 0.000 | 321 | 174 | 0.000 | 0.000 |
| V9 COACH-RHPPO | evasive | 0.000 | 0.979 | 13 | 0 | 0.000 | 0.000 |
| V11.2 episode-latched | hold | 0.714 | 0.000 | 314 | 138 | 0.000 | 0.000 |
| V11.2 episode-latched | pursuit | 0.797 | 0.000 | 323 | 156 | 0.000 | 0.000 |
| V11.2 episode-latched | evasive | 0.260 | 0.693 | 138 | 50 | 0.000 | 0.000 |
| V11.3 sustained | hold | 0.693 | 0.000 | 309 | 133 | 0.000 | 0.000 |
| V11.3 sustained | pursuit | 0.828 | 0.000 | 321 | 162 | 0.000 | 0.000 |
| V11.3 sustained | evasive | 0.214 | 0.760 | 129 | 41 | 0.000 | 0.000 |
| V12 B3 shield-only diagnostic | hold | 0.500 | 0.000 | 297 | 96 | 0.338 | 0.164 |
| V12 B3 shield-only diagnostic | pursuit | 0.812 | 0.000 | 321 | 164 | 0.207 | 0.158 |
| V12 B3 shield-only diagnostic | evasive | 0.073 | 0.880 | 58 | 14 | 0.604 | 0.572 |
| V12-Frozen B1 (seed85) | hold | 0.469 | 0.000 | 341 | 94 | 0.718 | 0.065 |
| V12-Frozen B1 (seed85) | pursuit | 0.781 | 0.000 | 353 | 156 | 0.583 | 0.073 |
| V12-Frozen B1 (seed85) | evasive | 0.172 | 0.781 | 114 | 33 | 0.798 | 0.269 |
| V12 B2 cost-regularized (seed85) | hold | 0.396 | 0.000 | 373 | 80 | 1.000 | 0.000 |
| V12 B2 cost-regularized (seed85) | pursuit | 0.620 | 0.000 | 365 | 130 | 1.000 | 0.000 |
| V12 B2 cost-regularized (seed85) | evasive | 0.250 | 0.693 | 171 | 48 | 1.000 | 0.000 |

## 训练日志完整性

本阶段仅执行结构诊断评估，不产生训练更新。

详细机器可读结果见 `summary.json` 与 `fixed-opponent-comparison.csv`。
