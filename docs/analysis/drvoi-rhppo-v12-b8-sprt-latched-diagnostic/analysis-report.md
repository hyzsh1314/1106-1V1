# DRVOI-RHPPO V12 MRAT B8-SPRT-LATCHED-DIAGNOSTIC 审计报告

> B8-SPRT-LATCHED-DIAGNOSTIC 为开发实验，不构成正式多种子算法优越性证据。

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
| V12 B8 SPRT-latched diagnostic | hold | 0.531 | 0.000 | 312 | 103 | 0.477 | 0.147 |
| V12 B8 SPRT-latched diagnostic | pursuit | 0.818 | 0.000 | 317 | 165 | 0.201 | 0.146 |
| V12 B8 SPRT-latched diagnostic | evasive | 0.146 | 0.807 | 110 | 28 | 0.880 | 0.397 |
| V12 B7 gate-latched diagnostic | hold | 0.438 | 0.000 | 344 | 86 | 0.816 | 0.063 |
| V12 B7 gate-latched diagnostic | pursuit | 0.677 | 0.000 | 335 | 143 | 0.598 | 0.073 |
| V12 B7 gate-latched diagnostic | evasive | 0.219 | 0.729 | 156 | 42 | 0.986 | 0.047 |
| V12 B6 on-policy-gated smoke | hold | 0.536 | 0.000 | 308 | 105 | 0.520 | 0.128 |
| V12 B6 on-policy-gated smoke | pursuit | 0.781 | 0.000 | 330 | 156 | 0.365 | 0.127 |
| V12 B6 on-policy-gated smoke | evasive | 0.083 | 0.870 | 77 | 16 | 0.839 | 0.261 |
| V12 B5 behavior-gated smoke | hold | 0.510 | 0.000 | 295 | 98 | 0.332 | 0.168 |
| V12 B5 behavior-gated smoke | pursuit | 0.812 | 0.000 | 319 | 163 | 0.206 | 0.157 |
| V12 B5 behavior-gated smoke | evasive | 0.073 | 0.880 | 61 | 14 | 0.657 | 0.345 |
| V12 B4 return-gated smoke | hold | 0.438 | 0.000 | 346 | 87 | 0.878 | 0.044 |
| V12 B4 return-gated smoke | pursuit | 0.688 | 0.000 | 351 | 145 | 0.777 | 0.042 |
| V12 B4 return-gated smoke | evasive | 0.188 | 0.766 | 122 | 36 | 0.872 | 0.198 |
| V12-Frozen B1 (seed85) | hold | 0.469 | 0.000 | 341 | 94 | 0.718 | 0.065 |
| V12-Frozen B1 (seed85) | pursuit | 0.781 | 0.000 | 353 | 156 | 0.583 | 0.073 |
| V12-Frozen B1 (seed85) | evasive | 0.172 | 0.781 | 114 | 33 | 0.798 | 0.269 |

## 训练日志完整性

本阶段仅执行结构诊断评估，不产生训练更新。

详细机器可读结果见 `summary.json` 与 `fixed-opponent-comparison.csv`。
