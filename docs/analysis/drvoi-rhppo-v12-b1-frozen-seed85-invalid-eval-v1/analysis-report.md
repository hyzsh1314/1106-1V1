# DRVOI-RHPPO V12 MRAT B1-FROZEN-SEED85 审计报告

> B1-FROZEN-SEED85 为开发实验，不构成正式多种子算法优越性证据。

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
| V12-Frozen B1 (seed85) | hold | 0.469 | 0.000 | 341 | 94 | 0.718 | 0.065 |
| V12-Frozen B1 (seed85) | pursuit | 0.781 | 0.000 | 353 | 156 | 0.583 | 0.073 |
| V12-Frozen B1 (seed85) | evasive | 0.172 | 0.781 | 114 | 33 | 0.798 | 0.269 |

## 训练日志完整性

`metrics.jsonl` 共 41 次更新，全部通过 V12 强制字段与有限值校验。

详细机器可读结果见 `summary.json` 与 `fixed-opponent-comparison.csv`。
