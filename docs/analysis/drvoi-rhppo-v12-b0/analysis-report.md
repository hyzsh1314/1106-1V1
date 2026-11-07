# DRVOI-RHPPO V12 MRAT B0 审计报告

> B0 仅用于工程验收，不构成算法优越性证据。

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
| V12 MRAT B0 | hold | 0.500 | 0.000 | 298 | 96 | 0.338 | 0.161 |
| V12 MRAT B0 | pursuit | 0.807 | 0.000 | 319 | 162 | 0.208 | 0.153 |
| V12 MRAT B0 | evasive | 0.073 | 0.880 | 58 | 14 | 0.602 | 0.540 |

## 训练日志完整性

`metrics.jsonl` 共 4 次更新，全部通过 V12 强制字段与有限值校验。

详细机器可读结果见 `summary.json` 与 `fixed-opponent-comparison.csv`。
