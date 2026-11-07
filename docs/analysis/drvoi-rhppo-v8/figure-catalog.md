# DRVOI-RHPPO V8 图目录

## figure-01-fixed-opponent-comparison.pdf

- Purpose: 比较 B0 与 V8 在三类固定对手上的终局质量与武器效率。
- Data source: 六个 `eval-*.json`，每个柱 192 局；命中率分母为 accepted launches。
- Caption requirements: 误差条为局级或发射事件级 Wilson 95% 区间，不是种子方差；训练种子 n=1。
- Key observation: pursuit 改善，hold 超时恶化，evasive 未改善。
- Interpretation checklist: 不得把局级区间解释为算法跨种子显著性。

## figure-02-online-voi-dynamics.pdf

- Purpose: 检查冻结 VOI 模型在在线策略分布移动时是否保持有效覆盖。
- Data source: V8 `train.json` 的 41 次更新，不做平滑。
- Caption requirements: 横轴为累计环境决策；四条曲线均为每次 rollout 的统计量。
- Key observation: VOI 覆盖和选项分歧总体下降，Q 折差距保持较高。
- Interpretation checklist: 时间序列相关，不能把更新轮次当作独立重复；趋势用于机制诊断而非显著性声明。
