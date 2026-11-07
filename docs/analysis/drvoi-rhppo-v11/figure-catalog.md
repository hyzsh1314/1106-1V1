# 图目录与解释

## Figure 01 — `figure-01-fixed-opponent-win-rates`

- Purpose: 比较 V9--V11.3 在三类固定对手上的条件胜率，展示方法×对手交互而非单一平均分。
- Data source: 各运行目录的 `eval-{hold,pursuit,evasive}.json`，每格 192 局。
- Caption requirements: 误差条为 episode-level 95% Wilson CI；所有方法只有 seed=85；evaluation seed=10085；区间不含训练 seed 方差。
- Key observation: V10/V11.2/V11.3 首次在 Evasive 上获得非零胜率；V10 同时破坏 Hold/Pursuit，V11.2/V11.3 恢复了大部分简单场景性能。
- Interpretation checklist:
  1. 图存在是为了检验“困难场景改善是否以简单场景退化为代价”。
  2. 读者应注意 V11.2 与 V11.3 的跨场景平衡，而不是只看 Evasive 最高柱。
  3. 该观察支持继续研究仲裁终止器，但不支持跨 seed 显著性声明。
- Caveat: 同一开发评估被用于提出后续变体，存在选择偏差。

## Figure 02 — `figure-02-easy-hard-tradeoff`

- Purpose: 显示平均 Hold/Pursuit 胜率与 Evasive 胜率之间的 Pareto 权衡。
- Data source: `exact-summary.csv`。
- Caption requirements: 横轴为 Hold 与 Pursuit 的算术平均，纵轴为 Evasive 胜率；每点为单个 seed=85 checkpoint/仲裁器，无误差条。
- Key observation: V10 获得困难场景性能但位于低 easy-performance 区域；V11.2 与 V11.3 移到右上区域，V11.3 更偏向保留简单场景。
- Interpretation checklist:
  1. 图存在是为了避免用三场景宏平均掩盖策略权衡。
  2. 读者应注意 V11.2 与 V11.3 均非绝对支配关系。
  3. 该观察改变的决策是：V11.3 做主线，V11.2 做困难场景上界消融。
- Caveat: 点的位置没有 seed-level 不确定性，不能据此声称 Pareto 前沿稳定。

## Figure 03 — `figure-03-intervention-fraction`

- Purpose: 解释 V11 系列性能变化是否与动作介入覆盖率一致。
- Data source: V11--V11.3 评估 JSON 中的 `action_intervention_fraction`。
- Caption requirements: 介入定义为仲裁后动作四个离散分量中至少一个不同于 V9 输出的环境步比例。
- Key observation: 单步 V11 在 Evasive 介入 65.30%仍为零胜；V11.2/V11.3 达到约 90%且恢复 26.04%/21.35%胜率，说明持续性而非单步触发次数是关键。
- Interpretation checklist:
  1. 图存在是为了定位机制，而不是再次展示胜率。
  2. 读者应注意介入率与胜率并非线性关系。
  3. 该观察支持把下一步建模重点放在 option termination，而不是奖励或触发阈值。
- Caveat: 介入率是环境步加权，长超时局会占更高权重。
