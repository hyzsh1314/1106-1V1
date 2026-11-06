# DRVOI-RHPPO V7.1 测试与数据访问清单

## W0 冻结

- [ ] 分支基点严格等于 `73733ddcbfe48b04b9c267d4f86e8bda976d2896`
- [ ] V7完整validation、门禁JSON、摘要CSV和SHA-256清单入库
- [ ] V7.1配置与协议在任何新确认性test生成前提交
- [ ] 输出只写入 `runs/drvoi_rhppo_v7_1/`
- [ ] P1Q、O0、短筛选和长训练授权均为false

## W1 历史与因果对齐

- [ ] t=0/1的零观测、中性动作和有效掩码正确
- [ ] reset、done、auto-reset后历史全部清零
- [ ] 并行环境的episode_id和历史互不串线
- [ ] H0/H1动作权重置零后初始函数完全一致
- [ ] 正确历史包含 `a_(t-2),a_(t-1),a_t`
- [ ] wrong-delay控制不包含正确对齐的完整动作历史
- [ ] 同对手×阶段置换不跨分层
- [ ] alt/heading/speed/fire单头删除与置换可复现
- [ ] snapshot/restore覆盖飞机、导弹、低层RNN、环境RNG和回合状态
- [ ] 同一快照重复执行产生逐位一致结果

## W2 开发集

- [ ] 只加载train/validation，代码路径不能解析test对象
- [ ] reactive正确历史优于H0和wrong-delay
- [ ] H1在D-onpolicy validation具有稳定配对增益
- [ ] 无NaN/Inf、原始梯度未解释值不超过1000
- [ ] sigma上下边界触及率已记录
- [ ] 确认性配置在生成test前提交并记录Git SHA

## W3 确认性评估

- [ ] 使用10185/11185/12185/13185/14185五个新采集seed
- [ ] 数据生成策略固定为B0 seed85
- [ ] 五折留一采集seed，test不进入训练、早停或校准
- [ ] 每个外层test只读取一次
- [ ] 报告5个seed效应、t区间和正效应seed数量
- [ ] 主ΔNLL、置换、reactive、G2和数值门禁均为机器可读JSON
- [ ] 逐维、逐对手、逐阶段结果完整输出但不替代主检验

## W4 条件授权

- [ ] 只有所有响应门禁通过才修改P1Q
- [ ] Q结构化HOT翻转与决策无关扰动均通过
- [ ] 固定上下文动作排序准确率和argmax遗憾达到冻结门槛
- [ ] P1Q通过后才允许O0 10–20 update烟雾测试
