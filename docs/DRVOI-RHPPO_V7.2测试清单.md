# DRVOI-RHPPO V7.2 测试清单

- [x] V7.1 P1Q no-go 已冻结
- [x] V7.2 配置、表示、种子和门槛在数据生成前写入
- [x] 四种表示的维度和字段顺序有单元测试
- [x] 同组90动作共享完整环境状态、策略隐状态和低层隐状态
- [x] train/validation/formal-test seed 强隔离
- [x] formal-test 在开发冻结前由代码拒绝生成
- [x] 所有 MC 分支到达真实终止
- [x] 四种 critic 使用相同训练预算和 validation 早停规则
- [x] 输出训练拟合、validation 排序、区间、遗憾和随机对照
- [x] oracle 与可部署表示分别作门禁判断
- [x] 开发失败时 P1Q/O0/长训练保持 false
