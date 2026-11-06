# DRVOI-RHPPO V7.1 P1Q 开发阶段执行结果

## 结论

V7.1 已完成 Q-ready 软件修复、HOT 结构化正/负控制、W2 train/validation TD 开发、同状态 90 动作终局 Monte Carlo 探针，以及两轮 critic 结构和数据覆盖修复。P1Q 开发门禁未通过，因此冻结为 `P1Q_DEVELOPMENT_NO_GO`。

确认性 Q test 未生成、未访问；O0、短训练和长训练均不授权。

## 已完成的修复

1. 修复跨 Δ 终端部分回报窗口，终端元组不 bootstrap。
2. 在 validation 上比较目标网络更新周期 1、2、4，冻结为 4 epoch。
3. 将逐元组 Q 数据构造张量化，开发运行由约 10 分钟降至约 31 秒。
4. 增加 HOT 结构化 argmax 翻转、决策无关扰动和随机动作对照。
5. 增加完整环境同状态 90 动作 MC 探针，复制飞机、导弹、武器、回合、RNG 相关状态，并要求所有分支到达真实终止。
6. 依次验证 action-embedding、factorized-dueling-pairwise 和 joint-dueling critic。
7. 建立独立 MC train/validation 采集协议；train seeds 为 43185/44185，validation seed 为 45185，formal-test seed 46185 保持锁定。

## 关键结果

| 阶段 | 上下文数 | 成对排序准确率 | 归一化 argmax 遗憾 | 结论 |
|---|---:|---:|---:|---|
| 完整终局固定探针（factorized） | 12 | 0.501 | 0.464 | 失败 |
| factorized MC 校准 | 192 train / 48 val | 0.525 | 0.471 | 失败 |
| joint MC 校准 | 192 train / 48 val | 0.510 | 0.426 | 失败 |
| joint MC 校准扩展 | 768 train / 48 val | 0.514 | 0.415 | 失败 |

预注册门槛为：排序准确率不低于 0.60 且 95% 区间下界高于 0.50；归一化 argmax 遗憾不高于 0.10，并优于随机合法动作。扩展数据后仍远未达到门槛。

## 失败解释

W2 TD 开发的回合级 Spearman 可以通过，但它主要反映状态好坏，不能证明同一状态下动作价值排序正确。直接加入同状态 MC 监督后，训练拟合改善而独立 validation 不改善；训练上下文由 192 扩大到 768 后仍无稳定泛化。这表明当前 `belief + action + 12-D response` 上下文对未来动作价值不具备充分、平滑且可识别的状态语义。

当前结果不支持通过继续增加 epoch、降低阈值或查看 formal test 来推进。下一步若继续研究，应新建 V7.2 协议，先验证更充分的未来状态表示（例如可审计的未来 belief/低层控制器状态或其预测量）能否在开发集上恢复反事实动作价值，再重新预注册并生成新的确认性 test。

机器可读冻结结果见 `results/drvoi_rhppo_v7_1/p1q/p1q_development_no_go.json` 和 `results/drvoi_rhppo_v7_1/p1q/development_freeze.json`。
