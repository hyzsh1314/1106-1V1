# 1v1 WVR UAV Air-Combat Research Platform

本仓库包含无人机近距空战 1v1 研究平台、Torch 6DoF GPU 环境、JSBSim 参考后端、导弹与奖励模型，以及 PPO、分层 PPO、RHPPO 和 DRVOI-RHPPO 的实验实现与测试。

## 主要目录

- `envs/torch6dof/`：批量化 Torch 6DoF 与导弹环境，面向 GPU 并行训练。
- `envs/JSBSim/`：JSBSim 参考环境及低层控制模型。
- `benchmarks/`：训练、评估、消融、性能测试和数据分析脚本。
- `benchmarks/drvoi/`：独立的 DRVOI/RHPPO 响应、Q、VOI 与循环策略组件。
- `configs/`：平台和 DRVOI-RHPPO V6 适配配置。
- `scripts/`：Windows/WSL 训练、评估和监督脚本。
- `tests/`：环境、武器、奖励、PPO、分层 PPO 与 DRVOI 单元/回归测试。
- `docs/`：平台建模、验证协议和算法适配说明。
- `runs/drvoi_rhppo/`：经过筛选的小型 JSON/Markdown 实验记录；checkpoint 与大日志不入库。

## 当前 DRVOI-RHPPO 结论

V6 WVR 适配已完成 P0 和 P1。G3–G7 通过，但回合级 G1 动作条件 ΔNLL 与 G2 side 校准未通过，因此严格按照预注册停止规则，未启动 O0/O2/O3/O4 的 1000 万步长训练。详见：

- `runs/drvoi_rhppo/P1_GATE_DECISION.md`
- `runs/drvoi_rhppo/p1-gate-report.json`
- `docs/DRVOI_RHPPO_V6_WVR平台适配与执行协议.md`

## 快速验证

```bash
python -m pytest -q
```

当前完整回归测试结果为 155 项通过。WSL GPU 使用方式见 `WSL_GPU_运行说明.md`。

## 数据与模型说明

仓库保留运行配置、聚合指标、门禁 JSON、CSV 和小型分析图。训练 checkpoint、大型日志、缓存、Tacview 文件与原始轨迹默认忽略，以避免把不可审计的大型二进制产物混入源码版本。低层控制所必需的小型 `.pt` 模型位于 `envs/JSBSim/model/`。
