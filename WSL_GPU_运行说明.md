# WSL GPU/CPU运行说明

## 运行架构

JSBSim飞行动力学只能在CPU运行，不能迁移到CUDA。本平台采用以下异构流水线提高GPU利用率：

```text
多个WSL CPU进程：并行运行JSBSim环境
                 ↓ 批量观测
主进程RTX GPU：批量策略推理、GAE后的PPO大批量更新、AMP
                 ↓ 批量动作
多个WSL CPU进程：继续推进物理仿真
```

低层分层控制器保留在各CPU环境进程中；只有需要训练的高层PPO位于GPU。不要把GPU利用率低误判为JSBSim没有正确安装。

## 本机状态

当前已检测到：

- WSL2 Ubuntu 24.04；
- Linux内核6.18；
- NVIDIA GeForce RTX 4090，显存24 GB；
- Windows NVIDIA驱动610.88；
- WSL GPU透传正常；
- WSL Python 3.12.3；
- 已有虚拟环境`~/.venvs/bvr1022`；
- 其中PyTorch为`2.14.1+cu130`，并已成功识别RTX 4090。

本机实测结果：

- CUDA矩阵计算：通过；
- JSBSim环境reset/step：通过；
- GPU PPO短测：4个并行环境，512个智能体转移，4次有效导弹发射，约207.4次环境决策/秒；
- GPU短测中高层PPO参数发生更新，冻结低层控制器参数变化为0；
- WSL CPU回退短测：通过；
- WSL完整回归测试：56项全部通过。

## 一键安装

在Ubuntu WSL终端中进入项目目录：

```bash
cd '/mnt/d/Ph.D/毕业/WVR-1102/WVR-Env'
bash scripts/setup_wsl_gpu.sh
```

脚本会优先复用本机已有环境：

```text
~/.venvs/bvr1022
```

若该环境不存在，才会新建`~/.venvs/wvr-env`。也可用`WVR_VENV=/path/to/venv`显式指定。虚拟环境位于Linux文件系统，可避免把大量Python小文件放在`/mnt/d`导致导入和元数据访问变慢。脚本检测到可用的CUDA版PyTorch时会直接复用，只补齐项目依赖，最后执行CUDA矩阵运算和JSBSim一步测试。

默认PyTorch wheel索引是CUDA 13.0。如官方稳定wheel发生变化，可覆盖：

```bash
WVR_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
  bash scripts/setup_wsl_gpu.sh
```

WSL使用Windows侧NVIDIA驱动映射出的CUDA驱动。不要在WSL内安装`nvidia-driver`或包含Linux驱动的`cuda-drivers`元包。

## 诊断

```bash
source ~/.venvs/bvr1022/bin/activate
python -m benchmarks.diagnose_wsl_runtime --require-cuda --environment-smoke
```

成功时必须出现：

```text
cuda_available=True
cuda_compute=PASS
environment=PASS
```

## GPU并行PPO测试

```bash
bash scripts/run_wsl_gpu_ppo.sh
```

默认设置：

- 16个CPU环境进程（CPU不足时自动减少）；
- 每个worker只使用1个BLAS/PyTorch CPU线程；
- 主进程使用CUDA；
- PPO更新启用FP16 AMP；
- rollout长度512；
- minibatch 4096。

可以根据CPU和GPU监控结果调整：

```bash
WVR_NUM_ENVS=32 \
WVR_ROLLOUT_STEPS=1024 \
WVR_MINIBATCH_SIZE=8192 \
WVR_UPDATES=20 \
bash scripts/run_wsl_gpu_ppo.sh
```

另开终端监控：

```bash
watch -n 1 nvidia-smi
htop
```

调优顺序应为：先增加`WVR_NUM_ENVS`，直到CPU接近饱和或吞吐不再提升；再增加rollout和minibatch，使PPO更新阶段形成较大的GPU矩阵计算。不要仅为提高GPU占用率而随意扩大网络，因为那会改变论文算法容量和比较公平性。

## CPU模式

启动脚本可以通过环境变量切换到WSL CPU：

```bash
WVR_DEVICE=cpu WVR_NUM_ENVS=8 bash scripts/run_wsl_gpu_ppo.sh
```

同一训练入口也可直接在Windows、WSL CPU或无GPU Linux运行：

```bash
python -m benchmarks.train_ppo_1v1_shoot_smoke \
  --device cpu --no-amp --num-envs 8 --minibatch-size 1024
```

当前脚本是训练链路与性能验证入口，不包含LAG完整FSP对手池、长期checkpoint管理和正式多种子实验编排。
