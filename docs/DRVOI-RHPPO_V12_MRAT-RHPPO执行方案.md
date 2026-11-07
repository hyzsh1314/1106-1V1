# V12：MRAT-RHPPO 模型驱动可达性仲裁与可学习终止执行方案

> 版本：V12.0-protocol-draft  
> 日期：2026-11-07  
> 状态：执行前方案；本文件不代表算法已经实现或实验已经完成  
> 工作名称：**MRAT-RHPPO（Model-based Reachability Arbitration and Termination RHPPO，模型驱动可达性仲裁与终止循环分层 PPO）**

## 1. 执行结论

现有数据已经足以停止围绕固定阈值继续堆叠 V11.4、V11.5。V11.3 的价值不在于 `6 km、17 m/s、连续 12 步` 本身，而在于它证明了三个机制事实：

1. Evasive 场景并非武器链或动作空间不可解。纯追击为 `0/192`，LOS 超前拦截为 `48/192`；
2. 单步切换无效，持续执行拦截 option 才能恢复非零击毁；
3. 全局或过早接管会损伤 Hold/Pursuit，因而真正的问题是**何时进入 LOS 专家、保持多久、何时交还学习策略**。

V12 将研究对象从“修改 PPO 奖励或重新学习完整低层动作”收敛为一个可证伪问题：

> 在不修改动力学、武器、观测和环境奖励的前提下，能否用模型驱动的连续可达性量代替固定距离—闭合速度阈值，并用半马尔可夫高层策略学习专家进入与终止，使策略在 Evasive 上保留拦截能力，同时减少 Hold/Pursuit 中不必要的 LOS 接管？

V12 的两个执行专家保持明确：

- 学习专家：V9 COACH-RHPPO，记为 \(\pi_{\mathrm L}\)；
- 物理专家：确定性三维 LOS 超前拦截控制器，记为 \(\pi_{\mathrm I}\)。

V12 新增的不是第三套低层飞行策略，而是：

- 短时域物理可达性教师；
- 可达性蒸馏网络及不确定性下界；
- 专家路由策略；
- 可学习 option 终止器；
- 防止抖振、过度接管和不可达状态持续化的约束目标。

V7.1–V7.3 的响应预测与动作 Q 失败结果继续作为负结果保留，不重新包装为 V12 的正证据。V12 预测的是**技能级物理可达性**，而不是 90 个联合动作的终局回报排序，因此与已失败的动作级 DRVOI 假设不同。

---

## 2. 现有证据与算法决策

### 2.1 平台与当前协议

当前 Torch WVR 环境采用：

| 项目 | 当前实现 |
|---|---|
| 动力学 | GPU 批量 Torch F-16 六自由度动力学 |
| 仿真频率 | 60 Hz |
| 高层决策周期 | 12 个仿真子步，即 0.2 s、5 Hz |
| 单局上限 | 1000 个决策步，即 200 s |
| 武器 | 每机 2 枚导弹，PN 制导、推力/阻力/质量消耗、脱靶与命中状态 |
| 观测 | V4，39 维；包括 LOS 角/角速率、闭合速度和来袭导弹特征 |
| 动态发射包线 | 启用 |
| 奖励 | 竞争式零和 PBRS + 终局胜负；V12 不修改奖励 |
| 并行环境 | 默认 96 |

当前发射包线和可达性计算需要的量已经存在于执行观测中：距离、进入角、目标方位角、LOS 角速率、闭合速度、剩余导弹、合法发射标志和来袭导弹特征。V12 不读取真实对手策略标签，也不允许使用未来状态作为在线输入。

### 2.2 V7–V11.3 数据链

| 阶段 | 关键数据 | 支持的结论 |
|---|---|---|
| V7 B0，3 种子、每种子约 100.76 万步 | Hold `31.08%±16.06 pp`；Pursuit `73.61%±1.20 pp`；Evasive `0%`，超时 `84.72%` | 骨干可训练，但困难对手存在稳定覆盖缺口 |
| V7.1 | 三步动作历史的响应信息可预测；后续动作 Q 排序未通过 | 预测信息不等于可用的决策价值 |
| V7.2/V7.3 | 增加状态和改变 3/12/30/120 步回报仍未恢复低遗憾动作排序 | 不应继续扩张动作级 VOI/Q 主线 |
| V8，seed 85、约 100.76 万步 | Pursuit `75.00%→83.85%`；Hold 超时 `8.33%→20.83%`；Evasive 仍 `0%` | 软 VOI 改善局部效率但未解决持续拦截 |
| V9 COACH-RHPPO，seed 85 | Hold `60.94%`；Pursuit `89.06%`；Evasive `0%` | 学习分层对容易场景有效，困难场景仍退化 |
| V10 强制 LOS | Hold `40.10%`；Pursuit `62.50%`；Evasive `25.00%` | LOS 语义可实现拦截，但全局覆盖破坏已有能力 |
| V11 单步仲裁 | Evasive 介入 `65.30%`，胜率仍 `0%` | 触发次数不是充分条件；单步切换破坏 option 语义 |
| V11.1 滞回 | Hold `76.56%`；Pursuit `84.38%`；Evasive `3.65%` | 延长执行有效，但终止仍可能过早 |
| V11.2 整局锁定 | Hold `71.35%`；Pursuit `79.69%`；Evasive `26.04%` | 持续拦截最强，但简单场景存在过度接管 |
| V11.3 持续证书 | Hold `69.27%`；Pursuit `82.81%`；Evasive `21.35%` | 过滤瞬时触发后形成当前较均衡方案 |

V9–V11.3 的结果都只覆盖训练种子 85，并反复使用开发评估种子 `10085`。其中 V9.1 和 V10 分别完成了自己的开发训练，V11–V11.3 则复用冻结的 V9 checkpoint 并只改变运行时仲裁。它们足以指导 V12 设计，但不能支持算法显著优越、跨种子稳定或对未知对手泛化的结论。

### 2.3 从数据到设计的逻辑链

现有证据支持以下设计，而不支持继续无边界调阈值：

1. **专家选择粒度应为完整技能而非单步动作。** 动作级 Q 在 V7.1–V7.3 失败；LOS 技能级对照却产生 `0/192→48/192` 的可观测差异。
2. **持续时间是核心状态变量。** V11 与 V11.2 的主要差异不是专家类型，而是专家能否持续执行。
3. **进入和退出必须分开建模。** V11.2 说明长保持有效，V10/V11.2 的简单场景损失说明永久保持不是最终解。
4. **物理模型用于定义可达性，不用于替代终局价值。** 预测能否在有限时域内形成合法发射条件，比直接估计终局胜负更容易审计。
5. **固定阈值只能作为安全下界。** `6 km/17 m/s/12 步` 是开发启发式，不应继续被描述为一般性最优参数。

### 2.4 与公开研究的技术边界

V12 不把“分层 PPO、专家知识或动态切换”单独作为创新。Lockheed Martin 的空战系统已经结合分层强化学习、模块化策略和专家奖励塑形；E2L/H3E 已将行为克隆或规则知识融入分层空战学习；MEOL 已研究战术 option 的动态选择与终止；ICS-RL 已采用多专家 Max-Advantage 动态切换；基于控制屏障函数的空战研究也已使用学习价值限制危险动作。[Lockheed Martin](https://arxiv.org/abs/2105.00990)，[E2L](https://www.sciencedirect.com/science/article/pii/S092523122400362X)，[H3E](https://www.sciencedirect.com/science/article/abs/pii/S0957417423035868)，[MEOL](https://doi.org/10.3390/drones9050384)，[ICS-RL](https://arxiv.org/abs/2603.00974)，[air-combat CBF](https://kth.diva-portal.org/smash/get/diva2%3A1547485/FULLTEXT01.pdf)

V12 的待验证差异是：在带真实导弹飞行过程的 WVR 仿真中，用动态发射包线和短时域物理展开定义两个完整技能的可达性差，再以持续保护下界约束循环高层策略学习进入和终止。该组合是否形成足够的论文创新，最终仍取决于多种子效果、机制消融和公开文献检索，方案阶段不作“首次”声明。

---

## 3. 研究问题和可证伪假设

### 3.1 研究问题

- **RQ1：** 短时域模型可达性是否比当前二值 guard 更准确地预测“未来有限时间内能否形成合法发射条件”？
- **RQ2：** 学习终止器能否在保持 Evasive 拦截能力的同时降低 Hold/Pursuit 的不必要专家接管？
- **RQ3：** 性能变化主要来自可达性表示、持续执行、可学习终止还是额外训练预算？
- **RQ4：** V12 是否能在未参与开发的初始条件、对手参数和动力学扰动下保持跨场景平衡？

### 3.2 假设

- **H1（可达性识别）：** 在按完整回合隔离的验证集上，V12 可达性模型对“\(H\) 步内出现合法发射机会”的 Brier 分数和 AUROC 均优于 V11.3 二值 guard。
- **H2（困难场景）：** 相同训练预算下，V12 的 Evasive 胜率高于 V9，并降低 Evasive 超时率。
- **H3（能力保持）：** 相比 V11.2 整局锁定，V12 减少 Hold/Pursuit 的 LOS 接管占比和平均持续时间，同时不降低三场景宏平均胜率。
- **H4（机制归因）：** 删除可学习终止器会复现过度接管；删除持续保持会复现 V11 单步仲裁失败；删除物理可达性输入会增加错误切换和技能级反事实遗憾。
- **H5（训练预算控制）：** 在累计在线环境决策预算一致时，V12 相对 V9-continued 的差异不能仅由初始化 checkpoint 或额外低层训练步数解释。

H1–H5 是待检验命题。执行前不得把它们改写为已证明的算法优势。

---

## 4. 问题形式化

### 4.1 决策过程

用 \(n\) 表示 60 Hz 仿真子步，用 \(t\) 表示 5 Hz 策略决策步。两者关系为：

\[
n=12t,\qquad \Delta t=\frac{12}{60}=0.2\ \mathrm{s}.
\]

环境真实状态为 \(x_t\)，学习方只能获得 39 维局部观测 \(o_t\in\mathbb R^{39}\)。循环编码器产生历史表示：

\[
h_t=f_{\mathrm{GRU}}(h_{t-1},o_t,a_{t-1},d_{t-1}),
\]

其中 \(d_{t-1}\) 是上一决策步的终止/重置标志。V12 不把 \(h_t\) 称为完备 belief；它只是由可用历史形成的循环特征。

环境动作是四个离散头组成的混合离散动作：

\[
a_t=(a_t^{\mathrm{alt}},a_t^{\mathrm{head}},a_t^{\mathrm{speed}},a_t^{\mathrm{fire}}).
\]

低层冻结控制器将前三个高层命令转换为实际舵面和油门输入；发射头必须通过环境合法发射掩码。环境奖励 \(r_t^{\mathrm{ext}}\) 和终止条件保持当前实现不变。

### 4.2 半马尔可夫专家 option

专家集合为：

\[
\mathcal E=\{\mathrm L,\mathrm I\},
\]

其中 \(\mathrm L\) 为 COACH-RHPPO，\(\mathrm I\) 为确定性 LOS 超前拦截。高层 option 边界为 \(\tau_k\)，第 \(k\) 个 option 为 \(e_k\)，持续时间为：

\[
D_k=\tau_{k+1}-\tau_k.
\]

在 \(\tau_k\le t<\tau_{k+1}\) 内：

\[
a_t=
\begin{cases}
\pi_{\mathrm L}(a_t\mid h_t,o_t), & e_k=\mathrm L,\\
\pi_{\mathrm I}(o_t), & e_k=\mathrm I.
\end{cases}
\]

LOS 专家的发射头仍由 \(o_t[23]\) 的合法发射标志约束。任何路由器均不得绕过环境武器状态机。

---

## 5. 模型驱动连续可达性

### 5.1 即时动态发射裕度

从当前环境实现恢复物理量：

\[
R_t=10{,}000\,o_t[13],\qquad
V_{c,t}=340\,o_t[28],
\]

其中 \(R_t\) 的单位为 m，\(V_{c,t}>0\) 表示正在闭合。LOS 角速度为：

\[
\dot\lambda_t=\sqrt{o_t[26]^2+o_t[27]^2}.
\]

当前动态最大攻击距离为：

\[
R_{\mathrm{dyn},t}=\operatorname{clip}\left(
7500+8\operatorname{clip}(V_{c,t},-150,550)
-2200\frac{1-\cos\theta_{\mathrm{TA},t}}{2}
-5000\operatorname{clip}\left(\frac{\dot\lambda_t}{0.20},0,1\right),
2500,14000\right).
\]

定义归一化距离、进入角和闭合裕度：

\[
m_R(t)=\frac{R_{\mathrm{dyn},t}-R_t}{14000},
\]

\[
m_A(t)=\frac{\alpha_{\max}-\alpha_{\mathrm{AO},t}}{\alpha_{\max}},
\qquad \alpha_{\max}=45^\circ,
\]

\[
m_C(t)=\tanh\left(\frac{V_{c,t}}{250}\right),\qquad
m_\lambda(t)=1-\operatorname{clip}\left(\frac{\dot\lambda_t}{0.20},0,1\right).
\]

即时发射可行性使用平滑下界而不是任意加权和：

\[
q_t=\operatorname{softmin}_{\kappa}
\{m_R(t),m_A(t),m_C(t),m_\lambda(t)\},
\]

\[
\operatorname{softmin}_{\kappa}(z_1,\ldots,z_M)
=-\frac{1}{\kappa}\log\left[
\frac{1}{M}\sum_{j=1}^{M}\exp(-\kappa z_j)\right].
\]

\(q_t>0\) 表示各项均趋于可行，但它不是导弹必杀概率，也不构成真实系统安全保证。

### 5.2 技能级短时域物理教师

在 option 边界复制当前 Torch 环境张量状态和 V9 执行状态，并分别让两个专家执行 \(H_R\) 个决策步。V9 执行状态至少包括 GRU hidden、上一执行动作、内部 active option、option age、响应历史和低层控制器 hidden；否则两个分支并不是从同一决策状态出发。初始建议：

\[
H_R=30\quad (6\ \mathrm{s}).
\]

对手未来运动不读取真实策略标签，而在预定义假设集合中展开：

\[
\mathcal U=\{\text{hold},\text{pursuit},\text{evasive-left},
\text{evasive-right},\text{bounded-turn continuation}\}.
\]

令 \(\zeta_t\) 表示 V9、响应历史和低层控制器的内部执行状态，物理教师的增广状态为 \(\chi_t=(x_t,\zeta_t)\)。策略仍然只读取观测及自身可用的循环状态，不允许把环境真实状态直接输入策略。对于专家 \(e\) 和对手假设 \(u\)，模型轨迹为：

\[
\chi_{t+j+1}^{e,u}=F_{\mathrm{aug}}\left(
\chi_{t+j}^{e,u},
\pi_e(O(x_{t+j}^{e,u}),\zeta_{t+j}^{e,u}),
\pi_u(O_u(x_{t+j}^{e,u}))\right).
\]

单轨迹可达性目标定义为：

\[
\rho_t^{e,u}=
\max_{1\le j\le H_R}\left[
\gamma_R^{j-1}q(x_{t+j}^{e,u})
+b_{\mathrm{legal}}\mathbf 1\{\ell_{t+j}^{e,u}=1\}
-b_{\mathrm{crash}}\mathbf 1\{c_{t+j}^{e,u}=1\}
\right],
\]

其中 \(\ell=1\) 表示形成合法发射机会，\(c=1\) 表示预测坠毁。推荐初值为 \(\gamma_R=0.98\)，\(b_{\mathrm{legal}}=1\)，\(b_{\mathrm{crash}}=1\)；这些只定义教师标签，不进入环境奖励。

对手假设上的保守聚合采用温度可控的 soft-min：

\[
\rho_t^e=-\tau_U\log\sum_{u\in\mathcal U}
w_u\exp\left(-\frac{\rho_t^{e,u}}{\tau_U}\right).
\]

同时记录两个辅助监督目标。先在每个对手假设上定义：

\[
y_t^{e,u}=\mathbf 1\{\exists j\le H_R:\ell_{t+j}^{e,u}=1\},
\]

\[
T_t^{e,u}=\min\{j:\ell_{t+j}^{e,u}=1\}.
\]

不向路由器泄露真实对手类型时，合法机会概率使用假设权重聚合：

\[
y_t^e=\sum_{u\in\mathcal U}w_u y_t^{e,u}.
\]

到达时间只在成功假设上计算条件加权均值：

\[
T_t^e=
\frac{\sum_{u\in\mathcal U}w_u y_t^{e,u}T_t^{e,u}}
{\sum_{u\in\mathcal U}w_u y_t^{e,u}+\varepsilon}.
\]

失败假设保留为右删失样本。不能把未形成合法发射机会的 \(T_t^{e,u}\) 简单填为 \(H_R\) 后当作普通回归真值。

### 5.3 可达性蒸馏与不确定性下界

在线逐边界展开 \(2|\mathcal U|H_R\) 条轨迹会降低训练吞吐，因此用三成员轻量 ensemble 蒸馏教师：

\[
f_{\phi_j}(g_t,e)\rightarrow
(\hat\rho_{t,j}^e,\hat p_{t,j}^e,\hat T_{t,j}^e),
\]

\[
g_t=[o_t,h_t,e_{t-1},D_{t-1},q_t,\Delta q_t].
\]

实际实现将该输入展开为当前观测、V9 hidden、上一实际动作、V9 内部 active option/option age、低层控制器 hidden 的压缩特征、外层专家及持续时间。它们都是执行系统自身状态；不能加入对手脚本 ID、未来轨迹或环境隐藏真值。

每个成员的损失为：

\[
\mathcal L_{\mathrm{reach}}^{(j)}=
\lambda_\rho\operatorname{Huber}(\hat\rho_{t,j}^e,\rho_t^e)
+\lambda_p\operatorname{BCE}(\hat p_{t,j}^e,y_t^e)
+\lambda_T y_t^e\operatorname{Huber}(\hat T_{t,j}^e,T_t^e)
+\lambda_{\mathrm{cens}}\mathcal L_{\mathrm{censored}}.
\]

ensemble 均值和标准差为：

\[
\mu_t^e=\frac{1}{J}\sum_{j=1}^{J}\hat\rho_{t,j}^e,
\qquad
\sigma_t^e=\sqrt{\frac{1}{J-1}\sum_j(\hat\rho_{t,j}^e-\mu_t^e)^2}.
\]

路由器使用下置信界：

\[
\underline\rho_t^e=\mu_t^e-z_\alpha\sigma_t^e.
\]

\(z_\alpha\) 只在开发集选择一次并冻结。论文中应称其为“经验校准的可达性下置信界”，不得称为形式化安全证书。

---

## 6. 可达性仲裁与可学习终止

### 6.1 路由状态

路由器输入为：

\[
s_t^{\mathrm G}=
[h_t,o_t,\zeta_t^{\mathrm{exec}},
\underline\rho_t^{\mathrm L},
\underline\rho_t^{\mathrm I},
\Delta\rho_t,e_{t-1},D_{t-1},u_t^{\mathrm{missile}}],
\]

其中：

\[
\Delta\rho_t=\underline\rho_t^{\mathrm I}
-\underline\rho_t^{\mathrm L},
\]

\(\zeta_t^{\mathrm{exec}}\) 是上一实际动作、V9 内部 option 状态和低层控制器 hidden 的执行期压缩特征；\(u_t^{\mathrm{missile}}\) 由 V4 观测中来袭导弹数量、方位、闭合速度和 time-to-go 构成。所有输入在执行时均可获得。

### 6.2 专家选择与终止

在 option 边界，高层路由策略为：

\[
e_k\sim\pi_{\omega}(e\mid s_{\tau_k}^{\mathrm G}),
\qquad e\in\{\mathrm L,\mathrm I\}.
\]

最低保持时间取：

\[
D_{\min}=6\quad (1.2\ \mathrm{s}),
\]

达到最低保持后，终止器输出：

\[
b_t\sim\operatorname{Bernoulli}
\left(\beta_\psi(s_t^{\mathrm G},e_k,D_k)\right).
\]

若 \(b_t=1\)，下一步成为新的 option 边界。最大保持时间 \(D_{\max}\) 仅用于避免实现错误造成永久锁死，建议取 150 步（30 s），不能替代学习终止。

### 6.3 物理保护下界

V11.3 保留为保护下界，而不是主决策器。定义：

\[
z_t^{\mathrm{force}}=
\mathbf 1\left\{
\underline\rho_t^{\mathrm L}<\rho_{\mathrm{bad}}
\ \land\ 
\Delta\rho_t>\delta_\rho
\right\}.
\]

只有当该条件连续 \(K_{\mathrm{force}}\) 步成立时才强制进入 LOS 专家：

\[
\sum_{j=0}^{K_{\mathrm{force}}-1}
z_{t-j}^{\mathrm{force}}=K_{\mathrm{force}}.
\]

初始 \(K_{\mathrm{force}}=12\)，与 V11.3 的 2.4 s 持续条件对齐。区别在于触发量由固定距离—闭合速度变为两种专家的可达性差。

强制接管期间，只有满足恢复条件：

\[
\underline\rho_t^{\mathrm L}>\rho_{\mathrm{safe}},
\qquad
\rho_{\mathrm{safe}}>\rho_{\mathrm{bad}},
\]

并连续 \(K_{\mathrm{release}}\) 步后，终止器才拥有交还权。迟滞区间防止边界抖振。强制接管、路由器主动选择和终止器释放必须分别记录。

### 6.4 动作执行优先级

每个决策步按以下固定顺序执行：

1. 计算 V9 学习专家动作；
2. 计算 LOS 专家动作；
3. 更新物理保护状态；
4. 若强制接管成立，执行 LOS；否则执行当前 option；
5. 发射头通过合法掩码；
6. 记录原始动作、最终动作、接管来源和 reachability 特征；
7. 环境执行 12 个 60 Hz 子步。

V12 是 V9 内部分层策略之外的外层路由器。当 LOS 专家接管时，V9 仍以当前观测和**最终实际执行动作**更新 GRU、响应历史、内部 option 及 option age，但其候选动作不送入环境，也不产生低层 actor 梯度。这种影子同步保证交还 V9 时循环状态没有停留在接管前。该顺序和状态更新语义必须在训练、物理教师和评估脚本中一致。

---

## 7. 半马尔可夫 PPO 与约束目标

### 7.1 option 片段回报

第 \(k\) 个 option 的折扣片段回报为：

\[
R_k=\sum_{j=0}^{D_k-1}\gamma^j
r_{\tau_k+j}^{\mathrm{ext}},
\qquad \Gamma_k=\gamma^{D_k}.
\]

高层评论家输出专家条件价值 \(Q_\nu(s^{\mathrm G},e)\)，并定义路由策略下的状态基线：

\[
V_\nu(s^{\mathrm G})=
\sum_{e\in\mathcal E}\pi_\omega(e\mid s^{\mathrm G})
Q_\nu(s^{\mathrm G},e).
\]

高层 TD 误差为：

\[
\delta_k=R_k+\Gamma_k(1-d_k)
V_\nu(s_{\tau_{k+1}}^{\mathrm G})
-V_\nu(s_{\tau_k}^{\mathrm G}),
\]

其中 \(d_k\) 表示 option 片段后环境是否终止。可变时长 GAE 为：

\[
A_k^{\mathrm G}=\delta_k+
(\gamma\lambda)^{D_k}(1-d_k)A_{k+1}^{\mathrm G}.
\]

路由器 PPO 比率为：

\[
r_k(\omega)=
\frac{\pi_\omega(e_k\mid s_{\tau_k}^{\mathrm G})}
{\pi_{\omega_{\mathrm{old}}}(e_k\mid s_{\tau_k}^{\mathrm G})}.
\]

高层目标为：

\[
L_{\mathrm{route}}^{\mathrm{PPO}}=
\mathbb E_k\left[
\min\left(r_kA_k^{\mathrm G},
\operatorname{clip}(r_k,1-\epsilon,1+\epsilon)A_k^{\mathrm G}
\right)\right].
\]

终止器只在满足最低保持时间、未被物理保护锁定的步上更新。其优势使用“继续当前 option”与“在当前状态重新选择”的价值差：

\[
A_t^{\mathrm{term}}=
Q_\nu(s_t^{\mathrm G},e_t)-V_\nu(s_t^{\mathrm G}).
\]

若 \(A_t^{\mathrm{term}}>0\)，继续当前 option 更有利；若小于零，终止更有利。实现时必须统一终止损失的符号，并用合成样例验证梯度方向。

采用最小化形式时，终止损失定义为：

\[
\mathcal L_\beta=
\mathbb E_t\left[
m_t^{\mathrm{term}}\,
\beta_\psi(s_t^{\mathrm G},e_t,D_t)
\left(A_t^{\mathrm{term}}+\xi_{\mathrm{delib}}\right)
\right],
\]

其中 \(m_t^{\mathrm{term}}\) 只在最低保持时间已满足、没有强制锁定时为 1，\(\xi_{\mathrm{delib}}\ge0\) 是切换审议成本。最小化该式时，正优势抑制终止概率，负优势提高终止概率。

### 7.2 约束与正则项

定义四类诊断成本：

\[
c_t^{\mathrm{bad}}=
\max(0,\rho_{\mathrm{bad}}-\underline\rho_t^{e_t}),
\]

\[
c_t^{\mathrm{int}}=\mathbf 1\{e_t=\mathrm I\},
\qquad
c_t^{\mathrm{switch}}=\mathbf 1\{e_t\ne e_{t-1}\},
\]

\[
c_t^{\mathrm{crash}}=\mathbf 1\{\text{aircraft crashes at }t\}.
\]

优化问题为：

\[
\max_{\omega,\psi,\theta}\quad
J_R
-\eta_{\mathrm{int}}J_{\mathrm{int}}
-\eta_{\mathrm{switch}}J_{\mathrm{switch}},
\]

\[
\text{s.t.}\quad
J_{\mathrm{bad}}\le d_{\mathrm{bad}},
\qquad
J_{\mathrm{crash}}\le d_{\mathrm{crash}}.
\]

拉格朗日乘子使用投影更新：

\[
\lambda_c\leftarrow
\left[\lambda_c+\alpha_\lambda(J_c-d_c)\right]_+.
\]

接管和切换采用小权重正则而不是硬约束，避免路由器为了减少接管而重新退化为 V9。不可达成本和坠毁成本使用约束，因为它们对应明确的物理失败模式。

### 7.3 反事实预训练

对同一 option 边界状态执行成对物理展开，得到：

\[
e_t^*=\arg\max_{e\in\mathcal E}\rho_t^e,
\qquad
w_t=|\rho_t^{\mathrm I}-\rho_t^{\mathrm L}|.
\]

路由器初始监督损失为：

\[
L_{\mathrm{CF}}=-\mathbb E_t\left[
\operatorname{stopgrad}(w_t)
\log\pi_\omega(e_t^*\mid s_t^{\mathrm G})
\right].
\]

\(L_{\mathrm{CF}}\) 只用于冷启动并在前 10% 训练预算内退火到零。物理教师不直接决定最终胜负策略，PPO 仍由真实环境外在回报更新。

### 7.4 总目标和梯度路径

最小化形式的总损失为：

\[
\mathcal L=
-L_{\mathrm{route}}^{\mathrm{PPO}}
-L_{\mathrm{low}}^{\mathrm{PPO}}
+c_V\mathcal L_V+c_\beta\mathcal L_\beta
+c_{\mathrm{CF}}\mathcal L_{\mathrm{CF}}
+c_R\mathcal L_{\mathrm{reach}}
+\lambda_{\mathrm{bad}}L_{\mathrm{bad}}
+\lambda_{\mathrm{crash}}L_{\mathrm{crash}}
+\eta_{\mathrm{int}}J_{\mathrm{int}}
+\eta_{\mathrm{switch}}J_{\mathrm{switch}}.
\]

梯度边界如下：

- LOS 专家始终冻结；
- 物理展开教师始终 `stop-gradient`；
- 可达性 ensemble 只由 \(L_{\mathrm{reach}}\) 更新；
- 路由器和终止器由高层 PPO、终止损失与退火的反事实预训练更新；
- V9 学习专家在第一阶段冻结；进入联合微调后，只在实际执行 \(e=\mathrm L\) 的样本上做低层 PPO 更新；
- 联合微调使用较小学习率和相对初始 V9 的 KL 锚定；
- 任何 LOS 执行步不得伪造为 V9 actor 采样动作来计算低层 PPO 比率。

---

## 8. 算法执行流程

### 8.1 总体伪代码

```text
Input:
    frozen/initialized V9 policy πL
    deterministic LOS expert πI
    Torch WVR transition model F
    reachability ensemble fφ
    router πω, termination βψ, high-level critic Vν

Stage A: build reachability dataset
    for each complete episode from V9/V11.3/scripted mixtures:
        at registered option boundaries:
            snapshot all environment tensors
            for expert e in {L, I}:
                for opponent hypothesis u in U:
                    restore identical snapshot
                    roll F for HR decisions
                    record ρ(e,u), legal-within-H, time-to-legal, crash
            save one paired record; keep episode/reset ID
    split train/validation/test by complete episode and reset ID

Stage B: pretrain and calibrate
    train reachability ensemble on train episodes
    select capacity/early stop on validation episodes
    calibrate zα and probability temperature on calibration episodes
    evaluate once on reachability test episodes
    pretrain router from paired expert labels; anneal this loss later

Stage C: constrained SMDP-PPO
    reset vector environments and recurrent states
    while environment decisions < budget:
        compute πL action and πI action
        update reachability estimates and physical-force counters
        at option boundary:
            sample/choose expert from πω unless force condition overrides
        execute selected expert action through legal fire mask
        step environment for 12 substeps
        store low-level transition and option-segment statistics
        if minimum dwell reached and release is legal:
            sample termination βψ
        update reachability model from replayed teacher labels
        update router/termination/value with variable-duration PPO
        update πL only on L-expert samples after frozen warm-up
        update constraint multipliers
        save latest.pt, progress.json and event-level metrics

Stage D: frozen evaluation
    disable sampling and all parameter updates
    evaluate paired reset IDs against fixed, randomized and held-out opponents
    save one record per episode and one record per option segment
```

### 8.2 训练阶段

| 阶段 | 预算 | 目的 | 是否允许改变算法 |
|---|---:|---|---|
| A0 单元测试 | 无训练 | 状态复制、恢复、终止梯度、动作掩码、resume 一致性 | 允许修工程错误 |
| A1 教师数据 | 约 5–10 万个成对边界状态 | 建立技能级物理标签 | 不看最终确认集 |
| A2 可达性模型 | 监督训练 | 校准概率、时延和 LCB | 只用 train/validation/calibration |
| B0 烟雾训练 | 49,152–98,304 步，seed 85 | 检查数值、存取和吞吐 | 允许修工程错误 |
| B1 开发筛选 | 100 万步，seeds 85/185/285 | 选择少量预注册超参数和消融 | 使用开发评估集 |
| C1 长训练 | 总在线预算 1000 万步，5 个训练种子 | 主算法与主要基线 | 配置冻结后不改 |
| C2 确认评估 | 未接触的 evaluation seeds | 形成论文结果 | 不允许回写训练选择 |

每个正式种子先生成相同的约 100.76 万步 V9 初始化 checkpoint。该初始化步数以及 A1 教师采集主轨迹实际执行的在线环境决策均计入 1000 万总在线环境决策预算；教师反事实分支另计为 (N_{\mathrm{model}})，不混入 (N_{\mathrm{env}})。V12 只继续训练到累计 1000 万步，而不是在 V9 的 100 万步和 A1 主轨迹之外再获得额外 1000 万步。V12 专属续训预算的前 10% 冻结 V9，只学习 reachability/router/termination；其余阶段解冻 V9 的学习专家分支，学习率取路由器的 0.1 倍，并加入对初始化策略的 KL 锚定。`V12-Frozen` 消融在全程冻结 V9 的条件下运行，用于区分“路由收益”和“额外低层微调收益”。

### 8.3 只保留三类硬停止条件

为避免再次形成过度门禁，只保留以下硬停止条件：

1. **软件正确性失败：** NaN/Inf、非法动作、武器事件不守恒、checkpoint 无法精确恢复、训练/评估动作优先级不一致；
2. **可达性模型无信息：** 在完整回合隔离测试上，AUROC 不高于随机或 Brier 分数劣于恒定基率预测；
3. **系统性灾难退化：** 100 万步三种子筛选中所有种子均出现坠毁率显著上升、合法发射链破坏或三场景均低于 V9。

其他指标只用于选择、解释和完整报告，不以观察结果后新增的阈值阻止实验。

### 8.4 初始超参数与冻结规则

下表是首轮实现初值，不是已经验证的最优值：

| 类别 | 参数 | 初值/候选 |
|---|---|---|
| 环境 | 并行环境、rollout | `96`, `256` 决策步 |
| 环境 | 仿真/决策频率 | `60 Hz / 5 Hz` |
| PPO | \(\gamma,\lambda,\epsilon\) | `0.99, 0.95, 0.20` |
| PPO | 序列/烧入/epoch | `64 / 20 / 2` |
| 路由器 | 学习率、熵系数、target KL | `2e-4, 1e-3, 0.015` |
| V9 微调 | 学习率 | `2e-5` |
| Reachability | ensemble 成员 | `3` |
| Reachability | \(H_R\) | 开发比较 `{12,30,60}`，选择最短有效时域 |
| Reachability | \(\kappa\) | `10` |
| Option | \(D_{\min},D_{\max}\) | `6, 150` 决策步 |
| Shield | \(K_{\mathrm{force}},K_{\mathrm{release}}\) | `12, 6` 决策步 |
| LCB | \(z_\alpha\) | 校准集在 `{1.0,1.64}` 中选择 |
| 预算 | 总在线环境决策 | `10,000,000`，包含 V9 初始化训练 |

仅允许在 100 万步开发阶段选择 \(H_R,z_\alpha,\rho_{\mathrm{bad}},\rho_{\mathrm{safe}},\delta_\rho\) 和两个接管/切换正则系数。阈值选择目标固定为“降低技能级 reachability regret，同时约束切换频率”，不得直接在确认胜率上逐点扫描。进入正式训练后配置写入 JSON、计算 SHA-256 并冻结。

---

## 9. 对比算法

### 9.1 必须完成的主对比

| ID | 方法 | 比较目的 | 预算要求 |
|---|---|---|---|
| B1 | 标准 PPO | 非循环、非分层基线 | 与 V12 相同环境步数 |
| B2 | 现有分层 PPO | 检验普通技能分层是否足够 | 与 V12 相同环境步数 |
| B3 | B0-RHPPO | 循环策略骨干 | 同种子、累计到相同总在线预算 |
| B4 | V9 COACH-RHPPO | 当前最强纯学习专家 | 同种子、累计到相同总在线预算 |
| B5 | V11.3 RACE-RHPPO | 固定持续证书仲裁 | 包装对应种子的等预算 V9 checkpoint |
| B6 | V11.2 Episode-latched | 困难场景上界/过度接管参考 | 包装对应种子的等预算 V9 checkpoint |
| M1 | V12 MRAT-RHPPO | 完整方法 | 主要方法 |

旧 checkpoint 可以用于机制复核，但正式 1000 万步对比不得把 100 万步旧结果与 1000 万步新方法直接当作等预算算法比较。标准 PPO 和分层 PPO 也按“包括其初始化/预训练在内的累计在线环境决策”计费；任何额外行为克隆或模型展开计算单独报告。

### 9.2 强结构基线

为回应相邻文献，新增两个独立脚本，但不声称是原论文逐行复现：

1. **Learned-option baseline：** 两个相同专家、相同最短持续时间，由普通 option policy 学习选择和终止，但不输入模型可达性、没有物理保护下界。它对应动态 option 终止这一技术族，与空战 MEOL 的“战术选择—终止”思想可比。[相关论文](https://doi.org/10.3390/drones9050384)
2. **Max-advantage router：** 为两个专家训练独立价值头，执行时选择估计优势最大的专家，不使用固定规则和可达性教师。它对应多专家动态切换这一技术族，与 ICS-RL 的 Max-Advantage 切换思想可比。[相关论文](https://arxiv.org/abs/2603.00974)

E2L/H3E 使用专家知识进行行为克隆或分层训练，可在相关工作中比较，但若没有完整复现其训练协议，不应把“教师辅助 PPO”消融命名为 E2L 或 H3E。[E2L](https://www.sciencedirect.com/science/article/pii/S092523122400362X)，[H3E](https://www.sciencedirect.com/science/article/abs/pii/S0957417423035868)

### 9.3 核心消融

| 消融 | 删除/替换内容 | 回答的问题 |
|---|---|---|
| A1 Old-guard | 用 `R≥6 km ∧ Vc≤17 m/s` 替代连续 reachability | 模型可达性是否优于旧启发式 |
| A2 No-teacher | 路由器只读原始观测和历史 | 物理教师/蒸馏是否提供额外决策信息 |
| A3 No-termination | 一旦进入 LOS 就保持到回合结束 | 学习终止是否修复简单场景过度接管 |
| A4 Stepwise | 每步重新选择专家，无最短持续时间 | option 持续语义是否必要 |
| A5 No-shield | 删除强制接管下界 | 性能是否只来自学习路由，保护层作用多大 |
| A6 No-CF | 删除反事实预训练 | 冷启动收益是否来自物理标签 |
| A7 No-cost | 删除接管/切换成本 | 抑制抖振与过度接管是否必要 |
| A8 Frozen-base | V9 全程冻结 | 区分路由收益与额外低层训练收益 |

开发阶段先以 100 万步、3 种子做 A1–A8 筛选。正式长训练至少保留 A2、A3、A4、A8；其余消融根据开发效应和计算预算决定，但必须报告筛选规则。

---

## 10. 训练与评估数据划分

### 10.1 种子

- 开发训练种子：`85, 185, 285`；
- 正式训练建议增加：`385, 485`，共 5 个独立训练种子；
- 已反复使用的 evaluation seed `10085` 只保留为开发集；
- 新确认评估预留：`110085, 120085, 130085, 140085, 150085`；执行前再次确认这些种子未被任何开发脚本读取。

训练种子是算法比较的主要独立单位。每个 checkpoint 的数百局评估只能刻画该冻结策略，不能替代训练种子方差。

### 10.2 对手集合

主要算法比较的训练分布固定为当前 Hold/Pursuit/Evasive 等比例混合，使用与 V9 一致的静态 GPU batch 分配。这样 V12 与历史基线的变化只来自算法，而不是新增对手课程。

参数随机化对手和历史策略池作为第二阶段扩展：

1. 参数随机化对手改变 Evasive 转向方向、切换周期、航向幅度、速度和开火行为；
2. 历史策略池由 PPO、RHPPO、分层 PPO 和 V9 不同训练阶段的冻结快照组成；
3. 如果扩展阶段把这些对手加入训练，所有学习基线必须使用相同采样分布并重新训练；
4. 若没有 payoff/meta-solver/response/population update 闭环，只称历史对手池，不称为 PSRO。

确认评估包含：

- 三类固定对手；
- 未见参数组合的随机化对手；
- 历史策略跨博弈；
- 颜色互换和相同初始条件配对；
- 可选的动力学/传感器扰动集。

### 10.3 初始条件和配对

每个评估回合保存：

- reset-pool 索引与完整初始状态哈希；
- learner 颜色；
- 对手类型和参数；
- evaluation seed；
- checkpoint 哈希；
- 环境配置哈希。

不同算法使用相同的回合 ID、初始状态和对手参数，以支持逐局配对差异分析。现有评估脚本需要补充逐局记录，不能继续只保存汇总 JSON。

---

## 11. 对比指标

### 11.1 主要任务指标

对手类型 \(k\in\{H,P,E\}\) 的胜率为：

\[
W_k=\frac{N_{\mathrm{win},k}}{N_k}.
\]

宏平均胜率为：

\[
W_{\mathrm{macro}}=\frac{W_H+W_P+W_E}{3}.
\]

困难场景性能使用 \(W_E\)，跨场景下界使用：

\[
W_{\min}=\min(W_H,W_P,W_E).
\]

同时报告每类对手的超时率、失败率、平局率和坠毁率。不得只报告宏平均而隐藏某个场景的退化。

主比较为：

1. V12 vs V9：检验可达性仲裁是否修复困难场景；
2. V12 vs V11.3：检验学习终止是否优于固定持续证书；
3. V12 vs V11.2：检验是否在保留 Evasive 能力的同时减少简单场景损失。

### 11.2 武器链指标

- 发射请求合法率：

\[
L_{\mathrm{req}}=rac{N_{\mathrm{accepted}}}{N_{\mathrm{fire\ request}}};
\]

- 命中/已接受发射：

\[
H_{\mathrm{launch}}=rac{N_{\mathrm{hit}}}{N_{\mathrm{accepted}}};
\]

- 每局合法发射机会步数；
- 首次合法发射机会时间；
- 首次实际发射时间；
- 合法机会到实际发射的延迟；
- 发射时距离、进入角、LOS 角速率、闭合速度和攻击质量；
- 每次击毁消耗的导弹数；
- 双方导弹耗尽后终止比例。

命中率的分母必须明确为已接受发射，不能与“有合法发射机会的回合数”混用。

### 11.3 仲裁机制指标

- LOS 动作介入比例；
- 路由器主动选择比例；
- 物理保护强制接管比例；
- 每局切换次数；
- option 持续时间的均值、中位数、四分位数和尾部分布；
- 进入 LOS 前后的 \(R,V_c,q,\underline\rho^{\mathrm L},\underline\rho^{\mathrm I}\)；
- 恢复条件满足到实际交还的延迟；
- 交还后 5/10/20 步内重新强制接管率；
- 错误接管：教师和后验轨迹均显示学习专家更优但选择 LOS；
- 错误放任：LOS 可达性显著更高但仍持续选择学习专家；
- 技能级反事实遗憾：

\[
\operatorname{Regret}_{\rho,t}=
\max_e\rho_t^e-\rho_t^{e_t}.
\]

### 11.4 可达性模型指标

- `legal-within-H` 的 AUROC、AUPRC、Brier 分数和校准误差；
- 到首次合法发射机会的 MAE/中位绝对误差，成功样本与删失样本分开；
- \(\rho\) 的 MAE、Spearman 排序相关和符号一致率；
- ensemble 覆盖率和区间宽度；
- 按对手、距离区间、闭合速度区间和颜色分层的误差；
- 与 V11.3 二值 guard、即时 \(q_t\) 和无历史 MLP 的比较。

### 11.5 训练指标

- 外在回报、终局胜率和超时率；
- 低层/高层 policy loss、value loss、explained variance；
- 低层/高层 KL、clip fraction、entropy、gradient norm；
- 终止器熵、终止比例、强制接管比例；
- 两个约束成本和拉格朗日乘子；
- 反事实预训练损失与退火系数；
- 环境决策/s、仿真子步/s、GPU 利用率、显存、CPU 利用率；
- rollout、物理教师、可达性网络和 PPO 更新各自耗时；
- 在线环境决策数 \(N_{\mathrm{env}}\)、教师分支模型决策数 \(N_{\mathrm{model}}\) 及二者之和；
- 1000 万步总墙钟时间和能耗（若可采集）。

### 11.6 泛化与稳健性指标

泛化和稳健性分开报告：

- **对手泛化：** 未见 Evasive 转向周期/幅度、未见历史策略；
- **初始条件泛化：** 距离、速度差、高度差和初始态势超出训练分位区间；
- **动力学稳健性：** 飞机质量/推力/气动系数和导弹推力/阻力在明确范围内扰动；
- **观测稳健性：** LOS 角、闭合速度的零均值噪声和固定决策延迟；
- **计算稳健性：** 不同并行环境数下的吞吐和结果一致性。

每项必须记录扰动变量、幅值、训练是否见过以及性能下降量，不能只写“具有鲁棒性”。

---

## 12. 统计分析协议

### 12.1 单冻结策略

每个冻结 checkpoint、每类对手的胜率/超时率报告 Wilson 95% 区间。这只描述当前策略在当前评估分布下的回合不确定性，不代表算法跨训练种子的区间。

### 12.2 算法级比较

正式算法比较使用训练种子为外层单位，并利用相同 seed、相同回合 ID 的配对设计。主要效应为：

\[
\Delta W_{m,k,s}=
W_{\mathrm{V12},k,s}-W_{m,k,s},
\]

其中 \(m\in\{\mathrm{V9},\mathrm{V11.3},\mathrm{V11.2}\}\)，\(k\) 为对手类型，\(s\) 为训练种子。

报告：

- 每个训练种子的原始结果；
- 配对均值/中位数差和 95% 区间；
- 分层 bootstrap：外层重采训练种子，内层按 evaluation seed 和回合 ID 成对重采；
- 概率改进或性能 profile 作为补充，不把它解释为空战逐局获胜概率；
- 主比较固定为 V12 vs V9 和 V12 vs V11.3；其余消融使用 Holm 校正控制多重比较。

5 个训练种子适合报告效应方向和区间，但对小效应的显著性检验能力有限。因此论文主结论优先报告效应量、种子一致性和区间，而不是只报告 p 值。

### 12.3 检查点选择

- 开发阶段允许使用开发评估集选择结构和超参数；
- 正式训练在训练前冻结 checkpoint 选择规则；
- 推荐同时报告 `final checkpoint` 和按独立 validation score 选择的 `selected checkpoint`；
- 确认评估只能执行一次，不能用于返回选择 checkpoint；
- 学习曲线按种子展示原始曲线及跨种子聚合，平滑曲线不能替代原始点。

---

## 13. 公平性与计算预算

### 13.1 训练公平性

所有学习方法统一：

- 60 Hz 动力学、5 Hz 决策；
- 96 个并行环境；
- 相同初始状态分布；
- 相同训练对手采样；
- 相同环境奖励、终止和武器配置；
- 相同累计 1000 万在线环境决策预算，继承 checkpoint 的历史训练步数计入总额；
- 相同 evaluation seeds 和回合 ID；
- 相同硬件和混合精度设置。

V12 的物理教师额外使用模型展开，因此同时报告在线环境决策预算 \(N_{\mathrm{env}}\)、教师模型分支预算 \(N_{\mathrm{model}}\) 和总 GPU 墙钟预算。主任务训练预算按 \(N_{\mathrm{env}}\) 对齐，但样本/计算效率讨论必须同时给出 \(N_{\mathrm{model}}\)，不能把模型生成标签描述为零成本数据。

### 13.2 预计时间

V9 当前单次更新包含 `96×256=24,576` 个环境决策，已记录的 rollout 与 PPO 更新时间约为 `10.8 s + 2.4 s`，对应约 `1.86k` 环境决策/s。按该开发测量，从零累计到 1000 万步的纯 V9 约需 1.5 h；若从已完成的 100.76 万步初始化点继续到累计 1000 万步，剩余理想时间约 1.35 h。V12 加入可达性推理和路由后预计更慢，实际时间必须在 B0 烟雾阶段重新测量。

长训练不在方案中承诺固定完成时间。执行时先以烟雾阶段实测吞吐计算：

\[
T_{10M}=\frac{10^7}{\operatorname{throughput}_{\mathrm{smoke}}}.
\]

若模型展开使吞吐下降超过 30%，优先增加蒸馏缓存命中率、只在 option 边界调用教师、批量复制张量状态；不得通过降低 60 Hz 动力学频率换取速度。

---

## 14. 代码实施边界与文件规划

现有 PPO、分层 PPO、V9、V11 系列训练和评估脚本保持不改。V12 使用独立模块和入口：

```text
benchmarks/drvoi_v12/
    __init__.py
    snapshot.py              # 环境、V9、低层控制器执行状态快照/恢复
    reachability_teacher.py  # 成对物理展开和标签
    reachability_model.py    # ensemble 和校准
    router.py                # 路由器、终止器、物理保护状态机
    buffer.py                # 低层 rollout 与 option segment buffer
    losses.py                # SMDP-GAE、PPO 和约束更新

benchmarks/
    collect_drvoi_v12_reachability.py
    train_drvoi_v12_reachability.py
    train_torch_wvr_mrat_rhppo.py
    evaluate_torch_wvr_mrat_rhppo.py
    analyze_drvoi_v12_results.py

configs/
    drvoi_rhppo_v12_mrat.json

scripts/
    run_drvoi_v12_smoke_wsl.sh
    run_drvoi_v12_development_wsl.sh
    run_drvoi_v12_formal_wsl.sh
    status_drvoi_v12_wsl.sh

tests/
    test_drvoi_v12_snapshot.py
    test_drvoi_v12_reachability.py
    test_drvoi_v12_router.py
    test_drvoi_v12_termination.py
    test_drvoi_v12_resume.py
    test_drvoi_v12_metrics.py
```

环境层只允许新增只读诊断接口或状态快照接口；不得修改动力学方程、武器参数、奖励系数、初始状态分布和终止条件。若后续发现平台错误，必须作为独立环境修复提交，并让所有基线重新评估，不能只让 V12 使用修复后的环境。

### 14.1 必须通过的工程测试

1. 快照—恢复后，同一动作序列产生逐元素一致的状态、奖励、武器事件和终止；
2. 两专家成对展开使用完全相同的环境状态、V9 循环/option 状态、低层控制器状态与对手假设；
3. reset 后路由、终止、持续计数和 GRU hidden 全部清零；
4. 强制接管优先级在训练与评估一致；
5. LOS 专家只能在合法时发射；
6. SMDP-GAE 在 \(D_k=1\) 时退化为普通 GAE；
7. 终止优势正负与梯度方向通过合成测试；
8. LOS 步不进入 V9 actor PPO 比率；
9. LOS 接管期间 V9 影子状态按最终执行动作更新，交还后不出现 stale hidden；
10. 中断恢复后的首个 rollout 与不中断运行一致；
11. 每局武器事件满足请求、拒绝、接受、命中和剩余弹药守恒。

---

## 15. 执行顺序与决策规则

### 阶段 0：冻结历史证据

- 固化 V7–V11.3 数据清单、checkpoint SHA-256、代码 commit 和评估协议；
- 把 `10085` 标为开发评估种子；
- 生成一张“证据—结论—不能支持的结论”表；
- 不重写旧结果，不把 V11.3 重新标成训练算法。

### 阶段 1：物理教师与快照系统

- 实现环境张量快照、批量分支和状态恢复；
- 在纯追击与 LOS 已知对照上验证教师排序方向；
- 扫描 \(H_R\in\{12,30,60\}\)，只用开发集选择能够区分两专家且计算可接受的最短时域；
- 保存每个标签的对手假设、轨迹和构成项，保证可审计。

### 阶段 2：可达性蒸馏

- 按完整回合划分 train/validation/calibration/test；
- 训练 ensemble 并校准 LCB；
- 与旧 guard、即时 \(q_t\)、无历史 MLP 比较；
- 若可达性模型只学到对手脚本标签而不能跨随机化参数工作，则返回修改教师输入/目标，不启动 PPO 长训。

### 阶段 3：路由与终止烟雾测试

- 单元测试后运行 49,152–98,304 步；
- 检查动作、终止、强制接管、resume、NaN 和吞吐；
- 12–24 局/对手只作链路验证，不据此判断算法优越。

### 阶段 4：100 万步三种子开发

- 完整 V12 与 A1–A8 消融；
- 每种方法对 Hold/Pursuit/Evasive 各 192 局，使用开发种子；
- 决策依据是跨种子方向、Pareto 权衡、错误接管/放任和 reachability regret，而非单一最高胜率；
- 冻结进入长训练的方法、超参数和 checkpoint 选择规则。

### 阶段 5：累计 1000 万步正式训练

优先顺序：

1. V12 完整方法；
2. V9-continued；
3. V11.3；
4. 标准 PPO；
5. 分层 PPO；
6. 核心消融 A2/A3/A4/A8；
7. 其余结构基线。

每个方法按 5 个训练种子顺序运行；每种方法的初始化、预训练和续训在线交互合计为 1000 万环境决策。每次完成后立即检查工件完整性，但不根据中间确认结果修改后续种子的超参数。

### 阶段 6：一次性确认评估与论文分析

- 使用预留 evaluation seeds 和逐局配对初始条件；
- 输出固定对手、随机化对手、历史策略池和扰动测试；
- 生成胜率/超时、Pareto、option 持续时间、reachability 校准、训练吞吐和消融图；
- 统计单位、分母和区间在图表中明确；
- 只有完成多种子确认后，才能形成算法优越性、泛化或稳健性结论。

---

## 16. 预期论文贡献与失败解释

若 H1–H5 获得数据支持，论文贡献可表述为：

1. 提出面向导弹 WVR 空战的模型驱动技能可达性表示，把动态发射包线、短时域物理展开和预测不确定性结合为可审计的路由依据；
2. 提出带物理保护下界的专家路由与可学习终止机制，在循环分层策略和确定性 LOS 专家之间执行 option 级仲裁；
3. 通过持续时间、终止、保护层和反事实预训练消融，解释困难场景零胜与 option 语义丢失之间的关系；
4. 在统一 GPU 六自由度动力学、导弹和多种对手协议下，报告性能、武器效率、机制行为和计算代价。

若 V12 失败，结果仍按机制解释：

- 可达性模型失败：短时域物理标签不足以预测后续交战机会；
- 可达性准确但路由失败：监督目标与终局外在回报仍然错位；
- Evasive 改善但简单场景退化：终止器/接管成本没有解决专家覆盖冲突；
- 所有方法多种子方差大：主要瓶颈可能仍是基础策略训练稳定性或对手分布；
- V12 与 V11.3 相当：经验阈值已捕获主要机制，新增学习模块没有提供足够收益。

这些失败结论不得通过追加新版本名称或降低统计要求隐藏。

---

## 17. 当前允许和禁止的表述

### 允许

- 单种子开发结果显示，持续 LOS 仲裁在当前 Evasive 对手上恢复了非零击毁；
- 现有消融支持 option 持续时间和终止逻辑是当前性能瓶颈；
- V12 将固定启发式仲裁升级为模型可达性驱动的可学习路由与终止框架；
- V12 的正式性能仍待多种子实验验证。

### 禁止

- V11.3 或 V12 已显著优于 PPO/RHPPO；
- `6 km/17 m/s/12 步` 是理论最优证书；
- 可达性 LCB 对真实飞行提供形式化安全保证；
- 192 局评估等于 192 个独立算法重复；
- V12 是首个专家切换空战算法；
- 训练回报收敛等于达到纳什均衡或形成真实作战能力。

---

## 18. 方案验收交付物

后续执行完成后，至少交付：

1. 冻结配置、commit、环境和 checkpoint manifest；
2. 物理教师数据集清单及 train/validation/calibration/test 划分；
3. 可达性模型校准报告；
4. 训练/评估/分析独立脚本；
5. 每回合和每 option 级原始记录；
6. 5 种子主结果与核心消融；
7. 固定、随机化、历史策略和扰动评估；
8. 奖励曲线、胜率/超时曲线、Pareto 图、校准图、option 持续时间图、介入与 regret 图；
9. 训练吞吐和计算开销报告；
10. 可直接用于论文方法与实验章节的证据映射表。

本方案冻结后，下一步应从**阶段 0：历史证据 manifest 与阶段 1：环境快照/物理教师实现**开始，不直接启动 1000 万步训练。
