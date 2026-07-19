# 研究进展清单（截至 2026-07-18）

## 项目概况

- **课题**：面向多楼层场景的多机器人协同配送与计算资源联合调度
- **学生**：史佳玺，西安交通大学，控制工程硕士
- **导师**：胡剑晨 教授
- **代码量**：31 个 Python 文件，约 9,926 行
- **已发表论文**：IFAC 2026（GPU Cluster Scheduling via Checkpoint-based Live Migration）

---

## 一、已完成的研究

---

### 1.1 电梯群控系统建模

| 项目 | 详情 |
|:---|:---|
| **状态** | ✅ 已完成 |
| **脚本文件** | `module1/elevator_group_control.py`（424 行） |
| **实现内容** | 3 种经典派车算法（ETA / NC / SCAN），完整电梯状态机（IDLE→MOVING_TO_CALL→DOOR_OPENING→LOADING→MOVING_TO_DEST→UNLOADING），能耗追踪 |
| **实验脚本** | `module1/full_experiment.py` → dispatch comparison 部分 |
| **实验参数** | 12 层、10 机器人、4 电梯、30 任务，EBO+Relay 优化 |
| **结果文件** | `module1/results/full_experiment_v2/all_results.json` → `dispatch_comparison` |
| **关键结果** | ETA: Makespan=318s, avg_wait=3.1s; SCAN: 353.5s, 4.3s; NC: 470s, 5.5s |
| **计算资源** | 单次约 123-129s（含 EBO 优化），CPU |

---

### 1.2 MILP 精确求解（电梯不相交约束）

| 项目 | 详情 |
|:---|:---|
| **状态** | ✅ 已完成 |
| **脚本文件** | `module1/milp_v2.py`（520 行），`module1/milp_solver.py`（329 行，V1） |
| **实现内容** | V2 严格 MILP：电梯 disjunctive 约束（不可抢占机器建模）、电梯重定位时间、精确 GPU 交互时序（approach 8s + press 7s + exit 5s） |
| **实验脚本** | `module1/run_experiment.py` |
| **实验参数** | small (5F/5R/2E/10T)、medium (10F/10R/4E/30T) |
| **结果文件** | `module1/results/small_summary.json`、`module1/results/medium_summary.json` |
| **关键结果** | Small: MILP Makespan=288.3s vs Greedy=1020.5s（71.7%↓）；Medium: MILP=290.0s vs Greedy=740.5s（60.8%↓）；加权延迟均接近零 |
| **计算资源** | Small: 3.5s；Medium: 119s（CPU，PuLP/CBC） |
| **局限** | Large 规模（50T）超时未跑 |

---

### 1.3 NSGA-II 多目标进化算法

| 项目 | 详情 |
|:---|:---|
| **状态** | ✅ 已完成 |
| **脚本文件** | `module1/nsga2.py`（468 行） |
| **实现内容** | 3 层染色体编码（gene_assign / gene_elev / gene_priority），均匀交叉 + SBX 混合，4 种变异算子，Monte Carlo 超体积计算 |
| **实验脚本** | `module1/full_experiment.py`（V1 部分，`results/full_experiment/`） |
| **实验参数** | 3 规模 × 5 seeds，pop=50-100，gen=50-150 |
| **结果文件** | `module1/results/full_experiment/all_results.json` |
| **关键结果** | S2_medium: NSGA-II Makespan=399-457s（5 seeds），Pareto size=51-60；vs Greedy=495-652s |
| **计算资源** | 8-15s/run（CPU） |

---

### 1.4 ALNS 自适应大邻域搜索

| 项目 | 详情 |
|:---|:---|
| **状态** | ✅ 已完成 |
| **脚本文件** | `module1/alns.py`（586 行） |
| **实现内容** | 4 破坏算子（random / worst / related / elevator-bottleneck）+ 3 修复算子（greedy / regret-2 / elevator-aware），模拟退火接受准则，自适应权重 |
| **实验脚本** | `module1/full_experiment.py`（V1 部分） |
| **结果文件** | `module1/results/full_experiment/all_results.json` |
| **关键结果** | 能耗最低但 Makespan 较差（S2: Makespan=540-684s），Pareto size 不稳定（18-381） |
| **计算资源** | 2-19s/run（CPU） |

---

### 1.5 NSGA-II-EBO 核心算法（双层优化 + 电梯瓶颈）

| 项目 | 详情 |
|:---|:---|
| **状态** | ✅ 已完成 |
| **脚本文件** | `module1/nsga2_ebo.py`（517 行），`module1/ebdco.py`（711 行，下层优化器），`module1/problem_analysis.py`（390 行，问题分析） |
| **实现内容** | 上层 NSGA-II 搜索任务分配 + 下层权重启发式优化序列；冲突感知交叉、负载均衡变异、策略参数进化；接力配送染色体扩展（gene_relay / gene_relay_robot） |
| **实验脚本** | `module1/full_experiment.py` → algorithm_comparison 部分 |
| **实验参数** | 3 规模 × 3 方法 × 5 seeds |
| **结果文件** | `module1/results/full_experiment_v2/all_results.json` → `algorithm_comparison` |
| **关键结果** |（mean ± std across 5 seeds）|

| 规模 | Greedy Makespan | EBO Makespan | EBO+Relay Makespan | EBO+Relay 改善率 |
|:---|---:|---:|---:|:---:|
| S1 (10R/2E/20T) | 650.7±253.6 | 191.5±194.4 | 178.3±189.6 | -6.9% vs EBO |
| S2 (15R/4E/50T) | 684.9±71.6 | 220.9±21.1 | 214.2±35.6 | -3.0% vs EBO |
| S3 (25R/4E/80T) | 1002.6±72.9 | 342.4±69.2 | **234.9±54.4** | **-31.4% vs EBO** |

| 项目 | 详情（续） |
|:---|:---|
| **计算资源** | S1: ~90s/run, S2: ~300s/run, S3: ~500s/run（CPU） |
| **备注** | 所有 EBO/EBO+Relay 的加权延迟均为 0 |

---

### 1.6 多机器人接力配送（Relay Delivery）

| 项目 | 详情 |
|:---|:---|
| **状态** | ✅ 已完成 |
| **脚本文件** | `module1/config.py`（TaskSegment / RelayPlan / split_task），`module1/nsga2_ebo.py`（relay 染色体 + 变异算子），`module1/simulator_v2.py`（RELAY_DROPOFF 阶段 + relay_buffer） |
| **实验脚本** | `module1/full_experiment.py` → EBO+Relay |
| **结果文件** | 同 1.5 |
| **关键结果** | S3 大规模下 Makespan 从 342.4s→234.9s（降低 31.4%），平均启用 5.8 个中继任务 |

---

### 1.7 24 小时周期性任务流建模

| 项目 | 详情 |
|:---|:---|
| **状态** | ✅ 已完成 |
| **脚本文件** | `module1/dynamic_scenario.py`（242 行） |
| **实现内容** | 6 个时段周期建模（06-08 / 08-12 / 12-13 / 13-17 / 17-20 / 20-06），Poisson 到达，方向偏置（上午上行 60%、下午下行 60%） |
| **实验脚本** | `module1/day_experiment.py`（228 行） |
| **实验参数** | 15 层、20 机器人、4 电梯、~319 任务/天，峰谷比 8.2× |
| **结果文件** | `module1/results/day_experiment/day_results.json` |
| **关键结果** |（调度方法对比）|

| 组合 | 完成数 / 总数 | 完成率 |
|:---|:---:|:---:|
| ETA + Greedy | 257 / 319 | 80.6% |
| ETA + EBO+Relay | 279 / 319 | 87.5% |
| NC + EBO+Relay | 299 / 319 | 93.7% |
| SCAN + EBO+Relay | **300 / 319** | **94.0%** |

| 项目 | 详情（续） |
|:---|:---|
| **图表** | `docs/figures/daily_pattern.png`（任务到达分布图） |

---

### 1.8 V2 群控仿真器

| 项目 | 详情 |
|:---|:---|
| **状态** | ✅ 已完成 |
| **脚本文件** | `module1/simulator_v2.py`（676 行） |
| **实现内容** | 离散时间仿真（dt=0.5s），集成电梯群控，批量取货，多站送货，接力配送交接，GPU 需求自动标注，12 种机器人状态 |
| **测试** | 通过 `full_experiment.py` 和 `day_experiment.py` 间接验证 |

---

### 1.9 GPU 推理调度策略（基于机器人动作时间线）

| 项目 | 详情 |
|:---|:---|
| **状态** | ✅ 已完成 |
| **脚本文件** | `module2/gpu_scheduler.py`（458 行），`module2/gpu_config.py`（109 行） |
| **实现内容** | 5 种策略：FragAware（碎片感知）、Predictive（预测预加载）、OnDemand、Fixed、RoundRobin |
| **实验脚本** | `module2/full_gpu_experiment.py`（GPU sensitivity），`module1/day_experiment.py`（24h GPU） |
| **实验参数** | (1) 10F/15R/4E/40T，4 种 GPU 紧张度 × 4 策略；(2) 24h/20R/4E/319T，紧张(2GPU/4并发) |
| **结果文件** | `module2/results/gpu_experiment/gpu_sensitivity.json`、`module1/results/day_experiment/day_results.json` → gpu_comparison |
| **关键结果** |（紧张配置 2GPU / 4 并发，1026 需求段）|

| 策略 | 等待延迟 | 冷启动 | 碎片率 |
|:---|---:|---:|---:|
| Predictive | **0s** | **0** | 0.026 |
| FragAware | 0s | 683 | 0.002 |
| OnDemand | **2394s** | 228 | 0.937 |

---

### 1.10 IFAC 论文独立验证实验（φ(k) 碎片化模型）

| 项目 | 详情 |
|:---|:---|
| **状态** | ✅ 已完成 |
| **脚本文件** | `module2/frag_cluster.py`（381 行），`module2/thesis_experiment.py`（213 行），`module2/gpu_figures.py`（579 行） |
| **实现内容** | φ(k) 碎片化代价函数、One-hot 线性化 MILP、事件驱动集群仿真器、4 种方法（FragMig / BF / SC / MismatchPred） |

#### Exp 1: 主实验

| 参数 | 值 |
|:---|:---|
| 服务器 | 50 × 8 GPU = 400 GPU |
| 仿真时长 | 100 小时 |
| 到达率 | 10 个/小时 |
| 随机种子数 | 10 |
| 分布 | inv / norm / dir |
| MILP 时限 | 10s |
| ε | 0.01 |
| 结果文件 | `module2/results/thesis_gpu/exp1_main.json` |

**关键结果**（norm 分布，最具区分度）：

| 方法 | $\bar{w}$ (h) | Migrations |
|:---|---:|---:|
| **FragMig (Ours)** | **3.55±1.10** | **69±21** |
| Best Fit | 3.96±1.11 | 0 |
| Server Consolidation | 3.67±1.04 | **1378±287** |
| MismatchPred | 3.50±1.05 | 59±21 |

改善：等待时间比 BF 降低 10.3%，迁移次数仅为 SC 的 5.0%。

| 参数 | 详情 |
|:---|:---|
| **计算资源** | 总计约 43 分钟（CPU，主要在 MILP 求解） |

#### Exp 2: 规模敏感性

| 参数 | 值 |
|:---|:---|
| 服务器数 | [10, 25, 50, 100] |
| 分布 | norm |
| Seeds | 5 |
| 结果文件 | `module2/results/thesis_gpu/exp2_scale.json` |

**关键结果**：P=50 是 FragMig 价值最大的工作点（$\bar{w}$: 3.44 vs BF 3.87，11.1%↓）

#### Exp 3: 参数消融

| 参数 | 值 |
|:---|:---|
| ε 值 | [0.0, 0.001, 0.01, 0.1, 1.0] |
| 检查点间隔 | [0.5, 1.0, 2.0, 4.0] 小时 |
| Seeds | 5 |
| 结果文件 | `module2/results/thesis_gpu/exp3_ablation.json` |

**关键结果**：ε=0.001 最优（$\bar{w}$=3.39，迁移 76 次）；检查点间隔 2.0h 最优（$\bar{w}$=3.35）

---

### 1.11 可视化图表

| 项目 | 详情 |
|:---|:---|
| **状态** | ✅ 已完成 |
| **脚本文件** | `docs/generate_all_figures.py`（374 行），`module2/gpu_figures.py`（579 行） |

**模块 1 图表**（7 张，`docs/figures/`）：

| 文件名 | 内容 |
|:---|:---|
| `daily_pattern.png` | 24h 任务到达分布 |
| `gantt_v2.png` | 机器人 Gantt 图（GPU 需求标注） |
| `elevator_spacetime.png` | 电梯时空图 |
| `algorithm_comparison.png` | 算法对比柱状图 |
| `pareto_comparison.png` | Pareto 前沿对比 |
| `gpu_comparison.png` | GPU 调度策略对比 |
| `dispatch_comparison.png` | 派车算法对比 |

**模块 2 图表**（13 图 + 1 LaTeX 表，`module2/results/thesis_gpu/figures/`）：

| 文件名 | 内容 |
|:---|:---|
| `phi_curves.pdf/png` | φ(k) 阶梯函数 |
| `demand_distributions.pdf/png` | 三种需求分布 |
| `counterproductive_consolidation.pdf/png` | SC 反例 |
| `algorithm_flowchart.pdf/png` | Algorithm 1 流程图 |
| `utilization_timeseries.pdf/png` | 利用率时序 |
| `waiting_time.pdf/png` | 等待时间对比 |
| `queue_cardtime.pdf/png` | 队列卡时对比 |
| `utilization.pdf/png` | 利用率对比 |
| `makespan.pdf/png` | Makespan 对比 |
| `scale_sensitivity.pdf/png` | 规模敏感性 |
| `epsilon_ablation.pdf/png` | ε 消融 |
| `checkpoint_ablation.pdf/png` | 检查点消融 |
| `table_main.tex` | 主实验 LaTeX 表 |

---

### 1.12 中期报告

| 项目 | 详情 |
|:---|:---|
| **状态** | ✅ 已完成 |
| **脚本文件** | `write_midterm.py`（338 行） |
| **输出** | `midterm.docx`（43KB） |

---

## 二、未完成 / 待改进的研究

---

### 2.1 PredictiveScheduler 未纳入正式实验

| 项目 | 详情 |
|:---|:---|
| **状态** | ⚠️ 代码已实现，但实验不完整 |
| **问题** | `gpu_scheduler.py` 中 `PredictiveScheduler` 已实现（95 行），但 `full_gpu_experiment.py` 的调度器字典中**未包含**该策略——仅在 `day_experiment.py` 中使用。`gpu_sensitivity` 实验只对比了 FragAware / OnDemand / Fixed / RoundRobin 四种 |
| **建议** | 将 PredictiveScheduler 加入 `full_gpu_experiment.py` 的对比实验，在所有 GPU 紧张度配置下与 FragAware 对比 |

---

### 2.2 GPU MILP 精确解未作为基线对比

| 项目 | 详情 |
|:---|:---|
| **状态** | ⚠️ 代码已实现，结果不完整 |
| **问题** | `module2/gpu_milp.py`（173 行）实现了离线 MILP 精确 GPU 分配，`full_gpu_experiment.py` 中有调用但结果独立存放（非对比格式），且目标函数中 beta3（延迟）定义了但未加入目标 |
| **建议** | 修复 MILP 目标函数，将其作为上界/最优解基线加入对比 |

---

### 2.3 IFAC φ(k) 模型与机器人推理调度的直接集成

| 项目 | 详情 |
|:---|:---|
| **状态** | ⚠️ 概念映射已完成，代码未直接打通 |
| **问题** | `frag_cluster.py`（IFAC 论文实现）和 `gpu_scheduler.py`（机器人推理调度）是**两套独立的系统**，API 不同、数据结构不同。报告中描述的"乘电梯窗口=检查点"映射停留在概念层面，没有代码将 `frag_cluster.py` 的 MILP 迁移直接应用于 `gpu_scheduler.py` 的推理场景 |
| **建议** | 实现一个 `FragMigScheduler`，在机器人乘电梯时调用 `solve_migration_milp()` 做显存重分配，与现有 FragAware/Predictive 对比 |

---

### 2.4 大规模统计实验（Wilcoxon 检验）

| 项目 | 详情 |
|:---|:---|
| **状态** | ❌ 未完成 |
| **问题** | `IDEA_REPORT.md` 计划了 4 规模 × 4 方法 × 5 seeds + Wilcoxon 检验的统计显著性分析。当前 full_experiment_v2 有 3 规模 × 3 方法 × 5 seeds，但缺少 NSGA-II（标准版）和 ALNS 在 V2 仿真器下的对比 |
| **建议** | 在 V2 仿真器下补充 NSGA-II 和 ALNS 基线，并做 Wilcoxon 检验 |

---

### 2.5 MILP 大规模实例

| 项目 | 详情 |
|:---|:---|
| **状态** | ❌ 未完成 |
| **问题** | `run_experiment.py` 定义了 large 场景（15F/15R/4E/50T），但无结果文件（MILP 超时）。MILP-仿真 gap 分析缺失 |
| **建议** | 用松弛/分解方法获取 MILP 下界，计算 EBO 的 optimality gap |

---

### 2.6 单目深度估计模型

| 项目 | 详情 |
|:---|:---|
| **状态** | ❌ 未完成 |
| **问题** | `write_midterm.py` 的未来计划中列出了"单目深度估计模型开发（VAE+DiT 架构）"，但项目中无任何深度估计相关代码 |
| **备注** | 这是论文的第二个支柱（路径规划），与当前配送+GPU 调度工作独立 |

---

### 2.7 理论分析

| 项目 | 详情 |
|:---|:---|
| **状态** | ❌ 未完成 |
| **问题** | NSGA-II-EBO 的收敛性、计算复杂度缺少理论分析。φ(k) 的 approximation ratio 未证明 |
| **建议** | 至少给出计算复杂度分析和收敛性实验（超体积随代数变化） |

---

### 2.8 scale_experiment（模块 2 集成）

| 项目 | 详情 |
|:---|:---|
| **状态** | ❌ 代码存在但未运行 |
| **问题** | `full_gpu_experiment.py` 中 `run_scale_experiment()` 测试 10/20/30 机器人的 GPU 调度，但无结果文件 |
| **建议** | 运行该实验，验证 GPU 调度在不同机器人规模下的表现 |

---

### 2.9 Git 未提交文件

| 项目 | 详情 |
|:---|:---|
| **状态** | ⚠️ 待提交 |
| **文件列表** | `IFAC_sjx.pdf`、`module2/GPU_SCHEDULING_REPORT.md`、`module2/frag_cluster.py`、`module2/gpu_figures.py`、`module2/thesis_experiment.py`、`module2/results/thesis_gpu/`（3 JSON + 27 图表文件） |
| **备注** | 这些是今天新增的 IFAC 论文独立验证工作，需要提交 |

---

## 三、文件清单

### 模块 1：配送调度（23 文件，7,696 行）

| 文件 | 行数 | 类型 | 说明 |
|:---|---:|:---|:---|
| `config.py` | 231 | 核心 | 数据结构（Robot, Elevator, Task, TaskSegment, RelayPlan） |
| `elevator_group_control.py` | 424 | 核心 | 电梯群控系统（ETA/SCAN/NC） |
| `simulator.py` | 540 | 核心 | V1 事件驱动仿真器 |
| `simulator_v2.py` | 676 | 核心 | V2 群控仿真器（批量+接力+GPU标注） |
| `milp_solver.py` | 329 | 算法 | V1 MILP |
| `milp_v2.py` | 520 | 算法 | V2 MILP（disjunctive 电梯约束） |
| `nsga2.py` | 468 | 算法 | NSGA-II 多目标进化 |
| `nsga2_ebo.py` | 517 | 算法 | NSGA-II-EBO（核心算法，含接力） |
| `alns.py` | 586 | 算法 | ALNS 自适应大邻域搜索 |
| `ebdco.py` | 711 | 算法 | EBDCO 分解协同优化 / 下层优化器 |
| `problem_analysis.py` | 390 | 分析 | 电梯瓶颈分析、冲突图、关键任务 |
| `baselines.py` | 212 | 基线 | Greedy-FCFS / Independent |
| `dynamic_scenario.py` | 242 | 场景 | 24h 周期任务生成 |
| `timeline_generator.py` | 447 | 工具 | 动作时间线生成 + GPU 需求提取 |
| `visualize.py` | 256 | 可视化 | Gantt / 时空图 / Pareto / 收敛 |
| `run_experiment.py` | 171 | 实验 | MILP + 基线对比 |
| `full_experiment.py` | 234 | 实验 | V2 算法全面对比 |
| `day_experiment.py` | 228 | 实验 | 24h 全链路实验 |
| `test_nsga2.py` | 93 | 测试 | NSGA-II 多规模测试 |
| `test_alns.py` | 85 | 测试 | ALNS 测试 |
| `test_ebdco.py` | 101 | 测试 | EBDCO 测试 |
| `test_milp_v2.py` | 121 | 测试 | MILP V2 测试 |
| `test_simulator.py` | 114 | 测试 | 仿真器验证 |

### 模块 2：GPU 调度（8 文件，2,230 行）

| 文件 | 行数 | 类型 | 说明 |
|:---|---:|:---|:---|
| `gpu_config.py` | 109 | 核心 | GPU 集群配置、需求数据结构 |
| `gpu_scheduler.py` | 458 | 核心 | 5 种调度策略（含 Predictive） |
| `gpu_milp.py` | 173 | 算法 | GPU 分配 MILP 精确解 |
| `frag_cluster.py` | 381 | 算法 | IFAC 论文核心（φ(k) + MILP + 仿真） |
| `thesis_experiment.py` | 213 | 实验 | IFAC 独立验证（3 个实验） |
| `full_gpu_experiment.py` | 194 | 实验 | 模块 1→2 集成实验 |
| `run_gpu_experiment.py` | 123 | 实验 | 简单 GPU 实验入口 |
| `gpu_figures.py` | 579 | 可视化 | 13 图 + 1 表生成 |

### 顶层文档

| 文件 | 说明 |
|:---|:---|
| `RESEARCH_BRIEF.md` | 完整数学建模（452 行） |
| `PROJECT.md` | 项目概览 |
| `SYSTEM_MODEL.md` | 系统建模详细文档 |
| `GPU_SCHEDULING_REPORT.md` | 三层联合系统报告 |
| `IDEA_REPORT.md` | 初期实验报告 |
| `AUTO_REVIEW.md` | 审查记录（Round 1: 2/10） |
| `IFAC_sjx.pdf` | 已发表 IFAC 论文 PDF |
| `write_midterm.py` | 中期报告生成脚本 |

### 实验结果文件

| 路径 | 来源 | 大小 |
|:---|:---|:---|
| `module1/results/small_summary.json` | run_experiment.py | 小 |
| `module1/results/medium_summary.json` | run_experiment.py | 小 |
| `module1/results/small_timelines.json` | run_experiment.py | 23KB |
| `module1/results/medium_timelines.json` | run_experiment.py | 69KB |
| `module1/results/sim_small_timelines.json` | test_simulator.py | 15KB |
| `module1/results/small_gpu_demands.json` | run_experiment.py | 11KB |
| `module1/results/medium_gpu_demands.json` | run_experiment.py | 34KB |
| `module1/results/small_gpu_scheduling_results.json` | run_gpu_experiment.py | 小 |
| `module1/results/medium_gpu_scheduling_results.json` | run_gpu_experiment.py | 小 |
| `module1/results/full_experiment/all_results.json` | full_experiment.py (V1) | 中 |
| `module1/results/full_experiment_v2/all_results.json` | full_experiment.py (V2) | 中 |
| `module1/results/day_experiment/day_results.json` | day_experiment.py | 中 |
| `module2/results/gpu_experiment/gpu_sensitivity.json` | full_gpu_experiment.py | 小 |
| `module2/results/thesis_gpu/exp1_main.json` | thesis_experiment.py | 30KB |
| `module2/results/thesis_gpu/exp2_scale.json` | thesis_experiment.py | 中 |
| `module2/results/thesis_gpu/exp3_ablation.json` | thesis_experiment.py | 中 |

---

## 四、优先级建议

| 优先级 | 事项 | 工作量 | 影响 |
|:---|:---|:---|:---|
| **P0** | 将 `frag_cluster.py` 的 MILP 迁移直接集成到机器人推理调度 | 2-3 天 | 将 IFAC 论文从"概念映射"变为"代码集成" |
| **P0** | Git 提交当前未跟踪文件 | 5 分钟 | 防止丢失工作 |
| **P1** | PredictiveScheduler 加入 GPU sensitivity 对比实验 | 1 小时 | 补全实验数据 |
| **P1** | V2 仿真器下补充 NSGA-II / ALNS 基线 + Wilcoxon 检验 | 1 天 | 统计严谨性 |
| **P2** | GPU scale experiment 运行 | 2 小时 | 多规模验证 |
| **P2** | 理论分析（复杂度 + 收敛性） | 3-5 天 | 论文要求 |
| **P3** | 单目深度估计模型 | 数周 | 独立支柱 |
