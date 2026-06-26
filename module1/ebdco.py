"""EBDCO: 电梯瓶颈分解协同优化算法
(Elevator-Bottleneck Decomposition Cooperative Optimization)

核心思想：
  本问题的数学结构表明电梯是系统瓶颈（电梯下界 > 机器人下界）。
  因此将问题分解为两层：
    上层（Master）：任务-机器人分配 + 电梯选择 → 决定电梯负载
    下层（Slave）：给定分配后，优化每个机器人的任务序列 + 每部电梯的服务顺序

  与NSGA-II/ALNS的本质区别：
    - NSGA-II/ALNS把所有决策放在同一层搜索 → 搜索空间 O(M^N × N! × E^N)
    - EBDCO分解后：上层 O(M^N × E^N)，下层 O(N_j! per robot × disjunctive on E)
    - 下层可以用多项式时间算法（优先级规则+电梯序列优化）精确/近似求解
    → 有效搜索空间大幅缩小

  利用的问题特征：
    1. 电梯瓶颈下界 → 剪枝不均衡分配
    2. 任务冲突图 → 引导交叉/变异避免高冲突对
    3. 关键任务优先 → 先确定关键任务分配，再填充其余任务
    4. 楼层空间局部性 → 同方向任务优先共享电梯行程

  多目标处理：MOEA/D分解框架
    - 生成均匀分布的权重向量
    - 每个权重向量对应一个标量子问题
    - 邻域协作：相邻子问题共享解信息
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
import time

from config import (Robot, Elevator, Task, Priority, PRIORITY_WEIGHT,
                    FLOOR_DISTANCE, INTER_STATION_DISTANCE)
from problem_analysis import ProblemAnalyzer
from simulator import Simulator


# =========================================================================
# 下层求解器：给定任务分配，优化序列和电梯调度
# =========================================================================

class LowerLevelSolver:
    """
    给定任务-机器人分配和电梯选择，优化：
      (1) 每个机器人的任务执行顺序
      (2) 每部电梯的服务顺序

    使用组合优先级规则 + 局部搜索。
    """

    def __init__(self, robots: List[Robot], elevators: List[Elevator], tasks: List[Task]):
        self.robots = robots
        self.elevators = elevators
        self.tasks = tasks

    def optimize_sequence(self, assignment: Dict[int, int],
                           elevator_assignment: Dict[int, int],
                           weight: np.ndarray = None) -> Dict[int, List[int]]:
        """
        权重感知的序列优化。

        weight[0] 高(makespan) → 减少最长机器人的执行时间，并行化
        weight[1] 高(energy) → 最近邻，最小化总转移距离
        weight[2] 高(tardiness) → 严格按截止时间排序(EDD规则)
        """
        if weight is None:
            weight = np.array([0.33, 0.34, 0.33])

        task_sequence = {j: [] for j in range(len(self.robots))}

        for j in range(len(self.robots)):
            assigned = [i for i in range(len(self.tasks)) if assignment.get(i) == j]
            if not assigned:
                continue

            if weight[2] > 0.5:
                # 延迟导向：严格EDD
                task_sequence[j] = sorted(assigned, key=lambda i: (
                    0 if self.tasks[i].priority == Priority.URGENT else 1,
                    self.tasks[i].deadline))
            elif weight[1] > 0.5:
                # 能耗导向：最近邻（最小转移距离）
                task_sequence[j] = self._nearest_neighbor_sequence(j, assigned)
            else:
                # makespan导向：紧急优先 + 楼层局部性
                urgent = [i for i in assigned if self.tasks[i].priority == Priority.URGENT]
                others = [i for i in assigned if self.tasks[i].priority != Priority.URGENT]
                urgent.sort(key=lambda i: self.tasks[i].deadline)
                others = self._nearest_neighbor_sequence(j, others) if others else []
                task_sequence[j] = urgent + others

            # 2-opt
            if len(task_sequence[j]) >= 4:
                task_sequence[j] = self._two_opt(j, task_sequence[j])

        return task_sequence

    def _nearest_neighbor_sequence(self, robot_id: int, task_ids: List[int]) -> List[int]:
        """最近邻启发式：从机器人当前位置出发，每次选最近的未访问任务"""
        robot = self.robots[robot_id]
        current_floor = robot.init_floor
        remaining = set(task_ids)
        sequence = []

        while remaining:
            best = min(remaining,
                       key=lambda i: abs(self.tasks[i].origin_floor - current_floor))
            sequence.append(best)
            remaining.remove(best)
            current_floor = self.tasks[best].dest_floor

        return sequence

    def _two_opt(self, robot_id: int, sequence: List[int]) -> List[int]:
        """2-opt局部搜索：交换序列中两个位置，保留紧急任务的优先位置"""
        n = len(sequence)
        # 找到紧急任务的结束位置，2-opt只在非紧急部分执行
        urgent_end = 0
        for idx, i in enumerate(sequence):
            if self.tasks[i].priority == Priority.URGENT:
                urgent_end = idx + 1

        best_seq = list(sequence)
        improved = True
        while improved:
            improved = False
            for i in range(urgent_end, n - 1):
                for j in range(i + 1, n):
                    new_seq = best_seq[:i] + best_seq[i:j+1][::-1] + best_seq[j+1:]
                    # 快速评估：比较总楼层跳转距离
                    old_cost = self._sequence_cost(robot_id, best_seq)
                    new_cost = self._sequence_cost(robot_id, new_seq)
                    if new_cost < old_cost:
                        best_seq = new_seq
                        improved = True

        return best_seq

    def _sequence_cost(self, robot_id: int, sequence: List[int]) -> float:
        """快速评估序列质量：总楼层跳转距离"""
        if not sequence:
            return 0
        robot = self.robots[robot_id]
        cost = abs(robot.init_floor - self.tasks[sequence[0]].origin_floor)
        for idx in range(len(sequence) - 1):
            cost += abs(self.tasks[sequence[idx]].dest_floor -
                        self.tasks[sequence[idx + 1]].origin_floor)
        return cost

    def optimize_elevator_order(self, task_sequence: Dict[int, List[int]],
                                 elevator_assignment: Dict[int, int]) -> Dict[int, int]:
        """
        优化电梯分配：基于负载均衡+方向性考虑。

        对于每个跨楼层任务，重新考虑是否切换到更空闲的电梯。
        """
        # 统计各电梯负载
        elev_load = {e: 0.0 for e in range(len(self.elevators))}
        cf_tasks = []

        for j, seq in task_sequence.items():
            for i in seq:
                if self.tasks[i].is_cross_floor:
                    e = elevator_assignment.get(i, 0)
                    elev_load[e] += self.elevators[e].travel_time(
                        self.tasks[i].origin_floor, self.tasks[i].dest_floor)
                    cf_tasks.append(i)

        # 按负载差异重新分配
        new_elev = dict(elevator_assignment)
        for i in cf_tasks:
            current_e = new_elev.get(i, 0)
            current_load = elev_load[current_e]
            t = self.tasks[i]

            best_e = current_e
            best_score = current_load
            for e in range(len(self.elevators)):
                if e == current_e:
                    continue
                ride_time = self.elevators[e].travel_time(t.origin_floor, t.dest_floor)
                score = elev_load[e] + ride_time
                if score < best_score - 5.0:  # 只有显著改善才切换
                    best_score = score
                    best_e = e

            if best_e != current_e:
                old_ride = self.elevators[current_e].travel_time(t.origin_floor, t.dest_floor)
                new_ride = self.elevators[best_e].travel_time(t.origin_floor, t.dest_floor)
                elev_load[current_e] -= old_ride
                elev_load[best_e] += new_ride
                new_elev[i] = best_e

        return new_elev


# =========================================================================
# 上层搜索 + MOEA/D框架
# =========================================================================

@dataclass
class EBDCOIndividual:
    """个体编码：仅上层决策（分配+电梯选择）"""
    gene_assign: np.ndarray     # (N,): task → robot
    gene_elev: np.ndarray       # (N,): task → elevator

    objectives: np.ndarray = field(default_factory=lambda: np.array([np.inf, np.inf, np.inf]))
    scalar_obj: float = np.inf
    task_sequence: Optional[Dict] = None

    def copy(self):
        ind = EBDCOIndividual(
            gene_assign=self.gene_assign.copy(),
            gene_elev=self.gene_elev.copy(),
        )
        return ind


class EBDCOSolver:
    """EBDCO主算法"""

    def __init__(self, robots: List[Robot], elevators: List[Elevator], tasks: List[Task],
                 pop_size: int = 100, max_gen: int = 200,
                 n_neighbors: int = 20,
                 crossover_rate: float = 0.9, mutation_rate: float = 0.4,
                 seed: int = 42):
        self.robots = robots
        self.elevators = elevators
        self.tasks = tasks
        self.N = len(tasks)
        self.M = len(robots)
        self.E = len(elevators)
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.n_neighbors = min(n_neighbors, pop_size)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.rng = np.random.RandomState(seed)

        # 问题分析
        self.analyzer = ProblemAnalyzer(robots, elevators, tasks)
        self.analysis = self.analyzer.full_analysis(verbose=False)
        self.conflict_matrix, _ = self.analyzer.build_conflict_graph()
        self.critical_tasks = self.analysis['critical_tasks']['critical_order']
        self.criticality = self.analysis['critical_tasks']['criticality_scores']

        # 下层求解器
        self.lower_solver = LowerLevelSolver(robots, elevators, tasks)

        self.cf_ids = set(i for i, t in enumerate(tasks) if t.is_cross_floor)

        # MOEA/D权重向量
        self.weight_vectors = self._generate_weight_vectors()
        self.n_subproblems = len(self.weight_vectors)
        self.neighborhoods = self._compute_neighborhoods()

        # 统计
        self.eval_count = 0
        self.gen_history = []

    def _generate_weight_vectors(self) -> np.ndarray:
        """生成均匀分布的3目标权重向量（单纯形上）"""
        H = int(np.ceil(self.pop_size ** (1/2)))
        vectors = []
        for i in range(H + 1):
            for j in range(H + 1 - i):
                k = H - i - j
                w = np.array([i, j, k], dtype=float) / H
                if np.min(w) >= 0.02:  # 排除极端权重
                    vectors.append(w)

        # 如果数量不够，补充随机的
        while len(vectors) < self.pop_size:
            w = self.rng.dirichlet([1, 1, 1])
            w = np.maximum(w, 0.02)
            w /= w.sum()
            vectors.append(w)

        return np.array(vectors[:self.pop_size])

    def _compute_neighborhoods(self) -> List[List[int]]:
        """基于权重向量的欧式距离计算邻域"""
        n = len(self.weight_vectors)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(
                    self.weight_vectors[i] - self.weight_vectors[j])

        neighborhoods = []
        for i in range(n):
            sorted_indices = np.argsort(distances[i])
            neighborhoods.append(sorted_indices[:self.n_neighbors].tolist())

        return neighborhoods

    # ===================================================================
    # 初始化策略
    # ===================================================================

    def _init_critical_first(self) -> EBDCOIndividual:
        """
        关键任务优先初始化。

        策略：先确定关键任务（高载重、紧急、长途）的分配，
        再用贪心填充其余任务。利用冲突图避免高冲突任务分给同一机器人。
        """
        gene_assign = np.zeros(self.N, dtype=int)
        gene_elev = np.zeros(self.N, dtype=int)
        robot_load = np.zeros(self.M)

        assigned = set()

        # 阶段1：按关键度顺序分配关键任务
        for i in self.critical_tasks:
            feasible = self.analyzer.feasible_robots[i]
            # 选冲突最小+负载最低的机器人
            best_j = None
            best_score = float('inf')
            for j in feasible:
                # 与已分配给j的任务的冲突总和
                conflict_sum = sum(
                    self.conflict_matrix[i, k]
                    for k in assigned if gene_assign[k] == j
                )
                score = conflict_sum * 10 + robot_load[j]
                if score < best_score:
                    best_score = score
                    best_j = j

            gene_assign[i] = best_j
            robot_load[best_j] += 1
            assigned.add(i)

            # 电梯：选最近/最空闲的
            if i in self.cf_ids:
                gene_elev[i] = self.rng.randint(self.E)

        return EBDCOIndividual(gene_assign=gene_assign, gene_elev=gene_elev)

    def _init_floor_cluster(self) -> EBDCOIndividual:
        """
        楼层聚类初始化。

        按起点楼层将任务分组，每组分配给一个机器人，
        减少跨楼层空跑。
        """
        gene_assign = np.zeros(self.N, dtype=int)
        gene_elev = np.zeros(self.N, dtype=int)

        # 按起点楼层聚类
        floor_tasks = {}
        for i in range(self.N):
            f = self.tasks[i].origin_floor
            if f not in floor_tasks:
                floor_tasks[f] = []
            floor_tasks[f].append(i)

        # 轮流分配给机器人
        robot_idx = 0
        for f in sorted(floor_tasks.keys()):
            for i in floor_tasks[f]:
                feasible = self.analyzer.feasible_robots[i]
                # 从robot_idx开始找可行的
                for offset in range(self.M):
                    j = (robot_idx + offset) % self.M
                    if j in feasible:
                        gene_assign[i] = j
                        break
                if i in self.cf_ids:
                    gene_elev[i] = self.rng.randint(self.E)
            robot_idx = (robot_idx + 1) % self.M

        return EBDCOIndividual(gene_assign=gene_assign, gene_elev=gene_elev)

    def _init_random(self) -> EBDCOIndividual:
        gene_assign = np.zeros(self.N, dtype=int)
        gene_elev = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            feasible = self.analyzer.feasible_robots[i]
            gene_assign[i] = feasible[self.rng.randint(len(feasible))]
            if i in self.cf_ids:
                gene_elev[i] = self.rng.randint(self.E)
        return EBDCOIndividual(gene_assign=gene_assign, gene_elev=gene_elev)

    # ===================================================================
    # 评估（上层决策 → 下层优化 → 仿真评估）
    # ===================================================================

    def _evaluate(self, ind: EBDCOIndividual, weight: np.ndarray):
        """两阶段评估：权重感知的下层优化 → 仿真"""
        assignment = {i: int(ind.gene_assign[i]) for i in range(self.N)}
        elevator_assignment = {i: int(ind.gene_elev[i]) for i in self.cf_ids}

        # 下层优化（权重感知）：
        # 高makespan权重 → 优先紧急任务，减少序列长度
        # 高energy权重 → 优先空间局部性，减少跨楼层转移
        # 高tardiness权重 → 严格按截止时间排序
        task_sequence = self.lower_solver.optimize_sequence(
            assignment, elevator_assignment, weight=weight)
        elevator_assignment = self.lower_solver.optimize_elevator_order(
            task_sequence, elevator_assignment)

        # 仿真评估
        sim = Simulator(self.robots, self.elevators, self.tasks)
        sim.load_schedule(assignment, task_sequence, elevator_assignment)
        result = sim.run()

        ind.objectives = np.array([
            result['makespan'], result['energy'], result['weighted_tardiness']
        ])
        ind.scalar_obj = float(self._tchebycheff(ind.objectives, weight))
        ind.task_sequence = task_sequence

        # 更新电梯分配（下层可能修改了）
        for i, e in elevator_assignment.items():
            ind.gene_elev[i] = e

        self.eval_count += 1

    def _tchebycheff(self, objectives: np.ndarray, weight: np.ndarray) -> float:
        """切比雪夫分解函数（MOEA/D核心）"""
        diff = objectives - self.ideal_point
        # 防止除零：用nadir-ideal做归一化
        range_val = self.nadir_point - self.ideal_point
        range_val = np.maximum(range_val, 1e-6)
        normalized = diff / range_val
        return float(np.max(weight * normalized + 1e-6))

    # ===================================================================
    # 进化算子（利用问题结构）
    # ===================================================================

    def _crossover(self, p1: EBDCOIndividual, p2: EBDCOIndividual) -> EBDCOIndividual:
        """
        冲突感知交叉算子。

        对于低冲突任务：从两个父代随机选择分配
        对于高冲突任务：从适应度更好的父代继承
        """
        child = p1.copy()

        if self.rng.random() > self.crossover_rate:
            return child

        for i in range(self.N):
            # 获取任务i与其他任务的平均冲突度
            avg_conflict = np.mean(self.conflict_matrix[i, self.conflict_matrix[i] > 0]) \
                if np.any(self.conflict_matrix[i] > 0) else 0

            if avg_conflict > 0.3:
                # 高冲突任务：从更好的父代继承
                if p2.scalar_obj < p1.scalar_obj:
                    child.gene_assign[i] = p2.gene_assign[i]
                    if i in self.cf_ids:
                        child.gene_elev[i] = p2.gene_elev[i]
            else:
                # 低冲突任务：50/50随机
                if self.rng.random() < 0.5:
                    child.gene_assign[i] = p2.gene_assign[i]
                    if i in self.cf_ids:
                        child.gene_elev[i] = p2.gene_elev[i]

        self._repair(child)
        return child

    def _mutate(self, ind: EBDCOIndividual):
        """
        结构感知变异。

        4种变异策略，按问题特征选择：
        """
        if self.rng.random() > self.mutation_rate:
            return

        strategy = self.rng.randint(5)

        if strategy == 0:
            # 关键任务重分配：随机重新分配一个关键任务
            critical = self.critical_tasks[:max(5, self.N // 5)]
            i = critical[self.rng.randint(len(critical))]
            feasible = self.analyzer.feasible_robots[i]
            ind.gene_assign[i] = feasible[self.rng.randint(len(feasible))]

        elif strategy == 1:
            # 负载均衡变异：从最忙的机器人转移任务到最闲的
            load = np.bincount(ind.gene_assign, minlength=self.M)
            busiest = np.argmax(load)
            lightest = np.argmin(load)
            tasks_on_busiest = [i for i in range(self.N) if ind.gene_assign[i] == busiest]
            if tasks_on_busiest:
                movable = [i for i in tasks_on_busiest
                           if lightest in self.analyzer.feasible_robots[i]]
                if movable:
                    i = movable[self.rng.randint(len(movable))]
                    ind.gene_assign[i] = lightest

        elif strategy == 2:
            # 电梯均衡变异：将任务从最忙电梯移到最闲电梯
            elev_count = np.zeros(self.E)
            for i in self.cf_ids:
                elev_count[int(ind.gene_elev[i])] += 1
            if elev_count.max() > elev_count.min() + 1:
                busiest_e = np.argmax(elev_count)
                lightest_e = np.argmin(elev_count)
                candidates = [i for i in self.cf_ids if int(ind.gene_elev[i]) == busiest_e]
                if candidates:
                    i = candidates[self.rng.randint(len(candidates))]
                    ind.gene_elev[i] = lightest_e

        elif strategy == 3:
            # 空间局部性变异：将一个任务移到处理相邻楼层任务的机器人
            i = self.rng.randint(self.N)
            target_floor = self.tasks[i].origin_floor
            # 找处理同楼层或邻近楼层任务最多的机器人
            robot_floor_affinity = np.zeros(self.M)
            for k in range(self.N):
                if k == i:
                    continue
                j = int(ind.gene_assign[k])
                dist = abs(self.tasks[k].origin_floor - target_floor)
                if dist <= 2:
                    robot_floor_affinity[j] += 1.0 / (dist + 1)
            feasible = self.analyzer.feasible_robots[i]
            feasible_affinity = [(j, robot_floor_affinity[j]) for j in feasible]
            if feasible_affinity:
                best_j = max(feasible_affinity, key=lambda x: x[1])[0]
                ind.gene_assign[i] = best_j

        elif strategy == 4:
            # 随机变异（exploration保证）
            i = self.rng.randint(self.N)
            feasible = self.analyzer.feasible_robots[i]
            ind.gene_assign[i] = feasible[self.rng.randint(len(feasible))]

    def _repair(self, ind: EBDCOIndividual):
        """修复不可行解"""
        for i in range(self.N):
            j = int(ind.gene_assign[i])
            if j not in self.analyzer.feasible_robots[i]:
                feasible = self.analyzer.feasible_robots[i]
                ind.gene_assign[i] = feasible[self.rng.randint(len(feasible))]
            if i in self.cf_ids:
                ind.gene_elev[i] = int(ind.gene_elev[i]) % self.E

    # ===================================================================
    # 主循环（MOEA/D框架）
    # ===================================================================

    def solve(self, verbose: bool = True) -> Dict:
        t_start = time.time()
        N_sub = min(self.n_subproblems, self.pop_size)

        if verbose:
            lb = self.analysis['bottleneck_bounds']['lb_combined']
            print(f"EBDCO: pop={N_sub}, gen={self.max_gen}, "
                  f"N={self.N}, M={self.M}, E={self.E}")
            print(f"  电梯瓶颈下界: {lb:.0f}s")
            print(f"  关键任务数: {len(self.critical_tasks[:max(5, self.N//5)])}")
            print(f"  高冲突对数: {self.analysis['conflict']['high_conflict_pairs']}")

        # 初始化种群
        pop = []
        # 30%关键任务优先, 20%楼层聚类, 50%随机
        for _ in range(N_sub * 3 // 10):
            pop.append(self._init_critical_first())
        for _ in range(N_sub * 2 // 10):
            pop.append(self._init_floor_cluster())
        while len(pop) < N_sub:
            pop.append(self._init_random())
        pop = pop[:N_sub]

        # 初始评估 — 先用加权和做粗评估找到理想点/nadir点
        self.ideal_point = np.array([np.inf, np.inf, np.inf])
        self.nadir_point = np.array([-np.inf, -np.inf, -np.inf])

        for idx, ind in enumerate(pop):
            self._evaluate(ind, self.weight_vectors[idx])
            self.ideal_point = np.minimum(self.ideal_point, ind.objectives)
            self.nadir_point = np.maximum(self.nadir_point, ind.objectives)

        # 给理想点留余量
        self.ideal_point *= 0.95
        # nadir点稍微扩大
        self.nadir_point *= 1.05

        # 用正确的理想/nadir点重新计算标量值
        for idx, ind in enumerate(pop):
            ind.scalar_obj = self._tchebycheff(ind.objectives, self.weight_vectors[idx])

        # Pareto archive
        archive = self._update_archive([], pop)

        if verbose:
            best_ms = min(ind.objectives[0] for ind in pop)
            best_en = min(ind.objectives[1] for ind in pop)
            best_td = min(ind.objectives[2] for ind in pop)
            print(f"  初始: best=[{best_ms:.0f}, {best_en:.0f}, {best_td:.0f}], "
                  f"archive={len(archive)}")

        # 主循环
        for gen in range(self.max_gen):
            perm = self.rng.permutation(N_sub)

            for idx in perm:
                # 从邻域选两个父代
                neighbors = self.neighborhoods[idx]
                p1_idx, p2_idx = self.rng.choice(neighbors, 2, replace=False)

                # 交叉 + 变异
                child = self._crossover(pop[p1_idx], pop[p2_idx])
                self._mutate(child)

                # 评估（包含下层优化）
                self._evaluate(child, self.weight_vectors[idx])

                # 更新理想点和nadir点
                self.ideal_point = np.minimum(self.ideal_point, child.objectives * 0.95)
                self.nadir_point = np.maximum(self.nadir_point, child.objectives * 1.05)

                # 更新邻域（MOEA/D核心）
                for nb_idx in neighbors:
                    child_scalar = self._tchebycheff(child.objectives, self.weight_vectors[nb_idx])
                    if child_scalar < pop[nb_idx].scalar_obj:
                        pop[nb_idx] = child.copy()
                        pop[nb_idx].scalar_obj = child_scalar

            # 更新archive
            archive = self._update_archive(archive, pop)

            # 记录
            all_objs = np.array([ind.objectives for ind in pop])
            best_ms = np.min(all_objs[:, 0])
            best_en = np.min(all_objs[:, 1])
            best_td = np.min(all_objs[:, 2])

            self.gen_history.append({
                'gen': gen,
                'best_makespan': float(best_ms),
                'best_energy': float(best_en),
                'best_tardiness': float(best_td),
                'archive_size': len(archive),
                'evals': self.eval_count,
            })

            if verbose and (gen % 20 == 0 or gen == self.max_gen - 1):
                elapsed = time.time() - t_start
                print(f"  Gen {gen:>3d}: best=[{best_ms:.0f}, {best_en:.0f}, {best_td:.0f}], "
                      f"archive={len(archive)}, evals={self.eval_count}, {elapsed:.1f}s")

        total_time = time.time() - t_start

        # 结果
        archive_objs = np.array([obj for obj in archive])
        result = {
            "pareto_front": archive_objs.tolist(),
            "pareto_size": len(archive),
            "best_makespan": archive_objs[np.argmin(archive_objs[:, 0])].tolist() if len(archive) > 0 else None,
            "best_energy": archive_objs[np.argmin(archive_objs[:, 1])].tolist() if len(archive) > 0 else None,
            "best_tardiness": archive_objs[np.argmin(archive_objs[:, 2])].tolist() if len(archive) > 0 else None,
            "total_evals": self.eval_count,
            "total_time": total_time,
            "gen_history": self.gen_history,
            "analysis": {
                "lb_combined": self.analysis['bottleneck_bounds']['lb_combined'],
                "critical_tasks": len(self.critical_tasks[:max(5, self.N // 5)]),
            },
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"EBDCO 完成: {total_time:.1f}s, {self.eval_count} evaluations")
            print(f"Pareto archive: {len(archive)} 个解")
            if len(archive) > 0:
                print(f"最优Makespan: {archive_objs[np.argmin(archive_objs[:,0])]}")
                print(f"最优能耗:     {archive_objs[np.argmin(archive_objs[:,1])]}")
                print(f"最优延迟:     {archive_objs[np.argmin(archive_objs[:,2])]}")
                lb = self.analysis['bottleneck_bounds']['lb_combined']
                gap = (archive_objs[:, 0].min() - lb) / lb * 100
                print(f"Makespan vs 下界: {archive_objs[:,0].min():.0f} vs {lb:.0f} (gap={gap:.1f}%)")

        return result

    def _update_archive(self, archive: List[np.ndarray],
                         pop: List[EBDCOIndividual]) -> List[np.ndarray]:
        """更新Pareto archive"""
        candidates = list(archive) + [ind.objectives.copy() for ind in pop]
        # 非支配筛选
        result = []
        for i, ci in enumerate(candidates):
            dominated = False
            for j, cj in enumerate(candidates):
                if i == j:
                    continue
                if np.all(cj <= ci) and np.any(cj < ci):
                    dominated = True
                    break
            if not dominated:
                # 去重
                is_dup = False
                for r in result:
                    if np.allclose(r, ci, atol=0.1):
                        is_dup = True
                        break
                if not is_dup:
                    result.append(ci)
        return result
