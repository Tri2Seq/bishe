"""NSGA-II 多目标进化算法

染色体编码（三层）：
  Layer 1: 任务-机器人分配 gene_assign[i] ∈ {0,...,M-1}
  Layer 2: 每个机器人的任务执行顺序 gene_order[j] = permutation of assigned tasks
  Layer 3: 跨楼层任务的电梯选择 gene_elev[i] ∈ {0,...,E-1}

适应度评估：通过仿真器计算三个目标值
  f1 = makespan, f2 = energy, f3 = weighted_tardiness

专用算子：
  - 交叉: 任务分配层用均匀交叉 + 序列层用OX(Order Crossover)
  - 变异: 任务重分配 + 序列内交换 + 电梯重选
  - 修复: 载重不可行 → 找最近可行机器人

参考: Deb et al., 2002, "A fast and elitist multiobjective genetic algorithm: NSGA-II"
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import copy
import time

from config import Robot, Elevator, Task, Priority, PRIORITY_WEIGHT, FLOOR_DISTANCE, INTER_STATION_DISTANCE
from simulator import Simulator


@dataclass
class Individual:
    """个体（染色体）"""
    gene_assign: np.ndarray    # shape (N,): task i → robot j
    gene_elev: np.ndarray      # shape (N,): task i → elevator e (仅跨楼层有效)
    gene_priority: np.ndarray  # shape (N,): 任务优先级键（用于序列排序）

    objectives: np.ndarray = field(default_factory=lambda: np.array([np.inf, np.inf, np.inf]))
    rank: int = 0
    crowding_dist: float = 0.0
    feasible: bool = True

    def copy(self):
        return Individual(
            gene_assign=self.gene_assign.copy(),
            gene_elev=self.gene_elev.copy(),
            gene_priority=self.gene_priority.copy(),
        )


class NSGA2Solver:
    """NSGA-II求解器"""

    def __init__(self, robots: List[Robot], elevators: List[Elevator], tasks: List[Task],
                 pop_size: int = 100, max_gen: int = 200,
                 crossover_rate: float = 0.9, mutation_rate: float = 0.3,
                 seed: int = 42):
        self.robots = robots
        self.elevators = elevators
        self.tasks = tasks
        self.N = len(tasks)
        self.M = len(robots)
        self.E = len(elevators)
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.rng = np.random.RandomState(seed)

        # 预计算可行分配（考虑载重约束）
        self.feasible_robots = {}
        for i in range(self.N):
            self.feasible_robots[i] = [j for j in range(self.M)
                                        if tasks[i].weight <= robots[j].payload_max]

        self.cf_ids = set(i for i, t in enumerate(tasks) if t.is_cross_floor)

        # 统计
        self.eval_count = 0
        self.gen_history = []

    def _init_individual(self) -> Individual:
        """随机初始化一个可行个体"""
        gene_assign = np.zeros(self.N, dtype=int)
        gene_elev = np.zeros(self.N, dtype=int)
        gene_priority = self.rng.rand(self.N)

        for i in range(self.N):
            feasible = self.feasible_robots[i]
            gene_assign[i] = feasible[self.rng.randint(len(feasible))]
            if i in self.cf_ids:
                gene_elev[i] = self.rng.randint(self.E)

        return Individual(gene_assign=gene_assign, gene_elev=gene_elev,
                          gene_priority=gene_priority)

    def _init_heuristic_individual(self) -> Individual:
        """启发式初始化：按优先级+距离贪心分配"""
        gene_assign = np.zeros(self.N, dtype=int)
        gene_elev = np.zeros(self.N, dtype=int)

        task_order = sorted(range(self.N),
                            key=lambda i: (self.tasks[i].priority,
                                           self.tasks[i].deadline))
        robot_load = {j: 0.0 for j in range(self.M)}
        robot_floor = {j: self.robots[j].init_floor for j in range(self.M)}

        for i in task_order:
            task = self.tasks[i]
            best_j = None
            best_cost = float('inf')
            for j in self.feasible_robots[i]:
                dist = abs(robot_floor[j] - task.origin_floor)
                cost = robot_load[j] + dist * 10
                if cost < best_cost:
                    best_cost = cost
                    best_j = j
            gene_assign[i] = best_j
            robot_load[best_j] += 1
            robot_floor[best_j] = task.dest_floor

            if i in self.cf_ids:
                gene_elev[i] = self.rng.randint(self.E)

        gene_priority = np.array([self.tasks[i].priority * 100 + self.tasks[i].deadline / 10
                                   for i in range(self.N)])
        # 归一化到[0,1]
        if gene_priority.max() > gene_priority.min():
            gene_priority = (gene_priority - gene_priority.min()) / (gene_priority.max() - gene_priority.min())
        else:
            gene_priority = self.rng.rand(self.N)

        return Individual(gene_assign=gene_assign, gene_elev=gene_elev,
                          gene_priority=gene_priority)

    def _repair(self, ind: Individual):
        """修复不可行的个体"""
        for i in range(self.N):
            j = ind.gene_assign[i]
            if j not in self.feasible_robots[i]:
                ind.gene_assign[i] = self.feasible_robots[i][
                    self.rng.randint(len(self.feasible_robots[i]))]
            if i in self.cf_ids:
                ind.gene_elev[i] = ind.gene_elev[i] % self.E

    def _decode(self, ind: Individual) -> Tuple[Dict, Dict, Dict]:
        """解码染色体为调度方案"""
        assignment = {i: int(ind.gene_assign[i]) for i in range(self.N)}

        elevator_assignment = {}
        for i in self.cf_ids:
            elevator_assignment[i] = int(ind.gene_elev[i])

        # 根据gene_priority确定每个机器人的任务执行顺序
        task_sequence = {j: [] for j in range(self.M)}
        for j in range(self.M):
            assigned = [i for i in range(self.N) if assignment[i] == j]
            assigned.sort(key=lambda i: ind.gene_priority[i])
            task_sequence[j] = assigned

        return assignment, elevator_assignment, task_sequence

    def _evaluate(self, ind: Individual):
        """用仿真器评估个体"""
        assignment, elevator_assignment, task_sequence = self._decode(ind)

        sim = Simulator(self.robots, self.elevators, self.tasks)
        sim.load_schedule(assignment, task_sequence, elevator_assignment)
        result = sim.run(verbose=False)

        ind.objectives = np.array([
            result['makespan'],
            result['energy'],
            result['weighted_tardiness'],
        ])
        ind.feasible = result['completed_tasks'] == self.N
        self.eval_count += 1

    def _crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        """交叉算子"""
        c1, c2 = p1.copy(), p2.copy()

        if self.rng.random() > self.crossover_rate:
            return c1, c2

        # Layer 1: 均匀交叉（任务分配）
        mask = self.rng.random(self.N) < 0.5
        c1.gene_assign[mask] = p2.gene_assign[mask]
        c2.gene_assign[mask] = p1.gene_assign[mask]

        # Layer 2: 混合优先级（SBX-like）
        beta = self.rng.random(self.N)
        c1.gene_priority = beta * p1.gene_priority + (1 - beta) * p2.gene_priority
        c2.gene_priority = (1 - beta) * p1.gene_priority + beta * p2.gene_priority

        # Layer 3: 电梯选择均匀交叉
        elev_mask = self.rng.random(self.N) < 0.5
        c1.gene_elev[elev_mask] = p2.gene_elev[elev_mask]
        c2.gene_elev[elev_mask] = p1.gene_elev[elev_mask]

        self._repair(c1)
        self._repair(c2)
        return c1, c2

    def _mutate(self, ind: Individual):
        """变异算子"""
        if self.rng.random() > self.mutation_rate:
            return

        mutation_type = self.rng.randint(4)

        if mutation_type == 0:
            # 任务重分配：随机选一个任务换机器人
            i = self.rng.randint(self.N)
            ind.gene_assign[i] = self.feasible_robots[i][
                self.rng.randint(len(self.feasible_robots[i]))]

        elif mutation_type == 1:
            # 序列交换：随机选同一机器人的两个任务交换优先级
            j = self.rng.randint(self.M)
            assigned = [i for i in range(self.N) if ind.gene_assign[i] == j]
            if len(assigned) >= 2:
                a, b = self.rng.choice(assigned, 2, replace=False)
                ind.gene_priority[a], ind.gene_priority[b] = \
                    ind.gene_priority[b], ind.gene_priority[a]

        elif mutation_type == 2:
            # 电梯重选
            cf_list = list(self.cf_ids)
            if cf_list:
                i = cf_list[self.rng.randint(len(cf_list))]
                ind.gene_elev[i] = self.rng.randint(self.E)

        elif mutation_type == 3:
            # 优先级微扰
            i = self.rng.randint(self.N)
            ind.gene_priority[i] += self.rng.normal(0, 0.1)
            ind.gene_priority[i] = np.clip(ind.gene_priority[i], 0, 1)

    def _fast_nondominated_sort(self, pop: List[Individual]) -> List[List[int]]:
        """快速非支配排序"""
        n = len(pop)
        domination_count = [0] * n
        dominated_set = [[] for _ in range(n)]
        fronts = [[]]

        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if self._dominates(pop[p], pop[q]):
                    dominated_set[p].append(q)
                elif self._dominates(pop[q], pop[p]):
                    domination_count[p] += 1

            if domination_count[p] == 0:
                pop[p].rank = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in dominated_set[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        pop[q].rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return [f for f in fronts if f]

    def _dominates(self, a: Individual, b: Individual) -> bool:
        """a是否支配b（所有目标不差，至少一个更好）"""
        if not a.feasible and b.feasible:
            return False
        if a.feasible and not b.feasible:
            return True
        return (np.all(a.objectives <= b.objectives) and
                np.any(a.objectives < b.objectives))

    def _crowding_distance(self, pop: List[Individual], front: List[int]):
        """计算拥挤度距离"""
        n = len(front)
        if n <= 2:
            for idx in front:
                pop[idx].crowding_dist = float('inf')
            return

        for idx in front:
            pop[idx].crowding_dist = 0.0

        num_obj = len(pop[0].objectives)
        for m in range(num_obj):
            sorted_front = sorted(front, key=lambda idx: pop[idx].objectives[m])
            pop[sorted_front[0]].crowding_dist = float('inf')
            pop[sorted_front[-1]].crowding_dist = float('inf')

            obj_range = (pop[sorted_front[-1]].objectives[m] -
                         pop[sorted_front[0]].objectives[m])
            if obj_range < 1e-10:
                continue

            for i in range(1, n - 1):
                pop[sorted_front[i]].crowding_dist += \
                    (pop[sorted_front[i + 1]].objectives[m] -
                     pop[sorted_front[i - 1]].objectives[m]) / obj_range

    def _tournament_select(self, pop: List[Individual]) -> Individual:
        """锦标赛选择（rank优先，相同rank选拥挤度大的）"""
        a, b = self.rng.randint(len(pop), size=2)
        if pop[a].rank < pop[b].rank:
            return pop[a]
        elif pop[a].rank > pop[b].rank:
            return pop[b]
        elif pop[a].crowding_dist > pop[b].crowding_dist:
            return pop[a]
        else:
            return pop[b]

    def solve(self, verbose: bool = True) -> Dict:
        """运行NSGA-II"""
        t_start = time.time()

        # 初始化种群
        pop = []
        # 20%启发式初始化
        for _ in range(self.pop_size // 5):
            ind = self._init_heuristic_individual()
            pop.append(ind)
        # 80%随机初始化
        for _ in range(self.pop_size - len(pop)):
            ind = self._init_individual()
            pop.append(ind)

        # 评估
        for ind in pop:
            self._evaluate(ind)

        if verbose:
            print(f"NSGA-II: pop={self.pop_size}, gen={self.max_gen}, "
                  f"N={self.N}, M={self.M}, E={self.E}")

        for gen in range(self.max_gen):
            # 生成子代
            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self._tournament_select(pop)
                p2 = self._tournament_select(pop)
                c1, c2 = self._crossover(p1, p2)
                self._mutate(c1)
                self._mutate(c2)
                self._evaluate(c1)
                self._evaluate(c2)
                offspring.extend([c1, c2])

            offspring = offspring[:self.pop_size]

            # 合并父代和子代
            combined = pop + offspring
            fronts = self._fast_nondominated_sort(combined)

            # 选择下一代
            new_pop = []
            for front in fronts:
                if len(new_pop) + len(front) <= self.pop_size:
                    self._crowding_distance(combined, front)
                    new_pop.extend([combined[idx] for idx in front])
                else:
                    self._crowding_distance(combined, front)
                    remaining = self.pop_size - len(new_pop)
                    sorted_front = sorted(front, key=lambda idx: -combined[idx].crowding_dist)
                    new_pop.extend([combined[idx] for idx in sorted_front[:remaining]])
                    break

            pop = new_pop

            # 记录
            pareto_front = [ind for ind in pop if ind.rank == 0]
            best_makespan = min(ind.objectives[0] for ind in pareto_front)
            best_energy = min(ind.objectives[1] for ind in pareto_front)
            best_tard = min(ind.objectives[2] for ind in pareto_front)

            self.gen_history.append({
                'gen': gen,
                'pareto_size': len(pareto_front),
                'best_makespan': best_makespan,
                'best_energy': best_energy,
                'best_tardiness': best_tard,
                'evals': self.eval_count,
            })

            if verbose and (gen % 20 == 0 or gen == self.max_gen - 1):
                elapsed = time.time() - t_start
                print(f"  Gen {gen:>3d}: Pareto={len(pareto_front):>3d}, "
                      f"best=[{best_makespan:.0f}, {best_energy:.0f}, {best_tard:.0f}], "
                      f"evals={self.eval_count}, {elapsed:.1f}s")

        total_time = time.time() - t_start

        # 提取Pareto前沿
        pareto_front = [ind for ind in pop if ind.rank == 0]
        pareto_objectives = np.array([ind.objectives for ind in pareto_front])

        # 找三个极端解
        best_makespan_idx = np.argmin(pareto_objectives[:, 0])
        best_energy_idx = np.argmin(pareto_objectives[:, 1])
        best_tard_idx = np.argmin(pareto_objectives[:, 2])

        result = {
            "pareto_front": pareto_objectives.tolist(),
            "pareto_size": len(pareto_front),
            "best_makespan": pareto_objectives[best_makespan_idx].tolist(),
            "best_energy": pareto_objectives[best_energy_idx].tolist(),
            "best_tardiness": pareto_objectives[best_tard_idx].tolist(),
            "total_evals": self.eval_count,
            "total_time": total_time,
            "gen_history": self.gen_history,
            "pareto_individuals": pareto_front,
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"NSGA-II 完成: {total_time:.1f}s, {self.eval_count} evaluations")
            print(f"Pareto前沿大小: {len(pareto_front)}")
            print(f"最优Makespan: {pareto_objectives[best_makespan_idx]}")
            print(f"最优能耗:     {pareto_objectives[best_energy_idx]}")
            print(f"最优延迟:     {pareto_objectives[best_tard_idx]}")

        return result


def compute_hypervolume(pareto_front: np.ndarray, ref_point: np.ndarray) -> float:
    """计算2D/3D超体积指标（精确计算，仅支持2-3目标）

    使用简化的包含-排除方法。对于论文级实验，建议用pygmo或deap的HV实现。
    这里提供一个近似版本：归一化后用蒙特卡洛估算。
    """
    if pareto_front.shape[0] == 0:
        return 0.0

    n_obj = pareto_front.shape[1]
    n_points = pareto_front.shape[0]

    # 归一化到 [0, 1]
    normalized = pareto_front.copy()
    for m in range(n_obj):
        rng = ref_point[m] - pareto_front[:, m].min()
        if rng > 1e-10:
            normalized[:, m] = (pareto_front[:, m] - pareto_front[:, m].min()) / rng
        else:
            normalized[:, m] = 0

    ref_norm = np.ones(n_obj)

    # 蒙特卡洛估算
    n_samples = 100000
    rng = np.random.RandomState(42)
    samples = rng.random((n_samples, n_obj))

    dominated_count = 0
    for s in range(n_samples):
        for p in range(n_points):
            if np.all(normalized[p] <= samples[s]):
                dominated_count += 1
                break

    total_volume = np.prod(ref_norm)
    return dominated_count / n_samples * total_volume
