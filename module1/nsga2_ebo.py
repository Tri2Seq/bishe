"""NSGA-II-EBO: 基于电梯瓶颈优化的NSGA-II混合算法

核心创新：
  标准NSGA-II的适应度评估 = 直接仿真（黑箱）
  NSGA-II-EBO的适应度评估 = 下层结构优化 → 仿真（灰箱）

  下层优化器利用三个问题结构特征：
    1. 权重感知序列优化 — 根据子问题权重选择不同排序策略
    2. 电梯负载均衡 — 重新分配电梯以均衡各电梯服务时间
    3. 关键任务优先 — 高关键度任务先确定位置

  效果：搜索空间从 O(M^N × N! × E^N) 降为 O(M^N × E^N)
        序列维度 N! 被下层优化器处理，不需要进化搜索

  vs 标准NSGA-II：同样评估次数下，每次评估质量更高
  vs EBDCO：用NSGA-II框架替代MOEA/D，获得更好的Pareto多样性

算法名称：NSGA-II-EBO (NSGA-II with Elevator-Bottleneck Optimization)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import time

from config import (Robot, Elevator, Task, Priority, PRIORITY_WEIGHT,
                    FLOOR_DISTANCE, INTER_STATION_DISTANCE)
from problem_analysis import ProblemAnalyzer
from ebdco import LowerLevelSolver
from simulator import Simulator


@dataclass
class EBOIndividual:
    gene_assign: np.ndarray
    gene_elev: np.ndarray
    objectives: np.ndarray = field(default_factory=lambda: np.array([np.inf, np.inf, np.inf]))
    rank: int = 0
    crowding_dist: float = 0.0

    def copy(self):
        return EBOIndividual(
            gene_assign=self.gene_assign.copy(),
            gene_elev=self.gene_elev.copy(),
        )


class NSGA2EBOSolver:

    def __init__(self, robots: List[Robot], elevators: List[Elevator], tasks: List[Task],
                 pop_size: int = 100, max_gen: int = 200,
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
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.rng = np.random.RandomState(seed)

        # 问题分析
        self.analyzer = ProblemAnalyzer(robots, elevators, tasks)
        self.analysis = self.analyzer.full_analysis(verbose=False)
        self.conflict_matrix, _ = self.analyzer.build_conflict_graph()
        self.criticality = self.analysis['critical_tasks']['criticality_scores']
        self.critical_tasks = self.analysis['critical_tasks']['critical_order']

        # 下层优化器
        self.lower_solver = LowerLevelSolver(robots, elevators, tasks)

        self.cf_ids = set(i for i, t in enumerate(tasks) if t.is_cross_floor)

        self.eval_count = 0
        self.gen_history = []

    # ===================================================================
    # 初始化（利用问题结构）
    # ===================================================================

    def _init_critical_first(self) -> EBOIndividual:
        """关键任务优先 + 冲突感知初始化"""
        gene_assign = np.zeros(self.N, dtype=int)
        gene_elev = np.zeros(self.N, dtype=int)
        robot_load = np.zeros(self.M)
        assigned = set()

        for i in self.critical_tasks:
            feasible = self.analyzer.feasible_robots[i]
            best_j, best_score = feasible[0], float('inf')
            for j in feasible:
                conflict_sum = sum(
                    self.conflict_matrix[i, k] for k in assigned if gene_assign[k] == j)
                score = conflict_sum * 10 + robot_load[j]
                if score < best_score:
                    best_score, best_j = score, j
            gene_assign[i] = best_j
            robot_load[best_j] += 1
            assigned.add(i)
            if i in self.cf_ids:
                gene_elev[i] = self.rng.randint(self.E)

        return EBOIndividual(gene_assign=gene_assign, gene_elev=gene_elev)

    def _init_floor_cluster(self) -> EBOIndividual:
        gene_assign = np.zeros(self.N, dtype=int)
        gene_elev = np.zeros(self.N, dtype=int)
        floor_tasks = {}
        for i in range(self.N):
            f = self.tasks[i].origin_floor
            floor_tasks.setdefault(f, []).append(i)

        robot_idx = 0
        for f in sorted(floor_tasks):
            for i in floor_tasks[f]:
                feasible = self.analyzer.feasible_robots[i]
                for offset in range(self.M):
                    j = (robot_idx + offset) % self.M
                    if j in feasible:
                        gene_assign[i] = j
                        break
                if i in self.cf_ids:
                    gene_elev[i] = self.rng.randint(self.E)
            robot_idx = (robot_idx + 1) % self.M
        return EBOIndividual(gene_assign=gene_assign, gene_elev=gene_elev)

    def _init_random(self) -> EBOIndividual:
        gene_assign = np.zeros(self.N, dtype=int)
        gene_elev = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            feasible = self.analyzer.feasible_robots[i]
            gene_assign[i] = feasible[self.rng.randint(len(feasible))]
            if i in self.cf_ids:
                gene_elev[i] = self.rng.randint(self.E)
        return EBOIndividual(gene_assign=gene_assign, gene_elev=gene_elev)

    # ===================================================================
    # 评估：上层编码 → 下层优化 → 仿真
    # ===================================================================

    def _evaluate(self, ind: EBOIndividual):
        """灰箱评估 = 下层结构优化 + 仿真"""
        assignment = {i: int(ind.gene_assign[i]) for i in range(self.N)}
        elevator_assignment = {i: int(ind.gene_elev[i]) for i in self.cf_ids}

        # 下层优化1：序列优化（多策略取最优）
        best_seq = None
        best_obj = np.inf

        # 尝试3种权重方向的序列，取仿真结果最优的
        for w in [np.array([0.7, 0.2, 0.1]),
                  np.array([0.2, 0.6, 0.2]),
                  np.array([0.1, 0.2, 0.7])]:
            seq = self.lower_solver.optimize_sequence(assignment, elevator_assignment, weight=w)
            # 快速代价估算（不用完整仿真）
            cost = self._quick_estimate(assignment, seq, elevator_assignment)
            if cost < best_obj:
                best_obj = cost
                best_seq = seq

        # 下层优化2：电梯负载均衡
        elevator_assignment = self.lower_solver.optimize_elevator_order(
            best_seq, elevator_assignment)

        # 更新电梯基因
        for i, e in elevator_assignment.items():
            ind.gene_elev[i] = e

        # 完整仿真评估
        sim = Simulator(self.robots, self.elevators, self.tasks)
        sim.load_schedule(assignment, best_seq, elevator_assignment)
        result = sim.run()

        ind.objectives = np.array([
            result['makespan'], result['energy'], result['weighted_tardiness']
        ])
        self.eval_count += 1

    def _quick_estimate(self, assignment, task_sequence, elevator_assignment) -> float:
        """快速代价估算（避免完整仿真，O(N)时间）"""
        robot_time = np.zeros(self.M)
        total_energy = 0

        for j in range(self.M):
            current_floor = self.robots[j].init_floor
            t = 0.0
            for i in task_sequence.get(j, []):
                task = self.tasks[i]
                # 到达取货点的时间
                if current_floor != task.origin_floor:
                    t += abs(current_floor - task.origin_floor) * 3.0 + 20.0  # 电梯交互
                t += 10.0  # 取货
                if task.is_cross_floor:
                    e = elevator_assignment.get(i, 0)
                    t += abs(task.origin_floor - task.dest_floor) / self.elevators[e].speed
                    t += 25.0  # 电梯交互开销
                t += 10.0  # 送货
                current_floor = task.dest_floor

                dist = abs(task.origin_floor - task.dest_floor) * 3.0 + 30.0
                total_energy += dist * self.robots[j].energy_per_m

            robot_time[j] = t

        makespan = robot_time.max()
        return makespan * 0.5 + total_energy * 0.3

    # ===================================================================
    # 结构感知交叉/变异
    # ===================================================================

    def _crossover(self, p1: EBOIndividual, p2: EBOIndividual) -> Tuple[EBOIndividual, EBOIndividual]:
        c1, c2 = p1.copy(), p2.copy()
        if self.rng.random() > self.crossover_rate:
            return c1, c2

        for i in range(self.N):
            # 关键任务：从更好的父代继承（强exploitation）
            if self.criticality[i] > 0.35:
                if np.sum(p2.objectives) < np.sum(p1.objectives):
                    c1.gene_assign[i] = p2.gene_assign[i]
                    c1.gene_elev[i] = p2.gene_elev[i]
                if np.sum(p1.objectives) < np.sum(p2.objectives):
                    c2.gene_assign[i] = p1.gene_assign[i]
                    c2.gene_elev[i] = p1.gene_elev[i]
            else:
                # 普通任务：均匀交叉
                if self.rng.random() < 0.5:
                    c1.gene_assign[i] = p2.gene_assign[i]
                    c1.gene_elev[i] = p2.gene_elev[i]
                if self.rng.random() < 0.5:
                    c2.gene_assign[i] = p1.gene_assign[i]
                    c2.gene_elev[i] = p1.gene_elev[i]

        self._repair(c1)
        self._repair(c2)
        return c1, c2

    def _mutate(self, ind: EBOIndividual):
        if self.rng.random() > self.mutation_rate:
            return

        strategy = self.rng.randint(4)
        if strategy == 0:
            # 关键任务重分配
            critical = self.critical_tasks[:max(5, self.N // 5)]
            i = critical[self.rng.randint(len(critical))]
            feasible = self.analyzer.feasible_robots[i]
            ind.gene_assign[i] = feasible[self.rng.randint(len(feasible))]
        elif strategy == 1:
            # 负载均衡
            load = np.bincount(ind.gene_assign, minlength=self.M)
            busiest = np.argmax(load)
            lightest = np.argmin(load)
            movable = [i for i in range(self.N)
                       if ind.gene_assign[i] == busiest
                       and lightest in self.analyzer.feasible_robots[i]]
            if movable:
                ind.gene_assign[movable[self.rng.randint(len(movable))]] = lightest
        elif strategy == 2:
            # 电梯均衡
            cf_list = list(self.cf_ids)
            if cf_list:
                i = cf_list[self.rng.randint(len(cf_list))]
                ind.gene_elev[i] = self.rng.randint(self.E)
        else:
            # 随机
            i = self.rng.randint(self.N)
            feasible = self.analyzer.feasible_robots[i]
            ind.gene_assign[i] = feasible[self.rng.randint(len(feasible))]

    def _repair(self, ind: EBOIndividual):
        for i in range(self.N):
            j = int(ind.gene_assign[i])
            if j not in self.analyzer.feasible_robots[i]:
                ind.gene_assign[i] = self.analyzer.feasible_robots[i][0]

    # ===================================================================
    # NSGA-II框架
    # ===================================================================

    def _fast_nondominated_sort(self, pop):
        n = len(pop)
        dom_count = [0] * n
        dom_set = [[] for _ in range(n)]
        fronts = [[]]
        for p in range(n):
            for q in range(n):
                if p == q: continue
                if self._dominates(pop[p], pop[q]):
                    dom_set[p].append(q)
                elif self._dominates(pop[q], pop[p]):
                    dom_count[p] += 1
            if dom_count[p] == 0:
                pop[p].rank = 0
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            nxt = []
            for p in fronts[i]:
                for q in dom_set[p]:
                    dom_count[q] -= 1
                    if dom_count[q] == 0:
                        pop[q].rank = i + 1
                        nxt.append(q)
            i += 1
            fronts.append(nxt)
        return [f for f in fronts if f]

    def _dominates(self, a, b):
        return (np.all(a.objectives <= b.objectives) and
                np.any(a.objectives < b.objectives))

    def _crowding_distance(self, pop, front):
        n = len(front)
        if n <= 2:
            for idx in front:
                pop[idx].crowding_dist = float('inf')
            return
        for idx in front:
            pop[idx].crowding_dist = 0.0
        for m in range(3):
            sf = sorted(front, key=lambda idx: pop[idx].objectives[m])
            pop[sf[0]].crowding_dist = float('inf')
            pop[sf[-1]].crowding_dist = float('inf')
            rng = pop[sf[-1]].objectives[m] - pop[sf[0]].objectives[m]
            if rng < 1e-10: continue
            for i in range(1, n - 1):
                pop[sf[i]].crowding_dist += (
                    pop[sf[i+1]].objectives[m] - pop[sf[i-1]].objectives[m]) / rng

    def _tournament(self, pop):
        a, b = self.rng.randint(len(pop), size=2)
        if pop[a].rank < pop[b].rank: return pop[a]
        if pop[a].rank > pop[b].rank: return pop[b]
        return pop[a] if pop[a].crowding_dist > pop[b].crowding_dist else pop[b]

    # ===================================================================
    # 主循环
    # ===================================================================

    def solve(self, verbose=True):
        t_start = time.time()

        # 混合初始化
        pop = []
        for _ in range(self.pop_size * 3 // 10):
            pop.append(self._init_critical_first())
        for _ in range(self.pop_size * 2 // 10):
            pop.append(self._init_floor_cluster())
        while len(pop) < self.pop_size:
            pop.append(self._init_random())

        for ind in pop:
            self._evaluate(ind)

        if verbose:
            lb = self.analysis['bottleneck_bounds']['lb_combined']
            print(f"NSGA-II-EBO: pop={self.pop_size}, gen={self.max_gen}, "
                  f"N={self.N}, M={self.M}, E={self.E}")
            print(f"  电梯瓶颈下界: {lb:.0f}s, 关键任务: {len(self.critical_tasks[:max(5,self.N//5)])}")

        for gen in range(self.max_gen):
            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self._tournament(pop)
                p2 = self._tournament(pop)
                c1, c2 = self._crossover(p1, p2)
                self._mutate(c1)
                self._mutate(c2)
                self._evaluate(c1)
                self._evaluate(c2)
                offspring.extend([c1, c2])
            offspring = offspring[:self.pop_size]

            combined = pop + offspring
            fronts = self._fast_nondominated_sort(combined)

            new_pop = []
            for front in fronts:
                if len(new_pop) + len(front) <= self.pop_size:
                    self._crowding_distance(combined, front)
                    new_pop.extend([combined[idx] for idx in front])
                else:
                    self._crowding_distance(combined, front)
                    remaining = self.pop_size - len(new_pop)
                    sf = sorted(front, key=lambda idx: -combined[idx].crowding_dist)
                    new_pop.extend([combined[idx] for idx in sf[:remaining]])
                    break
            pop = new_pop

            pareto = [ind for ind in pop if ind.rank == 0]
            best_ms = min(ind.objectives[0] for ind in pareto)
            best_en = min(ind.objectives[1] for ind in pareto)
            best_td = min(ind.objectives[2] for ind in pareto)

            self.gen_history.append({
                'gen': gen, 'pareto_size': len(pareto),
                'best_makespan': best_ms, 'best_energy': best_en,
                'best_tardiness': best_td, 'evals': self.eval_count,
            })

            if verbose and (gen % 20 == 0 or gen == self.max_gen - 1):
                elapsed = time.time() - t_start
                print(f"  Gen {gen:>3d}: Pareto={len(pareto):>3d}, "
                      f"best=[{best_ms:.0f}, {best_en:.0f}, {best_td:.0f}], "
                      f"evals={self.eval_count}, {elapsed:.1f}s")

        total_time = time.time() - t_start
        pareto = [ind for ind in pop if ind.rank == 0]
        pf = np.array([ind.objectives for ind in pareto])

        result = {
            "pareto_front": pf.tolist(),
            "pareto_size": len(pareto),
            "best_makespan": pf[np.argmin(pf[:,0])].tolist(),
            "best_energy": pf[np.argmin(pf[:,1])].tolist(),
            "best_tardiness": pf[np.argmin(pf[:,2])].tolist(),
            "total_evals": self.eval_count,
            "total_time": total_time,
            "gen_history": self.gen_history,
        }

        if verbose:
            lb = self.analysis['bottleneck_bounds']['lb_combined']
            print(f"\n{'='*60}")
            print(f"NSGA-II-EBO 完成: {total_time:.1f}s, {self.eval_count} evals")
            print(f"Pareto: {len(pareto)}, best=[{pf[:,0].min():.0f}, {pf[:,1].min():.0f}, {pf[:,2].min():.0f}]")
            print(f"Makespan vs 下界: {pf[:,0].min():.0f} vs {lb:.0f} (gap={(pf[:,0].min()-lb)/lb*100:.1f}%)")

        return result
