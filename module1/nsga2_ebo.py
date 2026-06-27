"""NSGA-II-EBO v2: 基于电梯群控系统的混合优化算法

v2改进（相比v1）：
  1. 使用SimulatorV2（电梯群控系统），不再由算法选电梯
  2. 移除gene_elev — 搜索空间从 O(M^N × E^N) 降为 O(M^N)
  3. 新增gene_stagger — 每台机器人的错峰延迟参数
  4. 下层优化器增加批量取货感知
  5. 能耗包含电梯群控系统的能耗
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import time

from config import (Robot, Elevator, Task, Priority, PRIORITY_WEIGHT,
                    FLOOR_DISTANCE, INTER_STATION_DISTANCE)
from problem_analysis import ProblemAnalyzer
from ebdco import LowerLevelSolver
from simulator_v2 import SimulatorV2, RobotStrategy
from elevator_group_control import DispatchAlgorithm


@dataclass
class EBOIndividual:
    gene_assign: np.ndarray     # (N,): task → robot
    gene_stagger: float = 0.0   # 全局错峰延迟参数
    objectives: np.ndarray = field(default_factory=lambda: np.array([np.inf, np.inf, np.inf]))
    rank: int = 0
    crowding_dist: float = 0.0

    def copy(self):
        return EBOIndividual(
            gene_assign=self.gene_assign.copy(),
            gene_stagger=self.gene_stagger,
        )


class NSGA2EBOSolver:

    def __init__(self, robots: List[Robot], elevators: List[Elevator], tasks: List[Task],
                 pop_size: int = 100, max_gen: int = 200,
                 crossover_rate: float = 0.9, mutation_rate: float = 0.4,
                 dispatch_algorithm: DispatchAlgorithm = DispatchAlgorithm.ETA,
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
        self.dispatch_algorithm = dispatch_algorithm
        self.rng = np.random.RandomState(seed)

        self.analyzer = ProblemAnalyzer(robots, elevators, tasks)
        self.analysis = self.analyzer.full_analysis(verbose=False)
        self.conflict_matrix, _ = self.analyzer.build_conflict_graph()
        self.criticality = self.analysis['critical_tasks']['criticality_scores']
        self.critical_tasks = self.analysis['critical_tasks']['critical_order']

        self.lower_solver = LowerLevelSolver(robots, elevators, tasks)
        self.cf_ids = set(i for i, t in enumerate(tasks) if t.is_cross_floor)

        self.eval_count = 0
        self.gen_history = []

    def _init_critical_first(self) -> EBOIndividual:
        gene_assign = np.zeros(self.N, dtype=int)
        robot_load = np.zeros(self.M)
        assigned = set()
        for i in self.critical_tasks:
            feasible = self.analyzer.feasible_robots[i]
            if not feasible:
                assigned.add(i)
                continue
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
        return EBOIndividual(gene_assign=gene_assign,
                             gene_stagger=self.rng.uniform(0, 5))

    def _init_floor_cluster(self) -> EBOIndividual:
        gene_assign = np.zeros(self.N, dtype=int)
        floor_tasks = {}
        for i in range(self.N):
            floor_tasks.setdefault(self.tasks[i].origin_floor, []).append(i)
        robot_idx = 0
        for f in sorted(floor_tasks):
            for i in floor_tasks[f]:
                feasible = self.analyzer.feasible_robots[i]
                if not feasible:
                    continue
                for offset in range(self.M):
                    j = (robot_idx + offset) % self.M
                    if j in feasible:
                        gene_assign[i] = j
                        break
            robot_idx = (robot_idx + 1) % self.M
        return EBOIndividual(gene_assign=gene_assign,
                             gene_stagger=self.rng.uniform(0, 5))

    def _init_random(self) -> EBOIndividual:
        gene_assign = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            feasible = self.analyzer.feasible_robots[i]
            if feasible:
                gene_assign[i] = feasible[self.rng.randint(len(feasible))]
        return EBOIndividual(gene_assign=gene_assign,
                             gene_stagger=self.rng.uniform(0, 10))

    def _evaluate(self, ind: EBOIndividual):
        """下层结构优化 → V2群控仿真"""
        assignment = {i: int(ind.gene_assign[i]) for i in range(self.N)}

        # 下层优化：多策略序列，取快速估算最优的
        best_seq, best_cost = None, np.inf
        for w in [np.array([0.7, 0.2, 0.1]),
                  np.array([0.2, 0.6, 0.2]),
                  np.array([0.1, 0.2, 0.7])]:
            seq = self.lower_solver.optimize_sequence(assignment, {}, weight=w)
            cost = self._quick_estimate(assignment, seq)
            if cost < best_cost:
                best_cost = cost
                best_seq = seq

        # V2仿真（群控系统决定电梯）
        strategy = RobotStrategy(stagger_delay=ind.gene_stagger)
        sim = SimulatorV2(self.robots, self.elevators, self.tasks,
                          dispatch_algorithm=self.dispatch_algorithm,
                          dt=0.5, strategy=strategy)
        sim.load_schedule(assignment, best_seq)
        result = sim.run(max_time=10000)

        ind.objectives = np.array([
            result['makespan'],
            result['energy'],
            result['weighted_tardiness'],
        ])
        self.eval_count += 1

    def _quick_estimate(self, assignment, task_sequence) -> float:
        robot_time = np.zeros(self.M)
        total_energy = 0.0
        for j in range(self.M):
            current_floor = self.robots[j].init_floor
            t = 0.0
            for i in task_sequence.get(j, []):
                task = self.tasks[i]
                if current_floor != task.origin_floor:
                    t += abs(current_floor - task.origin_floor) * 3.0 + 25.0
                t += 10.0
                if task.is_cross_floor:
                    t += abs(task.origin_floor - task.dest_floor) / 0.5 + 25.0
                t += 10.0
                current_floor = task.dest_floor
                total_energy += (abs(task.origin_floor - task.dest_floor) * 3.0 + 30.0) * self.robots[j].energy_per_m
            robot_time[j] = t
        return robot_time.max() * 0.5 + total_energy * 0.3

    def _crossover(self, p1: EBOIndividual, p2: EBOIndividual) -> Tuple[EBOIndividual, EBOIndividual]:
        c1, c2 = p1.copy(), p2.copy()
        if self.rng.random() > self.crossover_rate:
            return c1, c2
        for i in range(self.N):
            if self.criticality[i] > 0.35:
                if np.sum(p2.objectives) < np.sum(p1.objectives):
                    c1.gene_assign[i] = p2.gene_assign[i]
                if np.sum(p1.objectives) < np.sum(p2.objectives):
                    c2.gene_assign[i] = p1.gene_assign[i]
            else:
                if self.rng.random() < 0.5:
                    c1.gene_assign[i] = p2.gene_assign[i]
                if self.rng.random() < 0.5:
                    c2.gene_assign[i] = p1.gene_assign[i]
        # 策略参数混合
        alpha = self.rng.random()
        c1.gene_stagger = alpha * p1.gene_stagger + (1 - alpha) * p2.gene_stagger
        c2.gene_stagger = (1 - alpha) * p1.gene_stagger + alpha * p2.gene_stagger
        self._repair(c1)
        self._repair(c2)
        return c1, c2

    def _mutate(self, ind: EBOIndividual):
        if self.rng.random() > self.mutation_rate:
            return
        strategy = self.rng.randint(4)
        if strategy == 0:
            critical = self.critical_tasks[:max(5, self.N // 5)]
            i = critical[self.rng.randint(len(critical))]
            feasible = self.analyzer.feasible_robots[i]
            if feasible:
                ind.gene_assign[i] = feasible[self.rng.randint(len(feasible))]
        elif strategy == 1:
            load = np.bincount(ind.gene_assign, minlength=self.M)
            busiest, lightest = int(np.argmax(load)), int(np.argmin(load))
            movable = [i for i in range(self.N)
                       if ind.gene_assign[i] == busiest
                       and lightest in self.analyzer.feasible_robots[i]]
            if movable:
                ind.gene_assign[movable[self.rng.randint(len(movable))]] = lightest
        elif strategy == 2:
            ind.gene_stagger = np.clip(ind.gene_stagger + self.rng.normal(0, 2), 0, 15)
        else:
            i = self.rng.randint(self.N)
            feasible = self.analyzer.feasible_robots[i]
            if feasible:
                ind.gene_assign[i] = feasible[self.rng.randint(len(feasible))]

    def _repair(self, ind: EBOIndividual):
        for i in range(self.N):
            j = int(ind.gene_assign[i])
            feasible = self.analyzer.feasible_robots[i]
            if feasible and j not in feasible:
                ind.gene_assign[i] = feasible[0]

    def _fast_nondominated_sort(self, pop):
        n = len(pop)
        dom_count = [0] * n
        dom_set = [[] for _ in range(n)]
        fronts = [[]]
        for p in range(n):
            for q in range(n):
                if p == q: continue
                if np.all(pop[p].objectives <= pop[q].objectives) and np.any(pop[p].objectives < pop[q].objectives):
                    dom_set[p].append(q)
                elif np.all(pop[q].objectives <= pop[p].objectives) and np.any(pop[q].objectives < pop[p].objectives):
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

    def solve(self, verbose=True):
        t_start = time.time()
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
            print(f"NSGA-II-EBO v2: pop={self.pop_size}, gen={self.max_gen}, "
                  f"N={self.N}, M={self.M}, E={self.E}, dispatch={self.dispatch_algorithm.value}")
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
            print(f"\nNSGA-II-EBO v2 完成: {total_time:.1f}s, {self.eval_count} evals")
            print(f"Pareto: {len(pareto)}, best=[{pf[:,0].min():.0f}, {pf[:,1].min():.0f}, {pf[:,2].min():.0f}]")
            print(f"Makespan vs 下界: {pf[:,0].min():.0f} vs {lb:.0f} (gap={(pf[:,0].min()-lb)/lb*100:.1f}%)")

        return result
