"""ALNS 自适应大邻域搜索

框架：
  1. 初始解（贪心构造）
  2. 每次迭代：选破坏算子 → 移除部分任务 → 选修复算子 → 重新插入
  3. 模拟退火接受准则
  4. 自适应权重调整算子选择概率

破坏算子（4个）：
  D1: 随机移除 — 随机选q个任务移除
  D2: 最差移除 — 移除对目标贡献最差的q个任务
  D3: 相关移除 — 移除一组相互关联（同楼层/同电梯）的任务
  D4: 电梯瓶颈移除 — 移除使用最拥挤电梯的任务

修复算子（3个）：
  R1: 贪心插入 — 每次插入代价最小的任务到最佳位置
  R2: 后悔值插入 — 优先插入"后悔值"最大的任务（只有一个好位置的）
  R3: 电梯感知插入 — 插入时同时优化电梯选择

多目标扩展：加权和目标，通过改变权重扫描Pareto前沿

参考: Ropke & Pisinger, 2006; François et al., 2019 (MTVRP)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
import copy
import time

from config import (Robot, Elevator, Task, Priority, PRIORITY_WEIGHT,
                    FLOOR_DISTANCE, INTER_STATION_DISTANCE)
from simulator import Simulator


@dataclass
class ALNSSolution:
    """ALNS解"""
    assignment: Dict[int, int]           # task → robot
    task_sequence: Dict[int, List[int]]  # robot → ordered task list
    elevator_assignment: Dict[int, int]  # task → elevator

    objectives: np.ndarray = field(default_factory=lambda: np.array([np.inf, np.inf, np.inf]))
    weighted_obj: float = np.inf

    def copy(self):
        return ALNSSolution(
            assignment=dict(self.assignment),
            task_sequence={j: list(seq) for j, seq in self.task_sequence.items()},
            elevator_assignment=dict(self.elevator_assignment),
            objectives=self.objectives.copy(),
            weighted_obj=self.weighted_obj,
        )


class ALNSSolver:

    def __init__(self, robots: List[Robot], elevators: List[Elevator], tasks: List[Task],
                 max_iter: int = 5000, segment_size: int = 100,
                 removal_pct: Tuple[float, float] = (0.1, 0.4),
                 sa_start_temp: float = 100.0, sa_cooling: float = 0.9995,
                 weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
                 seed: int = 42):
        self.robots = robots
        self.elevators = elevators
        self.tasks = tasks
        self.N = len(tasks)
        self.M = len(robots)
        self.E = len(elevators)
        self.max_iter = max_iter
        self.segment_size = segment_size
        self.removal_pct = removal_pct
        self.sa_temp = sa_start_temp
        self.sa_cooling = sa_cooling
        self.weights = np.array(weights)
        self.rng = np.random.RandomState(seed)

        self.feasible_robots = {
            i: [j for j in range(self.M) if tasks[i].weight <= robots[j].payload_max]
            for i in range(self.N)
        }
        self.cf_ids = set(i for i, t in enumerate(tasks) if t.is_cross_floor)

        # 算子
        self.destroy_ops = [
            self._destroy_random,
            self._destroy_worst,
            self._destroy_related,
            self._destroy_elevator_bottleneck,
        ]
        self.repair_ops = [
            self._repair_greedy,
            self._repair_regret,
            self._repair_elevator_aware,
        ]
        self.n_destroy = len(self.destroy_ops)
        self.n_repair = len(self.repair_ops)

        # 自适应权重
        self.destroy_weights = np.ones(self.n_destroy)
        self.repair_weights = np.ones(self.n_repair)
        self.destroy_scores = np.zeros(self.n_destroy)
        self.repair_scores = np.zeros(self.n_repair)
        self.destroy_uses = np.zeros(self.n_destroy)
        self.repair_uses = np.zeros(self.n_repair)

        # 奖励参数
        self.SCORE_BEST = 3      # 找到全局最优
        self.SCORE_BETTER = 2    # 改进当前解
        self.SCORE_ACCEPT = 1    # 被接受但未改进
        self.REACTION_FACTOR = 0.1

        self.eval_count = 0
        self.history = []

    def _evaluate(self, sol: ALNSSolution):
        """仿真评估"""
        sim = Simulator(self.robots, self.elevators, self.tasks)
        sim.load_schedule(sol.assignment, sol.task_sequence, sol.elevator_assignment)
        result = sim.run()
        sol.objectives = np.array([
            result['makespan'], result['energy'], result['weighted_tardiness']
        ])
        sol.weighted_obj = float(self.weights @ sol.objectives)
        self.eval_count += 1

    def _removal_count(self) -> int:
        lo = max(1, int(self.N * self.removal_pct[0]))
        hi = max(lo + 1, int(self.N * self.removal_pct[1]))
        return self.rng.randint(lo, hi)

    # ==================== 破坏算子 ====================

    def _destroy_random(self, sol: ALNSSolution) -> Set[int]:
        q = self._removal_count()
        return set(self.rng.choice(self.N, q, replace=False))

    def _destroy_worst(self, sol: ALNSSolution) -> Set[int]:
        """移除对目标贡献最差的任务"""
        q = self._removal_count()
        costs = []
        for i in range(self.N):
            j = sol.assignment[i]
            task = self.tasks[i]
            dist = FLOOR_DISTANCE * 2 if task.is_cross_floor else INTER_STATION_DISTANCE
            energy_cost = dist * self.robots[j].energy_per_m
            # 简化评估：从序列中移除i后的改进估算
            seq = sol.task_sequence[j]
            idx = seq.index(i) if i in seq else -1
            position_cost = 0
            if idx > 0:
                prev = self.tasks[seq[idx - 1]]
                position_cost += abs(prev.dest_floor - task.origin_floor) * 5
            if idx < len(seq) - 1 and idx >= 0:
                nxt = self.tasks[seq[idx + 1]]
                position_cost += abs(task.dest_floor - nxt.origin_floor) * 5
            costs.append((i, energy_cost + position_cost + self.rng.random() * 10))

        costs.sort(key=lambda x: -x[1])
        return set(c[0] for c in costs[:q])

    def _destroy_related(self, sol: ALNSSolution) -> Set[int]:
        """移除一组相关联的任务（同起点楼层或同终点楼层）"""
        q = self._removal_count()
        seed_task = self.rng.randint(self.N)
        seed_floor = self.tasks[seed_task].origin_floor

        relatedness = []
        for i in range(self.N):
            if i == seed_task:
                continue
            t = self.tasks[i]
            score = (10 - abs(t.origin_floor - seed_floor) +
                     10 - abs(t.dest_floor - self.tasks[seed_task].dest_floor) +
                     self.rng.random() * 3)
            relatedness.append((i, score))

        relatedness.sort(key=lambda x: -x[1])
        removed = {seed_task}
        for i, _ in relatedness:
            if len(removed) >= q:
                break
            removed.add(i)
        return removed

    def _destroy_elevator_bottleneck(self, sol: ALNSSolution) -> Set[int]:
        """移除使用最拥挤电梯的任务"""
        q = self._removal_count()
        elev_usage = {e: [] for e in range(self.E)}
        for i in self.cf_ids:
            e = sol.elevator_assignment.get(i, 0)
            elev_usage[e].append(i)

        busiest = max(elev_usage.keys(), key=lambda e: len(elev_usage[e]))
        candidates = elev_usage[busiest]
        if len(candidates) <= q:
            removed = set(candidates)
        else:
            removed = set(self.rng.choice(candidates, q, replace=False))

        # 补足q个
        while len(removed) < q:
            i = self.rng.randint(self.N)
            removed.add(i)
        return removed

    # ==================== 修复算子 ====================

    def _insert_cost(self, sol: ALNSSolution, task_id: int,
                     robot_id: int, position: int, elev_id: int) -> float:
        """估算将任务插入指定位置的代价"""
        task = self.tasks[task_id]
        robot = self.robots[robot_id]
        seq = sol.task_sequence[robot_id]

        if task.weight > robot.payload_max:
            return float('inf')

        cost = 0.0
        # 距离代价
        if position == 0:
            cost += abs(robot.init_floor - task.origin_floor) * 5
        else:
            prev = self.tasks[seq[position - 1]]
            cost += abs(prev.dest_floor - task.origin_floor) * 5

        if position < len(seq):
            nxt = self.tasks[seq[position]]
            cost += abs(task.dest_floor - nxt.origin_floor) * 5
            # 移除原来的直接连接代价
            if position > 0:
                prev = self.tasks[seq[position - 1]]
                cost -= abs(prev.dest_floor - nxt.origin_floor) * 5

        # 能耗代价
        dist = FLOOR_DISTANCE * 2 if task.is_cross_floor else INTER_STATION_DISTANCE
        cost += dist * robot.energy_per_m * self.weights[1]

        return cost

    def _find_best_insertion(self, sol: ALNSSolution, task_id: int) -> Tuple[int, int, int, float]:
        """找到任务的最佳插入位置: (robot, position, elevator, cost)"""
        best = (0, 0, 0, float('inf'))
        for j in self.feasible_robots[task_id]:
            seq = sol.task_sequence[j]
            for pos in range(len(seq) + 1):
                for e in range(self.E) if task_id in self.cf_ids else [0]:
                    cost = self._insert_cost(sol, task_id, j, pos, e)
                    if cost < best[3]:
                        best = (j, pos, e, cost)
        return best

    def _repair_greedy(self, sol: ALNSSolution, removed: Set[int]):
        """贪心插入：每次插入代价最小的任务"""
        remaining = list(removed)
        self.rng.shuffle(remaining)

        for task_id in remaining:
            j, pos, e, cost = self._find_best_insertion(sol, task_id)
            sol.assignment[task_id] = j
            sol.task_sequence[j].insert(pos, task_id)
            if task_id in self.cf_ids:
                sol.elevator_assignment[task_id] = e

    def _repair_regret(self, sol: ALNSSolution, removed: Set[int]):
        """后悔值插入：优先插入只有一个好位置的任务（regret-2）"""
        remaining = list(removed)

        while remaining:
            best_regret = -float('inf')
            best_task = None
            best_insert = None

            for task_id in remaining:
                insertions = []
                for j in self.feasible_robots[task_id]:
                    seq = sol.task_sequence[j]
                    for pos in range(len(seq) + 1):
                        for e in range(self.E) if task_id in self.cf_ids else [0]:
                            cost = self._insert_cost(sol, task_id, j, pos, e)
                            insertions.append((j, pos, e, cost))

                insertions.sort(key=lambda x: x[3])
                if not insertions:
                    continue

                best_cost = insertions[0][3]
                second_cost = insertions[1][3] if len(insertions) > 1 else best_cost * 2
                regret = second_cost - best_cost

                if regret > best_regret:
                    best_regret = regret
                    best_task = task_id
                    best_insert = insertions[0]

            if best_task is not None and best_insert is not None:
                j, pos, e, _ = best_insert
                sol.assignment[best_task] = j
                sol.task_sequence[j].insert(pos, best_task)
                if best_task in self.cf_ids:
                    sol.elevator_assignment[best_task] = e
                remaining.remove(best_task)
            else:
                # fallback: 随机插入剩余任务
                for tid in remaining:
                    j = self.feasible_robots[tid][0]
                    sol.assignment[tid] = j
                    sol.task_sequence[j].append(tid)
                    if tid in self.cf_ids:
                        sol.elevator_assignment[tid] = self.rng.randint(self.E)
                break

    def _repair_elevator_aware(self, sol: ALNSSolution, removed: Set[int]):
        """电梯感知插入：插入时平衡电梯负载"""
        # 统计当前各电梯使用量
        elev_load = {e: 0 for e in range(self.E)}
        for i in self.cf_ids:
            if i not in removed and i in sol.elevator_assignment:
                elev_load[sol.elevator_assignment[i]] += 1

        remaining = list(removed)
        self.rng.shuffle(remaining)

        for task_id in remaining:
            best = (0, 0, 0, float('inf'))
            for j in self.feasible_robots[task_id]:
                seq = sol.task_sequence[j]
                for pos in range(len(seq) + 1):
                    if task_id in self.cf_ids:
                        # 选负载最轻的电梯
                        e = min(range(self.E), key=lambda e: elev_load[e])
                    else:
                        e = 0
                    cost = self._insert_cost(sol, task_id, j, pos, e)
                    if task_id in self.cf_ids:
                        cost += elev_load[e] * 2  # 负载惩罚
                    if cost < best[3]:
                        best = (j, pos, e, cost)

            j, pos, e, _ = best
            sol.assignment[task_id] = j
            sol.task_sequence[j].insert(pos, task_id)
            if task_id in self.cf_ids:
                sol.elevator_assignment[task_id] = e
                elev_load[e] += 1

    # ==================== 核心逻辑 ====================

    def _select_operator(self, weights: np.ndarray) -> int:
        probs = weights / weights.sum()
        return self.rng.choice(len(weights), p=probs)

    def _update_weights(self):
        for i in range(self.n_destroy):
            if self.destroy_uses[i] > 0:
                self.destroy_weights[i] = (
                    self.destroy_weights[i] * (1 - self.REACTION_FACTOR) +
                    self.REACTION_FACTOR * self.destroy_scores[i] / self.destroy_uses[i]
                )
        for i in range(self.n_repair):
            if self.repair_uses[i] > 0:
                self.repair_weights[i] = (
                    self.repair_weights[i] * (1 - self.REACTION_FACTOR) +
                    self.REACTION_FACTOR * self.repair_scores[i] / self.repair_uses[i]
                )
        # 最小权重
        self.destroy_weights = np.maximum(self.destroy_weights, 0.1)
        self.repair_weights = np.maximum(self.repair_weights, 0.1)
        # 重置
        self.destroy_scores[:] = 0
        self.repair_scores[:] = 0
        self.destroy_uses[:] = 0
        self.repair_uses[:] = 0

    def _construct_initial(self) -> ALNSSolution:
        """贪心构造初始解"""
        assignment = {}
        task_sequence = {j: [] for j in range(self.M)}
        elevator_assignment = {}
        robot_floor = {j: self.robots[j].init_floor for j in range(self.M)}
        robot_load = {j: 0 for j in range(self.M)}

        order = sorted(range(self.N), key=lambda i: (self.tasks[i].priority, self.tasks[i].deadline))

        for i in order:
            task = self.tasks[i]
            best_j = None
            best_cost = float('inf')
            for j in self.feasible_robots[i]:
                dist = abs(robot_floor[j] - task.origin_floor)
                cost = dist * 5 + robot_load[j] * 20
                if cost < best_cost:
                    best_cost = cost
                    best_j = j

            assignment[i] = best_j
            task_sequence[best_j].append(i)
            robot_floor[best_j] = task.dest_floor
            robot_load[best_j] += 1

            if i in self.cf_ids:
                elevator_assignment[i] = self.rng.randint(self.E)

        sol = ALNSSolution(assignment, task_sequence, elevator_assignment)
        self._evaluate(sol)
        return sol

    def _remove_tasks(self, sol: ALNSSolution, removed: Set[int]):
        """从解中移除一组任务"""
        for i in removed:
            j = sol.assignment[i]
            if i in sol.task_sequence[j]:
                sol.task_sequence[j].remove(i)
            del sol.assignment[i]
            if i in sol.elevator_assignment:
                del sol.elevator_assignment[i]

    def solve(self, verbose: bool = True) -> Dict:
        """运行ALNS"""
        t_start = time.time()

        current = self._construct_initial()
        best = current.copy()
        best.objectives = current.objectives.copy()
        best.weighted_obj = current.weighted_obj

        if verbose:
            print(f"ALNS: iter={self.max_iter}, N={self.N}, M={self.M}, E={self.E}")
            print(f"  初始解: [{current.objectives[0]:.0f}, {current.objectives[1]:.0f}, "
                  f"{current.objectives[2]:.0f}] = {current.weighted_obj:.1f}")

        # 收集所有访问过的非支配解（用于Pareto前沿）
        archive: List[np.ndarray] = [best.objectives.copy()]

        for it in range(self.max_iter):
            candidate = current.copy()

            # 选算子
            d_idx = self._select_operator(self.destroy_weights)
            r_idx = self._select_operator(self.repair_weights)

            # 破坏
            removed = self.destroy_ops[d_idx](candidate)
            self._remove_tasks(candidate, removed)

            # 修复
            self.repair_ops[r_idx](candidate, removed)

            # 评估
            self._evaluate(candidate)

            # 接受准则（模拟退火）
            score = 0
            delta = candidate.weighted_obj - current.weighted_obj
            accepted = False

            if candidate.weighted_obj < best.weighted_obj:
                best = candidate.copy()
                best.objectives = candidate.objectives.copy()
                best.weighted_obj = candidate.weighted_obj
                current = candidate
                score = self.SCORE_BEST
                accepted = True
            elif delta < 0:
                current = candidate
                score = self.SCORE_BETTER
                accepted = True
            elif self.rng.random() < np.exp(-delta / max(self.sa_temp, 0.01)):
                current = candidate
                score = self.SCORE_ACCEPT
                accepted = True

            # 更新archive
            if accepted:
                dominated = False
                new_archive = []
                for obj in archive:
                    if np.all(candidate.objectives <= obj) and np.any(candidate.objectives < obj):
                        continue  # candidate支配obj
                    if np.all(obj <= candidate.objectives) and np.any(obj < candidate.objectives):
                        dominated = True
                    new_archive.append(obj)
                if not dominated:
                    new_archive.append(candidate.objectives.copy())
                archive = new_archive

            # 更新算子分数
            self.destroy_scores[d_idx] += score
            self.repair_scores[r_idx] += score
            self.destroy_uses[d_idx] += 1
            self.repair_uses[r_idx] += 1

            # 降温
            self.sa_temp *= self.sa_cooling

            # 段结束更新权重
            if (it + 1) % self.segment_size == 0:
                self._update_weights()

            # 日志
            if verbose and (it % 500 == 0 or it == self.max_iter - 1):
                elapsed = time.time() - t_start
                print(f"  Iter {it:>5d}: best=[{best.objectives[0]:.0f}, "
                      f"{best.objectives[1]:.0f}, {best.objectives[2]:.0f}], "
                      f"current={current.weighted_obj:.1f}, temp={self.sa_temp:.2f}, "
                      f"archive={len(archive)}, {elapsed:.1f}s")

            self.history.append({
                'iter': it,
                'best_obj': best.weighted_obj,
                'current_obj': current.weighted_obj,
                'best_objs': best.objectives.tolist(),
                'archive_size': len(archive),
                'temp': self.sa_temp,
            })

        total_time = time.time() - t_start

        pareto_front = np.array(archive)
        result = {
            "best_weighted": best.weighted_obj,
            "best_objectives": best.objectives.tolist(),
            "best_solution": best,
            "pareto_front": pareto_front.tolist() if len(pareto_front) > 0 else [],
            "pareto_size": len(archive),
            "total_evals": self.eval_count,
            "total_time": total_time,
            "history": self.history,
            "destroy_weights": self.destroy_weights.tolist(),
            "repair_weights": self.repair_weights.tolist(),
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"ALNS 完成: {total_time:.1f}s, {self.eval_count} evaluations")
            print(f"最优加权目标: {best.weighted_obj:.1f}")
            print(f"最优目标: [{best.objectives[0]:.0f}, {best.objectives[1]:.0f}, {best.objectives[2]:.0f}]")
            print(f"Pareto archive: {len(archive)} 个解")
            print(f"算子权重:")
            names_d = ["Random", "Worst", "Related", "ElevBottle"]
            names_r = ["Greedy", "Regret", "ElevAware"]
            print(f"  破坏: {dict(zip(names_d, [f'{w:.2f}' for w in self.destroy_weights]))}")
            print(f"  修复: {dict(zip(names_r, [f'{w:.2f}' for w in self.repair_weights]))}")

        return result


def alns_pareto_sweep(robots, elevators, tasks, n_weights: int = 11,
                      max_iter: int = 3000, seed: int = 42, verbose: bool = True):
    """通过改变权重扫描Pareto前沿"""
    all_solutions = []

    weight_sets = []
    for a1 in np.linspace(0.2, 0.8, n_weights):
        for a3 in np.linspace(0.0, 0.6, max(1, n_weights // 2)):
            a2 = 1.0 - a1 - a3
            if a2 >= 0.05:
                weight_sets.append((a1, a2, a3))

    if verbose:
        print(f"Pareto sweep: {len(weight_sets)} 组权重")

    for idx, w in enumerate(weight_sets):
        solver = ALNSSolver(robots, elevators, tasks,
                            max_iter=max_iter, weights=w, seed=seed + idx)
        result = solver.solve(verbose=False)
        all_solutions.append(result['best_objectives'])
        if verbose and idx % 5 == 0:
            print(f"  [{idx+1}/{len(weight_sets)}] w={w}, obj={result['best_objectives']}")

    # 提取非支配集
    solutions = np.array(all_solutions)
    pareto = []
    for i in range(len(solutions)):
        dominated = False
        for j in range(len(solutions)):
            if i == j:
                continue
            if (np.all(solutions[j] <= solutions[i]) and
                    np.any(solutions[j] < solutions[i])):
                dominated = True
                break
        if not dominated:
            pareto.append(solutions[i])

    return np.array(pareto)
