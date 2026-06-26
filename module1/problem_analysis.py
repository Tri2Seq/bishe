"""问题数学特征分析

本模块深度剖析多机器人多楼层配送调度问题的数学结构，
为设计针对性算法提供理论基础。

分析内容：
  1. 电梯瓶颈下界 — 基于电梯总服务时间的makespan下界
  2. 任务冲突图 — 空间竞争+电梯竞争+资源竞争
  3. 机器人可行域分析 — 载重约束导致的分配限制
  4. 楼层流量矩阵 — 跨楼层需求的方向性与平衡性
  5. 关键任务识别 — 对目标影响最大的任务子集
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from config import Robot, Elevator, Task, Priority, FLOOR_DISTANCE, INTER_STATION_DISTANCE


class ProblemAnalyzer:
    """问题结构分析器"""

    def __init__(self, robots: List[Robot], elevators: List[Elevator], tasks: List[Task]):
        self.robots = robots
        self.elevators = elevators
        self.tasks = tasks
        self.N = len(tasks)
        self.M = len(robots)
        self.E = len(elevators)
        self.F = max(max(t.origin_floor, t.dest_floor) for t in tasks)

        self.cf_ids = [i for i, t in enumerate(tasks) if t.is_cross_floor]
        self.sf_ids = [i for i, t in enumerate(tasks) if not t.is_cross_floor]

        # 预计算
        self._compute_feasible_robots()
        self._compute_floor_flow()
        self._compute_task_properties()

    def _compute_feasible_robots(self):
        """计算每个任务的可行机器人集合"""
        self.feasible_robots: Dict[int, List[int]] = {}
        self.robot_feasible_tasks: Dict[int, List[int]] = {}

        for i in range(self.N):
            self.feasible_robots[i] = [
                j for j in range(self.M)
                if self.tasks[i].weight <= self.robots[j].payload_max
            ]

        for j in range(self.M):
            self.robot_feasible_tasks[j] = [
                i for i in range(self.N)
                if j in self.feasible_robots[i]
            ]

    def _compute_floor_flow(self):
        """计算楼层间流量矩阵"""
        self.flow_matrix = np.zeros((self.F + 1, self.F + 1))
        for t in self.tasks:
            if t.is_cross_floor:
                self.flow_matrix[t.origin_floor, t.dest_floor] += 1

        self.floor_demand = np.zeros(self.F + 1)
        for t in self.tasks:
            self.floor_demand[t.origin_floor] += 1

    def _compute_task_properties(self):
        """预计算任务属性"""
        self.task_elev_time = {}  # task → 最快电梯行程时间
        for i in self.cf_ids:
            t = self.tasks[i]
            self.task_elev_time[i] = min(
                e.travel_time(t.origin_floor, t.dest_floor)
                for e in self.elevators
            )

    # ===================================================================
    # 特征1：电梯瓶颈下界
    # ===================================================================

    def elevator_bottleneck_bound(self) -> Dict:
        """
        计算基于电梯瓶颈的makespan下界。

        定理：对于任意可行调度方案，makespan满足：
          C_max ≥ max_e { Σ_{i∈S_e} [τ_e(i) + δ_repo(i)] / Q_e }

        其中 S_e 是使用电梯e的任务集，τ_e(i)是服务时间，
        δ_repo是重定位时间，Q_e是容量。

        更紧的界考虑电梯交互开销（approach+press+exit）。
        """
        approach_press_exit = 8.0 + 7.0 + 5.0  # GPU交互总开销

        # 每部电梯的最小总服务时间（假设所有跨楼层任务均匀分配）
        total_elev_service = sum(
            self.task_elev_time[i] + approach_press_exit
            for i in self.cf_ids
        )

        # 电梯总容量
        total_capacity = sum(e.capacity for e in self.elevators)

        # 下界1：电梯瓶颈（均匀分配假设）
        lb_elevator = total_elev_service / max(total_capacity, 1)

        # 下界2：最慢机器人瓶颈（所有任务分配给最少机器人）
        avg_tasks_per_robot = self.N / self.M
        avg_task_time = np.mean([
            self.task_elev_time.get(i, 0) + approach_press_exit + 20.0
            if self.tasks[i].is_cross_floor
            else INTER_STATION_DISTANCE / max(r.speed for r in self.robots) + 20.0
            for i in range(self.N)
        ])
        lb_robot = avg_tasks_per_robot * avg_task_time

        # 下界3：最紧截止时间的逆推
        min_deadline = min(t.deadline for t in self.tasks)

        # 综合下界
        lb = max(lb_elevator, lb_robot)

        # 电梯负载均衡分析
        per_elev_min_load = total_elev_service / self.E
        elev_balance_ratio = per_elev_min_load / max(total_elev_service, 1)

        return {
            "lb_elevator": lb_elevator,
            "lb_robot": lb_robot,
            "lb_combined": lb,
            "total_cf_tasks": len(self.cf_ids),
            "total_elev_service_time": total_elev_service,
            "avg_elev_load_per_elevator": per_elev_min_load,
            "cross_floor_ratio": len(self.cf_ids) / self.N,
        }

    # ===================================================================
    # 特征2：任务冲突图
    # ===================================================================

    def build_conflict_graph(self) -> Tuple[np.ndarray, Dict]:
        """
        构建任务冲突强度矩阵。

        冲突定义：两个任务i和j之间的冲突度量化为它们对调度方案的竞争程度。

        冲突源：
          (1) 楼层竞争 — 同一起点楼层的任务竞争电梯资源
          (2) 电梯竞争 — 跨楼层方向相近的任务竞争电梯时段
          (3) 机器人竞争 — 只有少数机器人能执行的任务互相竞争
          (4) 时间窗竞争 — 截止时间相近的紧急任务

        冲突强度 ∈ [0, 1]，越高表示越不应该分配给同一机器人或同一时段。
        """
        conflict = np.zeros((self.N, self.N))

        for i in range(self.N):
            for j in range(i + 1, self.N):
                ti, tj = self.tasks[i], self.tasks[j]
                score = 0.0

                # (1) 楼层竞争：同起点楼层
                if ti.origin_floor == tj.origin_floor:
                    score += 0.3
                # 起点楼层相近（1层内）
                elif abs(ti.origin_floor - tj.origin_floor) <= 1:
                    score += 0.15

                # (2) 电梯竞争：都是跨楼层且方向相近
                if ti.is_cross_floor and tj.is_cross_floor:
                    # 方向相同（都上行或都下行）
                    dir_i = np.sign(ti.dest_floor - ti.origin_floor)
                    dir_j = np.sign(tj.dest_floor - tj.origin_floor)
                    if dir_i == dir_j:
                        score += 0.2
                    # 路径重叠度
                    overlap = max(0, min(ti.dest_floor, tj.dest_floor) -
                                  max(ti.origin_floor, tj.origin_floor))
                    score += overlap * 0.05

                # (3) 机器人竞争：可行机器人集合的重叠度
                fi = set(self.feasible_robots[i])
                fj = set(self.feasible_robots[j])
                if len(fi) > 0 and len(fj) > 0:
                    overlap_ratio = len(fi & fj) / len(fi | fj)
                    # 如果两个任务只能被少数相同的机器人执行，竞争激烈
                    scarcity = 1.0 - min(len(fi), len(fj)) / self.M
                    score += scarcity * overlap_ratio * 0.2

                # (4) 时间窗竞争
                deadline_diff = abs(ti.deadline - tj.deadline)
                max_deadline = max(t.deadline for t in self.tasks)
                time_proximity = 1.0 - deadline_diff / max(max_deadline, 1)
                if ti.priority == Priority.URGENT and tj.priority == Priority.URGENT:
                    score += time_proximity * 0.3

                conflict[i, j] = min(score, 1.0)
                conflict[j, i] = conflict[i, j]

        # 聚类分析：高冲突任务组
        clusters = self._extract_conflict_clusters(conflict, threshold=0.4)

        return conflict, {
            "avg_conflict": np.mean(conflict[conflict > 0]),
            "max_conflict": np.max(conflict),
            "high_conflict_pairs": int(np.sum(conflict > 0.5)),
            "conflict_clusters": clusters,
        }

    def _extract_conflict_clusters(self, conflict: np.ndarray,
                                    threshold: float = 0.4) -> List[Set[int]]:
        """提取高冲突任务簇（连通分量）"""
        visited = set()
        clusters = []

        for i in range(self.N):
            if i in visited:
                continue
            cluster = set()
            stack = [i]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                cluster.add(node)
                for j in range(self.N):
                    if j not in visited and conflict[node, j] > threshold:
                        stack.append(j)
            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters

    # ===================================================================
    # 特征3：楼层流量分析
    # ===================================================================

    def floor_flow_analysis(self) -> Dict:
        """
        分析楼层间的流量模式。

        关键指标：
          - 流量方向性：上行 vs 下行比例
          - 流量集中度：是否集中在少数楼层对
          - 热点楼层：哪些楼层是主要的取货/送货点
        """
        upward = sum(1 for t in self.tasks if t.is_cross_floor and t.dest_floor > t.origin_floor)
        downward = sum(1 for t in self.tasks if t.is_cross_floor and t.dest_floor < t.origin_floor)

        # 热点楼层
        origin_counts = defaultdict(int)
        dest_counts = defaultdict(int)
        for t in self.tasks:
            origin_counts[t.origin_floor] += 1
            dest_counts[t.dest_floor] += 1

        hotspot_origin = sorted(origin_counts.items(), key=lambda x: -x[1])[:3]
        hotspot_dest = sorted(dest_counts.items(), key=lambda x: -x[1])[:3]

        # 流量集中度（基尼系数的简化版）
        if len(self.cf_ids) > 0:
            flow_vals = self.flow_matrix[self.flow_matrix > 0]
            if len(flow_vals) > 0:
                sorted_flow = np.sort(flow_vals)
                n = len(sorted_flow)
                cumsum = np.cumsum(sorted_flow)
                gini = 1 - 2 * np.sum(cumsum) / (n * cumsum[-1]) + 1 / n
            else:
                gini = 0
        else:
            gini = 0

        return {
            "upward_tasks": upward,
            "downward_tasks": downward,
            "direction_ratio": upward / max(downward, 1),
            "hotspot_origins": hotspot_origin,
            "hotspot_dests": hotspot_dest,
            "flow_concentration_gini": gini,
        }

    # ===================================================================
    # 特征4：关键任务识别
    # ===================================================================

    def identify_critical_tasks(self) -> Dict:
        """
        识别对调度质量影响最大的"关键任务"。

        关键任务特征：
          (1) 重载任务 — 只有少数机器人能执行
          (2) 紧急任务 — 截止时间紧迫
          (3) 长途任务 — 占用电梯时间长
          (4) 瓶颈任务 — 同时满足以上多个条件
        """
        criticality = np.zeros(self.N)

        for i in range(self.N):
            t = self.tasks[i]
            # 机器人稀缺性
            robot_scarcity = 1.0 - len(self.feasible_robots[i]) / self.M
            # 时间紧迫性
            if t.is_cross_floor:
                min_exec = self.task_elev_time[i] + 20.0 + 20.0  # 电梯+取送+交互
            else:
                min_exec = INTER_STATION_DISTANCE / max(r.speed for r in self.robots) + 20.0
            time_pressure = min_exec / max(t.deadline, 1)
            # 电梯占用度
            elev_occupation = self.task_elev_time.get(i, 0) / max(
                max(self.task_elev_time.values()) if self.task_elev_time else 1, 1)
            # 优先级权重
            priority_weight = {Priority.URGENT: 1.0, Priority.NORMAL: 0.5, Priority.BATCH: 0.2}

            criticality[i] = (
                0.3 * robot_scarcity +
                0.3 * time_pressure +
                0.2 * elev_occupation +
                0.2 * priority_weight.get(t.priority, 0.5)
            )

        # 排序
        critical_order = np.argsort(-criticality)
        top_critical = critical_order[:max(3, self.N // 5)]

        return {
            "criticality_scores": criticality,
            "critical_order": critical_order.tolist(),
            "top_critical_tasks": top_critical.tolist(),
            "avg_criticality": float(np.mean(criticality)),
            "max_criticality": float(np.max(criticality)),
        }

    # ===================================================================
    # 综合分析报告
    # ===================================================================

    def full_analysis(self, verbose: bool = True) -> Dict:
        """完整问题分析"""
        bounds = self.elevator_bottleneck_bound()
        conflict_matrix, conflict_info = self.build_conflict_graph()
        flow = self.floor_flow_analysis()
        critical = self.identify_critical_tasks()

        report = {
            "problem_size": {"N": self.N, "M": self.M, "E": self.E, "F": self.F},
            "bottleneck_bounds": bounds,
            "conflict": conflict_info,
            "flow": flow,
            "critical_tasks": critical,
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"问题数学特征分析")
            print(f"{'='*60}")
            print(f"规模: {self.N}任务, {self.M}机器人, {self.E}电梯, {self.F}层")
            print(f"跨楼层比例: {bounds['cross_floor_ratio']:.1%}")

            print(f"\n--- 电梯瓶颈分析 ---")
            print(f"  电梯服务总时间: {bounds['total_elev_service_time']:.0f}s")
            print(f"  均匀分配时每电梯负载: {bounds['avg_elev_load_per_elevator']:.0f}s")
            print(f"  Makespan下界(电梯): {bounds['lb_elevator']:.0f}s")
            print(f"  Makespan下界(机器人): {bounds['lb_robot']:.0f}s")
            print(f"  综合下界: {bounds['lb_combined']:.0f}s")

            print(f"\n--- 冲突分析 ---")
            print(f"  平均冲突强度: {conflict_info['avg_conflict']:.3f}")
            print(f"  高冲突对数(>0.5): {conflict_info['high_conflict_pairs']}")
            print(f"  冲突簇数: {len(conflict_info['conflict_clusters'])}")

            print(f"\n--- 流量分析 ---")
            print(f"  上行/下行: {flow['upward_tasks']}/{flow['downward_tasks']}")
            print(f"  方向比: {flow['direction_ratio']:.2f}")
            print(f"  流量集中度(Gini): {flow['flow_concentration_gini']:.3f}")
            print(f"  热点取货楼层: {flow['hotspot_origins']}")

            print(f"\n--- 关键任务 ---")
            print(f"  平均关键度: {critical['avg_criticality']:.3f}")
            print(f"  最高关键度: {critical['max_criticality']:.3f}")
            top = critical['top_critical_tasks'][:5]
            for i in top:
                t = self.tasks[i]
                print(f"    T{i}: F{t.origin_floor}→F{t.dest_floor}, "
                      f"{t.weight}kg, {t.priority.name}, "
                      f"可行机器人={len(self.feasible_robots[i])}, "
                      f"关键度={critical['criticality_scores'][i]:.3f}")

        return report
