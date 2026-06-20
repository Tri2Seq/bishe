"""模块1：多机器人跨楼层配送调度 MILP求解器

决策变量：
  x[i,j]    — 任务i分配给机器人j
  y[i,i',j] — 机器人j在任务i后紧接执行任务i'
  z[i,e]    — 任务i使用电梯e（仅跨楼层任务）
  s[i]      — 任务i的开始时间
  c[i]      — 任务i的完成时间
  C_max     — makespan

目标函数（加权和）：
  min α₁·C_max + α₂·E_total + α₃·Tardiness
"""

import pulp
import numpy as np
from itertools import product
from typing import List, Tuple, Dict, Optional
from config import (Robot, Elevator, Task, Priority, PRIORITY_WEIGHT,
                    FLOOR_DISTANCE, INTER_STATION_DISTANCE)


def estimate_task_distance(task: Task) -> float:
    """估算任务总行程距离"""
    intra_floor = FLOOR_DISTANCE * 2  # 取货站→电梯+电梯→送货站
    if task.is_cross_floor:
        return intra_floor + INTER_STATION_DISTANCE
    return INTER_STATION_DISTANCE


def travel_time_between_tasks(t1: Task, t2: Task, robot: Robot,
                               elevators: List[Elevator]) -> float:
    """估算机器人从任务t1送达点到任务t2取货点的转移时间"""
    if t1.dest_floor == t2.origin_floor:
        return INTER_STATION_DISTANCE / robot.speed
    floor_diff = abs(t1.dest_floor - t2.origin_floor)
    fastest_elevator_time = min(e.travel_time(t1.dest_floor, t2.origin_floor)
                                for e in elevators)
    walk_to_elevator = FLOOR_DISTANCE / robot.speed
    return walk_to_elevator * 2 + fastest_elevator_time


def task_execution_time(task: Task, robot: Robot, elevator: Elevator) -> float:
    """计算任务执行时间（从取货到送达）"""
    pickup_time = 10.0  # 取货操作时间
    deliver_time = 10.0
    walk_time = FLOOR_DISTANCE / robot.speed  # 站点到电梯
    if task.is_cross_floor:
        elevator_time = elevator.travel_time(task.origin_floor, task.dest_floor)
        return pickup_time + walk_time + elevator_time + walk_time + deliver_time
    return pickup_time + INTER_STATION_DISTANCE / robot.speed + deliver_time


def solve_milp(robots: List[Robot], elevators: List[Elevator], tasks: List[Task],
               alpha: Tuple[float, float, float] = (0.5, 0.3, 0.2),
               time_limit: int = 120, verbose: bool = True) -> Optional[Dict]:
    """
    求解多机器人跨楼层配送调度MILP

    Args:
        robots: 机器人列表
        elevators: 电梯列表
        tasks: 任务列表
        alpha: 三目标权重 (makespan, energy, tardiness)
        time_limit: 求解时间限制(秒)
        verbose: 是否输出求解过程

    Returns:
        求解结果字典，包含分配方案、时序、目标值
    """
    N = len(tasks)
    M = len(robots)
    E = len(elevators)

    if verbose:
        print(f"问题规模: {N}任务, {M}机器人, {E}电梯")

    BIG_M = 10000.0

    prob = pulp.LpProblem("MultiRobot_MultiFloor_Delivery", pulp.LpMinimize)

    # ========== 决策变量 ==========

    # x[i,j]: 任务i分配给机器人j
    x = pulp.LpVariable.dicts("x", ((i, j) for i in range(N) for j in range(M)),
                               cat="Binary")

    # z[i,e]: 跨楼层任务i使用电梯e
    cross_floor_tasks = [t for t in tasks if t.is_cross_floor]
    cf_ids = [t.id for t in cross_floor_tasks]
    z = pulp.LpVariable.dicts("z", ((i, e) for i in cf_ids for e in range(E)),
                               cat="Binary")

    # s[i]: 任务i开始时间
    s = pulp.LpVariable.dicts("s", range(N), lowBound=0, cat="Continuous")

    # c[i]: 任务i完成时间
    c = pulp.LpVariable.dicts("c", range(N), lowBound=0, cat="Continuous")

    # y[i,i',j]: 机器人j在任务i后执行任务i'
    y = pulp.LpVariable.dicts("y", ((i, ip, j)
                                     for i in range(N) for ip in range(N) for j in range(M)
                                     if i != ip),
                               cat="Binary")

    # C_max: makespan
    C_max = pulp.LpVariable("C_max", lowBound=0, cat="Continuous")

    # tardiness[i]: 任务i的延迟
    tardiness = pulp.LpVariable.dicts("tard", range(N), lowBound=0, cat="Continuous")

    # ========== 预计算 ==========

    # 任务执行时间矩阵 proc[i,j,e]
    proc = {}
    for i in range(N):
        for j in range(M):
            if tasks[i].is_cross_floor:
                for e in range(E):
                    proc[i, j, e] = task_execution_time(tasks[i], robots[j], elevators[e])
            else:
                for e in range(E):
                    proc[i, j, e] = task_execution_time(tasks[i], robots[j], elevators[0])

    # 任务间转移时间 trans[i,i',j]
    trans = {}
    for i in range(N):
        for ip in range(N):
            if i == ip:
                continue
            for j in range(M):
                trans[i, ip, j] = travel_time_between_tasks(tasks[i], tasks[ip],
                                                             robots[j], elevators)

    # 机器人初始位置到任务i取货点的时间 init_time[i,j]
    init_time = {}
    for i in range(N):
        for j in range(M):
            if robots[j].init_floor == tasks[i].origin_floor:
                init_time[i, j] = INTER_STATION_DISTANCE / robots[j].speed
            else:
                floor_diff = abs(robots[j].init_floor - tasks[i].origin_floor)
                fastest = min(el.travel_time(robots[j].init_floor, tasks[i].origin_floor)
                              for el in elevators)
                init_time[i, j] = FLOOR_DISTANCE / robots[j].speed + fastest + \
                                  FLOOR_DISTANCE / robots[j].speed

    # 任务距离
    task_dist = {i: estimate_task_distance(tasks[i]) for i in range(N)}

    # ========== 约束 ==========

    # C1: 任务唯一性
    for i in range(N):
        prob += pulp.lpSum(x[i, j] for j in range(M)) == 1, f"unique_{i}"

    # C2: 载重约束（简化：单任务载重不超过机器人容量）
    for i in range(N):
        for j in range(M):
            if tasks[i].weight > robots[j].payload_max:
                prob += x[i, j] == 0, f"payload_{i}_{j}"

    # C3: 续航约束
    for j in range(M):
        prob += pulp.lpSum(x[i, j] * task_dist[i] for i in range(N)) <= \
                robots[j].range_max, f"range_{j}"

    # 跨楼层任务的电梯选择唯一性
    for i in cf_ids:
        prob += pulp.lpSum(z[i, e] for e in range(E)) == 1, f"elev_unique_{i}"

    # 电梯选择与任务分配关联：只有被分配了才需要选电梯
    # (如果任务i不是跨楼层的，不需要电梯)

    # C5: 时间窗
    for i in range(N):
        prob += s[i] >= tasks[i].earliest_start, f"earliest_{i}"

    # 完成时间 = 开始时间 + 执行时间
    # 使用线性化：对于跨楼层任务，c[i] 取决于分配的机器人和电梯
    for i in range(N):
        if tasks[i].is_cross_floor:
            for j in range(M):
                for e in range(E):
                    prob += c[i] >= s[i] + proc[i, j, e] - \
                            BIG_M * (2 - x[i, j] - z[i, e]), \
                            f"completion_{i}_{j}_{e}"
        else:
            for j in range(M):
                prob += c[i] >= s[i] + proc[i, j, 0] - \
                        BIG_M * (1 - x[i, j]), f"completion_{i}_{j}"

    # Makespan
    for i in range(N):
        prob += C_max >= c[i], f"makespan_{i}"

    # 延迟
    for i in range(N):
        prob += tardiness[i] >= c[i] - tasks[i].deadline, f"tardiness_{i}"

    # 排序约束：y[i,i',j]的一致性
    for i in range(N):
        for ip in range(N):
            if i == ip:
                continue
            for j in range(M):
                prob += y[i, ip, j] <= x[i, j], f"seq_assign1_{i}_{ip}_{j}"
                prob += y[i, ip, j] <= x[ip, j], f"seq_assign2_{i}_{ip}_{j}"

    # 每个任务在机器人j的序列中最多有一个后继
    for i in range(N):
        for j in range(M):
            prob += pulp.lpSum(y[i, ip, j] for ip in range(N) if ip != i) <= 1, \
                    f"one_succ_{i}_{j}"

    # 每个任务在机器人j的序列中最多有一个前驱
    for ip in range(N):
        for j in range(M):
            prob += pulp.lpSum(y[i, ip, j] for i in range(N) if i != ip) <= 1, \
                    f"one_pred_{ip}_{j}"

    # 时序一致性：如果y[i,i',j]=1, 则 s[i'] >= c[i] + trans[i,i',j]
    for i in range(N):
        for ip in range(N):
            if i == ip:
                continue
            for j in range(M):
                prob += s[ip] >= c[i] + trans[i, ip, j] - \
                        BIG_M * (1 - y[i, ip, j]), f"seq_time_{i}_{ip}_{j}"

    # 机器人的第一个任务：s[i] >= init_time[i,j]
    # 简化：如果任务i是机器人j的第一个任务（没有前驱），则 s[i] >= init_time[i,j]
    for i in range(N):
        for j in range(M):
            has_pred = pulp.lpSum(y[ip, i, j] for ip in range(N) if ip != i)
            prob += s[i] >= init_time[i, j] - BIG_M * (1 - x[i, j] + has_pred), \
                    f"init_time_{i}_{j}"

    # ========== 目标函数 ==========

    # 能耗
    E_robot = pulp.lpSum(x[i, j] * task_dist[i] * robots[j].energy_per_m
                          for i in range(N) for j in range(M))
    E_elev = pulp.lpSum(z[i, e] * tasks[i].floor_diff * 10.0  # 电梯能耗系数
                         for i in cf_ids for e in range(E))
    E_total = E_robot + E_elev

    # 加权延迟
    weighted_tardiness = pulp.lpSum(PRIORITY_WEIGHT[tasks[i].priority] * tardiness[i]
                                    for i in range(N))

    # 归一化（用估算的上界）
    makespan_ub = max(t.deadline for t in tasks) * 2
    energy_ub = sum(task_dist[i] * max(r.energy_per_m for r in robots) for i in range(N)) + \
                sum(t.floor_diff * 10.0 for t in cross_floor_tasks)
    tardiness_ub = sum(PRIORITY_WEIGHT[t.priority] * t.deadline for t in tasks)

    prob += alpha[0] * (C_max / makespan_ub) + \
            alpha[1] * (E_total / max(energy_ub, 1)) + \
            alpha[2] * (weighted_tardiness / max(tardiness_ub, 1)), "objective"

    # ========== 求解 ==========

    solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=verbose)
    status = prob.solve(solver)

    if verbose:
        print(f"求解状态: {pulp.LpStatus[status]}")
        print(f"目标值: {pulp.value(prob.objective):.4f}")

    if status not in [pulp.constants.LpStatusOptimal]:
        if verbose:
            print("未找到最优解")
        if status == pulp.constants.LpStatusNotSolved:
            return None

    # ========== 提取结果 ==========

    assignment = {}  # task_id -> robot_id
    for i in range(N):
        for j in range(M):
            if pulp.value(x[i, j]) is not None and pulp.value(x[i, j]) > 0.5:
                assignment[i] = j

    elevator_assignment = {}  # task_id -> elevator_id
    for i in cf_ids:
        for e in range(E):
            if pulp.value(z[i, e]) is not None and pulp.value(z[i, e]) > 0.5:
                elevator_assignment[i] = e

    task_sequence = {j: [] for j in range(M)}
    for j in range(M):
        assigned_tasks = [i for i in range(N) if assignment.get(i) == j]
        if not assigned_tasks:
            continue
        ordered = sorted(assigned_tasks, key=lambda i: pulp.value(s[i]) or 0)
        task_sequence[j] = ordered

    start_times = {i: pulp.value(s[i]) or 0 for i in range(N)}
    completion_times = {i: pulp.value(c[i]) or 0 for i in range(N)}
    tardiness_vals = {i: pulp.value(tardiness[i]) or 0 for i in range(N)}

    result = {
        "status": pulp.LpStatus[status],
        "makespan": pulp.value(C_max),
        "energy": pulp.value(E_total),
        "weighted_tardiness": pulp.value(weighted_tardiness),
        "objective": pulp.value(prob.objective),
        "assignment": assignment,
        "elevator_assignment": elevator_assignment,
        "task_sequence": task_sequence,
        "start_times": start_times,
        "completion_times": completion_times,
        "tardiness": tardiness_vals,
    }

    if verbose:
        print(f"\n=== 求解结果 ===")
        print(f"Makespan: {result['makespan']:.1f}s")
        print(f"总能耗: {result['energy']:.1f}J")
        print(f"加权延迟: {result['weighted_tardiness']:.1f}")
        print(f"\n任务分配:")
        for j in range(M):
            if task_sequence[j]:
                task_str = " → ".join(f"T{i}(F{tasks[i].origin_floor}→F{tasks[i].dest_floor})"
                                       for i in task_sequence[j])
                print(f"  机器人{j}({robots[j].type.name}): {task_str}")

    return result
