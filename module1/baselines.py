"""对比基线算法

B1: Greedy-FCFS    — 贪心分配 + 先到先服务电梯
B2: 单目标MILP      — 仅优化makespan
B3: 无电梯感知VRP   — 忽略电梯竞争，事后分配电梯
B4: 独立规划        — 每个机器人独立贪心，无协同
"""

from typing import List, Dict, Optional
import numpy as np
from config import (Robot, Elevator, Task, Priority, PRIORITY_WEIGHT,
                    FLOOR_DISTANCE, INTER_STATION_DISTANCE)
from milp_solver import estimate_task_distance, task_execution_time


def greedy_fcfs(robots: List[Robot], elevators: List[Elevator],
                tasks: List[Task]) -> Dict:
    """
    B1: 贪心最近分配 + FCFS电梯

    策略：按优先级排序任务，每个任务分配给当前"最近"的可用机器人。
    电梯按先到先服务分配。
    """
    sorted_tasks = sorted(range(len(tasks)), key=lambda i: (tasks[i].priority, tasks[i].deadline))

    assignment = {}
    elevator_assignment = {}
    robot_current_floor = {j: r.init_floor for j, r in enumerate(robots)}
    robot_current_load = {j: 0.0 for j in range(len(robots))}
    robot_available_time = {j: 0.0 for j in range(len(robots))}
    robot_total_dist = {j: 0.0 for j in range(len(robots))}
    task_sequence = {j: [] for j in range(len(robots))}
    start_times = {}
    completion_times = {}

    elevator_available_time = {e: 0.0 for e in range(len(elevators))}

    for task_id in sorted_tasks:
        task = tasks[task_id]
        best_robot = None
        best_cost = float("inf")

        for j, robot in enumerate(robots):
            if task.weight > robot.payload_max:
                continue
            dist = estimate_task_distance(task)
            if robot_total_dist[j] + dist > robot.range_max:
                continue

            floor_diff = abs(robot_current_floor[j] - task.origin_floor)
            time_to_reach = floor_diff * 3.0 + INTER_STATION_DISTANCE / robot.speed
            cost = robot_available_time[j] + time_to_reach

            if cost < best_cost:
                best_cost = cost
                best_robot = j

        if best_robot is None:
            # 分配给负载最轻的能承载的机器人
            for j, robot in enumerate(robots):
                if task.weight <= robot.payload_max:
                    best_robot = j
                    break
            if best_robot is None:
                continue

        assignment[task_id] = best_robot
        task_sequence[best_robot].append(task_id)

        # 电梯分配：FCFS
        if task.is_cross_floor:
            best_elev = min(range(len(elevators)),
                            key=lambda e: elevator_available_time[e])
            elevator_assignment[task_id] = best_elev
            elev = elevators[best_elev]
            elev_time = elev.travel_time(task.origin_floor, task.dest_floor)
        else:
            elev_time = 0

        # 计算时间
        robot = robots[best_robot]
        start = max(robot_available_time[best_robot] + best_cost - robot_available_time[best_robot],
                    task.earliest_start)
        start_times[task_id] = start

        exec_time = INTER_STATION_DISTANCE / robot.speed + 20.0  # 取+送
        if task.is_cross_floor:
            exec_time += FLOOR_DISTANCE / robot.speed * 2 + elev_time + 20.0  # 电梯交互
            elevator_available_time[elevator_assignment[task_id]] = start + exec_time

        end = start + exec_time
        completion_times[task_id] = end

        robot_available_time[best_robot] = end
        robot_current_floor[best_robot] = task.dest_floor
        robot_total_dist[best_robot] += estimate_task_distance(task)

    # 计算指标
    makespan = max(completion_times.values()) if completion_times else 0
    energy = sum(estimate_task_distance(tasks[i]) * robots[assignment[i]].energy_per_m
                 for i in assignment)
    energy += sum(tasks[i].floor_diff * 10.0 for i in elevator_assignment)
    weighted_tardiness = sum(
        PRIORITY_WEIGHT[tasks[i].priority] * max(0, completion_times[i] - tasks[i].deadline)
        for i in completion_times
    )

    return {
        "status": "Greedy-FCFS",
        "makespan": makespan,
        "energy": energy,
        "weighted_tardiness": weighted_tardiness,
        "assignment": assignment,
        "elevator_assignment": elevator_assignment,
        "task_sequence": task_sequence,
        "start_times": start_times,
        "completion_times": completion_times,
        "tardiness": {i: max(0, completion_times[i] - tasks[i].deadline)
                      for i in completion_times},
    }


def independent_planning(robots: List[Robot], elevators: List[Elevator],
                         tasks: List[Task]) -> Dict:
    """
    B4: 独立规划 — 每个机器人按距离贪心选任务，无全局协同

    模拟每个机器人独立决策，可能导致多个机器人抢同一任务/电梯。
    简化：轮流让机器人选任务，但不考虑其他机器人的计划。
    """
    remaining = set(range(len(tasks)))
    assignment = {}
    elevator_assignment = {}
    task_sequence = {j: [] for j in range(len(robots))}
    robot_floor = {j: r.init_floor for j, r in enumerate(robots)}
    robot_time = {j: 0.0 for j in range(len(robots))}
    robot_dist = {j: 0.0 for j in range(len(robots))}
    start_times = {}
    completion_times = {}
    elevator_avail = {e: 0.0 for e in range(len(elevators))}

    while remaining:
        assigned_this_round = False
        for j, robot in enumerate(robots):
            if not remaining:
                break
            # 选离当前位置最近的可行任务
            best_task = None
            best_dist = float("inf")
            for tid in remaining:
                if tasks[tid].weight > robot.payload_max:
                    continue
                d = abs(robot_floor[j] - tasks[tid].origin_floor)
                if d < best_dist:
                    best_dist = d
                    best_task = tid

            if best_task is None:
                continue

            remaining.discard(best_task)
            assignment[best_task] = j
            task_sequence[j].append(best_task)
            assigned_this_round = True

            task = tasks[best_task]
            # 计算到达时间
            travel = abs(robot_floor[j] - task.origin_floor) * 3.0 + \
                     INTER_STATION_DISTANCE / robot.speed
            start = max(robot_time[j] + travel, task.earliest_start)
            start_times[best_task] = start

            exec_time = 20.0 + INTER_STATION_DISTANCE / robot.speed
            if task.is_cross_floor:
                best_elev = min(range(len(elevators)),
                                key=lambda e: elevator_avail[e])
                elevator_assignment[best_task] = best_elev
                elev = elevators[best_elev]
                exec_time += FLOOR_DISTANCE / robot.speed * 2 + \
                             elev.travel_time(task.origin_floor, task.dest_floor) + 20.0
                elevator_avail[best_elev] = start + exec_time

            end = start + exec_time
            completion_times[best_task] = end
            robot_time[j] = end
            robot_floor[j] = task.dest_floor

        if not assigned_this_round:
            break

    makespan = max(completion_times.values()) if completion_times else 0
    energy = sum(estimate_task_distance(tasks[i]) * robots[assignment[i]].energy_per_m
                 for i in assignment)
    energy += sum(tasks[i].floor_diff * 10.0 for i in elevator_assignment)
    weighted_tardiness = sum(
        PRIORITY_WEIGHT[tasks[i].priority] * max(0, completion_times[i] - tasks[i].deadline)
        for i in completion_times
    )

    return {
        "status": "Independent",
        "makespan": makespan,
        "energy": energy,
        "weighted_tardiness": weighted_tardiness,
        "assignment": assignment,
        "elevator_assignment": elevator_assignment,
        "task_sequence": task_sequence,
        "start_times": start_times,
        "completion_times": completion_times,
        "tardiness": {i: max(0, completion_times[i] - tasks[i].deadline)
                      for i in completion_times},
    }
