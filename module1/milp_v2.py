"""模块1 v2：严格MILP建模

核心改进：
1. 电梯建模为不可抢占的 disjunctive machine — 同一电梯的任意两次使用不重叠
2. 电梯重定位时间 — 电梯从上一次服务的终点楼层移动到下一次服务的起点楼层
3. 任务执行时间精确分解 — 取货行走 + 电梯等待 + 乘梯 + 送货行走
4. 机器人任务间转移的精确建模 — 可能需要额外乘电梯
5. 电梯使用决策与时序的完整耦合

符号约定：
  N 任务数, M 机器人数, E 电梯数, F 楼层数
  i,i' 任务索引, j 机器人索引, e 电梯索引
"""

import pulp
import numpy as np
from typing import List, Dict, Optional, Tuple
from config import (Robot, Elevator, Task, Priority, PRIORITY_WEIGHT,
                    FLOOR_DISTANCE, INTER_STATION_DISTANCE)


class MILPModelV2:
    """严格MILP建模器"""

    def __init__(self, robots: List[Robot], elevators: List[Elevator],
                 tasks: List[Task], alpha: Tuple[float, float, float] = (0.5, 0.3, 0.2)):
        self.robots = robots
        self.elevators = elevators
        self.tasks = tasks
        self.alpha = alpha
        self.N = len(tasks)
        self.M = len(robots)
        self.E = len(elevators)

        # 跨楼层任务集合
        self.cf_tasks = [t for t in tasks if t.is_cross_floor]
        self.cf_ids = set(t.id for t in self.cf_tasks)

        # Big-M: 使用更紧的上界
        self._compute_big_m()

    def _compute_big_m(self):
        """计算尽可能紧的Big-M上界"""
        max_floor_diff = max((t.floor_diff for t in self.tasks), default=1)
        slowest_elev = min(e.speed for e in self.elevators)
        max_elev_time = max_floor_diff / slowest_elev + max(e.door_time for e in self.elevators)
        slowest_robot = min(r.speed for r in self.robots)

        # 单个任务最长执行时间
        self.max_task_time = (FLOOR_DISTANCE / slowest_robot * 2 +  # 走到电梯+走到站点
                              max_elev_time +                         # 电梯行程
                              20.0)                                   # 取货+送货
        # 总时间上界
        self.T_max = self.max_task_time * self.N + max_elev_time * self.N
        self.BIG_M = self.T_max

    def _intra_floor_time(self, robot: Robot) -> float:
        """机器人在楼层内从站点到电梯大厅的时间"""
        return FLOOR_DISTANCE / robot.speed

    def _task_exec_time(self, task: Task, robot: Robot, elev: Elevator) -> float:
        """任务i由机器人j使用电梯e的执行时间（从取货开始到送达完成）

        跨楼层分解：
          取货(10s) + 走到电梯(floor_dist/speed) + 接近电梯(8s) + 按按钮(7s) +
          电梯行程(|Δf|/u_e + δ_e) + 出电梯(5s) + 走到站点(floor_dist/speed) + 送货(10s)
        同层：
          取货(10s) + 走到站点(station_dist/speed) + 送货(10s)
        """
        pickup_op = 10.0
        deliver_op = 10.0
        approach_time = 8.0   # 接近电梯（需GPU）
        press_time = 7.0      # 按按钮（需GPU）
        exit_time = 5.0       # 出电梯（需GPU）
        door_overhead = elev.door_time if task.is_cross_floor else 0  # 开关门

        if task.is_cross_floor:
            walk_to_elev = self._intra_floor_time(robot)
            ride = abs(task.dest_floor - task.origin_floor) / elev.speed
            walk_to_dest = self._intra_floor_time(robot)
            return (pickup_op + walk_to_elev + approach_time + press_time +
                    door_overhead + ride + exit_time + walk_to_dest + deliver_op)
        else:
            walk = INTER_STATION_DISTANCE / robot.speed
            return pickup_op + walk + deliver_op

    def _transfer_time(self, task_from: Task, task_to: Task, robot: Robot) -> float:
        """机器人从task_from的送达点到task_to的取货点的转移时间

        同楼层：直接走
        跨楼层：走到电梯 + approach(8s) + press(7s) + 电梯行程 + exit(5s) + 走到站点
        """
        if task_from.dest_floor == task_to.origin_floor:
            return INTER_STATION_DISTANCE / robot.speed
        else:
            fastest_ride = min(
                abs(task_from.dest_floor - task_to.origin_floor) / e.speed + e.door_time
                for e in self.elevators)
            walk = self._intra_floor_time(robot) * 2
            elev_interact = 8.0 + 7.0 + 5.0  # approach + press + exit
            return walk + elev_interact + fastest_ride

    def _init_reach_time(self, task: Task, robot: Robot) -> float:
        """机器人从初始位置到达任务取货点的时间"""
        if robot.init_floor == task.origin_floor:
            return INTER_STATION_DISTANCE / robot.speed
        else:
            fastest_ride = min(
                abs(robot.init_floor - task.origin_floor) / e.speed + e.door_time
                for e in self.elevators)
            walk = self._intra_floor_time(robot) * 2
            elev_interact = 8.0 + 7.0 + 5.0
            return walk + elev_interact + fastest_ride

    def build_and_solve(self, time_limit: int = 300, verbose: bool = True) -> Optional[Dict]:
        """构建并求解完整MILP模型"""

        prob = pulp.LpProblem("MultiFloor_Delivery_V2", pulp.LpMinimize)

        N, M, E = self.N, self.M, self.E
        tasks = self.tasks
        robots = self.robots
        elevators = self.elevators
        BIG_M = self.BIG_M

        if verbose:
            print(f"MILP V2 构建中: {N}任务, {M}机器人, {E}电梯")
            print(f"  跨楼层任务: {len(self.cf_ids)}, Big-M={BIG_M:.0f}")

        # ================================================================
        # 决策变量
        # ================================================================

        # x[i,j]: 任务i分配给机器人j
        x = pulp.LpVariable.dicts("x", ((i, j) for i in range(N) for j in range(M)),
                                   cat="Binary")

        # y[i,i',j]: 机器人j在执行序列中i排在i'前面（紧邻）
        y = {}
        for j in range(M):
            for i in range(N):
                for ip in range(N):
                    if i != ip:
                        y[i, ip, j] = pulp.LpVariable(f"y_{i}_{ip}_{j}", cat="Binary")

        # z[i,e]: 跨楼层任务i使用电梯e
        z = {}
        for i in self.cf_ids:
            for e in range(E):
                z[i, e] = pulp.LpVariable(f"z_{i}_{e}", cat="Binary")

        # 时序变量
        s = pulp.LpVariable.dicts("s", range(N), lowBound=0)  # 任务开始时间（取货开始）
        c = pulp.LpVariable.dicts("c", range(N), lowBound=0)  # 任务完成时间（送达完成）

        # 电梯使用时序：
        # elev_start[i,e]: 任务i开始使用电梯e的时刻（机器人到达电梯层开始等待/登入）
        # elev_end[i,e]:   任务i结束使用电梯e的时刻（电梯到达目标层，机器人离开）
        elev_start = {}
        elev_end = {}
        for i in self.cf_ids:
            for e in range(E):
                elev_start[i, e] = pulp.LpVariable(f"es_{i}_{e}", lowBound=0)
                elev_end[i, e] = pulp.LpVariable(f"ee_{i}_{e}", lowBound=0)

        # 电梯不相交排序变量：
        # sigma[i,i',e]: 在电梯e上，任务i在任务i'之前使用（i先用e，i'后用e）
        sigma = {}
        cf_list = sorted(self.cf_ids)
        for idx_a, i in enumerate(cf_list):
            for idx_b, ip in enumerate(cf_list):
                if i < ip:  # 只对i<i'建立变量（对称性破除）
                    for e in range(E):
                        sigma[i, ip, e] = pulp.LpVariable(f"sig_{i}_{ip}_{e}", cat="Binary")

        # Makespan
        C_max = pulp.LpVariable("C_max", lowBound=0)

        # 延迟变量
        tard = pulp.LpVariable.dicts("tard", range(N), lowBound=0)

        # ================================================================
        # 约束
        # ================================================================

        # --- C1: 任务唯一性 ---
        for i in range(N):
            prob += pulp.lpSum(x[i, j] for j in range(M)) == 1, f"C1_unique_{i}"

        # --- C2: 载重可行性 ---
        for i in range(N):
            for j in range(M):
                if tasks[i].weight > robots[j].payload_max:
                    prob += x[i, j] == 0, f"C2_payload_{i}_{j}"

        # --- C3: 续航约束 ---
        task_dist = {}
        for i in range(N):
            if tasks[i].is_cross_floor:
                task_dist[i] = FLOOR_DISTANCE * 2 + INTER_STATION_DISTANCE
            else:
                task_dist[i] = INTER_STATION_DISTANCE
        for j in range(M):
            prob += pulp.lpSum(x[i, j] * task_dist[i] for i in range(N)) <= \
                    robots[j].range_max, f"C3_range_{j}"

        # --- C4: 跨楼层任务电梯选择唯一 ---
        for i in self.cf_ids:
            prob += pulp.lpSum(z[i, e] for e in range(E)) == 1, f"C4_elev_{i}"

        # --- C5: 时间窗 ---
        for i in range(N):
            prob += s[i] >= tasks[i].earliest_start, f"C5_earliest_{i}"

        # --- C6: 任务完成时间 = 开始时间 + 执行时间 ---
        # 线性化：c[i] >= s[i] + exec_time(i,j,e) - BIG_M*(2 - x[i,j] - z[i,e])
        for i in range(N):
            if i in self.cf_ids:
                for j in range(M):
                    for e in range(E):
                        et = self._task_exec_time(tasks[i], robots[j], elevators[e])
                        prob += c[i] >= s[i] + et - BIG_M * (2 - x[i, j] - z[i, e]), \
                                f"C6_comp_{i}_{j}_{e}"
            else:
                for j in range(M):
                    et = self._task_exec_time(tasks[i], robots[j], elevators[0])
                    prob += c[i] >= s[i] + et - BIG_M * (1 - x[i, j]), \
                            f"C6_comp_{i}_{j}"

        # --- C7: 排序约束 ---
        # y[i,i',j] <= x[i,j]  且  y[i,i',j] <= x[i',j]
        for j in range(M):
            for i in range(N):
                for ip in range(N):
                    if i == ip:
                        continue
                    prob += y[i, ip, j] <= x[i, j], f"C7a_{i}_{ip}_{j}"
                    prob += y[i, ip, j] <= x[ip, j], f"C7b_{i}_{ip}_{j}"

        # 每个任务在机器人j的序列中最多一个后继
        for i in range(N):
            for j in range(M):
                prob += pulp.lpSum(y[i, ip, j] for ip in range(N) if ip != i) <= 1, \
                        f"C7_succ_{i}_{j}"

        # 每个任务在机器人j的序列中最多一个前驱
        for ip in range(N):
            for j in range(M):
                prob += pulp.lpSum(y[i, ip, j] for i in range(N) if i != ip) <= 1, \
                        f"C7_pred_{ip}_{j}"

        # --- C8: 时序一致性 ---
        # 如果 y[i,i',j]=1，则 s[i'] >= c[i] + transfer_time(i→i', j)
        for j in range(M):
            for i in range(N):
                for ip in range(N):
                    if i == ip:
                        continue
                    tt = self._transfer_time(tasks[i], tasks[ip], robots[j])
                    prob += s[ip] >= c[i] + tt - BIG_M * (1 - y[i, ip, j]), \
                            f"C8_seq_{i}_{ip}_{j}"

        # --- C9: 首任务到达时间 ---
        for i in range(N):
            for j in range(M):
                init_t = self._init_reach_time(tasks[i], robots[j])
                has_pred = pulp.lpSum(y[ip, i, j] for ip in range(N) if ip != i)
                prob += s[i] >= init_t - BIG_M * (1 - x[i, j] + has_pred), \
                        f"C9_init_{i}_{j}"

        # --- C10: 电梯使用时序 ---
        # 电梯使用开始时间 >= 任务开始时间 + 取货操作 + 走到电梯
        # 电梯使用结束时间 = 电梯使用开始时间 + 电梯行程时间
        approach_time = 8.0
        press_time = 7.0
        exit_time_const = 5.0

        for i in self.cf_ids:
            for j in range(M):
                for e in range(E):
                    walk_time = self._intra_floor_time(robots[j])
                    pickup_time = 10.0
                    ride_time = abs(tasks[i].dest_floor - tasks[i].origin_floor) / elevators[e].speed
                    door_time = elevators[e].door_time

                    # elev_start = 机器人到达电梯准备登入的时刻
                    # = s[i] + pickup + walk + approach + press
                    pre_elev = pickup_time + walk_time + approach_time + press_time
                    prob += elev_start[i, e] >= s[i] + pre_elev - \
                            BIG_M * (2 - x[i, j] - z[i, e]), \
                            f"C10a_{i}_{j}_{e}"

                    # elev_end = elev_start + door_open/close + ride
                    prob += elev_end[i, e] >= elev_start[i, e] + door_time + ride_time - \
                            BIG_M * (1 - z[i, e]), \
                            f"C10b_{i}_{e}_{j}"

        # --- C11: 电梯不相交约束 (Disjunctive) ---
        # 核心约束：同一电梯e上，任务i和i'的使用时段不重叠
        # 且考虑电梯重定位时间
        #
        # 如果 sigma[i,i',e]=1 (i先于i'):
        #   elev_start[i',e] >= elev_end[i,e] + reposition_time(dest_i → origin_i')
        # 如果 sigma[i,i',e]=0 (i'先于i):
        #   elev_start[i,e] >= elev_end[i',e] + reposition_time(dest_i' → origin_i)
        #
        # 这些约束仅在i和i'都使用电梯e时激活

        for idx_a, i in enumerate(cf_list):
            for idx_b, ip in enumerate(cf_list):
                if i >= ip:
                    continue
                for e in range(E):
                    # 电梯从任务i的终点楼层移动到任务i'的起点楼层
                    repo_i_to_ip = elevators[e].travel_time(tasks[i].dest_floor,
                                                             tasks[ip].origin_floor)
                    # 反方向
                    repo_ip_to_i = elevators[e].travel_time(tasks[ip].dest_floor,
                                                             tasks[i].origin_floor)

                    # 约束仅在两个任务都使用电梯e时有效
                    # 使用 z[i,e] + z[i',e] <= 1 + (需要不相交) 的方式
                    # 当 sigma=1: elev_start[i',e] >= elev_end[i,e] + repo
                    prob += elev_start[ip, e] >= elev_end[i, e] + repo_i_to_ip - \
                            BIG_M * (3 - z[i, e] - z[ip, e] - sigma[i, ip, e]), \
                            f"C11a_{i}_{ip}_{e}"

                    # 当 sigma=0: elev_start[i,e] >= elev_end[i',e] + repo
                    prob += elev_start[i, e] >= elev_end[ip, e] + repo_ip_to_i - \
                            BIG_M * (2 - z[i, e] - z[ip, e] + sigma[i, ip, e]), \
                            f"C11b_{i}_{ip}_{e}"

        # --- C12: Makespan ---
        for i in range(N):
            prob += C_max >= c[i], f"C12_makespan_{i}"

        # --- C13: 延迟 ---
        for i in range(N):
            prob += tard[i] >= c[i] - tasks[i].deadline, f"C13_tard_{i}"

        # ================================================================
        # 目标函数
        # ================================================================

        # 能耗
        E_robot = pulp.lpSum(x[i, j] * task_dist[i] * robots[j].energy_per_m
                              for i in range(N) for j in range(M))
        E_elev = pulp.lpSum(z[i, e] * tasks[i].floor_diff *
                             (10.0 + elevators[e].door_time * 0.5)
                             for i in self.cf_ids for e in range(E))
        E_total = E_robot + E_elev

        # 加权延迟
        W_tard = pulp.lpSum(PRIORITY_WEIGHT[tasks[i].priority] * tard[i]
                             for i in range(N))

        # 归一化上界（更紧的估算）
        makespan_ub = max(t.deadline for t in tasks) * 3
        energy_ub = sum(task_dist[i] * max(r.energy_per_m for r in robots)
                        for i in range(N)) + \
                    sum(t.floor_diff * 15.0 for t in self.cf_tasks)
        tardiness_ub = sum(PRIORITY_WEIGHT[t.priority] * t.deadline * 2 for t in tasks)

        a1, a2, a3 = self.alpha
        prob += a1 * (C_max / max(makespan_ub, 1)) + \
                a2 * (E_total / max(energy_ub, 1)) + \
                a3 * (W_tard / max(tardiness_ub, 1)), "objective"

        # ================================================================
        # 统计变量/约束数量
        # ================================================================
        num_vars = len(prob.variables())
        num_constraints = len(prob.constraints)
        if verbose:
            print(f"  变量数: {num_vars} (其中二元变量占大部分)")
            print(f"  约束数: {num_constraints}")

        # ================================================================
        # 求解
        # ================================================================
        solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=verbose,
                                    options=["ratioGap", "0.05"])
        status = prob.solve(solver)

        if verbose:
            print(f"\n求解状态: {pulp.LpStatus[status]}")
            if pulp.value(prob.objective) is not None:
                print(f"目标值: {pulp.value(prob.objective):.6f}")

        if status not in [pulp.constants.LpStatusOptimal, 1]:  # 1 = Optimal
            # 也接受 time limit 下的可行解
            if pulp.value(C_max) is None:
                if verbose:
                    print("未找到可行解")
                return None

        # ================================================================
        # 提取结果
        # ================================================================
        return self._extract_results(prob, x, y, z, s, c, tard, C_max, E_total, W_tard,
                                      elev_start, elev_end, verbose)

    def _extract_results(self, prob, x, y, z, s, c, tard, C_max, E_total, W_tard,
                         elev_start, elev_end, verbose) -> Dict:
        N, M, E = self.N, self.M, self.E
        tasks = self.tasks

        assignment = {}
        for i in range(N):
            for j in range(M):
                val = pulp.value(x[i, j])
                if val is not None and val > 0.5:
                    assignment[i] = j

        elevator_assignment = {}
        for i in self.cf_ids:
            for e in range(E):
                val = pulp.value(z[i, e])
                if val is not None and val > 0.5:
                    elevator_assignment[i] = e

        task_sequence = {j: [] for j in range(M)}
        for j in range(M):
            assigned = [i for i in range(N) if assignment.get(i) == j]
            assigned.sort(key=lambda i: pulp.value(s[i]) or 0)
            task_sequence[j] = assigned

        start_times = {i: pulp.value(s[i]) or 0 for i in range(N)}
        completion_times = {i: pulp.value(c[i]) or 0 for i in range(N)}

        # 电梯使用时段
        elev_usage = {}
        for i in self.cf_ids:
            e = elevator_assignment.get(i)
            if e is not None:
                es = pulp.value(elev_start[i, e]) or 0
                ee = pulp.value(elev_end[i, e]) or 0
                elev_usage[i] = {"elevator": e, "start": es, "end": ee,
                                  "from": tasks[i].origin_floor, "to": tasks[i].dest_floor}

        makespan_val = pulp.value(C_max) or 0
        energy_val = pulp.value(E_total) or 0
        tardiness_val = pulp.value(W_tard) or 0

        result = {
            "status": pulp.LpStatus[prob.status],
            "makespan": makespan_val,
            "energy": energy_val,
            "weighted_tardiness": tardiness_val,
            "objective": pulp.value(prob.objective),
            "assignment": assignment,
            "elevator_assignment": elevator_assignment,
            "task_sequence": task_sequence,
            "start_times": start_times,
            "completion_times": completion_times,
            "tardiness": {i: pulp.value(tard[i]) or 0 for i in range(N)},
            "elevator_usage": elev_usage,
            "num_variables": len(prob.variables()),
            "num_constraints": len(prob.constraints),
        }

        if verbose:
            self._print_results(result)

        return result

    def _print_results(self, result: Dict):
        tasks = self.tasks
        robots = self.robots

        print(f"\n{'='*60}")
        print(f"MILP V2 求解结果")
        print(f"{'='*60}")
        print(f"  Makespan: {result['makespan']:.1f}s")
        print(f"  总能耗: {result['energy']:.1f}J")
        print(f"  加权延迟: {result['weighted_tardiness']:.1f}")
        print(f"  变量数: {result['num_variables']}")
        print(f"  约束数: {result['num_constraints']}")

        print(f"\n任务分配:")
        for j in range(self.M):
            seq = result['task_sequence'][j]
            if seq:
                items = []
                for i in seq:
                    t = tasks[i]
                    elev_info = ""
                    if i in result['elevator_usage']:
                        eu = result['elevator_usage'][i]
                        elev_info = f"[E{eu['elevator']}]"
                    items.append(f"T{i}(F{t.origin_floor}→F{t.dest_floor}){elev_info}")
                print(f"  R{j}({robots[j].type.name}): {' → '.join(items)}")

        print(f"\n电梯使用时序:")
        for e in range(self.E):
            usages = [(i, u) for i, u in result['elevator_usage'].items() if u['elevator'] == e]
            usages.sort(key=lambda x: x[1]['start'])
            if usages:
                print(f"  电梯E{e}:")
                for i, u in usages:
                    print(f"    T{i}: [{u['start']:.1f}s - {u['end']:.1f}s] "
                          f"F{u['from']}→F{u['to']}")
            else:
                print(f"  电梯E{e}: 未使用")

        # 验证电梯不相交
        for e in range(self.E):
            usages = [(i, u) for i, u in result['elevator_usage'].items() if u['elevator'] == e]
            usages.sort(key=lambda x: x[1]['start'])
            for idx in range(len(usages) - 1):
                i1, u1 = usages[idx]
                i2, u2 = usages[idx + 1]
                gap = u2['start'] - u1['end']
                if gap < -0.1:
                    print(f"  ⚠ 电梯E{e}冲突: T{i1}结束{u1['end']:.1f} > T{i2}开始{u2['start']:.1f}")
                else:
                    repo_time = self.elevators[e].travel_time(
                        tasks[i1].dest_floor, tasks[i2].origin_floor)
                    print(f"    间隔: {gap:.1f}s (重定位需{repo_time:.1f}s) "
                          f"{'✓' if gap >= repo_time - 0.1 else '⚠不足'}")
