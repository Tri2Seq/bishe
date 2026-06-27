"""仿真器V2：基于电梯群控系统的完整仿真

核心改进（相比V1）：
  1. 电梯由群控系统调度，机器人不直接选择电梯
  2. 机器人可以批量携带多个同方向货物（减少电梯使用次数）
  3. 机器人可以共乘（同方向、同电梯、电梯容量允许）
  4. 时间驱动仿真（固定步长推进），而非纯事件驱动
  5. 完整的能耗统计（机器人行走+电梯运行+GPU推理）
  6. 支持24小时连续仿真，任务动态到达

机器人决策层（在电梯群控下的策略选择）：
  - 何时去取货（是否等更多同方向任务一起取？）
  - 何时按电梯（是否等邻近楼层的任务合并？）
  - 电梯到了是否搭乘（如果方向不对，等下一班？）

这是模块1优化算法的目标：找到最优的任务分配+执行策略
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum, auto
import json

from config import Robot, Elevator, Task, Priority, PRIORITY_WEIGHT, FLOOR_DISTANCE, INTER_STATION_DISTANCE, RelayPlan
from elevator_group_control import (ElevatorGroupController, DispatchAlgorithm,
                                     Direction, HallCall, ElevatorPhase)


class RobotPhase(Enum):
    IDLE = auto()
    WALKING_TO_PICKUP = auto()
    PICKING_UP = auto()
    WALKING_TO_ELEVATOR = auto()
    APPROACHING_ELEVATOR = auto()   # 需要GPU
    PRESSING_BUTTON = auto()        # 需要GPU
    WAITING_ELEVATOR = auto()
    BOARDING = auto()
    RIDING = auto()
    EXITING_ELEVATOR = auto()       # 需要GPU
    WALKING_TO_DELIVERY = auto()
    DELIVERING = auto()
    RELAY_DROPOFF = auto()          # 在中继楼层放下货物


GPU_PHASES = {RobotPhase.APPROACHING_ELEVATOR, RobotPhase.PRESSING_BUTTON,
              RobotPhase.EXITING_ELEVATOR}

PICKUP_TIME = 10.0
DELIVER_TIME = 10.0
APPROACH_TIME = 8.0
PRESS_TIME = 7.0
EXIT_TIME = 5.0


@dataclass
class RobotStrategy:
    """
    机器人在电梯群控系统下的策略参数。
    这些参数由优化算法调节，不同取值产生不同的调度效果。

    batch_wait: 批量等待时间(s) — 取完一个货后等多久看有没有同楼层新任务
    skip_wrong_dir: 是否跳过方向不对的电梯
    stagger_delay: 同楼层多机器人时的错峰延迟(s)
    preemptive_call: 是否在取货前就提前呼叫电梯
    """
    batch_wait: float = 0.0
    skip_wrong_dir: bool = False
    stagger_delay: float = 0.0
    preemptive_call: bool = False


@dataclass
class RobotState:
    """机器人完整状态"""
    id: int
    robot: Robot
    floor: int
    phase: RobotPhase = RobotPhase.IDLE
    phase_remaining: float = 0.0     # 当前阶段剩余时间
    current_task: int = -1           # 正在执行的任务ID
    carrying: List[int] = field(default_factory=list)  # 正在携带的货物(任务ID)
    carrying_weight: float = 0.0
    task_queue: List[int] = field(default_factory=list)  # 待执行任务队列
    pending_call: Optional[HallCall] = None  # 待响应的电梯呼叫
    target_floor: int = -1
    delivery_route: List = field(default_factory=list)  # 多站送货路线

    # 统计
    total_distance: float = 0.0
    total_energy: float = 0.0
    total_idle_time: float = 0.0
    total_gpu_time: float = 0.0
    tasks_completed: int = 0


@dataclass
class ActionLog:
    """动作日志"""
    robot_id: int
    action: str
    start_time: float
    end_time: float
    floor: int
    needs_gpu: bool = False
    task_id: int = -1
    elevator_id: int = -1
    details: str = ""


class SimulatorV2:
    """基于电梯群控的仿真器"""

    def __init__(self, robots: List[Robot], elevators: List[Elevator],
                 tasks: List[Task], dispatch_algorithm: DispatchAlgorithm = DispatchAlgorithm.ETA,
                 dt: float = 0.5, strategy: Optional[RobotStrategy] = None):
        self.robots_config = robots
        self.elevators_config = elevators
        self.tasks = tasks
        self.dt = dt
        self.strategy = strategy or RobotStrategy()
        self.N = len(tasks)
        self.M = len(robots)
        self.E = len(elevators)

        # 电梯群控系统
        elev_configs = [
            {'init_floor': e.init_floor, 'speed': e.speed,
             'door_time': e.door_time, 'capacity': e.capacity}
            for e in elevators
        ]
        num_floors = max(max(t.origin_floor, t.dest_floor) for t in tasks)
        self.egcs = ElevatorGroupController(
            self.E, num_floors, dispatch_algorithm, elev_configs)

        # 机器人状态
        self.robot_states: List[RobotState] = [
            RobotState(id=j, robot=robots[j], floor=robots[j].init_floor)
            for j in range(self.M)
        ]

        # 任务状态
        self.task_completed = [False] * self.N
        self.task_start_times = {}
        self.task_completion_times = {}

        self.current_time = 0.0
        self.action_log: List[ActionLog] = []

        # GPU需求跟踪
        self.gpu_demand_log: List[Dict] = []

        # 中继缓冲区：{floor: [(task_id, weight, relay_robot_id), ...]}
        self.relay_buffer: Dict[int, List] = defaultdict(list)
        # 中继计划：{task_id: RelayPlan}
        self.relay_plans: Dict[int, 'RelayPlan'] = {}
        # 段完成状态：{seg_id: bool}
        self.seg_completed: Dict[int, bool] = {}

    def load_schedule(self, assignment: Dict[int, int],
                      task_sequence: Dict[int, List[int]],
                      elevator_assignment: Dict[int, int] = None,
                      relay_plans: Dict[int, any] = None):
        """加载调度方案。relay_plans: {task_id: RelayPlan}"""
        for j, seq in task_sequence.items():
            self.robot_states[j].task_queue = list(seq)
        if relay_plans:
            self.relay_plans = dict(relay_plans)

    def run(self, max_time: float = 86400.0, verbose: bool = False) -> Dict:
        """
        运行仿真

        Args:
            max_time: 最大仿真时间(秒)，默认24小时
        """
        self.current_time = 0.0

        # 启动所有有任务的机器人
        for rs in self.robot_states:
            if rs.task_queue:
                self._start_next_task(rs)

        while self.current_time < max_time:
            # 检查是否所有任务完成
            if all(self.task_completed):
                break

            # 推进一步
            self._step()
            self.current_time += self.dt

        return self._collect_results()

    def _step(self):
        """推进一个时间步"""
        # 1. 更新电梯群控系统
        self.egcs.step(self.dt)
        self.egcs.current_time = self.current_time

        # 2. 处理电梯到站事件
        for car in self.egcs.elevators:
            if car.phase == ElevatorPhase.DOOR_OPENING:
                floor = int(round(car.current_floor))

                # 先放人：当前楼层有乘客要下
                dropped = self.egcs.dropoff_at_floor(car)
                for call in dropped:
                    self._handle_dropoff(call)

                # 再接人：当前楼层有呼叫等待
                picked = self.egcs.pickup_at_floor(car)
                for call in picked:
                    self._handle_pickup(call)

                # 决定下一步：用stops列表确定方向
                self.egcs._update_stops(car.id)
                if car.passengers or car.stops:
                    car.phase = ElevatorPhase.MOVING_TO_DEST
                    if car.stops:
                        # 按当前方向找下一站
                        above = [s for s in car.stops if s > floor]
                        below = [s for s in car.stops if s < floor]
                        if car.direction == Direction.UP and above:
                            car.direction = Direction.UP
                        elif car.direction == Direction.DOWN and below:
                            car.direction = Direction.DOWN
                        elif above:
                            car.direction = Direction.UP
                        elif below:
                            car.direction = Direction.DOWN
                        else:
                            car.phase = ElevatorPhase.IDLE
                            car.direction = Direction.IDLE
                    else:
                        car.phase = ElevatorPhase.IDLE
                        car.direction = Direction.IDLE
                elif car.call_queue:
                    car.phase = ElevatorPhase.MOVING_TO_CALL
                    next_call = car.call_queue[0]
                    car.direction = Direction.UP if next_call.floor > floor else Direction.DOWN
                else:
                    car.phase = ElevatorPhase.IDLE
                    car.direction = Direction.IDLE

        # 3. 更新每个机器人
        for rs in self.robot_states:
            self._update_robot(rs)

    def _update_robot(self, rs: RobotState):
        """更新单个机器人状态"""
        if rs.phase == RobotPhase.IDLE:
            rs.total_idle_time += self.dt
            if rs.task_queue:
                self._start_next_task(rs)
            return

        # 事件驱动阶段：不用倒计时，等外部事件触发转换
        if rs.phase == RobotPhase.WAITING_ELEVATOR:
            return
        if rs.phase == RobotPhase.RIDING:
            return

        # GPU时间统计
        if rs.phase in GPU_PHASES:
            rs.total_gpu_time += self.dt

        # 倒计时阶段
        rs.phase_remaining -= self.dt
        if rs.phase_remaining > 0:
            if rs.phase in (RobotPhase.WALKING_TO_PICKUP, RobotPhase.WALKING_TO_ELEVATOR,
                            RobotPhase.WALKING_TO_DELIVERY):
                dist = rs.robot.speed * self.dt
                rs.total_distance += dist
                rs.total_energy += dist * rs.robot.energy_per_m
            return

        self._transition(rs)

    def _start_next_task(self, rs: RobotState):
        """开始下一个任务"""
        if not rs.task_queue:
            rs.phase = RobotPhase.IDLE
            return

        task_id = rs.task_queue[0]
        task = self.tasks[task_id]

        # 中继第二段：检查第一段是否已完成（货物是否在relay_buffer中）
        plan = self.relay_plans.get(task_id)
        if plan and plan.robot_2 == rs.id:
            # 这是中继第二段——检查relay_buffer中有没有货
            relay_ready = any(item['task_id'] == task_id
                              for item in self.relay_buffer.get(plan.relay_floor, []))
            if not relay_ready:
                rs.phase = RobotPhase.IDLE
                return  # 第一段还没送到，等

            # 货物已到中继楼层——修改任务的起点为中继楼层
            # 从buffer中取出
            buf = self.relay_buffer[plan.relay_floor]
            self.relay_buffer[plan.relay_floor] = [
                item for item in buf if item['task_id'] != task_id]

        # 动态任务：还没到达 → 保持IDLE，下一步再检查
        if task.earliest_start > self.current_time:
            rs.phase = RobotPhase.IDLE
            return

        rs.task_queue.pop(0)
        rs.current_task = task_id
        if task_id not in self.task_start_times:
            self.task_start_times[task_id] = self.current_time
        self._log_action(rs, "START_TASK", details=f"T{task_id}" +
                         (f" (relay seg2)" if plan and plan.robot_2 == rs.id else ""))

        # 确定取货楼层：中继第二段从relay_floor取
        pickup_floor = task.origin_floor
        if plan and plan.robot_2 == rs.id:
            pickup_floor = plan.relay_floor

        if rs.floor != pickup_floor:
            self._initiate_elevator_trip(rs, pickup_floor, "pickup_transfer")
        else:
            rs.phase = RobotPhase.WALKING_TO_PICKUP
            rs.phase_remaining = INTER_STATION_DISTANCE / rs.robot.speed

    def _scan_batchable_tasks(self, rs: RobotState) -> List[int]:
        """扫描队列中同楼层、可批量取货的任务（载重约束内）"""
        remaining_cap = rs.robot.payload_max - rs.carrying_weight
        batchable = []
        for tid in rs.task_queue:
            t = self.tasks[tid]
            if t.origin_floor == rs.floor and t.weight <= remaining_cap:
                if t.earliest_start <= self.current_time:
                    batchable.append(tid)
                    remaining_cap -= t.weight
        return batchable

    def _get_delivery_floor(self, task_id: int, robot_id: int) -> int:
        """获取任务的实际送货楼层（考虑中继）"""
        plan = self.relay_plans.get(task_id)
        if plan and plan.robot_1 == robot_id:
            return plan.relay_floor  # 第一段送到中继楼层
        return self.tasks[task_id].dest_floor  # 直送或第二段送到终点

    def _compute_delivery_route(self, rs: RobotState) -> List[Tuple[int, List[int]]]:
        """
        计算多站送货路线：按方向排序携带货物的目标楼层。
        考虑中继任务：第一段机器人送到relay_floor而非dest_floor。
        """
        if not rs.carrying:
            return []
        floor_tasks = defaultdict(list)
        has_cross_floor = False
        for tid in rs.carrying:
            delivery_floor = self._get_delivery_floor(tid, rs.id)
            floor_tasks[delivery_floor].append(tid)
            if delivery_floor != rs.floor:
                has_cross_floor = True

        if not has_cross_floor:
            return []

        route = sorted(floor_tasks.items(), key=lambda x: abs(x[0] - rs.floor))
        return route

    def _initiate_elevator_trip(self, rs: RobotState, target_floor: int, purpose: str):
        """发起一次电梯行程"""
        rs.target_floor = target_floor

        # 走到电梯大厅
        rs.phase = RobotPhase.WALKING_TO_ELEVATOR
        rs.phase_remaining = FLOOR_DISTANCE / rs.robot.speed
        self._log_action(rs, "WALK_TO_ELEVATOR")

    def _transition(self, rs: RobotState):
        """阶段转换逻辑"""
        task = self.tasks[rs.current_task] if rs.current_task >= 0 else None

        if rs.phase == RobotPhase.WALKING_TO_PICKUP:
            # 到达取货点 → 开始取货
            rs.phase = RobotPhase.PICKING_UP
            rs.phase_remaining = PICKUP_TIME
            self._log_action(rs, "PICKUP", task_id=rs.current_task)

        elif rs.phase == RobotPhase.PICKING_UP:
            # 取货完成 → 检查是否有同楼层可批量取的任务
            if task and rs.current_task not in rs.carrying:
                rs.carrying.append(rs.current_task)
                rs.carrying_weight += task.weight

            batch = self._scan_batchable_tasks(rs)
            if batch:
                # 批量取货：追加到carrying，额外取货时间
                for tid in batch:
                    rs.task_queue.remove(tid)
                    rs.carrying.append(tid)
                    rs.carrying_weight += self.tasks[tid].weight
                    self.task_start_times[tid] = self.current_time
                    self._log_action(rs, "BATCH_PICKUP", task_id=tid,
                                     details=f"batch +T{tid}")
                rs.phase_remaining = 5.0 * len(batch)  # 每个额外5s
                # 不转换阶段，继续PICKING_UP直到phase_remaining耗尽
                return

            # 所有同楼层任务都取完了，决定下一步
            route = self._compute_delivery_route(rs)
            if not route:
                # 只有同层任务
                if rs.carrying:
                    rs.phase = RobotPhase.WALKING_TO_DELIVERY
                    rs.phase_remaining = INTER_STATION_DISTANCE / rs.robot.speed
                else:
                    rs.phase = RobotPhase.IDLE
            else:
                # 有跨楼层任务：去第一个目标楼层
                first_floor = route[0][0]
                rs.target_floor = first_floor
                rs.delivery_route = route
                self._initiate_elevator_trip(rs, first_floor, "delivery")

        elif rs.phase == RobotPhase.WALKING_TO_ELEVATOR:
            # 到达电梯大厅 → 接近电梯（需GPU）
            rs.phase = RobotPhase.APPROACHING_ELEVATOR
            rs.phase_remaining = APPROACH_TIME
            self._log_action(rs, "APPROACH_ELEVATOR", needs_gpu=True)

        elif rs.phase == RobotPhase.APPROACHING_ELEVATOR:
            # 接近完成 → 按按钮（需GPU）
            rs.phase = RobotPhase.PRESSING_BUTTON
            rs.phase_remaining = PRESS_TIME
            self._log_action(rs, "PRESS_BUTTON", needs_gpu=True)

        elif rs.phase == RobotPhase.PRESSING_BUTTON:
            # 错峰呼叫：同楼层已有机器人在等电梯时，延迟一次
            if self.strategy.stagger_delay > 0 and not getattr(rs, '_stagger_applied', False):
                waiting_here = sum(1 for other in self.robot_states
                                   if other.id != rs.id
                                   and other.floor == rs.floor
                                   and other.phase == RobotPhase.WAITING_ELEVATOR)
                if waiting_here > 0:
                    rs.phase_remaining = self.strategy.stagger_delay * waiting_here
                    rs._stagger_applied = True
                    return
            rs._stagger_applied = False

            # 向群控系统注册呼叫 → 等待电梯（事件驱动）
            direction = Direction.UP if rs.target_floor > rs.floor else Direction.DOWN
            call = self.egcs.register_hall_call(
                floor=rs.floor, direction=direction,
                robot_id=rs.id, dest_floor=rs.target_floor,
                call_time=self.current_time, task_id=rs.current_task,
            )
            rs.pending_call = call
            rs.phase = RobotPhase.WAITING_ELEVATOR
            self._log_action(rs, "WAIT_ELEVATOR",
                             elevator_id=call.assigned_elevator,
                             details=f"assigned E{call.assigned_elevator}")

        elif rs.phase == RobotPhase.WALKING_TO_DELIVERY:
            # 到达送货点 → 送货
            rs.phase = RobotPhase.DELIVERING
            rs.phase_remaining = DELIVER_TIME
            self._log_action(rs, "DELIVER", task_id=rs.current_task)

        elif rs.phase == RobotPhase.RELAY_DROPOFF:
            # 中继放货完成 → 将任务放入中继缓冲区，通知第二段机器人
            task_id = rs.current_task
            plan = self.relay_plans.get(task_id)
            if plan:
                self.relay_buffer[rs.floor].append({
                    'task_id': task_id,
                    'weight': self.tasks[task_id].weight,
                    'dest_floor': self.tasks[task_id].dest_floor,
                    'robot_2': plan.robot_2,
                })
                # 将第二段加入robot_2的队列
                r2 = self.robot_states[plan.robot_2]
                r2.task_queue.append(task_id)
                self._log_action(rs, "RELAY_HANDOFF", task_id=task_id,
                                 details=f"→R{plan.robot_2} at F{rs.floor}")
            # 从当前机器人的carrying中移除
            if task_id in rs.carrying:
                rs.carrying.remove(task_id)
                rs.carrying_weight -= self.tasks[task_id].weight
            # 继续下一个任务
            rs.current_task = -1
            if rs.carrying:
                route = self._compute_delivery_route(rs)
                if route:
                    self._initiate_elevator_trip(rs, route[0][0], "delivery_next_stop")
                else:
                    rs.phase = RobotPhase.IDLE
            else:
                self._start_next_task(rs)

        elif rs.phase == RobotPhase.DELIVERING:
            # 送货完成——检查是否是中继任务的中间站
            task_id = rs.current_task
            plan = self.relay_plans.get(task_id)
            if plan and plan.relay_floor == rs.floor:
                # 这是中继楼层，不是最终目的地 → 放货交接
                rs.phase = RobotPhase.RELAY_DROPOFF
                rs.phase_remaining = 5.0
                self._log_action(rs, "RELAY_DROPOFF", task_id=task_id,
                                 details=f"relay at F{rs.floor}")
                return

            self._complete_task(rs, rs.current_task)

            # 检查当前楼层是否还有要送的货（多站送货，考虑中继）
            same_floor_carry = [tid for tid in rs.carrying
                                if self._get_delivery_floor(tid, rs.id) == rs.floor]
            if same_floor_carry:
                # 继续送同楼层的货
                next_tid = same_floor_carry[0]
                rs.current_task = next_tid
                rs.phase = RobotPhase.DELIVERING
                rs.phase_remaining = DELIVER_TIME
                self._log_action(rs, "DELIVER", task_id=next_tid,
                                 details="multi-stop same floor")
            elif rs.carrying:
                # 还有其他楼层的货 → 去下一个送货楼层
                route = self._compute_delivery_route(rs)
                if route:
                    next_floor = route[0][0]
                    rs.delivery_route = route
                    self._initiate_elevator_trip(rs, next_floor, "delivery_next_stop")
                else:
                    # 全是同层任务，走过去送
                    next_tid = rs.carrying[0]
                    rs.current_task = next_tid
                    rs.phase = RobotPhase.WALKING_TO_DELIVERY
                    rs.phase_remaining = INTER_STATION_DISTANCE / rs.robot.speed
            else:
                # 全部送完
                rs.current_task = -1
                rs.delivery_route = []
                self._start_next_task(rs)

        elif rs.phase == RobotPhase.EXITING_ELEVATOR:
            # 出电梯完成
            rs.floor = rs.target_floor
            self._log_action(rs, "EXIT_DONE")

            # 检查当前楼层有没有要送的货
            deliverable_here = [tid for tid in rs.carrying
                                if self.tasks[tid].dest_floor == rs.floor]
            if deliverable_here:
                # 有货要在这层送 → 走到送货站
                rs.current_task = deliverable_here[0]
                rs.phase = RobotPhase.WALKING_TO_DELIVERY
                rs.phase_remaining = FLOOR_DISTANCE / rs.robot.speed
            elif task and task.origin_floor == rs.floor and rs.current_task not in rs.carrying:
                # 到达取货楼层 → 走到取货点
                rs.phase = RobotPhase.WALKING_TO_PICKUP
                rs.phase_remaining = INTER_STATION_DISTANCE / rs.robot.speed
            else:
                # 默认：走到最近的工作站
                rs.phase = RobotPhase.WALKING_TO_DELIVERY
                rs.phase_remaining = FLOOR_DISTANCE / rs.robot.speed

    def _handle_pickup(self, call: HallCall):
        """电梯接上机器人——转为RIDING，由_handle_dropoff事件触发离开"""
        rs = self.robot_states[call.robot_id]
        if rs.phase == RobotPhase.WAITING_ELEVATOR:
            rs.phase = RobotPhase.RIDING
            # 不设倒计时——电梯到达目的楼层时由_handle_dropoff触发转换
            self._log_action(rs, "RIDE_ELEVATOR",
                             elevator_id=call.assigned_elevator,
                             details=f"F{call.floor}→F{call.dest_floor}")

    def _handle_dropoff(self, call: HallCall):
        """电梯放下机器人"""
        rs = self.robot_states[call.robot_id]
        if rs.phase == RobotPhase.RIDING:
            rs.phase = RobotPhase.EXITING_ELEVATOR
            rs.phase_remaining = EXIT_TIME
            rs.floor = call.dest_floor
            self._log_action(rs, "EXIT_ELEVATOR", needs_gpu=True,
                             elevator_id=call.assigned_elevator)

    def _complete_task(self, rs: RobotState, task_id: int):
        """完成一个任务"""
        if task_id in rs.carrying:
            rs.carrying.remove(task_id)
            rs.carrying_weight -= self.tasks[task_id].weight
        self.task_completed[task_id] = True
        self.task_completion_times[task_id] = self.current_time
        rs.tasks_completed += 1

    def _log_action(self, rs: RobotState, action: str, needs_gpu: bool = False,
                    task_id: int = -1, elevator_id: int = -1, details: str = ""):
        if task_id == -1:
            task_id = rs.current_task
        self.action_log.append(ActionLog(
            robot_id=rs.id, action=action,
            start_time=self.current_time, end_time=self.current_time,
            floor=rs.floor, needs_gpu=needs_gpu,
            task_id=task_id, elevator_id=elevator_id, details=details,
        ))

    def _collect_results(self) -> Dict:
        completed = sum(self.task_completed)
        makespan = max(self.task_completion_times.values()) if self.task_completion_times else 0

        # 加权延迟
        weighted_tardiness = 0
        for i in range(self.N):
            if i in self.task_completion_times:
                tard = max(0, self.task_completion_times[i] - self.tasks[i].deadline)
                weighted_tardiness += PRIORITY_WEIGHT[self.tasks[i].priority] * tard

        # 机器人能耗
        robot_energy = sum(rs.total_energy for rs in self.robot_states)

        # 电梯统计
        egcs_stats = self.egcs.get_statistics()

        # GPU统计
        total_gpu_time = sum(rs.total_gpu_time for rs in self.robot_states)

        # 等待时间统计
        wait_times = []
        for call in self.egcs.call_history:
            if call.pickup_time > 0:
                wait_times.append(call.pickup_time - call.call_time)

        total_energy = robot_energy + egcs_stats['total_energy_kwh'] * 3600
        return {
            "completed_tasks": completed,
            "total_tasks": self.N,
            "makespan": makespan,
            "energy": total_energy,
            "energy_robot": robot_energy,
            "energy_elevator_kwh": egcs_stats['total_energy_kwh'],
            "energy_total": total_energy,
            "weighted_tardiness": weighted_tardiness,
            "avg_elevator_wait": np.mean(wait_times) if wait_times else 0,
            "max_elevator_wait": max(wait_times) if wait_times else 0,
            "elevator_stats": egcs_stats,
            "gpu_time": total_gpu_time,
            "robot_stats": {
                rs.id: {
                    "completed": rs.tasks_completed,
                    "distance": rs.total_distance,
                    "energy": rs.total_energy,
                    "idle_time": rs.total_idle_time,
                    "gpu_time": rs.total_gpu_time,
                }
                for rs in self.robot_states
            },
        }

    def print_summary(self, result: Dict):
        print(f"\n{'='*60}")
        print(f"仿真V2结果 (基于电梯群控系统)")
        print(f"{'='*60}")
        print(f"  完成任务: {result['completed_tasks']}/{result['total_tasks']}")
        print(f"  Makespan: {result['makespan']:.1f}s ({result['makespan']/3600:.1f}h)")
        print(f"  机器人能耗: {result['energy_robot']:.1f}J")
        print(f"  电梯能耗: {result['energy_elevator_kwh']:.3f}kWh")
        print(f"  加权延迟: {result['weighted_tardiness']:.1f}")
        print(f"  平均电梯等待: {result['avg_elevator_wait']:.1f}s")
        print(f"  最大电梯等待: {result['max_elevator_wait']:.1f}s")
        print(f"  GPU总需求时长: {result['gpu_time']:.1f}s")

        es = result['elevator_stats']
        print(f"\n  电梯群控统计:")
        print(f"    总呼叫: {es['total_calls']}, 完成: {es['completed_calls']}")
        print(f"    平均等待: {es['avg_wait_time']:.1f}s")
        for eid, stats in es['per_elevator'].items():
            print(f"    E{eid}: {stats['trips']}趟, 利用率{stats['utilization']:.1%}, "
                  f"能耗{stats['energy_kwh']:.3f}kWh")
