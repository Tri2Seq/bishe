"""离散事件仿真器

逼真地模拟多机器人多楼层配送过程：
- 电梯状态机：IDLE/MOVING/DOOR_OPEN/DOOR_CLOSE
- 机器人状态机：IDLE/WALKING/WAITING_ELEVATOR/APPROACHING_ELEVATOR/
                RIDING/EXITING/PICKING/DELIVERING
- FCFS电梯请求队列
- 事件驱动推进

用途：
1. 验证MILP解的可行性（将MILP方案输入仿真，观察实际完成时间）
2. 评估启发式算法的方案质量
3. 生成精确的动作时间线供模块2使用
"""

import heapq
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import json

from config import (Robot, Elevator, Task, Priority, PRIORITY_WEIGHT,
                    FLOOR_DISTANCE, INTER_STATION_DISTANCE)


class EventType(Enum):
    ROBOT_ARRIVE_PICKUP = auto()
    ROBOT_PICKUP_DONE = auto()
    ROBOT_ARRIVE_ELEVATOR = auto()
    ROBOT_PRESS_BUTTON = auto()      # 开始按按钮（需要GPU）
    ROBOT_BOARD_ELEVATOR = auto()
    ROBOT_EXIT_ELEVATOR = auto()
    ROBOT_EXIT_DONE = auto()         # 出电梯完成（GPU释放）
    ROBOT_ARRIVE_DELIVER = auto()
    ROBOT_DELIVER_DONE = auto()
    ROBOT_TASK_COMPLETE = auto()
    ELEVATOR_ARRIVE = auto()
    ELEVATOR_DOOR_OPEN = auto()
    ELEVATOR_DOOR_CLOSE = auto()


class ElevatorState(Enum):
    IDLE = auto()
    MOVING = auto()
    DOOR_OPENING = auto()
    DOOR_CLOSING = auto()


class RobotState(Enum):
    IDLE = auto()
    WALKING = auto()
    WAITING_ELEVATOR = auto()
    APPROACHING_ELEVATOR = auto()  # 需要GPU
    PRESSING_BUTTON = auto()       # 需要GPU
    RIDING = auto()
    EXITING_ELEVATOR = auto()      # 需要GPU
    PICKING = auto()
    DELIVERING = auto()


@dataclass(order=True)
class Event:
    time: float
    priority: int = field(compare=True, default=0)
    type: EventType = field(compare=False, default=EventType.ROBOT_ARRIVE_PICKUP)
    robot_id: int = field(compare=False, default=-1)
    elevator_id: int = field(compare=False, default=-1)
    task_id: int = field(compare=False, default=-1)
    data: Any = field(compare=False, default=None)


@dataclass
class ElevatorRequest:
    robot_id: int
    from_floor: int
    to_floor: int
    request_time: float
    task_id: int


@dataclass
class ActionRecord:
    """记录一条动作用于时间线输出"""
    type: str
    start: float
    end: float
    floor: int
    robot_id: int
    needs_gpu: bool = False
    task_id: int = -1
    elevator_id: int = -1
    dest_floor: int = -1


# 时间参数
PICKUP_TIME = 10.0
DELIVER_TIME = 10.0
APPROACH_TIME = 8.0     # 接近电梯（需GPU）
PRESS_TIME = 7.0        # 按按钮（需GPU）
EXIT_TIME = 5.0         # 出电梯（需GPU）
DOOR_OPEN_TIME = 2.0
DOOR_CLOSE_TIME = 2.0


class Simulator:
    """离散事件仿真器"""

    def __init__(self, robots: List[Robot], elevators: List[Elevator], tasks: List[Task]):
        self.robots = robots
        self.elevators = elevators
        self.tasks = tasks
        self.M = len(robots)
        self.E = len(elevators)
        self.N = len(tasks)

        # 事件队列
        self.event_queue: List[Event] = []
        self.current_time = 0.0

        # 电梯状态
        self.elev_state = [ElevatorState.IDLE] * self.E
        self.elev_floor = [e.init_floor for e in elevators]
        self.elev_queue: List[List[ElevatorRequest]] = [[] for _ in range(self.E)]
        self.elev_current_passenger: List[Optional[int]] = [None] * self.E

        # 机器人状态
        self.robot_state = [RobotState.IDLE] * self.M
        self.robot_floor = [r.init_floor for r in robots]
        self.robot_current_task: List[Optional[int]] = [None] * self.M
        self.robot_task_queue: Dict[int, List[int]] = {j: [] for j in range(self.M)}

        # 统计
        self.task_start_times: Dict[int, float] = {}
        self.task_completion_times: Dict[int, float] = {}
        self.total_energy = 0.0
        self.total_elev_energy = 0.0
        self.action_records: Dict[int, List[ActionRecord]] = {j: [] for j in range(self.M)}

        # 电梯统计
        self.elev_trips = {e: 0 for e in range(self.E)}
        self.elev_busy_time = {e: 0.0 for e in range(self.E)}
        self.elev_wait_total = {e: 0.0 for e in range(self.E)}

    def _push_event(self, time: float, etype: EventType, robot_id: int = -1,
                    elevator_id: int = -1, task_id: int = -1, data=None, priority: int = 0):
        heapq.heappush(self.event_queue, Event(
            time=time, priority=priority, type=etype,
            robot_id=robot_id, elevator_id=elevator_id, task_id=task_id, data=data))

    def _record(self, robot_id: int, action_type: str, start: float, end: float,
                floor: int, needs_gpu: bool = False, task_id: int = -1,
                elevator_id: int = -1, dest_floor: int = -1):
        self.action_records[robot_id].append(ActionRecord(
            type=action_type, start=start, end=end, floor=floor,
            robot_id=robot_id, needs_gpu=needs_gpu, task_id=task_id,
            elevator_id=elevator_id, dest_floor=dest_floor))

    def load_schedule(self, assignment: Dict[int, int], task_sequence: Dict[int, List[int]],
                      elevator_assignment: Dict[int, int]):
        """加载MILP/启发式的调度方案"""
        self.assignment = assignment
        self.elevator_assignment = elevator_assignment
        for j, seq in task_sequence.items():
            self.robot_task_queue[j] = list(seq)

    def run(self, verbose: bool = False) -> Dict:
        """运行仿真"""
        # 初始化：每个有任务的机器人开始执行第一个任务
        for j in range(self.M):
            if self.robot_task_queue[j]:
                self._start_next_task(j, 0.0)

        # 事件循环
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time

            if verbose and event.type not in (EventType.ELEVATOR_ARRIVE,):
                pass  # 可选：打印事件日志

            self._handle_event(event)

        # 计算结果
        return self._compute_results()

    def _start_next_task(self, robot_id: int, time: float):
        """机器人开始执行下一个任务"""
        if not self.robot_task_queue[robot_id]:
            self.robot_state[robot_id] = RobotState.IDLE
            return

        task_id = self.robot_task_queue[robot_id].pop(0)
        self.robot_current_task[robot_id] = task_id
        task = self.tasks[task_id]

        # 是否需要先移动到取货楼层
        if self.robot_floor[robot_id] != task.origin_floor:
            self._go_to_floor(robot_id, task.origin_floor, time, task_id, "goto_pickup")
        else:
            walk_time = INTER_STATION_DISTANCE / self.robots[robot_id].speed
            self.robot_state[robot_id] = RobotState.WALKING
            self._record(robot_id, "WALK", time, time + walk_time,
                         self.robot_floor[robot_id], task_id=task_id)
            start_time = max(time + walk_time, task.earliest_start)
            self._push_event(start_time, EventType.ROBOT_ARRIVE_PICKUP,
                             robot_id=robot_id, task_id=task_id)

    def _go_to_floor(self, robot_id: int, target_floor: int, time: float,
                     task_id: int, purpose: str):
        """机器人需要去另一个楼层（走到电梯→乘电梯→到达）"""
        robot = self.robots[robot_id]
        walk_to_elev = FLOOR_DISTANCE / robot.speed

        self.robot_state[robot_id] = RobotState.WALKING
        self._record(robot_id, "WALK", time, time + walk_to_elev,
                     self.robot_floor[robot_id], task_id=task_id)

        # 接近电梯（需要GPU）
        approach_start = time + walk_to_elev
        self._record(robot_id, "APPROACH_ELEVATOR", approach_start,
                     approach_start + APPROACH_TIME, self.robot_floor[robot_id],
                     needs_gpu=True, task_id=task_id)

        # 按按钮（需要GPU）
        press_start = approach_start + APPROACH_TIME
        self._record(robot_id, "PRESS_BUTTON", press_start,
                     press_start + PRESS_TIME, self.robot_floor[robot_id],
                     needs_gpu=True, task_id=task_id)

        # 请求电梯
        request_time = press_start + PRESS_TIME
        # 选择电梯（用调度方案指定的，或选最近的）
        if task_id in self.elevator_assignment and purpose == "goto_deliver":
            elev_id = self.elevator_assignment[task_id]
        else:
            elev_id = self._select_nearest_elevator(self.robot_floor[robot_id])

        self._request_elevator(robot_id, elev_id, self.robot_floor[robot_id],
                               target_floor, request_time, task_id)

    def _select_nearest_elevator(self, floor: int) -> int:
        """选择最近的空闲电梯，如果都忙则选队列最短的"""
        best = 0
        best_score = float('inf')
        for e in range(self.E):
            if self.elev_state[e] == ElevatorState.IDLE:
                dist = abs(self.elev_floor[e] - floor)
                if dist < best_score:
                    best_score = dist
                    best = e
            else:
                queue_len = len(self.elev_queue[e]) + (1 if self.elev_current_passenger[e] is not None else 0)
                score = 1000 + queue_len * 100
                if score < best_score:
                    best_score = score
                    best = e
        return best

    def _request_elevator(self, robot_id: int, elev_id: int, from_floor: int,
                          to_floor: int, time: float, task_id: int):
        """向电梯发送请求"""
        req = ElevatorRequest(robot_id, from_floor, to_floor, time, task_id)
        self.robot_state[robot_id] = RobotState.WAITING_ELEVATOR

        self._record(robot_id, "WAIT_ELEVATOR", time, time,  # end会在boarding时更新
                     from_floor, task_id=task_id, elevator_id=elev_id)

        self.elev_queue[elev_id].append(req)

        # 如果电梯空闲，立即开始服务
        if self.elev_state[elev_id] == ElevatorState.IDLE:
            self._dispatch_elevator(elev_id, time)

    def _dispatch_elevator(self, elev_id: int, time: float):
        """调度电梯去服务下一个请求"""
        if not self.elev_queue[elev_id]:
            self.elev_state[elev_id] = ElevatorState.IDLE
            return

        # FCFS: 取队列第一个
        req = self.elev_queue[elev_id][0]

        # 电梯移动到请求楼层
        travel_time = abs(self.elev_floor[elev_id] - req.from_floor) / self.elevators[elev_id].speed
        if travel_time > 0.01:
            self.elev_state[elev_id] = ElevatorState.MOVING
            arrive_time = time + travel_time
            self._push_event(arrive_time, EventType.ELEVATOR_ARRIVE,
                             elevator_id=elev_id, data=("pickup", req))
        else:
            # 电梯已在请求楼层
            self._elevator_ready_for_boarding(elev_id, req, time)

    def _elevator_ready_for_boarding(self, elev_id: int, req: ElevatorRequest, time: float):
        """电梯到达请求楼层，准备载客"""
        self.elev_floor[elev_id] = req.from_floor
        door_time = self.elevators[elev_id].door_time / 2  # 开门时间

        self.elev_state[elev_id] = ElevatorState.DOOR_OPENING
        self._push_event(time + door_time, EventType.ROBOT_BOARD_ELEVATOR,
                         robot_id=req.robot_id, elevator_id=elev_id,
                         task_id=req.task_id, data=req)

    def _handle_event(self, event: Event):
        """处理事件"""
        handlers = {
            EventType.ROBOT_ARRIVE_PICKUP: self._on_arrive_pickup,
            EventType.ROBOT_PICKUP_DONE: self._on_pickup_done,
            EventType.ROBOT_BOARD_ELEVATOR: self._on_board_elevator,
            EventType.ELEVATOR_ARRIVE: self._on_elevator_arrive,
            EventType.ROBOT_EXIT_ELEVATOR: self._on_exit_elevator,
            EventType.ROBOT_EXIT_DONE: self._on_exit_done,
            EventType.ROBOT_ARRIVE_DELIVER: self._on_arrive_deliver,
            EventType.ROBOT_DELIVER_DONE: self._on_deliver_done,
        }
        handler = handlers.get(event.type)
        if handler:
            handler(event)

    def _on_arrive_pickup(self, event: Event):
        t, rid, tid = event.time, event.robot_id, event.task_id
        self.task_start_times[tid] = t
        self.robot_state[rid] = RobotState.PICKING
        self._record(rid, "PICKUP", t, t + PICKUP_TIME, self.robot_floor[rid], task_id=tid)
        self._push_event(t + PICKUP_TIME, EventType.ROBOT_PICKUP_DONE,
                         robot_id=rid, task_id=tid)

    def _on_pickup_done(self, event: Event):
        t, rid, tid = event.time, event.robot_id, event.task_id
        task = self.tasks[tid]

        if task.is_cross_floor:
            elev_id = self.elevator_assignment.get(tid, self._select_nearest_elevator(task.origin_floor))
            self._go_to_floor_for_delivery(rid, task, elev_id, t, tid)
        else:
            walk = INTER_STATION_DISTANCE / self.robots[rid].speed
            self._record(rid, "WALK", t, t + walk, self.robot_floor[rid], task_id=tid)
            self._push_event(t + walk, EventType.ROBOT_ARRIVE_DELIVER,
                             robot_id=rid, task_id=tid)

    def _go_to_floor_for_delivery(self, robot_id: int, task: Task,
                                   elev_id: int, time: float, task_id: int):
        """取货完成后去电梯送货"""
        robot = self.robots[robot_id]
        walk_to_elev = FLOOR_DISTANCE / robot.speed

        self.robot_state[robot_id] = RobotState.WALKING
        self._record(robot_id, "WALK", time, time + walk_to_elev,
                     self.robot_floor[robot_id], task_id=task_id)

        # 接近电梯
        approach_start = time + walk_to_elev
        self._record(robot_id, "APPROACH_ELEVATOR", approach_start,
                     approach_start + APPROACH_TIME, self.robot_floor[robot_id],
                     needs_gpu=True, task_id=task_id, elevator_id=elev_id)

        press_start = approach_start + APPROACH_TIME
        self._record(robot_id, "PRESS_BUTTON", press_start,
                     press_start + PRESS_TIME, self.robot_floor[robot_id],
                     needs_gpu=True, task_id=task_id, elevator_id=elev_id)

        request_time = press_start + PRESS_TIME
        req = ElevatorRequest(robot_id, task.origin_floor, task.dest_floor,
                              request_time, task_id)
        self.robot_state[robot_id] = RobotState.WAITING_ELEVATOR
        self._record(robot_id, "WAIT_ELEVATOR", request_time, request_time,
                     self.robot_floor[robot_id], task_id=task_id, elevator_id=elev_id)

        self.elev_queue[elev_id].append(req)
        if self.elev_state[elev_id] == ElevatorState.IDLE:
            self._dispatch_elevator(elev_id, request_time)

    def _on_board_elevator(self, event: Event):
        t, rid, eid = event.time, event.robot_id, event.elevator_id
        req: ElevatorRequest = event.data

        # 更新等待记录的结束时间
        for rec in reversed(self.action_records[rid]):
            if rec.type == "WAIT_ELEVATOR" and rec.end == rec.start:
                rec.end = t
                self.elev_wait_total[eid] += t - rec.start
                break

        # 从队列中移除
        self.elev_queue[eid] = [r for r in self.elev_queue[eid] if r.robot_id != rid]
        self.elev_current_passenger[eid] = rid

        # 电梯关门移动
        elev = self.elevators[eid]
        ride_time = abs(req.to_floor - req.from_floor) / elev.speed
        door_close = elev.door_time / 2

        self.elev_state[eid] = ElevatorState.MOVING
        self.robot_state[rid] = RobotState.RIDING

        self._record(rid, "RIDE_ELEVATOR", t + door_close,
                     t + door_close + ride_time, req.from_floor,
                     task_id=req.task_id, elevator_id=eid, dest_floor=req.to_floor)

        self.elev_trips[eid] += 1
        self.elev_busy_time[eid] += door_close + ride_time + elev.door_time / 2
        self.total_elev_energy += abs(req.to_floor - req.from_floor) * 10.0

        arrive_time = t + door_close + ride_time
        self._push_event(arrive_time, EventType.ROBOT_EXIT_ELEVATOR,
                         robot_id=rid, elevator_id=eid, task_id=req.task_id,
                         data=req.to_floor)

    def _on_elevator_arrive(self, event: Event):
        t, eid = event.time, event.elevator_id
        purpose, req = event.data
        self.elev_floor[eid] = req.from_floor
        self._elevator_ready_for_boarding(eid, req, t)

    def _on_exit_elevator(self, event: Event):
        t, rid, eid = event.time, event.robot_id, event.elevator_id
        dest_floor = event.data

        self.robot_floor[rid] = dest_floor
        self.elev_floor[eid] = dest_floor
        self.elev_current_passenger[eid] = None
        self.robot_state[rid] = RobotState.EXITING_ELEVATOR

        # 出电梯导航（需GPU）
        door_open = self.elevators[eid].door_time / 2
        exit_start = t + door_open
        self._record(rid, "EXIT_ELEVATOR", exit_start, exit_start + EXIT_TIME,
                     dest_floor, needs_gpu=True, task_id=event.task_id, elevator_id=eid)

        self._push_event(exit_start + EXIT_TIME, EventType.ROBOT_EXIT_DONE,
                         robot_id=rid, elevator_id=eid, task_id=event.task_id)

        # 电梯空闲，调度下一个请求
        self.elev_state[eid] = ElevatorState.IDLE
        if self.elev_queue[eid]:
            self._dispatch_elevator(eid, t + door_open)

    def _on_exit_done(self, event: Event):
        t, rid, tid = event.time, event.robot_id, event.task_id

        # 走到送货站
        walk = FLOOR_DISTANCE / self.robots[rid].speed
        self._record(rid, "WALK", t, t + walk, self.robot_floor[rid], task_id=tid)
        self._push_event(t + walk, EventType.ROBOT_ARRIVE_DELIVER,
                         robot_id=rid, task_id=tid)

    def _on_arrive_deliver(self, event: Event):
        t, rid, tid = event.time, event.robot_id, event.task_id
        self.robot_state[rid] = RobotState.DELIVERING
        self._record(rid, "DELIVER", t, t + DELIVER_TIME, self.robot_floor[rid], task_id=tid)
        self._push_event(t + DELIVER_TIME, EventType.ROBOT_DELIVER_DONE,
                         robot_id=rid, task_id=tid)

    def _on_deliver_done(self, event: Event):
        t, rid, tid = event.time, event.robot_id, event.task_id
        self.task_completion_times[tid] = t
        self.robot_floor[rid] = self.tasks[tid].dest_floor

        # 能耗
        task = self.tasks[tid]
        robot = self.robots[rid]
        dist = FLOOR_DISTANCE * 2 + INTER_STATION_DISTANCE if task.is_cross_floor else INTER_STATION_DISTANCE
        self.total_energy += dist * robot.energy_per_m

        # 执行下一个任务
        self._start_next_task(rid, t)

    def _compute_results(self) -> Dict:
        makespan = max(self.task_completion_times.values()) if self.task_completion_times else 0
        weighted_tardiness = sum(
            PRIORITY_WEIGHT[self.tasks[i].priority] *
            max(0, self.task_completion_times.get(i, 0) - self.tasks[i].deadline)
            for i in range(self.N)
        )

        # GPU统计
        total_gpu_time = 0
        total_gpu_demands = 0
        total_ride_time = 0
        for rid, records in self.action_records.items():
            for rec in records:
                if rec.needs_gpu:
                    total_gpu_time += rec.end - rec.start
                    total_gpu_demands += 1
                if rec.type == "RIDE_ELEVATOR":
                    total_ride_time += rec.end - rec.start

        return {
            "makespan": makespan,
            "energy": self.total_energy + self.total_elev_energy,
            "weighted_tardiness": weighted_tardiness,
            "task_completions": dict(self.task_completion_times),
            "task_starts": dict(self.task_start_times),
            "elevator_trips": dict(self.elev_trips),
            "elevator_busy_time": dict(self.elev_busy_time),
            "elevator_wait_total": dict(self.elev_wait_total),
            "gpu_demands": total_gpu_demands,
            "gpu_time": total_gpu_time,
            "ride_time": total_ride_time,
            "completed_tasks": len(self.task_completion_times),
        }

    def get_timelines_json(self) -> Dict:
        """导出时间线为JSON格式（供模块2使用）"""
        result = {}
        for rid, records in self.action_records.items():
            result[str(rid)] = [
                {
                    "type": r.type,
                    "start": round(r.start, 2),
                    "end": round(r.end, 2),
                    "floor": r.floor,
                    "robot_id": r.robot_id,
                    "needs_gpu": r.needs_gpu,
                    **({"task_id": r.task_id} if r.task_id >= 0 else {}),
                    **({"elevator_id": r.elevator_id} if r.elevator_id >= 0 else {}),
                    **({"dest_floor": r.dest_floor} if r.dest_floor >= 0 else {}),
                }
                for r in records
            ]
        return result

    def print_summary(self, result: Dict):
        print(f"\n{'='*60}")
        print(f"仿真结果")
        print(f"{'='*60}")
        print(f"  完成任务: {result['completed_tasks']}/{self.N}")
        print(f"  Makespan: {result['makespan']:.1f}s")
        print(f"  总能耗: {result['energy']:.1f}J")
        print(f"  加权延迟: {result['weighted_tardiness']:.1f}")
        print(f"\n  电梯统计:")
        for e in range(self.E):
            trips = result['elevator_trips'].get(e, 0)
            busy = result['elevator_busy_time'].get(e, 0)
            wait = result['elevator_wait_total'].get(e, 0)
            print(f"    E{e}: {trips}趟, 忙碌{busy:.1f}s, 总等待{wait:.1f}s")
        print(f"\n  GPU统计:")
        print(f"    GPU需求段数: {result['gpu_demands']}")
        print(f"    GPU需求总时长: {result['gpu_time']:.1f}s")
        print(f"    乘梯(GPU释放窗口): {result['ride_time']:.1f}s")
