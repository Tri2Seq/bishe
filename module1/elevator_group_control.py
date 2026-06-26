"""电梯群控系统 (Elevator Group Control System, EGCS)

真实的电梯群控调度建模：
  当机器人在某楼层按下电梯呼叫按钮（上行/下行），电梯群控系统
  根据派车算法决定派哪一部电梯响应。机器人不选择电梯——电梯系统选择。

实现3种经典派车算法：
  1. ETA (Estimated Time of Arrival) — 最短预估到达时间
  2. SCAN — 电梯沿一个方向扫描，到头再折返（类似磁盘SCAN）
  3. Nearest Car (NC) — 最近电梯优先

电梯状态机：
  IDLE → 收到呼叫 → MOVING_TO_CALL → 到达呼叫层 → DOOR_OPEN →
  乘客登入 → DOOR_CLOSE → MOVING_TO_DEST → 到达目的层 → DOOR_OPEN →
  乘客离开 → 检查队列 → IDLE 或继续服务

参考文献：
  - Barney, 2003. Elevator Traffic Handbook
  - Koehler & Otis, 2010. Group elevator scheduling
"""

import heapq
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
import numpy as np


class Direction(Enum):
    UP = 1
    DOWN = -1
    IDLE = 0


class ElevatorPhase(Enum):
    IDLE = auto()
    MOVING_TO_CALL = auto()     # 空驶去接人
    DOOR_OPENING = auto()
    LOADING = auto()            # 等待乘客登入
    DOOR_CLOSING = auto()
    MOVING_TO_DEST = auto()     # 载客运行
    UNLOADING = auto()          # 等待乘客离开


@dataclass
class HallCall:
    """楼层呼叫（机器人按了电梯按钮）"""
    floor: int
    direction: Direction       # UP 或 DOWN
    robot_id: int
    call_time: float
    dest_floor: int            # 机器人要去的楼层
    task_id: int = -1
    assigned_elevator: int = -1  # 群控系统分配的电梯
    pickup_time: float = -1     # 实际被接上的时间
    dropoff_time: float = -1    # 实际到达的时间


@dataclass
class ElevatorCar:
    """单部电梯状态"""
    id: int
    current_floor: float       # 当前楼层（可以是小数表示在运行中）
    direction: Direction = Direction.IDLE
    phase: ElevatorPhase = ElevatorPhase.IDLE
    speed: float = 0.5         # 层/秒
    door_time: float = 3.0     # 单次开门/关门时间(s)
    capacity: int = 1          # 最大同时搭载机器人数
    passengers: List[HallCall] = field(default_factory=list)  # 当前搭载
    call_queue: List[HallCall] = field(default_factory=list)  # 等待服务的呼叫
    stops: List[int] = field(default_factory=list)            # 计划停靠楼层

    # 统计
    total_trips: int = 0
    total_distance: float = 0.0  # 总运行层数
    total_idle_time: float = 0.0
    total_busy_time: float = 0.0
    energy_consumed: float = 0.0  # kWh

    @property
    def is_idle(self) -> bool:
        return self.phase == ElevatorPhase.IDLE and not self.passengers and not self.call_queue

    @property
    def load(self) -> int:
        return len(self.passengers)

    @property
    def has_capacity(self) -> bool:
        return self.load < self.capacity

    def travel_time(self, from_floor: int, to_floor: int) -> float:
        return abs(from_floor - to_floor) / self.speed

    def energy_per_floor(self, loaded: bool = True) -> float:
        """每层楼的能耗(kWh): 空载 vs 满载"""
        return 0.015 if loaded else 0.008  # 典型客梯能耗


class DispatchAlgorithm(Enum):
    ETA = "ETA"       # 最短预估到达时间
    SCAN = "SCAN"     # 扫描算法
    NC = "NC"         # 最近电梯


class ElevatorGroupController:
    """
    电梯群控系统

    核心职责：
      1. 接收楼层呼叫(hall call)
      2. 基于派车算法将呼叫分配给某部电梯
      3. 管理电梯运行状态
      4. 计算统计数据（等待时间、能耗等）
    """

    def __init__(self, num_elevators: int, num_floors: int,
                 algorithm: DispatchAlgorithm = DispatchAlgorithm.ETA,
                 elevator_configs: Optional[List[Dict]] = None):
        self.num_floors = num_floors
        self.algorithm = algorithm
        self.current_time = 0.0

        # 初始化电梯
        self.elevators: List[ElevatorCar] = []
        for i in range(num_elevators):
            cfg = elevator_configs[i] if elevator_configs and i < len(elevator_configs) else {}
            car = ElevatorCar(
                id=i,
                current_floor=cfg.get('init_floor', 1),
                speed=cfg.get('speed', 0.5),
                door_time=cfg.get('door_time', 3.0),
                capacity=cfg.get('capacity', 1),
            )
            self.elevators.append(car)

        # 全局呼叫历史
        self.call_history: List[HallCall] = []
        self.active_calls: Dict[int, HallCall] = {}  # call_id -> HallCall
        self.next_call_id = 0

        # 事件队列
        self.event_queue: List[Tuple[float, int, str, dict]] = []  # (time, priority, type, data)

    def register_hall_call(self, floor: int, direction: Direction,
                           robot_id: int, dest_floor: int,
                           call_time: float, task_id: int = -1) -> HallCall:
        """
        注册一个楼层呼叫。群控系统决定派哪部电梯。

        Returns:
            HallCall对象，其中assigned_elevator已填充
        """
        call = HallCall(
            floor=floor, direction=direction,
            robot_id=robot_id, dest_floor=dest_floor,
            call_time=call_time, task_id=task_id,
        )

        # 派车
        assigned = self._dispatch(call)
        call.assigned_elevator = assigned

        # 加入该电梯的队列
        self.elevators[assigned].call_queue.append(call)
        self._update_stops(assigned)

        self.call_history.append(call)
        call_id = self.next_call_id
        self.active_calls[call_id] = call
        self.next_call_id += 1

        return call

    def _dispatch(self, call: HallCall) -> int:
        """根据选定的派车算法分配电梯"""
        if self.algorithm == DispatchAlgorithm.ETA:
            return self._dispatch_eta(call)
        elif self.algorithm == DispatchAlgorithm.NC:
            return self._dispatch_nearest(call)
        elif self.algorithm == DispatchAlgorithm.SCAN:
            return self._dispatch_scan(call)
        return 0

    def _dispatch_eta(self, call: HallCall) -> int:
        """
        ETA派车：选预估到达时间最短的电梯。

        ETA计算考虑：
          1. 电梯当前位置到呼叫层的距离
          2. 电梯当前方向（顺路 vs 需折返）
          3. 中间需要停靠的站点数（每站增加开关门时间）
          4. 当前负载（满载不可接）
        """
        best_elev = 0
        best_eta = float('inf')

        for car in self.elevators:
            eta = self._estimate_arrival(car, call.floor, call.direction)
            if eta < best_eta and car.has_capacity:
                best_eta = eta
                best_elev = car.id

        return best_elev

    def _estimate_arrival(self, car: ElevatorCar, target_floor: int,
                           target_dir: Direction) -> float:
        """估算电梯到达目标楼层的时间"""
        if car.is_idle:
            return car.travel_time(int(car.current_floor), target_floor) + car.door_time

        # 电梯正在运行
        dist = abs(car.current_floor - target_floor)
        direct_time = dist / car.speed

        if car.direction == Direction.IDLE:
            return direct_time + car.door_time

        # 检查是否顺路
        if car.direction == Direction.UP:
            if target_floor >= car.current_floor and target_dir == Direction.UP:
                # 顺路向上
                intermediate_stops = sum(1 for s in car.stops
                                          if car.current_floor < s < target_floor)
                return direct_time + intermediate_stops * (car.door_time * 2) + car.door_time
            else:
                # 需要折返：先走到最高站再下来
                max_stop = max(car.stops) if car.stops else int(car.current_floor)
                up_time = (max_stop - car.current_floor) / car.speed
                down_time = abs(max_stop - target_floor) / car.speed
                stops = len([s for s in car.stops if s > car.current_floor])
                return up_time + down_time + stops * (car.door_time * 2) + car.door_time

        elif car.direction == Direction.DOWN:
            if target_floor <= car.current_floor and target_dir == Direction.DOWN:
                intermediate_stops = sum(1 for s in car.stops
                                          if target_floor < s < car.current_floor)
                return direct_time + intermediate_stops * (car.door_time * 2) + car.door_time
            else:
                min_stop = min(car.stops) if car.stops else int(car.current_floor)
                down_time = (car.current_floor - min_stop) / car.speed
                up_time = abs(target_floor - min_stop) / car.speed
                stops = len([s for s in car.stops if s < car.current_floor])
                return down_time + up_time + stops * (car.door_time * 2) + car.door_time

        return direct_time + car.door_time

    def _dispatch_nearest(self, call: HallCall) -> int:
        """最近电梯优先"""
        best = min(range(len(self.elevators)),
                   key=lambda i: abs(self.elevators[i].current_floor - call.floor)
                   if self.elevators[i].has_capacity else float('inf'))
        return best

    def _dispatch_scan(self, call: HallCall) -> int:
        """SCAN派车：选正在朝呼叫方向运行且未经过呼叫层的电梯"""
        candidates = []
        for car in self.elevators:
            if not car.has_capacity:
                continue
            if car.is_idle:
                candidates.append((car.id, abs(car.current_floor - call.floor)))
            elif car.direction == call.direction:
                if (call.direction == Direction.UP and car.current_floor <= call.floor) or \
                   (call.direction == Direction.DOWN and car.current_floor >= call.floor):
                    candidates.append((car.id, abs(car.current_floor - call.floor)))

        if candidates:
            return min(candidates, key=lambda x: x[1])[0]
        return self._dispatch_nearest(call)

    def _update_stops(self, elev_id: int):
        """更新电梯的停靠计划"""
        car = self.elevators[elev_id]
        stops = set()
        for call in car.call_queue:
            stops.add(call.floor)
        for passenger in car.passengers:
            stops.add(passenger.dest_floor)
        car.stops = sorted(stops)

    def step(self, dt: float):
        """推进仿真时间dt秒，更新所有电梯状态"""
        self.current_time += dt

        for car in self.elevators:
            self._step_elevator(car, dt)

    def _step_elevator(self, car: ElevatorCar, dt: float):
        """更新单部电梯状态"""
        if car.phase == ElevatorPhase.IDLE:
            if car.call_queue:
                # 有待处理的呼叫 → 开始移动
                next_call = car.call_queue[0]
                if abs(car.current_floor - next_call.floor) < 0.01:
                    car.phase = ElevatorPhase.DOOR_OPENING
                else:
                    car.phase = ElevatorPhase.MOVING_TO_CALL
                    car.direction = Direction.UP if next_call.floor > car.current_floor else Direction.DOWN
            else:
                car.total_idle_time += dt

        elif car.phase == ElevatorPhase.MOVING_TO_CALL:
            if not car.call_queue:
                car.phase = ElevatorPhase.IDLE
                car.direction = Direction.IDLE
                return
            target = car.call_queue[0].floor
            if abs(car.current_floor - target) < 0.05:
                car.current_floor = float(target)
                car.phase = ElevatorPhase.DOOR_OPENING
            else:
                direction = 1 if target > car.current_floor else -1
                move = car.speed * dt * direction
                car.current_floor += move
                car.total_distance += abs(move)
                car.energy_consumed += abs(move) * car.energy_per_floor(loaded=False)
                car.total_busy_time += dt
                if (direction > 0 and car.current_floor >= target - 0.05) or \
                   (direction < 0 and car.current_floor <= target + 0.05):
                    car.current_floor = float(target)
                    car.phase = ElevatorPhase.DOOR_OPENING

        elif car.phase == ElevatorPhase.MOVING_TO_DEST:
            if not car.stops and not car.passengers:
                car.phase = ElevatorPhase.IDLE
                car.direction = Direction.IDLE
                return

            # 如果stops为空但有乘客，重建stops
            if not car.stops and car.passengers:
                car.stops = sorted(set(p.dest_floor for p in car.passengers))

            if not car.stops:
                car.phase = ElevatorPhase.IDLE
                car.direction = Direction.IDLE
                return

            # 找当前方向的下一站
            if car.direction == Direction.UP:
                above = [s for s in car.stops if s > car.current_floor + 0.05]
                next_stop = min(above) if above else car.stops[-1]
            else:
                below = [s for s in car.stops if s < car.current_floor - 0.05]
                next_stop = max(below) if below else car.stops[0]

            # 移动
            if abs(car.current_floor - next_stop) < 0.05:
                car.current_floor = float(next_stop)
                car.phase = ElevatorPhase.DOOR_OPENING
            else:
                direction = 1 if next_stop > car.current_floor else -1
                move = car.speed * dt * direction
                car.current_floor += move
                car.total_distance += abs(move)
                car.energy_consumed += abs(move) * car.energy_per_floor(loaded=True)
                car.total_busy_time += dt

                # 检查是否到达或越过
                if (direction > 0 and car.current_floor >= next_stop - 0.05) or \
                   (direction < 0 and car.current_floor <= next_stop + 0.05):
                    car.current_floor = float(next_stop)
                    car.phase = ElevatorPhase.DOOR_OPENING

    def pickup_at_floor(self, car: ElevatorCar) -> List[HallCall]:
        """电梯在当前楼层接人"""
        floor = int(round(car.current_floor))
        picked_up = []
        remaining = []
        for call in car.call_queue:
            if call.floor == floor and car.has_capacity:
                call.pickup_time = self.current_time
                car.passengers.append(call)
                picked_up.append(call)
            else:
                remaining.append(call)
        car.call_queue = remaining
        self._update_stops(car.id)
        return picked_up

    def dropoff_at_floor(self, car: ElevatorCar) -> List[HallCall]:
        """电梯在当前楼层放人"""
        floor = int(round(car.current_floor))
        dropped = []
        remaining = []
        for p in car.passengers:
            if p.dest_floor == floor:
                p.dropoff_time = self.current_time
                dropped.append(p)
                car.total_trips += 1
            else:
                remaining.append(p)
        car.passengers = remaining
        self._update_stops(car.id)
        return dropped

    def get_statistics(self) -> Dict:
        """获取群控系统统计"""
        completed = [c for c in self.call_history if c.dropoff_time > 0]
        wait_times = [c.pickup_time - c.call_time for c in completed if c.pickup_time > 0]
        ride_times = [c.dropoff_time - c.pickup_time for c in completed if c.dropoff_time > 0]

        return {
            "total_calls": len(self.call_history),
            "completed_calls": len(completed),
            "avg_wait_time": np.mean(wait_times) if wait_times else 0,
            "max_wait_time": max(wait_times) if wait_times else 0,
            "avg_ride_time": np.mean(ride_times) if ride_times else 0,
            "total_energy_kwh": sum(c.energy_consumed for c in self.elevators),
            "per_elevator": {
                car.id: {
                    "trips": car.total_trips,
                    "distance_floors": car.total_distance,
                    "busy_time": car.total_busy_time,
                    "idle_time": car.total_idle_time,
                    "energy_kwh": car.energy_consumed,
                    "utilization": car.total_busy_time / max(car.total_busy_time + car.total_idle_time, 1),
                }
                for car in self.elevators
            },
        }
