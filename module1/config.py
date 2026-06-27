"""场景配置与问题参数"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple, Optional
import random
import numpy as np


class RobotType(IntEnum):
    HEAVY = 0   # 重载型
    STANDARD = 1  # 标准型
    EXPRESS = 2   # 快递型


class ElevatorType(IntEnum):
    FREIGHT = 0  # 货梯
    PASSENGER = 1  # 客梯


class Priority(IntEnum):
    URGENT = 1
    NORMAL = 2
    BATCH = 3


PRIORITY_WEIGHT = {Priority.URGENT: 10, Priority.NORMAL: 3, Priority.BATCH: 1}

ROBOT_SPECS = {
    RobotType.HEAVY:    {"payload": 80, "speed": 0.8, "range": 3000, "energy_per_m": 5.0},
    RobotType.STANDARD: {"payload": 30, "speed": 1.5, "range": 2500, "energy_per_m": 3.0},
    RobotType.EXPRESS:  {"payload": 10, "speed": 2.5, "range": 2000, "energy_per_m": 2.0},
}

ELEVATOR_SPECS = {
    ElevatorType.FREIGHT:   {"capacity": 2, "speed": 0.4, "door_time": 8},
    ElevatorType.PASSENGER: {"capacity": 1, "speed": 0.6, "door_time": 5},
}

FLOOR_DISTANCE = 50.0  # 每层楼内站点到电梯大厅的平均距离(m)
INTER_STATION_DISTANCE = 30.0  # 同层站点间平均距离(m)


@dataclass
class Robot:
    id: int
    type: RobotType
    payload_max: float
    speed: float
    range_max: float
    energy_per_m: float
    init_floor: int

    @classmethod
    def from_type(cls, id: int, rtype: RobotType, init_floor: int):
        spec = ROBOT_SPECS[rtype]
        return cls(id=id, type=rtype, payload_max=spec["payload"],
                   speed=spec["speed"], range_max=spec["range"],
                   energy_per_m=spec["energy_per_m"], init_floor=init_floor)


@dataclass
class Elevator:
    id: int
    type: ElevatorType
    capacity: int
    speed: float
    door_time: float
    init_floor: int

    @classmethod
    def from_type(cls, id: int, etype: ElevatorType, init_floor: int):
        spec = ELEVATOR_SPECS[etype]
        return cls(id=id, type=etype, capacity=spec["capacity"],
                   speed=spec["speed"], door_time=spec["door_time"],
                   init_floor=init_floor)

    def travel_time(self, floor_a: int, floor_b: int) -> float:
        return abs(floor_a - floor_b) / self.speed + self.door_time


@dataclass
class Task:
    id: int
    origin_floor: int
    dest_floor: int
    weight: float
    priority: Priority
    deadline: float
    earliest_start: float = 0.0

    @property
    def is_cross_floor(self) -> bool:
        return self.origin_floor != self.dest_floor

    @property
    def floor_diff(self) -> int:
        return abs(self.dest_floor - self.origin_floor)


@dataclass
class TaskSegment:
    """中继拆分后的子任务段"""
    seg_id: int
    original_task_id: int
    origin_floor: int
    dest_floor: int
    weight: float
    priority: Priority
    deadline: float
    earliest_start: float = 0.0
    is_first_seg: bool = True
    predecessor_seg_id: int = -1   # 第二段指向第一段的seg_id

    @property
    def is_cross_floor(self) -> bool:
        return self.origin_floor != self.dest_floor

    @property
    def floor_diff(self) -> int:
        return abs(self.dest_floor - self.origin_floor)


@dataclass
class RelayPlan:
    """中继决策"""
    task_id: int
    relay_floor: int
    robot_1: int
    robot_2: int


def split_task(task: Task, relay_floor: int, base_seg_id: int) -> Tuple[TaskSegment, TaskSegment]:
    """将一个任务在relay_floor处拆成两段

    Returns: (seg1: origin→relay, seg2: relay→dest)
    """
    seg1 = TaskSegment(
        seg_id=base_seg_id,
        original_task_id=task.id,
        origin_floor=task.origin_floor,
        dest_floor=relay_floor,
        weight=task.weight,
        priority=task.priority,
        deadline=task.deadline,
        earliest_start=task.earliest_start,
        is_first_seg=True,
    )
    seg2 = TaskSegment(
        seg_id=base_seg_id + 1,
        original_task_id=task.id,
        origin_floor=relay_floor,
        dest_floor=task.dest_floor,
        weight=task.weight,
        priority=task.priority,
        deadline=task.deadline,
        earliest_start=task.earliest_start,
        is_first_seg=False,
        predecessor_seg_id=base_seg_id,
    )
    return seg1, seg2


@dataclass
class ScenarioConfig:
    num_floors: int
    num_robots: int
    num_elevators: int
    num_tasks: int
    robot_type_ratio: Tuple[float, float, float] = (0.3, 0.5, 0.2)
    elevator_type_ratio: Tuple[float, float] = (0.4, 0.6)
    cross_floor_task_ratio: float = 0.75
    seed: int = 42


def generate_scenario(cfg: ScenarioConfig):
    """根据配置生成完整场景实例"""
    rng = np.random.RandomState(cfg.seed)

    # 生成机器人
    robots = []
    type_counts = [
        int(cfg.num_robots * cfg.robot_type_ratio[0]),
        int(cfg.num_robots * cfg.robot_type_ratio[1]),
        0,
    ]
    type_counts[2] = cfg.num_robots - type_counts[0] - type_counts[1]
    rid = 0
    for rtype, count in zip(RobotType, type_counts):
        for _ in range(count):
            init_floor = rng.randint(1, cfg.num_floors + 1)
            robots.append(Robot.from_type(rid, rtype, init_floor))
            rid += 1

    # 生成电梯
    elevators = []
    freight_count = max(1, int(cfg.num_elevators * cfg.elevator_type_ratio[0]))
    passenger_count = cfg.num_elevators - freight_count
    eid = 0
    for _ in range(freight_count):
        init_floor = rng.randint(1, cfg.num_floors + 1)
        elevators.append(Elevator.from_type(eid, ElevatorType.FREIGHT, init_floor))
        eid += 1
    for _ in range(passenger_count):
        init_floor = rng.randint(1, cfg.num_floors + 1)
        elevators.append(Elevator.from_type(eid, ElevatorType.PASSENGER, init_floor))
        eid += 1

    # 生成任务
    tasks = []
    for tid in range(cfg.num_tasks):
        origin = rng.randint(1, cfg.num_floors + 1)
        if rng.random() < cfg.cross_floor_task_ratio:
            dest = origin
            while dest == origin:
                dest = rng.randint(1, cfg.num_floors + 1)
        else:
            dest = origin

        weight = rng.choice([5, 8, 10, 12, 15, 20, 25, 30, 40, 50],
                            p=[0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05])
        priority = rng.choice(list(Priority), p=[0.2, 0.5, 0.3])
        base_time = abs(dest - origin) * 60 + 120
        deadline = base_time + rng.randint(60, 300)
        earliest = rng.randint(0, 60)

        tasks.append(Task(id=tid, origin_floor=origin, dest_floor=dest,
                          weight=float(weight), priority=Priority(priority),
                          deadline=float(deadline), earliest_start=float(earliest)))

    return robots, elevators, tasks
