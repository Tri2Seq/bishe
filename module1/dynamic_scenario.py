"""24小时动态任务场景建模

一天内的任务量不是平均的：
  06:00-08:00  早班启动期：少量任务（机器人充电/就位）
  08:00-12:00  上午高峰：大量入库任务（收货→存储），占比40%
  12:00-13:00  午间低谷：少量调拨任务
  13:00-17:00  下午高峰：大量出库任务（存储→打包→发货），占比40%
  17:00-20:00  晚间收尾：中等量级收尾任务
  20:00-06:00  夜间低谷：少量维护/补货任务

任务类型随时段变化：
  - 上午：以上行（收货→存储）为主
  - 下午：以下行（存储→发货）为主
  - 夜间：以同层调拨为主

GPU需求强度随之波动：
  高峰期 → 多机器人同时使用电梯 → GPU并发需求高
  低谷期 → 少数机器人工作 → GPU空闲 → 可释放/预加载
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from config import Task, Priority, Robot, Elevator, RobotType, ElevatorType, ROBOT_SPECS, ELEVATOR_SPECS


@dataclass
class TimePeriod:
    """时段定义"""
    name: str
    start_hour: float
    end_hour: float
    task_rate: float          # 每小时任务数
    up_ratio: float           # 上行任务比例
    down_ratio: float         # 下行任务比例
    same_floor_ratio: float   # 同层任务比例
    urgent_ratio: float       # 紧急任务比例
    weight_range: Tuple[float, float]  # 货物重量范围(kg)


# 24小时时段定义
DAY_SCHEDULE = [
    TimePeriod("早班启动", 6, 8, 8, 0.3, 0.2, 0.5, 0.1, (5, 15)),
    TimePeriod("上午高峰", 8, 12, 30, 0.6, 0.1, 0.3, 0.25, (5, 50)),
    TimePeriod("午间低谷", 12, 13, 5, 0.2, 0.2, 0.6, 0.05, (5, 15)),
    TimePeriod("下午高峰", 13, 17, 28, 0.1, 0.6, 0.3, 0.2, (5, 50)),
    TimePeriod("晚间收尾", 17, 20, 12, 0.2, 0.4, 0.4, 0.15, (5, 30)),
    TimePeriod("夜间低谷", 20, 6, 3, 0.1, 0.1, 0.8, 0.02, (5, 10)),
]


@dataclass
class DayScenario:
    """一整天的场景配置"""
    num_floors: int
    num_robots: int
    num_elevators: int
    num_gpu_servers: int
    gpus_per_server: int

    # 楼层功能定义
    receiving_floors: List[int] = None
    storage_floors: List[int] = None
    packing_floors: List[int] = None
    shipping_floors: List[int] = None

    def __post_init__(self):
        F = self.num_floors
        if self.receiving_floors is None:
            self.receiving_floors = list(range(1, max(2, F // 5) + 1))
        if self.storage_floors is None:
            self.storage_floors = list(range(max(2, F // 5) + 1, F * 3 // 4 + 1))
        if self.packing_floors is None:
            self.packing_floors = list(range(F * 3 // 4 + 1, F * 9 // 10 + 1))
        if self.shipping_floors is None:
            self.shipping_floors = list(range(F * 9 // 10 + 1, F + 1))


def generate_day_tasks(scenario: DayScenario, seed: int = 42) -> List[Task]:
    """
    生成一整天的任务流。

    每个任务带有精确的到达时间（秒），按时间排序。
    """
    rng = np.random.RandomState(seed)
    tasks = []
    tid = 0

    for period in DAY_SCHEDULE:
        # 处理跨日时段（夜间）
        if period.start_hour > period.end_hour:
            duration_hours = (24 - period.start_hour) + period.end_hour
        else:
            duration_hours = period.end_hour - period.start_hour

        n_tasks = int(period.task_rate * duration_hours)

        for _ in range(n_tasks):
            # 到达时间（均匀分布在时段内）
            if period.start_hour > period.end_hour:
                offset = rng.uniform(0, duration_hours)
                hour = (period.start_hour + offset) % 24
            else:
                hour = rng.uniform(period.start_hour, period.end_hour)
            arrival_time = hour * 3600  # 转换为秒

            # 任务方向
            all_floors = list(range(1, scenario.num_floors + 1))
            r = rng.random()
            if r < period.up_ratio:
                origin_pool = scenario.receiving_floors + scenario.storage_floors[:len(scenario.storage_floors)//2]
                dest_pool = scenario.storage_floors[len(scenario.storage_floors)//2:] + scenario.packing_floors
                origin = rng.choice(origin_pool) if origin_pool else rng.randint(1, scenario.num_floors + 1)
                dest = rng.choice(dest_pool) if dest_pool else rng.randint(1, scenario.num_floors + 1)
                if dest <= origin:
                    dest = min(origin + rng.randint(1, 4), scenario.num_floors)
            elif r < period.up_ratio + period.down_ratio:
                origin_pool = scenario.storage_floors + scenario.packing_floors
                dest_pool = scenario.packing_floors + scenario.shipping_floors
                origin = rng.choice(origin_pool) if origin_pool else rng.randint(1, scenario.num_floors + 1)
                dest = rng.choice(dest_pool) if dest_pool else rng.randint(1, scenario.num_floors + 1)
                if dest >= origin:
                    dest = max(origin - rng.randint(1, 4), 1)
            else:
                origin = rng.randint(1, scenario.num_floors + 1)
                dest = origin

            # 优先级
            priority = Priority.URGENT if rng.random() < period.urgent_ratio else \
                       (Priority.NORMAL if rng.random() < 0.6 else Priority.BATCH)

            # 重量
            weight = rng.uniform(*period.weight_range)
            weight = round(weight / 5) * 5  # 取整到5kg
            weight = max(5, weight)

            # 截止时间：到达时间 + 合理窗口
            floor_diff = abs(dest - origin)
            base_deadline = 120 + floor_diff * 60  # 基础截止窗口
            if priority == Priority.URGENT:
                deadline = arrival_time + base_deadline
            elif priority == Priority.NORMAL:
                deadline = arrival_time + base_deadline * 2
            else:
                deadline = arrival_time + base_deadline * 4

            tasks.append(Task(
                id=tid, origin_floor=origin, dest_floor=dest,
                weight=float(weight), priority=priority,
                deadline=deadline, earliest_start=arrival_time,
            ))
            tid += 1

    # 按到达时间排序
    tasks.sort(key=lambda t: t.earliest_start)
    # 重新编号
    for i, t in enumerate(tasks):
        t.id = i

    return tasks


def generate_day_robots(scenario: DayScenario, seed: int = 42) -> List[Robot]:
    """生成机器人车队"""
    rng = np.random.RandomState(seed)
    robots = []
    types = [RobotType.HEAVY] * int(scenario.num_robots * 0.3) + \
            [RobotType.STANDARD] * int(scenario.num_robots * 0.5) + \
            [RobotType.EXPRESS] * (scenario.num_robots - int(scenario.num_robots * 0.3) - int(scenario.num_robots * 0.5))

    for j, rtype in enumerate(types):
        init_floor = rng.randint(1, scenario.num_floors + 1)
        robots.append(Robot.from_type(j, rtype, init_floor))
    return robots


def generate_day_elevators(scenario: DayScenario, seed: int = 42) -> List[Elevator]:
    """生成电梯"""
    rng = np.random.RandomState(seed)
    elevators = []
    n_freight = max(1, scenario.num_elevators * 2 // 5)
    n_passenger = scenario.num_elevators - n_freight

    for e in range(n_freight):
        elevators.append(Elevator.from_type(e, ElevatorType.FREIGHT,
                                             rng.randint(1, scenario.num_floors + 1)))
    for e in range(n_passenger):
        elevators.append(Elevator.from_type(n_freight + e, ElevatorType.PASSENGER,
                                             rng.randint(1, scenario.num_floors + 1)))
    return elevators


def analyze_day_pattern(tasks: List[Task]) -> Dict:
    """分析24小时任务分布"""
    hourly_count = np.zeros(24)
    hourly_cf = np.zeros(24)
    hourly_urgent = np.zeros(24)

    for t in tasks:
        hour = int(t.earliest_start / 3600) % 24
        hourly_count[hour] += 1
        if t.is_cross_floor:
            hourly_cf[hour] += 1
        if t.priority == Priority.URGENT:
            hourly_urgent[hour] += 1

    peak_hour = int(np.argmax(hourly_count))
    valley_hour = int(np.argmin(hourly_count[6:20])) + 6  # 白天最低点

    return {
        "total_tasks": len(tasks),
        "hourly_distribution": hourly_count.tolist(),
        "hourly_crossfloor": hourly_cf.tolist(),
        "hourly_urgent": hourly_urgent.tolist(),
        "peak_hour": peak_hour,
        "peak_tasks": int(hourly_count[peak_hour]),
        "valley_hour": valley_hour,
        "valley_tasks": int(hourly_count[valley_hour]),
        "peak_valley_ratio": hourly_count[peak_hour] / max(hourly_count[valley_hour], 1),
        "cross_floor_ratio": sum(1 for t in tasks if t.is_cross_floor) / len(tasks),
    }


if __name__ == "__main__":
    scenario = DayScenario(
        num_floors=15, num_robots=20, num_elevators=4,
        num_gpu_servers=2, gpus_per_server=4,
    )
    tasks = generate_day_tasks(scenario, seed=42)
    pattern = analyze_day_pattern(tasks)

    print(f"24小时任务分析:")
    print(f"  总任务数: {pattern['total_tasks']}")
    print(f"  跨楼层比例: {pattern['cross_floor_ratio']:.1%}")
    print(f"  高峰时段: {pattern['peak_hour']}:00 ({pattern['peak_tasks']}个任务)")
    print(f"  低谷时段: {pattern['valley_hour']}:00 ({pattern['valley_tasks']}个任务)")
    print(f"  峰谷比: {pattern['peak_valley_ratio']:.1f}x")
    print(f"\n小时分布:")
    for h in range(24):
        bar = '█' * int(pattern['hourly_distribution'][h])
        cf = int(pattern['hourly_crossfloor'][h])
        print(f"  {h:02d}:00  {bar} ({int(pattern['hourly_distribution'][h])}个, 跨楼层{cf})")
