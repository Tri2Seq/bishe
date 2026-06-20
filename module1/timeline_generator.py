"""从MILP求解结果生成机器人动作时间线

这是模块1到模块2的接口：将调度方案转化为每台机器人的详细动作序列，
明确标注哪些时段需要深度估计推理。
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional
import json

from config import Robot, Elevator, Task, FLOOR_DISTANCE, INTER_STATION_DISTANCE


class ActionType(str, Enum):
    WALK = "WALK"                         # 楼层内行走
    WAIT_ELEVATOR = "WAIT_ELEVATOR"       # 等待电梯到达
    APPROACH_ELEVATOR = "APPROACH_ELEVATOR"  # 接近电梯准备交互 ★需要GPU★
    PRESS_BUTTON = "PRESS_BUTTON"         # 按电梯按钮       ★需要GPU★
    RIDE_ELEVATOR = "RIDE_ELEVATOR"       # 乘坐电梯
    EXIT_ELEVATOR = "EXIT_ELEVATOR"       # 出电梯导航       ★需要GPU★
    PICKUP = "PICKUP"                     # 取货
    DELIVER = "DELIVER"                   # 送货
    IDLE = "IDLE"                         # 空闲

    @property
    def needs_gpu(self) -> bool:
        return self in (ActionType.APPROACH_ELEVATOR,
                        ActionType.PRESS_BUTTON,
                        ActionType.EXIT_ELEVATOR)


@dataclass
class Action:
    type: ActionType
    start_time: float
    end_time: float
    floor: int
    robot_id: int
    task_id: Optional[int] = None
    elevator_id: Optional[int] = None
    dest_floor: Optional[int] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def needs_gpu(self) -> bool:
        return self.type.needs_gpu

    def to_dict(self) -> dict:
        d = {
            "type": self.type.value,
            "start": round(self.start_time, 2),
            "end": round(self.end_time, 2),
            "floor": self.floor,
            "robot_id": self.robot_id,
            "needs_gpu": self.needs_gpu,
        }
        if self.task_id is not None:
            d["task_id"] = self.task_id
        if self.elevator_id is not None:
            d["elevator_id"] = self.elevator_id
        if self.dest_floor is not None:
            d["dest_floor"] = self.dest_floor
        return d


# 电梯交互各阶段的典型时长
APPROACH_DURATION = 8.0   # 接近电梯（需要GPU）
PRESS_DURATION = 7.0      # 按按钮（需要GPU）
WAIT_DURATION = 0.0       # 等待电梯到达（动态计算）
EXIT_DURATION = 5.0       # 出电梯（需要GPU）
PICKUP_DURATION = 10.0
DELIVER_DURATION = 10.0


def generate_timeline(robots: List[Robot], elevators: List[Elevator],
                      tasks: List[Task], milp_result: Dict) -> Dict[int, List[Action]]:
    """
    根据MILP结果生成每台机器人的详细动作时间线

    Returns:
        {robot_id: [Action, ...]} 按时间排序的动作序列
    """
    assignment = milp_result["assignment"]
    elevator_assignment = milp_result["elevator_assignment"]
    task_sequence = milp_result["task_sequence"]
    start_times = milp_result["start_times"]

    timelines = {}

    for j, robot in enumerate(robots):
        actions = []
        seq = task_sequence.get(j, [])
        if not seq:
            timelines[j] = actions
            continue

        current_floor = robot.init_floor
        current_time = 0.0

        for idx, task_id in enumerate(seq):
            task = tasks[task_id]

            # === 阶段1: 移动到取货楼层 ===
            if current_floor != task.origin_floor:
                # 需要乘电梯去取货楼层
                walk_to_elevator = FLOOR_DISTANCE / robot.speed
                actions.append(Action(
                    type=ActionType.WALK,
                    start_time=current_time,
                    end_time=current_time + walk_to_elevator,
                    floor=current_floor,
                    robot_id=j,
                ))
                current_time += walk_to_elevator

                # 接近电梯（需要GPU）
                actions.append(Action(
                    type=ActionType.APPROACH_ELEVATOR,
                    start_time=current_time,
                    end_time=current_time + APPROACH_DURATION,
                    floor=current_floor,
                    robot_id=j,
                    elevator_id=0,  # 转移时使用任意可用电梯
                ))
                current_time += APPROACH_DURATION

                # 按按钮（需要GPU）
                actions.append(Action(
                    type=ActionType.PRESS_BUTTON,
                    start_time=current_time,
                    end_time=current_time + PRESS_DURATION,
                    floor=current_floor,
                    robot_id=j,
                    elevator_id=0,
                ))
                current_time += PRESS_DURATION

                # 等待电梯
                wait_time = 5.0  # 简化：固定等待时间
                if wait_time > 0:
                    actions.append(Action(
                        type=ActionType.WAIT_ELEVATOR,
                        start_time=current_time,
                        end_time=current_time + wait_time,
                        floor=current_floor,
                        robot_id=j,
                        elevator_id=0,
                    ))
                    current_time += wait_time

                # 乘坐电梯
                elev = elevators[0]
                ride_time = abs(current_floor - task.origin_floor) / elev.speed + elev.door_time
                actions.append(Action(
                    type=ActionType.RIDE_ELEVATOR,
                    start_time=current_time,
                    end_time=current_time + ride_time,
                    floor=current_floor,
                    robot_id=j,
                    elevator_id=0,
                    dest_floor=task.origin_floor,
                ))
                current_time += ride_time

                # 出电梯（需要GPU）
                actions.append(Action(
                    type=ActionType.EXIT_ELEVATOR,
                    start_time=current_time,
                    end_time=current_time + EXIT_DURATION,
                    floor=task.origin_floor,
                    robot_id=j,
                    elevator_id=0,
                ))
                current_time += EXIT_DURATION

                current_floor = task.origin_floor

            # === 阶段2: 楼层内走到取货站 ===
            walk_to_pickup = INTER_STATION_DISTANCE / robot.speed
            actions.append(Action(
                type=ActionType.WALK,
                start_time=current_time,
                end_time=current_time + walk_to_pickup,
                floor=current_floor,
                robot_id=j,
            ))
            current_time += walk_to_pickup

            # === 阶段3: 取货 ===
            task_start = max(current_time, start_times.get(task_id, 0),
                             task.earliest_start)
            if task_start > current_time:
                actions.append(Action(
                    type=ActionType.IDLE,
                    start_time=current_time,
                    end_time=task_start,
                    floor=current_floor,
                    robot_id=j,
                ))
                current_time = task_start

            actions.append(Action(
                type=ActionType.PICKUP,
                start_time=current_time,
                end_time=current_time + PICKUP_DURATION,
                floor=current_floor,
                robot_id=j,
                task_id=task_id,
            ))
            current_time += PICKUP_DURATION

            # === 阶段4: 跨楼层配送（如果需要）===
            if task.is_cross_floor:
                elev_id = elevator_assignment.get(task_id, 0)
                elev = elevators[elev_id]

                # 走到电梯
                walk_to_elev = FLOOR_DISTANCE / robot.speed
                actions.append(Action(
                    type=ActionType.WALK,
                    start_time=current_time,
                    end_time=current_time + walk_to_elev,
                    floor=current_floor,
                    robot_id=j,
                    task_id=task_id,
                ))
                current_time += walk_to_elev

                # 接近电梯（★需要GPU★）
                actions.append(Action(
                    type=ActionType.APPROACH_ELEVATOR,
                    start_time=current_time,
                    end_time=current_time + APPROACH_DURATION,
                    floor=current_floor,
                    robot_id=j,
                    task_id=task_id,
                    elevator_id=elev_id,
                ))
                current_time += APPROACH_DURATION

                # 按按钮（★需要GPU★）
                actions.append(Action(
                    type=ActionType.PRESS_BUTTON,
                    start_time=current_time,
                    end_time=current_time + PRESS_DURATION,
                    floor=current_floor,
                    robot_id=j,
                    task_id=task_id,
                    elevator_id=elev_id,
                ))
                current_time += PRESS_DURATION

                # 等待电梯
                wait_time = 5.0
                actions.append(Action(
                    type=ActionType.WAIT_ELEVATOR,
                    start_time=current_time,
                    end_time=current_time + wait_time,
                    floor=current_floor,
                    robot_id=j,
                    task_id=task_id,
                    elevator_id=elev_id,
                ))
                current_time += wait_time

                # 乘坐电梯（不需要GPU → 迁移窗口）
                ride_time = elev.travel_time(current_floor, task.dest_floor)
                actions.append(Action(
                    type=ActionType.RIDE_ELEVATOR,
                    start_time=current_time,
                    end_time=current_time + ride_time,
                    floor=current_floor,
                    robot_id=j,
                    task_id=task_id,
                    elevator_id=elev_id,
                    dest_floor=task.dest_floor,
                ))
                current_time += ride_time

                # 出电梯（★需要GPU★）
                actions.append(Action(
                    type=ActionType.EXIT_ELEVATOR,
                    start_time=current_time,
                    end_time=current_time + EXIT_DURATION,
                    floor=task.dest_floor,
                    robot_id=j,
                    task_id=task_id,
                    elevator_id=elev_id,
                ))
                current_time += EXIT_DURATION

                current_floor = task.dest_floor

                # 走到送货站
                walk_to_deliver = FLOOR_DISTANCE / robot.speed
                actions.append(Action(
                    type=ActionType.WALK,
                    start_time=current_time,
                    end_time=current_time + walk_to_deliver,
                    floor=current_floor,
                    robot_id=j,
                    task_id=task_id,
                ))
                current_time += walk_to_deliver
            else:
                # 同层配送
                walk_time = INTER_STATION_DISTANCE / robot.speed
                actions.append(Action(
                    type=ActionType.WALK,
                    start_time=current_time,
                    end_time=current_time + walk_time,
                    floor=current_floor,
                    robot_id=j,
                    task_id=task_id,
                ))
                current_time += walk_time

            # === 阶段5: 送货 ===
            actions.append(Action(
                type=ActionType.DELIVER,
                start_time=current_time,
                end_time=current_time + DELIVER_DURATION,
                floor=current_floor,
                robot_id=j,
                task_id=task_id,
            ))
            current_time += DELIVER_DURATION

        timelines[j] = actions

    return timelines


def extract_gpu_demands(timelines: Dict[int, List[Action]]) -> Dict[int, List[Dict]]:
    """
    从时间线中提取GPU需求时段和空闲窗口

    Returns:
        {robot_id: [{"type": "demand"/"idle", "start": ..., "end": ..., ...}, ...]}
    """
    gpu_schedule = {}

    for robot_id, actions in timelines.items():
        segments = []
        gpu_actions = [a for a in actions if a.needs_gpu]

        # 合并连续的GPU需求（如 APPROACH + PRESS 通常连续）
        merged_demands = []
        for a in gpu_actions:
            if merged_demands and a.start_time <= merged_demands[-1]["end"] + 0.1:
                merged_demands[-1]["end"] = a.end_time
                merged_demands[-1]["actions"].append(a.type.value)
            else:
                merged_demands.append({
                    "type": "demand",
                    "start": a.start_time,
                    "end": a.end_time,
                    "robot_id": robot_id,
                    "elevator_id": a.elevator_id,
                    "floor": a.floor,
                    "actions": [a.type.value],
                })

        # 提取GPU空闲窗口（两段需求之间的间隔）
        idle_windows = []
        for idx in range(len(merged_demands) - 1):
            gap_start = merged_demands[idx]["end"]
            gap_end = merged_demands[idx + 1]["start"]
            if gap_end - gap_start > 0.5:
                # 找这段间隔中的RIDE_ELEVATOR动作
                reason = "OTHER"
                for a in actions:
                    if a.start_time >= gap_start and a.end_time <= gap_end:
                        if a.type == ActionType.RIDE_ELEVATOR:
                            reason = "RIDING"
                            break
                        elif a.type == ActionType.WALK:
                            reason = "WALKING"
                        elif a.type == ActionType.IDLE:
                            reason = "IDLE"

                idle_windows.append({
                    "type": "idle",
                    "start": gap_start,
                    "end": gap_end,
                    "duration": gap_end - gap_start,
                    "robot_id": robot_id,
                    "reason": reason,
                })

        segments = sorted(merged_demands + idle_windows, key=lambda s: s["start"])
        gpu_schedule[robot_id] = segments

    return gpu_schedule


def save_timelines(timelines: Dict[int, List[Action]], filepath: str):
    """保存时间线到JSON文件"""
    data = {}
    for robot_id, actions in timelines.items():
        data[str(robot_id)] = [a.to_dict() for a in actions]
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_gpu_demands(gpu_schedule: Dict[int, List[Dict]], filepath: str):
    """保存GPU需求时间表到JSON文件"""
    data = {str(k): v for k, v in gpu_schedule.items()}
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def print_timeline_summary(timelines: Dict[int, List[Action]]):
    """打印时间线摘要"""
    print("\n=== 机器人动作时间线摘要 ===")
    for robot_id, actions in timelines.items():
        if not actions:
            continue
        gpu_actions = [a for a in actions if a.needs_gpu]
        total_gpu_time = sum(a.duration for a in gpu_actions)
        total_time = actions[-1].end_time - actions[0].start_time if actions else 0
        ride_actions = [a for a in actions if a.type == ActionType.RIDE_ELEVATOR]
        total_ride_time = sum(a.duration for a in ride_actions)

        print(f"\n机器人{robot_id}:")
        print(f"  总时长: {total_time:.1f}s")
        print(f"  动作数: {len(actions)}")
        print(f"  GPU需求时长: {total_gpu_time:.1f}s ({total_gpu_time/max(total_time,1)*100:.1f}%)")
        print(f"  乘梯时长(GPU释放窗口): {total_ride_time:.1f}s")
        print(f"  GPU需求段数: {len(gpu_actions)}")

        # 打印时间线条形图
        if total_time > 0:
            bar_width = 60
            bar = [" "] * bar_width
            for a in actions:
                start_pos = int(a.start_time / total_time * bar_width)
                end_pos = min(int(a.end_time / total_time * bar_width), bar_width - 1)
                char = "█" if a.needs_gpu else ("▓" if a.type == ActionType.RIDE_ELEVATOR else "░")
                for p in range(start_pos, end_pos + 1):
                    bar[p] = char
            print(f"  [{''.join(bar)}]")
            print(f"   █=需GPU  ▓=乘梯(可释放)  ░=其他")
