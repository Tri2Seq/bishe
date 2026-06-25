"""验证仿真器：将MILP解输入仿真，对比makespan等指标"""

import time
from config import ScenarioConfig, generate_scenario, Robot, Elevator, Task, Priority, RobotType, ElevatorType
from milp_v2 import MILPModelV2
from simulator import Simulator


def test_sim_vs_milp():
    """对比MILP解与仿真器结果"""
    print("=" * 60)
    print("测试: MILP V2 解 → 仿真器验证")
    print("=" * 60)

    # 手工实例
    robots = [
        Robot.from_type(0, RobotType.STANDARD, 1),
        Robot.from_type(1, RobotType.STANDARD, 1),
        Robot.from_type(2, RobotType.EXPRESS, 2),
    ]
    elevators = [
        Elevator.from_type(0, ElevatorType.PASSENGER, 1),
    ]
    tasks = [
        Task(0, origin_floor=1, dest_floor=3, weight=10, priority=Priority.URGENT, deadline=300),
        Task(1, origin_floor=1, dest_floor=3, weight=8, priority=Priority.URGENT, deadline=300),
        Task(2, origin_floor=1, dest_floor=2, weight=5, priority=Priority.NORMAL, deadline=400),
        Task(3, origin_floor=2, dest_floor=3, weight=5, priority=Priority.NORMAL, deadline=400),
    ]

    # 求解MILP
    print("\n--- MILP V2 求解 ---")
    model = MILPModelV2(robots, elevators, tasks, alpha=(0.6, 0.2, 0.2))
    milp_result = model.build_and_solve(time_limit=60, verbose=False)

    if not milp_result:
        print("MILP未找到解")
        return

    print(f"  MILP Makespan: {milp_result['makespan']:.1f}s")
    print(f"  MILP 能耗: {milp_result['energy']:.1f}J")
    print(f"  MILP 延迟: {milp_result['weighted_tardiness']:.1f}")

    # 仿真验证
    print("\n--- 仿真器验证 ---")
    sim = Simulator(robots, elevators, tasks)
    sim.load_schedule(
        assignment=milp_result['assignment'],
        task_sequence=milp_result['task_sequence'],
        elevator_assignment=milp_result['elevator_assignment']
    )
    sim_result = sim.run(verbose=False)
    sim.print_summary(sim_result)

    # 对比
    print(f"\n--- 对比 ---")
    print(f"{'指标':<15} {'MILP':>12} {'仿真':>12} {'差异':>12}")
    print("-" * 54)
    for key, label in [('makespan', 'Makespan(s)'), ('energy', '能耗(J)'),
                        ('weighted_tardiness', '加权延迟')]:
        m = milp_result[key]
        s = sim_result[key]
        diff = s - m
        pct = diff / m * 100 if m > 0 else 0
        print(f"{label:<15} {m:>12.1f} {s:>12.1f} {diff:>+8.1f} ({pct:>+.1f}%)")


def test_random_scenario():
    """随机场景测试"""
    print("\n" + "=" * 60)
    print("测试: 随机场景 (5楼,5机器人,2电梯,10任务)")
    print("=" * 60)

    cfg = ScenarioConfig(num_floors=5, num_robots=5, num_elevators=2, num_tasks=10, seed=42)
    robots, elevators, tasks = generate_scenario(cfg)

    model = MILPModelV2(robots, elevators, tasks, alpha=(0.5, 0.3, 0.2))
    milp_result = model.build_and_solve(time_limit=120, verbose=False)

    if not milp_result:
        print("MILP未找到解")
        return

    sim = Simulator(robots, elevators, tasks)
    sim.load_schedule(
        assignment=milp_result['assignment'],
        task_sequence=milp_result['task_sequence'],
        elevator_assignment=milp_result['elevator_assignment']
    )
    sim_result = sim.run()
    sim.print_summary(sim_result)

    print(f"\n--- MILP vs 仿真 ---")
    print(f"{'指标':<15} {'MILP':>12} {'仿真':>12} {'差异':>12}")
    print("-" * 54)
    for key, label in [('makespan', 'Makespan(s)'), ('energy', '能耗(J)'),
                        ('weighted_tardiness', '加权延迟')]:
        m = milp_result[key]
        s = sim_result[key]
        diff = s - m
        pct = diff / m * 100 if m > 0 else 0
        print(f"{label:<15} {m:>12.1f} {s:>12.1f} {diff:>+8.1f} ({pct:>+.1f}%)")

    # 导出时间线
    timelines = sim.get_timelines_json()
    import json
    with open("results/sim_small_timelines.json", "w") as f:
        json.dump(timelines, f, indent=2)
    print(f"\n仿真时间线已保存到 results/sim_small_timelines.json")


if __name__ == "__main__":
    test_sim_vs_milp()
    test_random_scenario()
