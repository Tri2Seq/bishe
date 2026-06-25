"""测试MILP V2：验证电梯不相交约束、与V1对比"""

import time
from config import ScenarioConfig, generate_scenario, Robot, Elevator, Task, Priority, RobotType, ElevatorType


def test_handcrafted():
    """手工构造的小实例，验证电梯不相交约束"""
    print("=" * 60)
    print("测试1: 手工实例 — 验证电梯时序不相交")
    print("=" * 60)

    # 3机器人, 1电梯, 3楼层, 4任务（全部跨楼层，必须共用1部电梯）
    robots = [
        Robot.from_type(0, RobotType.STANDARD, 1),
        Robot.from_type(1, RobotType.STANDARD, 1),
        Robot.from_type(2, RobotType.EXPRESS, 2),
    ]
    elevators = [
        Elevator.from_type(0, ElevatorType.PASSENGER, 1),  # 1部客梯，容量1
    ]
    tasks = [
        Task(0, origin_floor=1, dest_floor=3, weight=10, priority=Priority.URGENT, deadline=200),
        Task(1, origin_floor=1, dest_floor=3, weight=8, priority=Priority.URGENT, deadline=200),
        Task(2, origin_floor=1, dest_floor=2, weight=5, priority=Priority.NORMAL, deadline=300),
        Task(3, origin_floor=2, dest_floor=3, weight=5, priority=Priority.NORMAL, deadline=300),
    ]

    print(f"\n场景: 3机器人, 1电梯(容量1), 3楼层, 4跨楼层任务")
    print(f"所有任务都要用唯一的电梯 → 必须排队")
    print(f"电梯速度: {elevators[0].speed}层/s, 开关门: {elevators[0].door_time}s")
    print(f"F1→F3行程: {elevators[0].travel_time(1, 3):.1f}s")
    print(f"F1→F2行程: {elevators[0].travel_time(1, 2):.1f}s")

    from milp_v2 import MILPModelV2
    model = MILPModelV2(robots, elevators, tasks, alpha=(0.6, 0.2, 0.2))
    t0 = time.time()
    result = model.build_and_solve(time_limit=60, verbose=True)
    solve_time = time.time() - t0

    if result:
        print(f"\n求解时间: {solve_time:.2f}s")
        # 验证：电梯使用不重叠
        usages = sorted(result['elevator_usage'].items(), key=lambda x: x[1]['start'])
        print(f"\n验证电梯不相交:")
        for idx in range(len(usages) - 1):
            i1, u1 = usages[idx]
            i2, u2 = usages[idx + 1]
            gap = u2['start'] - u1['end']
            print(f"  T{i1} [{u1['start']:.1f}-{u1['end']:.1f}] → T{i2} [{u2['start']:.1f}-{u2['end']:.1f}] 间隔={gap:.1f}s")
            assert gap >= -0.1, f"电梯冲突！间隔={gap:.1f}s"
        print("  ✓ 电梯不相交约束验证通过")
    return result


def test_random_small():
    """随机小规模实例"""
    print("\n" + "=" * 60)
    print("测试2: 随机小规模 (5楼,5机器人,2电梯,10任务)")
    print("=" * 60)

    cfg = ScenarioConfig(num_floors=5, num_robots=5, num_elevators=2, num_tasks=10, seed=42)
    robots, elevators, tasks = generate_scenario(cfg)

    from milp_v2 import MILPModelV2
    model = MILPModelV2(robots, elevators, tasks, alpha=(0.5, 0.3, 0.2))
    t0 = time.time()
    result = model.build_and_solve(time_limit=120, verbose=True)
    solve_time = time.time() - t0

    if result:
        print(f"\n求解时间: {solve_time:.2f}s")

    # 对比V1
    print("\n--- V1 对比 ---")
    from milp_solver import solve_milp
    t0 = time.time()
    v1_result = solve_milp(robots, elevators, tasks, alpha=(0.5, 0.3, 0.2),
                            time_limit=120, verbose=False)
    v1_time = time.time() - t0

    if result and v1_result:
        print(f"\n{'指标':<15} {'V2(严格电梯)':>15} {'V1(简化)':>15}")
        print("-" * 48)
        print(f"{'Makespan(s)':<15} {result['makespan']:>15.1f} {v1_result['makespan']:>15.1f}")
        print(f"{'能耗(J)':<15} {result['energy']:>15.1f} {v1_result['energy']:>15.1f}")
        print(f"{'加权延迟':<15} {result['weighted_tardiness']:>15.1f} {v1_result['weighted_tardiness']:>15.1f}")
        print(f"{'求解时间(s)':<15} {solve_time:>15.2f} {v1_time:>15.2f}")
        print(f"{'变量数':<15} {result['num_variables']:>15} {'-':>15}")
        print(f"{'约束数':<15} {result['num_constraints']:>15} {'-':>15}")

    return result


def test_medium():
    """中型实例 — 测试可扩展性"""
    print("\n" + "=" * 60)
    print("测试3: 中型 (10楼,8机器人,3电梯,20任务)")
    print("=" * 60)

    cfg = ScenarioConfig(num_floors=10, num_robots=8, num_elevators=3, num_tasks=20, seed=123)
    robots, elevators, tasks = generate_scenario(cfg)

    from milp_v2 import MILPModelV2
    model = MILPModelV2(robots, elevators, tasks, alpha=(0.5, 0.3, 0.2))
    t0 = time.time()
    result = model.build_and_solve(time_limit=180, verbose=True)
    solve_time = time.time() - t0

    if result:
        print(f"\n求解时间: {solve_time:.2f}s")
    else:
        print(f"\n未找到可行解 (求解时间: {solve_time:.2f}s)")

    return result


if __name__ == "__main__":
    test_handcrafted()
    test_random_small()
    test_medium()
