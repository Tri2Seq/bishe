"""测试NSGA-II"""

import time
import numpy as np
from config import ScenarioConfig, generate_scenario
from nsga2 import NSGA2Solver, compute_hypervolume
from simulator import Simulator
from baselines import greedy_fcfs, independent_planning


def test_small():
    print("=" * 60)
    print("NSGA-II 测试: 小规模 (5楼,5机器人,2电梯,10任务)")
    print("=" * 60)

    cfg = ScenarioConfig(num_floors=5, num_robots=5, num_elevators=2, num_tasks=10, seed=42)
    robots, elevators, tasks = generate_scenario(cfg)

    solver = NSGA2Solver(robots, elevators, tasks,
                          pop_size=50, max_gen=50, seed=42)
    result = solver.solve(verbose=True)

    # 基线对比
    print("\n--- 基线对比 ---")
    greedy = greedy_fcfs(robots, elevators, tasks)
    indep = independent_planning(robots, elevators, tasks)

    # 用仿真器重新评估基线
    for name, baseline in [("Greedy-FCFS", greedy), ("Independent", indep)]:
        sim = Simulator(robots, elevators, tasks)
        sim.load_schedule(baseline['assignment'], baseline['task_sequence'],
                          baseline['elevator_assignment'])
        sim_res = sim.run()
        print(f"  {name}: Makespan={sim_res['makespan']:.0f}, "
              f"Energy={sim_res['energy']:.0f}, Tardiness={sim_res['weighted_tardiness']:.0f}")

    # Pareto前沿
    pf = np.array(result['pareto_front'])
    print(f"\n  NSGA-II Pareto前沿 ({len(pf)} 个解):")
    print(f"    Makespan范围: [{pf[:,0].min():.0f}, {pf[:,0].max():.0f}]")
    print(f"    能耗范围:     [{pf[:,1].min():.0f}, {pf[:,1].max():.0f}]")
    print(f"    延迟范围:     [{pf[:,2].min():.0f}, {pf[:,2].max():.0f}]")


def test_medium():
    print("\n" + "=" * 60)
    print("NSGA-II 测试: 中规模 (10楼,10机器人,4电梯,30任务)")
    print("=" * 60)

    cfg = ScenarioConfig(num_floors=10, num_robots=10, num_elevators=4,
                          num_tasks=30, seed=42)
    robots, elevators, tasks = generate_scenario(cfg)

    solver = NSGA2Solver(robots, elevators, tasks,
                          pop_size=80, max_gen=100, seed=42)
    result = solver.solve(verbose=True)

    pf = np.array(result['pareto_front'])
    print(f"\n  Pareto前沿 ({len(pf)} 个解):")
    print(f"    Makespan范围: [{pf[:,0].min():.0f}, {pf[:,0].max():.0f}]")
    print(f"    能耗范围:     [{pf[:,1].min():.0f}, {pf[:,1].max():.0f}]")
    print(f"    延迟范围:     [{pf[:,2].min():.0f}, {pf[:,2].max():.0f}]")

    # 超体积
    ref_point = np.array([pf[:,0].max()*1.1, pf[:,1].max()*1.1, pf[:,2].max()*1.1])
    hv = compute_hypervolume(pf, ref_point)
    print(f"    超体积(HV): {hv:.4f}")


def test_large():
    print("\n" + "=" * 60)
    print("NSGA-II 测试: 大规模 (15楼,15机器人,4电梯,50任务)")
    print("=" * 60)

    cfg = ScenarioConfig(num_floors=15, num_robots=15, num_elevators=4,
                          num_tasks=50, seed=42)
    robots, elevators, tasks = generate_scenario(cfg)

    solver = NSGA2Solver(robots, elevators, tasks,
                          pop_size=100, max_gen=150, seed=42)
    result = solver.solve(verbose=True)

    pf = np.array(result['pareto_front'])
    print(f"\n  Pareto前沿 ({len(pf)} 个解):")
    print(f"    Makespan范围: [{pf[:,0].min():.0f}, {pf[:,0].max():.0f}]")
    print(f"    能耗范围:     [{pf[:,1].min():.0f}, {pf[:,1].max():.0f}]")
    print(f"    延迟范围:     [{pf[:,2].min():.0f}, {pf[:,2].max():.0f}]")


if __name__ == "__main__":
    test_small()
    test_medium()
    test_large()
