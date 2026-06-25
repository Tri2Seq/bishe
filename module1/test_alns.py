"""测试ALNS并与NSGA-II对比"""

import numpy as np
import time
from config import ScenarioConfig, generate_scenario
from alns import ALNSSolver
from nsga2 import NSGA2Solver, compute_hypervolume
from simulator import Simulator
from baselines import greedy_fcfs


def test_and_compare(scenario_name, cfg, nsga_pop=80, nsga_gen=100,
                     alns_iter=5000):
    print(f"\n{'='*70}")
    print(f"场景: {scenario_name} ({cfg.num_floors}楼, {cfg.num_robots}机器人, "
          f"{cfg.num_elevators}电梯, {cfg.num_tasks}任务)")
    print(f"{'='*70}")

    robots, elevators, tasks = generate_scenario(cfg)

    # 基线
    greedy = greedy_fcfs(robots, elevators, tasks)
    sim = Simulator(robots, elevators, tasks)
    sim.load_schedule(greedy['assignment'], greedy['task_sequence'], greedy['elevator_assignment'])
    greedy_sim = sim.run()
    print(f"\nGreedy-FCFS (仿真): Makespan={greedy_sim['makespan']:.0f}, "
          f"Energy={greedy_sim['energy']:.0f}, Tard={greedy_sim['weighted_tardiness']:.0f}")

    # NSGA-II
    print(f"\n--- NSGA-II (pop={nsga_pop}, gen={nsga_gen}) ---")
    nsga = NSGA2Solver(robots, elevators, tasks, pop_size=nsga_pop, max_gen=nsga_gen, seed=42)
    nsga_result = nsga.solve(verbose=True)
    nsga_pf = np.array(nsga_result['pareto_front'])

    # ALNS
    print(f"\n--- ALNS (iter={alns_iter}) ---")
    alns = ALNSSolver(robots, elevators, tasks, max_iter=alns_iter, seed=42)
    alns_result = alns.solve(verbose=True)

    # 对比
    print(f"\n{'='*70}")
    print(f"对比总结: {scenario_name}")
    print(f"{'='*70}")
    print(f"{'方法':<18} {'Makespan':>10} {'能耗':>10} {'延迟':>10} {'时间(s)':>10} {'Pareto解':>8}")
    print("-" * 70)
    print(f"{'Greedy-FCFS':<18} {greedy_sim['makespan']:>10.0f} {greedy_sim['energy']:>10.0f} "
          f"{greedy_sim['weighted_tardiness']:>10.0f} {'<0.01':>10} {1:>8}")
    print(f"{'NSGA-II(best-ms)':<18} {nsga_pf[:,0].min():>10.0f} {'-':>10} "
          f"{'-':>10} {nsga_result['total_time']:>10.1f} {nsga_result['pareto_size']:>8}")
    print(f"{'NSGA-II(best-e)':<18} {'-':>10} {nsga_pf[:,1].min():>10.0f} "
          f"{'-':>10} {'-':>10} {'-':>8}")
    print(f"{'NSGA-II(best-td)':<18} {'-':>10} {'-':>10} "
          f"{nsga_pf[:,2].min():>10.0f} {'-':>10} {'-':>8}")
    print(f"{'ALNS(best)':<18} {alns_result['best_objectives'][0]:>10.0f} "
          f"{alns_result['best_objectives'][1]:>10.0f} "
          f"{alns_result['best_objectives'][2]:>10.0f} "
          f"{alns_result['total_time']:>10.1f} {alns_result['pareto_size']:>8}")

    # 超体积
    if len(nsga_pf) > 0:
        ref = np.array([
            max(nsga_pf[:,0].max(), alns_result['best_objectives'][0]) * 1.1,
            max(nsga_pf[:,1].max(), alns_result['best_objectives'][1]) * 1.1,
            max(nsga_pf[:,2].max(), alns_result['best_objectives'][2]) * 1.1,
        ])
        hv_nsga = compute_hypervolume(nsga_pf, ref)
        alns_pf = np.array(alns_result['pareto_front']) if alns_result['pareto_front'] else np.array([alns_result['best_objectives']])
        hv_alns = compute_hypervolume(alns_pf, ref)
        print(f"\n超体积(HV): NSGA-II={hv_nsga:.4f}, ALNS={hv_alns:.4f}")

    return nsga_result, alns_result


if __name__ == "__main__":
    test_and_compare("小型",
        ScenarioConfig(num_floors=5, num_robots=5, num_elevators=2, num_tasks=10, seed=42),
        nsga_pop=50, nsga_gen=50, alns_iter=2000)

    test_and_compare("中型",
        ScenarioConfig(num_floors=10, num_robots=10, num_elevators=4, num_tasks=30, seed=42),
        nsga_pop=80, nsga_gen=100, alns_iter=5000)

    test_and_compare("大型",
        ScenarioConfig(num_floors=15, num_robots=15, num_elevators=4, num_tasks=50, seed=42),
        nsga_pop=100, nsga_gen=150, alns_iter=8000)
