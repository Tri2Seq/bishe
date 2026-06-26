"""EBDCO对比测试：验证问题针对性算法的优势"""

import numpy as np
import time
from config import ScenarioConfig, generate_scenario
from ebdco import EBDCOSolver
from nsga2 import NSGA2Solver, compute_hypervolume
from alns import ALNSSolver
from simulator import Simulator
from baselines import greedy_fcfs


def compare_all(scenario_name, cfg, ebdco_pop=80, ebdco_gen=100,
                nsga_pop=80, nsga_gen=100, alns_iter=5000):
    print(f"\n{'='*70}")
    print(f"场景: {scenario_name} ({cfg.num_floors}F, {cfg.num_robots}R, "
          f"{cfg.num_elevators}E, {cfg.num_tasks}T)")
    print(f"{'='*70}")

    robots, elevators, tasks = generate_scenario(cfg)

    # Greedy
    greedy = greedy_fcfs(robots, elevators, tasks)
    sim = Simulator(robots, elevators, tasks)
    sim.load_schedule(greedy['assignment'], greedy['task_sequence'], greedy['elevator_assignment'])
    greedy_sim = sim.run()
    greedy_obj = np.array([greedy_sim['makespan'], greedy_sim['energy'],
                           greedy_sim['weighted_tardiness']])
    print(f"\nGreedy: [{greedy_obj[0]:.0f}, {greedy_obj[1]:.0f}, {greedy_obj[2]:.0f}]")

    # EBDCO
    print(f"\n--- EBDCO ---")
    ebdco = EBDCOSolver(robots, elevators, tasks,
                         pop_size=ebdco_pop, max_gen=ebdco_gen, seed=42)
    ebdco_result = ebdco.solve(verbose=True)
    ebdco_pf = np.array(ebdco_result['pareto_front'])

    # NSGA-II
    print(f"\n--- NSGA-II ---")
    nsga = NSGA2Solver(robots, elevators, tasks,
                        pop_size=nsga_pop, max_gen=nsga_gen, seed=42)
    nsga_result = nsga.solve(verbose=True)
    nsga_pf = np.array(nsga_result['pareto_front'])

    # ALNS
    print(f"\n--- ALNS ---")
    alns = ALNSSolver(robots, elevators, tasks, max_iter=alns_iter, seed=42)
    alns_result = alns.solve(verbose=True)
    alns_pf = np.array(alns_result['pareto_front']) if alns_result['pareto_front'] else np.array([alns_result['best_objectives']])

    # 超体积对比
    ref = np.array([
        max(ebdco_pf[:,0].max(), nsga_pf[:,0].max(), greedy_obj[0]) * 1.1,
        max(ebdco_pf[:,1].max(), nsga_pf[:,1].max(), greedy_obj[1]) * 1.1,
        max(ebdco_pf[:,2].max(), nsga_pf[:,2].max(), greedy_obj[2]) * 1.1,
    ])
    hv_ebdco = compute_hypervolume(ebdco_pf, ref)
    hv_nsga = compute_hypervolume(nsga_pf, ref)
    hv_alns = compute_hypervolume(alns_pf, ref)

    # 汇总
    print(f"\n{'='*70}")
    print(f"对比: {scenario_name}")
    print(f"{'='*70}")
    print(f"{'方法':<12} {'Makespan':>10} {'能耗':>10} {'延迟':>10} "
          f"{'时间(s)':>10} {'Pareto':>7} {'HV':>8}")
    print("-" * 72)
    print(f"{'Greedy':<12} {greedy_obj[0]:>10.0f} {greedy_obj[1]:>10.0f} "
          f"{greedy_obj[2]:>10.0f} {'<0.01':>10} {'1':>7} {'-':>8}")
    print(f"{'NSGA-II':<12} {nsga_pf[:,0].min():>10.0f} {nsga_pf[:,1].min():>10.0f} "
          f"{nsga_pf[:,2].min():>10.0f} {nsga_result['total_time']:>10.1f} "
          f"{nsga_result['pareto_size']:>7} {hv_nsga:>8.4f}")
    print(f"{'ALNS':<12} {alns_pf[:,0].min():>10.0f} {alns_pf[:,1].min():>10.0f} "
          f"{alns_pf[:,2].min():>10.0f} {alns_result['total_time']:>10.1f} "
          f"{alns_result['pareto_size']:>7} {hv_alns:>8.4f}")
    print(f"{'EBDCO':<12} {ebdco_pf[:,0].min():>10.0f} {ebdco_pf[:,1].min():>10.0f} "
          f"{ebdco_pf[:,2].min():>10.0f} {ebdco_result['total_time']:>10.1f} "
          f"{ebdco_result['pareto_size']:>7} {hv_ebdco:>8.4f}")

    # 改进比
    ns_ms = nsga_pf[:,0].min()
    eb_ms = ebdco_pf[:,0].min()
    print(f"\n  EBDCO vs NSGA-II Makespan: {eb_ms:.0f} vs {ns_ms:.0f} "
          f"({'↑' if eb_ms < ns_ms else '↓'}{abs(eb_ms-ns_ms)/ns_ms*100:.1f}%)")
    print(f"  EBDCO vs NSGA-II HV: {hv_ebdco:.4f} vs {hv_nsga:.4f} "
          f"({'↑' if hv_ebdco > hv_nsga else '↓'}{abs(hv_ebdco-hv_nsga)/hv_nsga*100:.1f}%)")
    if 'lb_combined' in ebdco_result.get('analysis', {}):
        lb = ebdco_result['analysis']['lb_combined']
        print(f"  EBDCO vs 下界: {eb_ms:.0f} vs {lb:.0f} (gap={((eb_ms-lb)/lb*100):.1f}%)")

    return ebdco_result, nsga_result, alns_result


if __name__ == "__main__":
    compare_all("中型",
        ScenarioConfig(num_floors=10, num_robots=10, num_elevators=4, num_tasks=30, seed=42),
        ebdco_pop=60, ebdco_gen=80, nsga_pop=60, nsga_gen=80, alns_iter=3000)

    compare_all("大型",
        ScenarioConfig(num_floors=15, num_robots=15, num_elevators=4, num_tasks=50, seed=42),
        ebdco_pop=80, ebdco_gen=100, nsga_pop=80, nsga_gen=100, alns_iter=5000)
