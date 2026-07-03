"""完整对比实验

实验1: V2算法对比 (Greedy vs EBO vs EBO+Relay) × 3规模 × 5 seeds
实验2: 派车算法对比 (ETA/NC/SCAN)
统计指标：均值、标准差、Wilcoxon秩和检验
"""

import sys
import os
import json
import time
import numpy as np
from scipy import stats
from typing import Dict, List

from config import ScenarioConfig, generate_scenario, Priority
from baselines import greedy_fcfs
from ebdco import LowerLevelSolver
from nsga2_ebo import NSGA2EBOSolver
from simulator_v2 import SimulatorV2, RobotStrategy
from elevator_group_control import DispatchAlgorithm


SCENARIOS = {
    "S1_small":  ScenarioConfig(num_floors=8,  num_robots=6,  num_elevators=3, num_tasks=15),
    "S2_medium": ScenarioConfig(num_floors=12, num_robots=10, num_elevators=4, num_tasks=30),
    "S3_large":  ScenarioConfig(num_floors=15, num_robots=12, num_elevators=5, num_tasks=50),
}

EBO_CONFIGS = {
    "S1_small":  {"pop_size": 40, "max_gen": 40},
    "S2_medium": {"pop_size": 60, "max_gen": 60},
    "S3_large":  {"pop_size": 60, "max_gen": 80},
}

N_RUNS = 5
SEEDS = [42, 123, 456, 789, 1024]


def run_greedy_v2(robots, elevators, tasks, dispatch_algo=DispatchAlgorithm.ETA):
    """Greedy-FCFS through V2 simulator"""
    greedy = greedy_fcfs(robots, elevators, tasks)
    lower = LowerLevelSolver(robots, elevators, tasks)
    seq = lower.optimize_sequence(greedy['assignment'], {}, weight=np.array([0.5, 0.3, 0.2]))
    sim = SimulatorV2(robots, elevators, tasks, dispatch_algorithm=dispatch_algo, dt=0.5)
    sim.load_schedule(greedy['assignment'], seq)
    return sim.run(max_time=10000)


def run_ebo_v2(robots, elevators, tasks, pop_size, max_gen, seed,
               dispatch_algo=DispatchAlgorithm.ETA, enable_relay=True):
    """NSGA-II-EBO through V2 simulator"""
    solver = NSGA2EBOSolver(robots, elevators, tasks,
                             pop_size=pop_size, max_gen=max_gen,
                             dispatch_algorithm=dispatch_algo, seed=seed)
    if not enable_relay:
        solver.relay_eligible = []
    result = solver.solve(verbose=False)
    assignment, seq, relay_plans, strategy = solver.get_best_schedule(objective_idx=0)

    sim = SimulatorV2(robots, elevators, tasks,
                      dispatch_algorithm=dispatch_algo, dt=0.5, strategy=strategy)
    sim.load_schedule(assignment, seq, relay_plans=relay_plans)
    sim_result = sim.run(max_time=10000)
    sim_result['pareto_size'] = result['pareto_size']
    sim_result['relay_active'] = result.get('relay_active_best', 0)
    sim_result['relay_eligible'] = result.get('relay_eligible', 0)
    return sim_result


def run_algorithm_comparison(output_dir):
    """实验1: 算法对比"""
    all_results = {}

    for scenario_name, cfg_template in SCENARIOS.items():
        print(f"\n{'#'*70}")
        print(f"场景: {scenario_name} ({cfg_template.num_floors}F, {cfg_template.num_robots}R, "
              f"{cfg_template.num_elevators}E, {cfg_template.num_tasks}T)")
        print(f"{'#'*70}")

        ebo_cfg = EBO_CONFIGS[scenario_name]
        scenario_results = {"Greedy": [], "EBO": [], "EBO+Relay": []}

        for run_idx, seed in enumerate(SEEDS):
            print(f"\n  --- Run {run_idx+1}/{N_RUNS} (seed={seed}) ---")
            cfg = ScenarioConfig(
                num_floors=cfg_template.num_floors,
                num_robots=cfg_template.num_robots,
                num_elevators=cfg_template.num_elevators,
                num_tasks=cfg_template.num_tasks,
                seed=seed
            )
            robots, elevators, tasks = generate_scenario(cfg)

            # Greedy
            t0 = time.time()
            r = run_greedy_v2(robots, elevators, tasks)
            scenario_results["Greedy"].append({
                "makespan": r['makespan'], "energy": r['energy'],
                "tardiness": r['weighted_tardiness'], "time": time.time() - t0,
            })

            # EBO (no relay)
            t0 = time.time()
            r = run_ebo_v2(robots, elevators, tasks,
                           pop_size=ebo_cfg['pop_size'], max_gen=ebo_cfg['max_gen'],
                           seed=seed, enable_relay=False)
            scenario_results["EBO"].append({
                "makespan": r['makespan'], "energy": r['energy'],
                "tardiness": r['weighted_tardiness'], "time": time.time() - t0,
                "pareto_size": r['pareto_size'],
            })

            # EBO + Relay
            t0 = time.time()
            r = run_ebo_v2(robots, elevators, tasks,
                           pop_size=ebo_cfg['pop_size'], max_gen=ebo_cfg['max_gen'],
                           seed=seed, enable_relay=True)
            scenario_results["EBO+Relay"].append({
                "makespan": r['makespan'], "energy": r['energy'],
                "tardiness": r['weighted_tardiness'], "time": time.time() - t0,
                "pareto_size": r['pareto_size'],
                "relay_active": r.get('relay_active', 0),
                "relay_eligible": r.get('relay_eligible', 0),
            })

            print(f"    Greedy: MS={scenario_results['Greedy'][-1]['makespan']:.0f}, "
                  f"EBO: MS={scenario_results['EBO'][-1]['makespan']:.0f}, "
                  f"EBO+R: MS={scenario_results['EBO+Relay'][-1]['makespan']:.0f} "
                  f"(relay={r.get('relay_active',0)})")

        all_results[scenario_name] = scenario_results

    # 统计分析
    print(f"\n\n{'='*90}")
    print("统计分析")
    print(f"{'='*90}")

    for scenario_name, results in all_results.items():
        print(f"\n--- {scenario_name} ---")
        print(f"{'方法':<14} {'Makespan':>20} {'能耗':>20} {'延迟':>20} {'时间(s)':>10}")
        print(f"{'':14} {'mean±std':>20} {'mean±std':>20} {'mean±std':>20} {'mean':>10}")
        print("-" * 90)

        for method, runs in results.items():
            ms = [r['makespan'] for r in runs]
            en = [r['energy'] for r in runs]
            td = [r['tardiness'] for r in runs]
            tm = [r['time'] for r in runs]
            relay_str = ""
            if 'relay_active' in runs[0]:
                avg_relay = np.mean([r.get('relay_active', 0) for r in runs])
                relay_str = f" relay={avg_relay:.1f}"
            print(f"{method:<14} {np.mean(ms):>8.0f}±{np.std(ms):>6.0f} "
                  f"{np.mean(en):>8.0f}±{np.std(en):>6.0f} "
                  f"{np.mean(td):>8.0f}±{np.std(td):>6.0f} "
                  f"{np.mean(tm):>10.1f}{relay_str}")

        # Wilcoxon检验
        print(f"\n  Wilcoxon秩和检验 (EBO+Relay vs others, Makespan):")
        ebo_r_ms = [r['makespan'] for r in results['EBO+Relay']]
        for other in ['Greedy', 'EBO']:
            other_ms = [r['makespan'] for r in results[other]]
            try:
                stat_val, p_val = stats.wilcoxon(ebo_r_ms, other_ms)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
                imp = (np.mean(other_ms) - np.mean(ebo_r_ms)) / np.mean(other_ms) * 100
                print(f"    vs {other:<12}: p={p_val:.4f} {sig}, 改进{imp:+.1f}%")
            except ValueError:
                print(f"    vs {other:<12}: 样本不足")

    return all_results


def run_dispatch_comparison(output_dir):
    """实验2: 派车算法对比 (ETA/NC/SCAN)"""
    print(f"\n\n{'='*70}")
    print("实验2: 派车算法对比")
    print(f"{'='*70}")

    cfg = ScenarioConfig(num_floors=12, num_robots=10, num_elevators=4, num_tasks=30, seed=42)
    robots, elevators, tasks = generate_scenario(cfg)

    results = {}
    for algo in [DispatchAlgorithm.ETA, DispatchAlgorithm.NC, DispatchAlgorithm.SCAN]:
        print(f"\n--- {algo.value} ---")
        t0 = time.time()
        r = run_ebo_v2(robots, elevators, tasks,
                       pop_size=40, max_gen=40, seed=42,
                       dispatch_algo=algo, enable_relay=True)
        elapsed = time.time() - t0
        results[algo.value] = {
            'makespan': r['makespan'], 'energy': r['energy'],
            'tardiness': r['weighted_tardiness'],
            'avg_wait': r['avg_elevator_wait'],
            'relay': r.get('relay_active', 0), 'time': elapsed,
        }
        print(f"  MS={r['makespan']:.0f}, Energy={r['energy']:.0f}, "
              f"AvgWait={r['avg_elevator_wait']:.1f}s, relay={r.get('relay_active',0)}")

    print(f"\n{'算法':<8} {'Makespan':>10} {'能耗':>10} {'延迟':>10} {'AvgWait':>8} {'Relay':>6}")
    print("-" * 56)
    for algo, r in results.items():
        print(f"{algo:<8} {r['makespan']:>10.0f} {r['energy']:>10.0f} "
              f"{r['tardiness']:>10.0f} {r['avg_wait']:>8.1f} {r['relay']:>6}")

    return results


def main():
    output_dir = "results/full_experiment_v2"
    os.makedirs(output_dir, exist_ok=True)

    # 实验1
    algo_results = run_algorithm_comparison(output_dir)

    # 实验2
    dispatch_results = run_dispatch_comparison(output_dir)

    # 保存
    all_data = {
        "algorithm_comparison": {},
        "dispatch_comparison": dispatch_results,
    }
    for s, results in algo_results.items():
        all_data["algorithm_comparison"][s] = results

    with open(os.path.join(output_dir, "all_results.json"), "w") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n结果已保存到 {output_dir}/all_results.json")


if __name__ == "__main__":
    main()
