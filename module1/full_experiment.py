"""完整对比实验

4个规模 × 4种方法 × 5次独立运行（seed不同）
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
from nsga2 import NSGA2Solver, compute_hypervolume
from alns import ALNSSolver
from simulator import Simulator
from baselines import greedy_fcfs, independent_planning


SCENARIOS = {
    "S1_small":  ScenarioConfig(num_floors=5,  num_robots=5,  num_elevators=2, num_tasks=10),
    "S2_medium": ScenarioConfig(num_floors=10, num_robots=10, num_elevators=4, num_tasks=30),
    "S3_large":  ScenarioConfig(num_floors=15, num_robots=15, num_elevators=4, num_tasks=50),
}

METHOD_CONFIGS = {
    "NSGA-II": {"pop_size": 60, "max_gen": 80},
    "ALNS": {"max_iter": 3000},
}

N_RUNS = 5
SEEDS = [42, 123, 456, 789, 1024]


def evaluate_baseline_with_sim(robots, elevators, tasks, baseline_result):
    sim = Simulator(robots, elevators, tasks)
    sim.load_schedule(baseline_result['assignment'], baseline_result['task_sequence'],
                      baseline_result['elevator_assignment'])
    return sim.run()


def run_full_experiment(output_dir="results/full_experiment"):
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for scenario_name, cfg_template in SCENARIOS.items():
        print(f"\n{'#'*70}")
        print(f"场景: {scenario_name}")
        print(f"{'#'*70}")

        scenario_results = {
            "Greedy-FCFS": [], "Independent": [], "NSGA-II": [], "ALNS": []
        }

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

            # Greedy-FCFS
            t0 = time.time()
            greedy = greedy_fcfs(robots, elevators, tasks)
            greedy_sim = evaluate_baseline_with_sim(robots, elevators, tasks, greedy)
            greedy_time = time.time() - t0
            scenario_results["Greedy-FCFS"].append({
                "makespan": greedy_sim['makespan'],
                "energy": greedy_sim['energy'],
                "tardiness": greedy_sim['weighted_tardiness'],
                "time": greedy_time,
            })

            # Independent
            t0 = time.time()
            indep = independent_planning(robots, elevators, tasks)
            indep_sim = evaluate_baseline_with_sim(robots, elevators, tasks, indep)
            indep_time = time.time() - t0
            scenario_results["Independent"].append({
                "makespan": indep_sim['makespan'],
                "energy": indep_sim['energy'],
                "tardiness": indep_sim['weighted_tardiness'],
                "time": indep_time,
            })

            # NSGA-II
            t0 = time.time()
            nsga_cfg = METHOD_CONFIGS["NSGA-II"]
            nsga = NSGA2Solver(robots, elevators, tasks,
                                pop_size=nsga_cfg["pop_size"],
                                max_gen=nsga_cfg["max_gen"],
                                seed=seed)
            nsga_result = nsga.solve(verbose=False)
            nsga_time = time.time() - t0
            pf = np.array(nsga_result['pareto_front'])
            scenario_results["NSGA-II"].append({
                "makespan": float(pf[:, 0].min()),
                "energy": float(pf[:, 1].min()),
                "tardiness": float(pf[:, 2].min()),
                "time": nsga_time,
                "pareto_size": nsga_result['pareto_size'],
                "hv": float(compute_hypervolume(pf,
                    np.array([pf[:,0].max()*1.2, pf[:,1].max()*1.2, pf[:,2].max()*1.2]))),
            })

            # ALNS
            t0 = time.time()
            alns_cfg = METHOD_CONFIGS["ALNS"]
            alns = ALNSSolver(robots, elevators, tasks,
                               max_iter=alns_cfg["max_iter"],
                               seed=seed)
            alns_result = alns.solve(verbose=False)
            alns_time = time.time() - t0
            scenario_results["ALNS"].append({
                "makespan": alns_result['best_objectives'][0],
                "energy": alns_result['best_objectives'][1],
                "tardiness": alns_result['best_objectives'][2],
                "time": alns_time,
                "pareto_size": alns_result['pareto_size'],
            })

            print(f"    Greedy: MS={greedy_sim['makespan']:.0f}, "
                  f"NSGA: MS={pf[:,0].min():.0f}, ALNS: MS={alns_result['best_objectives'][0]:.0f}")

        all_results[scenario_name] = scenario_results

    # 统计分析
    print(f"\n\n{'='*80}")
    print("统计分析")
    print(f"{'='*80}")

    stat_tables = {}
    for scenario_name, results in all_results.items():
        print(f"\n--- {scenario_name} ---")
        print(f"{'方法':<15} {'Makespan':>20} {'能耗':>20} {'延迟':>20} {'时间(s)':>12}")
        print(f"{'':15} {'mean±std':>20} {'mean±std':>20} {'mean±std':>20} {'mean':>12}")
        print("-" * 90)

        stat_table = {}
        for method, runs in results.items():
            ms = [r['makespan'] for r in runs]
            en = [r['energy'] for r in runs]
            td = [r['tardiness'] for r in runs]
            tm = [r['time'] for r in runs]
            stat_table[method] = {
                'makespan': (np.mean(ms), np.std(ms)),
                'energy': (np.mean(en), np.std(en)),
                'tardiness': (np.mean(td), np.std(td)),
                'time': np.mean(tm),
            }
            print(f"{method:<15} {np.mean(ms):>8.0f}±{np.std(ms):>6.0f} "
                  f"{np.mean(en):>8.0f}±{np.std(en):>6.0f} "
                  f"{np.mean(td):>8.0f}±{np.std(td):>6.0f} "
                  f"{np.mean(tm):>12.2f}")

        # Wilcoxon检验: NSGA-II vs 每个对比方法
        print(f"\n  Wilcoxon秩和检验 (NSGA-II vs others, Makespan):")
        nsga_ms = [r['makespan'] for r in results['NSGA-II']]
        for other in ['Greedy-FCFS', 'Independent', 'ALNS']:
            other_ms = [r['makespan'] for r in results[other]]
            if len(set(nsga_ms)) > 1 and len(set(other_ms)) > 1:
                try:
                    stat_val, p_val = stats.wilcoxon(nsga_ms, other_ms)
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
                    print(f"    vs {other:<15}: p={p_val:.4f} {sig}")
                except ValueError:
                    print(f"    vs {other:<15}: 样本不足/相同")
            else:
                print(f"    vs {other:<15}: 数据方差不足")

        stat_tables[scenario_name] = stat_table

    # 保存
    serializable = {}
    for s, results in all_results.items():
        serializable[s] = {}
        for method, runs in results.items():
            serializable[s][method] = runs
    with open(os.path.join(output_dir, "all_results.json"), "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {output_dir}/all_results.json")

    return all_results, stat_tables


if __name__ == "__main__":
    run_full_experiment()
