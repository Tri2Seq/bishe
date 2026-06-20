"""模块1实验主入口

运行不同规模实例，对比MILP求解与基线算法，生成动作时间线供模块2使用。
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List

from config import ScenarioConfig, generate_scenario, Priority
from milp_solver import solve_milp
from baselines import greedy_fcfs, independent_planning
from timeline_generator import (generate_timeline, extract_gpu_demands,
                                 save_timelines, save_gpu_demands,
                                 print_timeline_summary)


SCENARIOS = {
    "small": ScenarioConfig(num_floors=5, num_robots=5, num_elevators=2, num_tasks=10, seed=42),
    "medium": ScenarioConfig(num_floors=10, num_robots=10, num_elevators=4, num_tasks=30, seed=42),
    "large": ScenarioConfig(num_floors=15, num_robots=15, num_elevators=4, num_tasks=50, seed=42),
}


def run_single_experiment(scenario_name: str, cfg: ScenarioConfig,
                          output_dir: str = "results") -> Dict:
    """运行单个场景的实验"""
    print(f"\n{'='*60}")
    print(f"场景: {scenario_name}")
    print(f"  楼层={cfg.num_floors}, 机器人={cfg.num_robots}, "
          f"电梯={cfg.num_elevators}, 任务={cfg.num_tasks}")
    print(f"{'='*60}")

    robots, elevators, tasks = generate_scenario(cfg)

    # 统计
    cross_floor = sum(1 for t in tasks if t.is_cross_floor)
    print(f"\n任务统计:")
    print(f"  跨楼层任务: {cross_floor}/{len(tasks)} ({cross_floor/len(tasks)*100:.0f}%)")
    print(f"  紧急任务: {sum(1 for t in tasks if t.priority == Priority.URGENT)}")
    print(f"  普通任务: {sum(1 for t in tasks if t.priority == Priority.NORMAL)}")
    print(f"  批量任务: {sum(1 for t in tasks if t.priority == Priority.BATCH)}")
    print(f"  重量范围: {min(t.weight for t in tasks):.0f} ~ {max(t.weight for t in tasks):.0f} kg")

    results = {}

    # 1. MILP求解
    print(f"\n--- MILP求解 ---")
    t0 = time.time()
    milp_result = solve_milp(robots, elevators, tasks,
                              alpha=(0.5, 0.3, 0.2),
                              time_limit=120)
    milp_time = time.time() - t0
    if milp_result:
        milp_result["solve_time"] = milp_time
        results["MILP"] = milp_result

        # 生成时间线
        timelines = generate_timeline(robots, elevators, tasks, milp_result)
        print_timeline_summary(timelines)

        # 提取GPU需求
        gpu_demands = extract_gpu_demands(timelines)

        # 保存
        os.makedirs(output_dir, exist_ok=True)
        save_timelines(timelines, os.path.join(output_dir, f"{scenario_name}_timelines.json"))
        save_gpu_demands(gpu_demands, os.path.join(output_dir, f"{scenario_name}_gpu_demands.json"))

        # GPU需求统计
        total_gpu_demands = 0
        total_gpu_time = 0
        total_idle_time = 0
        for rid, segments in gpu_demands.items():
            for seg in segments:
                if seg["type"] == "demand":
                    total_gpu_demands += 1
                    total_gpu_time += seg["end"] - seg["start"]
                elif seg["type"] == "idle":
                    total_idle_time += seg["duration"]

        print(f"\n  GPU需求统计:")
        print(f"    总GPU需求段数: {total_gpu_demands}")
        print(f"    总GPU需求时长: {total_gpu_time:.1f}s")
        print(f"    总GPU空闲窗口时长: {total_idle_time:.1f}s")
        print(f"    GPU需求/空闲比: {total_gpu_time/max(total_idle_time,1):.2f}")

    # 2. Greedy-FCFS基线
    print(f"\n--- Greedy-FCFS ---")
    t0 = time.time()
    greedy_result = greedy_fcfs(robots, elevators, tasks)
    greedy_time = time.time() - t0
    greedy_result["solve_time"] = greedy_time
    results["Greedy-FCFS"] = greedy_result
    print(f"  Makespan: {greedy_result['makespan']:.1f}s")
    print(f"  能耗: {greedy_result['energy']:.1f}J")
    print(f"  加权延迟: {greedy_result['weighted_tardiness']:.1f}")
    print(f"  求解时间: {greedy_time:.3f}s")

    # 3. 独立规划基线
    print(f"\n--- 独立规划 ---")
    t0 = time.time()
    indep_result = independent_planning(robots, elevators, tasks)
    indep_time = time.time() - t0
    indep_result["solve_time"] = indep_time
    results["Independent"] = indep_result
    print(f"  Makespan: {indep_result['makespan']:.1f}s")
    print(f"  能耗: {indep_result['energy']:.1f}J")
    print(f"  加权延迟: {indep_result['weighted_tardiness']:.1f}")
    print(f"  求解时间: {indep_time:.3f}s")

    # 4. 仅Makespan的MILP
    print(f"\n--- 仅Makespan MILP ---")
    t0 = time.time()
    makespan_only = solve_milp(robots, elevators, tasks,
                                alpha=(1.0, 0.0, 0.0),
                                time_limit=120, verbose=False)
    makespan_time = time.time() - t0
    if makespan_only:
        makespan_only["solve_time"] = makespan_time
        results["MILP-Makespan"] = makespan_only
        print(f"  Makespan: {makespan_only['makespan']:.1f}s")
        print(f"  能耗: {makespan_only['energy']:.1f}J")
        print(f"  加权延迟: {makespan_only['weighted_tardiness']:.1f}")
        print(f"  求解时间: {makespan_time:.3f}s")

    # 汇总对比
    print(f"\n{'='*60}")
    print(f"对比汇总: {scenario_name}")
    print(f"{'='*60}")
    header = f"{'方法':<18} {'Makespan(s)':>12} {'能耗(J)':>10} {'加权延迟':>10} {'求解时间(s)':>12}"
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        print(f"{name:<18} {res['makespan']:>12.1f} {res['energy']:>10.1f} "
              f"{res['weighted_tardiness']:>10.1f} {res['solve_time']:>12.3f}")

    # 保存汇总
    summary = {name: {
        "makespan": res["makespan"],
        "energy": res["energy"],
        "weighted_tardiness": res["weighted_tardiness"],
        "solve_time": res["solve_time"],
    } for name, res in results.items()}
    with open(os.path.join(output_dir, f"{scenario_name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return results


def main():
    scenario = sys.argv[1] if len(sys.argv) > 1 else "small"
    output_dir = os.path.join(os.path.dirname(__file__), "results")

    if scenario == "all":
        for name, cfg in SCENARIOS.items():
            run_single_experiment(name, cfg, output_dir)
    elif scenario in SCENARIOS:
        run_single_experiment(scenario, SCENARIOS[scenario], output_dir)
    else:
        print(f"未知场景: {scenario}")
        print(f"可用场景: {list(SCENARIOS.keys())} 或 'all'")
        sys.exit(1)


if __name__ == "__main__":
    main()
