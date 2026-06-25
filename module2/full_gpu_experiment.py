"""模块2完整实验

对比5种GPU调度策略，在不同GPU紧张程度下测试。
使用模块1的仿真器生成更真实的GPU需求时间线。
"""

import sys
import os
import json
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'module1'))

from config import ScenarioConfig, generate_scenario
from simulator import Simulator
from baselines import greedy_fcfs
from nsga2 import NSGA2Solver
from gpu_config import GPUConfig, GPU_CONFIGS, load_gpu_demands, GPUDemand
from gpu_scheduler import (FragAwareScheduler, OnDemandScheduler,
                            FixedScheduler, RoundRobinScheduler)
from gpu_milp import solve_gpu_milp


def generate_demands_via_sim(robots, elevators, tasks, seed=42):
    """用NSGA-II+仿真器生成高质量调度方案的GPU需求"""
    nsga = NSGA2Solver(robots, elevators, tasks, pop_size=50, max_gen=30, seed=seed)
    result = nsga.solve(verbose=False)

    # 取Pareto前沿中makespan最优的解
    best_ind = result['pareto_individuals'][0]
    for ind in result['pareto_individuals']:
        if ind.objectives[0] < best_ind.objectives[0]:
            best_ind = ind

    # 解码并仿真
    assignment = {i: int(best_ind.gene_assign[i]) for i in range(len(tasks))}
    task_sequence = {j: [] for j in range(len(robots))}
    for j in range(len(robots)):
        assigned = [i for i in range(len(tasks)) if assignment[i] == j]
        assigned.sort(key=lambda i: best_ind.gene_priority[i])
        task_sequence[j] = assigned

    cf_ids = set(i for i, t in enumerate(tasks) if t.is_cross_floor)
    elevator_assignment = {i: int(best_ind.gene_elev[i]) for i in cf_ids}

    sim = Simulator(robots, elevators, tasks)
    sim.load_schedule(assignment, task_sequence, elevator_assignment)
    sim.run()

    # 提取GPU需求
    demands = {}
    for rid, records in sim.action_records.items():
        gpu_segs = []
        for rec in records:
            if rec.needs_gpu:
                gpu_segs.append(GPUDemand(
                    robot_id=rid, start=rec.start, end=rec.end,
                    demand_type="demand", elevator_id=rec.elevator_id, floor=rec.floor))
            elif rec.type == "RIDE_ELEVATOR":
                gpu_segs.append(GPUDemand(
                    robot_id=rid, start=rec.start, end=rec.end,
                    demand_type="idle", reason="RIDING"))
            elif rec.type == "WALK" and rec.end - rec.start > 1.0:
                gpu_segs.append(GPUDemand(
                    robot_id=rid, start=rec.start, end=rec.end,
                    demand_type="idle", reason="WALKING"))
        if gpu_segs:
            demands[rid] = gpu_segs

    return demands


def run_gpu_sensitivity(output_dir="results/gpu_experiment"):
    """GPU紧张程度灵敏度分析"""
    os.makedirs(output_dir, exist_ok=True)

    # 固定物理场景（中型），变化GPU配置
    cfg = ScenarioConfig(num_floors=10, num_robots=15, num_elevators=4, num_tasks=40, seed=42)
    robots, elevators, tasks = generate_scenario(cfg)

    print("生成GPU需求时间线...")
    demands = generate_demands_via_sim(robots, elevators, tasks)
    print(f"  机器人数: {len(demands)}")

    total_demands = sum(1 for segs in demands.values()
                        for s in segs if s.demand_type == "demand")
    print(f"  总GPU需求段: {total_demands}")

    # GPU配置梯度：从宽松到紧张
    gpu_configs = [
        ("宽松(8GPU)", GPUConfig(num_servers=2, gpus_per_server=4, vram_per_gpu=24, depth_model_vram=4)),
        ("适中(4GPU)", GPUConfig(num_servers=1, gpus_per_server=4, vram_per_gpu=24, depth_model_vram=4)),
        ("紧张(2GPU)", GPUConfig(num_servers=1, gpus_per_server=2, vram_per_gpu=24, depth_model_vram=4)),
        ("极紧(2GPU-大模型)", GPUConfig(num_servers=1, gpus_per_server=2, vram_per_gpu=16, depth_model_vram=6)),
    ]

    schedulers = {
        "FragAware": lambda cfg: FragAwareScheduler(cfg),
        "OnDemand": lambda cfg: OnDemandScheduler(cfg),
        "Fixed": lambda cfg: FixedScheduler(cfg),
        "RoundRobin": lambda cfg: RoundRobinScheduler(cfg),
    }

    all_results = {}

    for gpu_name, gpu_cfg in gpu_configs:
        print(f"\n{'='*60}")
        print(f"GPU配置: {gpu_name} (总并发={gpu_cfg.total_concurrent})")
        print(f"{'='*60}")

        config_results = {}
        for sched_name, make_sched in schedulers.items():
            scheduler = make_sched(gpu_cfg)
            metrics = scheduler.schedule(demands)
            summary = metrics.summary()
            config_results[sched_name] = summary

        # MILP（仅小规模）
        if total_demands <= 100:
            print("  运行GPU MILP...")
            milp_result = solve_gpu_milp(demands, gpu_cfg, time_limit=30, verbose=False)
            if milp_result:
                config_results["MILP"] = {
                    "total_demands": total_demands,
                    "migrations": milp_result['migrations'],
                    "avg_fragmentation": milp_result['imbalance'] / max(gpu_cfg.total_gpus, 1),
                    "gpu_load": milp_result['gpu_load_distribution'],
                }

        all_results[gpu_name] = config_results

        # 打印
        print(f"\n{'策略':<15} {'需求':>6} {'等待(s)':>10} {'冷启':>5} {'迁移':>5} {'碎片率':>8}")
        print("-" * 55)
        for name, s in config_results.items():
            if name == "MILP":
                print(f"{'MILP':<15} {s.get('total_demands',''):>6} {'N/A':>10} "
                      f"{'N/A':>5} {s.get('migrations',''):>5} {s.get('avg_fragmentation',0):>8.4f}")
            else:
                print(f"{name:<15} {s['total_demands']:>6} {s['total_wait_delay']:>10.2f} "
                      f"{s['cold_starts']:>5} {s['migrations']:>5} {s['avg_fragmentation']:>8.4f}")

    # 保存
    with open(os.path.join(output_dir, "gpu_sensitivity.json"), "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {output_dir}/gpu_sensitivity.json")

    return all_results


def run_scale_experiment(output_dir="results/gpu_experiment"):
    """规模扩展实验"""
    os.makedirs(output_dir, exist_ok=True)

    scales = [
        ("10robots",  ScenarioConfig(num_floors=5,  num_robots=10, num_elevators=2, num_tasks=20, seed=42)),
        ("20robots",  ScenarioConfig(num_floors=10, num_robots=20, num_elevators=4, num_tasks=50, seed=42)),
        ("30robots",  ScenarioConfig(num_floors=15, num_robots=30, num_elevators=4, num_tasks=80, seed=42)),
    ]

    gpu_cfg = GPUConfig(num_servers=1, gpus_per_server=4, vram_per_gpu=24, depth_model_vram=4)

    print(f"\n{'='*60}")
    print("GPU调度规模扩展实验")
    print(f"GPU配置: 4GPU, 24并发")
    print(f"{'='*60}")

    for scale_name, cfg in scales:
        print(f"\n--- {scale_name} ---")
        robots, elevators, tasks = generate_scenario(cfg)
        demands = generate_demands_via_sim(robots, elevators, tasks)

        total_d = sum(1 for segs in demands.values() for s in segs if s.demand_type == "demand")
        print(f"  机器人: {len(demands)}, GPU需求段: {total_d}")

        for sched_name, make_sched in [
            ("FragAware", lambda: FragAwareScheduler(gpu_cfg)),
            ("OnDemand", lambda: OnDemandScheduler(gpu_cfg)),
            ("RoundRobin", lambda: RoundRobinScheduler(gpu_cfg)),
        ]:
            t0 = time.time()
            scheduler = make_sched()
            metrics = scheduler.schedule(demands)
            dt = time.time() - t0
            s = metrics.summary()
            print(f"    {sched_name:<12}: wait={s['total_wait_delay']:.1f}s, "
                  f"cold={s['cold_starts']}, frag={s['avg_fragmentation']:.4f}, "
                  f"time={dt:.3f}s")


if __name__ == "__main__":
    run_gpu_sensitivity()
    run_scale_experiment()
