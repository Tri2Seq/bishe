"""24小时全链路实验

完整的一天仿真：
  1. 生成24h周期任务流(~300个任务)
  2. 分批用NSGA-II-EBO优化调度(含中继决策)
  3. V2群控仿真执行
  4. 收集GPU需求时间线 → 传给模块2
  5. 对比派车算法 × 优化方法 × GPU调度策略

实验矩阵：
  派车: ETA / NC / SCAN
  优化: Greedy / EBO+Relay
  GPU:  Predictive / FragAware / OnDemand
"""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'module2'))

from config import ScenarioConfig, generate_scenario
from dynamic_scenario import DayScenario, generate_day_tasks, generate_day_robots, generate_day_elevators, analyze_day_pattern
from simulator_v2 import SimulatorV2, RobotStrategy
from elevator_group_control import DispatchAlgorithm
from baselines import greedy_fcfs
from ebdco import LowerLevelSolver
from nsga2_ebo import NSGA2EBOSolver
from gpu_config import GPUConfig, GPUDemand
from gpu_scheduler import FragAwareScheduler, OnDemandScheduler, PredictiveScheduler


def run_greedy_schedule(robots, elevators, tasks):
    """Greedy + 下层序列优化"""
    greedy = greedy_fcfs(robots, elevators, tasks)
    lower = LowerLevelSolver(robots, elevators, tasks)
    seq = lower.optimize_sequence(greedy['assignment'], {}, weight=np.array([0.5, 0.3, 0.2]))
    return greedy['assignment'], seq, {}, RobotStrategy()


def run_ebo_schedule(robots, elevators, tasks, dispatch_algo, seed=42):
    """EBO + 中继优化"""
    solver = NSGA2EBOSolver(robots, elevators, tasks,
                             pop_size=30, max_gen=25,
                             dispatch_algorithm=dispatch_algo, seed=seed)
    solver.solve(verbose=False)
    return solver.get_best_schedule(objective_idx=0)


def simulate_day(robots, elevators, tasks, assignment, seq, relay_plans,
                 strategy, dispatch_algo, dt=1.0):
    """Run V2 simulation for a full day"""
    sim = SimulatorV2(robots, elevators, tasks,
                      dispatch_algorithm=dispatch_algo, dt=dt, strategy=strategy)
    sim.load_schedule(assignment, seq, relay_plans=relay_plans)
    result = sim.run(max_time=86400)
    return sim, result


def extract_gpu_demands(sim: SimulatorV2):
    """从V2仿真器提取GPU需求时间线"""
    demands = {}
    for rid, rs in enumerate(sim.robot_states):
        segs = []
        for a in sim.action_log:
            if a.robot_id == rid and a.needs_gpu:
                segs.append(GPUDemand(robot_id=rid, start=a.start_time, end=a.end_time,
                                      demand_type='demand'))
            elif a.robot_id == rid and a.action == 'RIDE_ELEVATOR':
                segs.append(GPUDemand(robot_id=rid, start=a.start_time, end=a.end_time,
                                      demand_type='idle', reason='RIDING'))
        if segs:
            demands[rid] = segs
    return demands


def run_day_experiment(output_dir="results/day_experiment"):
    os.makedirs(output_dir, exist_ok=True)

    scenario = DayScenario(num_floors=15, num_robots=20, num_elevators=4,
                           num_gpu_servers=1, gpus_per_server=4)
    all_tasks = generate_day_tasks(scenario, seed=42)
    robots = generate_day_robots(scenario, seed=42)
    elevators = generate_day_elevators(scenario, seed=42)
    pattern = analyze_day_pattern(all_tasks)

    print(f"{'='*60}")
    print(f"24小时全链路实验")
    print(f"{'='*60}")
    print(f"  楼层: {scenario.num_floors}, 机器人: {scenario.num_robots}, 电梯: {scenario.num_elevators}")
    print(f"  总任务: {len(all_tasks)}, 跨楼层: {pattern['cross_floor_ratio']:.0%}")
    print(f"  高峰: {pattern['peak_hour']}:00 ({pattern['peak_tasks']}个), "
          f"低谷: {pattern['valley_hour']}:00 ({pattern['valley_tasks']}个)")

    # ===== 准备任务列表 =====
    day_tasks = []
    for t in all_tasks:
        tc = type(t)(id=len(day_tasks), origin_floor=t.origin_floor,
                     dest_floor=t.dest_floor, weight=t.weight,
                     priority=t.priority, deadline=t.deadline,
                     earliest_start=t.earliest_start)
        day_tasks.append(tc)

    # ===== 实验1: 派车算法×优化方法 对比 =====
    print(f"\n{'='*60}")
    print("实验1: 派车算法×优化方法 对比")
    print(f"{'='*60}")

    results_scheduling = {}

    for algo in [DispatchAlgorithm.ETA, DispatchAlgorithm.NC, DispatchAlgorithm.SCAN]:
        for opt_name, opt_fn in [
            ("Greedy", lambda t, a: run_greedy_schedule(robots, elevators, t)),
            ("EBO+Relay", lambda t, a: run_ebo_schedule(robots, elevators, t, a, seed=42)),
        ]:
            key = f"{algo.value}+{opt_name}"
            print(f"\n--- {key} ---")
            t0 = time.time()
            assignment, seq, relay_plans, strategy = opt_fn(day_tasks, algo)
            opt_time = time.time() - t0
            sim, r = simulate_day(robots, elevators, day_tasks,
                                  assignment, seq, relay_plans, strategy, algo)
            total_time = time.time() - t0
            results_scheduling[key] = {
                'completed': r['completed_tasks'],
                'total': len(day_tasks),
                'makespan': r['makespan'],
                'energy': r['energy'],
                'tardiness': r['weighted_tardiness'],
                'avg_wait': r['avg_elevator_wait'],
                'gpu_time': r['gpu_time'],
                'opt_time': opt_time,
                'total_time': total_time,
            }
            print(f"  完成: {r['completed_tasks']}/{len(day_tasks)}, "
                  f"Makespan: {r['makespan']:.0f}s ({r['makespan']/3600:.1f}h), "
                  f"AvgWait: {r['avg_elevator_wait']:.1f}s, "
                  f"Energy: {r['energy']:.0f}J, "
                  f"Time: {total_time:.1f}s")

    # ===== 实验2: GPU调度对比 =====
    print(f"\n{'='*60}")
    print("实验2: GPU调度对比")
    print(f"{'='*60}")

    # 用ETA+EBO的仿真提取GPU需求
    print("  使用 ETA+EBO+Relay 仿真结果提取GPU需求...")
    assignment, seq, relay_plans, strategy = run_ebo_schedule(
        robots, elevators, day_tasks, DispatchAlgorithm.ETA, seed=42)
    sim_for_gpu, _ = simulate_day(robots, elevators, day_tasks,
                                  assignment, seq, relay_plans, strategy,
                                  DispatchAlgorithm.ETA)
    gpu_demands = extract_gpu_demands(sim_for_gpu)
    total_d = sum(1 for segs in gpu_demands.values() for s in segs if s.demand_type == 'demand')
    print(f"  GPU需求段: {total_d}")

    gpu_configs = [
        ("宽松(4GPU/24并发)", GPUConfig(num_servers=1, gpus_per_server=4)),
        ("紧张(2GPU/4并发)", GPUConfig(num_servers=1, gpus_per_server=2,
                                       vram_per_gpu=16, depth_model_vram=6)),
    ]

    results_gpu = {}
    for cfg_name, gpu_cfg in gpu_configs:
        print(f"\n  GPU配置: {cfg_name}")
        gpu_results = {}
        for sched_name, make_sched in [
            ("Predictive", lambda: PredictiveScheduler(gpu_cfg)),
            ("FragAware", lambda: FragAwareScheduler(gpu_cfg)),
            ("OnDemand", lambda: OnDemandScheduler(gpu_cfg)),
        ]:
            s = make_sched()
            m = s.schedule(gpu_demands)
            r = m.summary()
            gpu_results[sched_name] = r
            print(f"    {sched_name:<12}: wait={r['total_wait_delay']:.0f}s, "
                  f"cold={r['cold_starts']}, frag={r['avg_fragmentation']:.3f}")
        results_gpu[cfg_name] = gpu_results

    # ===== 保存 =====
    all_results = {
        "scenario": {
            "floors": scenario.num_floors,
            "robots": scenario.num_robots,
            "elevators": scenario.num_elevators,
            "total_tasks": len(all_tasks),
            "peak_hour": pattern['peak_hour'],
            "valley_hour": pattern['valley_hour'],
        },
        "scheduling_comparison": results_scheduling,
        "gpu_comparison": results_gpu,
        "task_pattern": {
            "hourly": pattern['hourly_distribution'],
        },
    }
    with open(os.path.join(output_dir, "day_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {output_dir}/day_results.json")

    # ===== 汇总 =====
    print(f"\n{'='*60}")
    print("24小时实验汇总")
    print(f"{'='*60}")

    print(f"\n调度对比:")
    print(f"{'组合':<18} {'完成':>6} {'Makespan':>10} {'能耗':>10} {'AvgWait':>8} {'时间(s)':>8}")
    print("-" * 64)
    for key, r in results_scheduling.items():
        print(f"{key:<18} {r['completed']:>6} {r['makespan']:>10.0f} "
              f"{r['energy']:>10.0f} {r['avg_wait']:>8.1f} {r['total_time']:>8.1f}")

    print(f"\nGPU调度对比 (紧张配置):")
    tight_name = [k for k in results_gpu if "紧张" in k]
    if tight_name:
        tight = results_gpu[tight_name[0]]
        print(f"{'策略':<12} {'等待(s)':>10} {'冷启动':>8} {'碎片率':>8}")
        print("-" * 42)
        for name, r in tight.items():
            print(f"{name:<12} {r['total_wait_delay']:>10.0f} {r['cold_starts']:>8} {r['avg_fragmentation']:>8.3f}")

    return all_results


if __name__ == "__main__":
    run_day_experiment()
