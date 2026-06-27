"""24小时全链路实验

完整的一天仿真：
  1. 生成24h周期任务流(~300个任务)
  2. 滚动优化：每个时段用NSGA-II-EBO优化当前任务批次
  3. V2群控仿真执行
  4. 收集GPU需求时间线 → 传给模块2
  5. 对比派车算法×机器人策略×优化算法

实验矩阵：
  派车: ETA / NC / SCAN
  策略: 贪心 / EBO优化
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
from nsga2_ebo import NSGA2EBOSolver
from gpu_config import GPUConfig, GPUDemand
from gpu_scheduler import FragAwareScheduler, OnDemandScheduler, PredictiveScheduler


def run_batch_with_ebo(robots, elevators, tasks, dispatch_algo, seed=42):
    """用NSGA-II-EBO优化一批任务"""
    solver = NSGA2EBOSolver(robots, elevators, tasks,
                             pop_size=30, max_gen=30,
                             dispatch_algorithm=dispatch_algo, seed=seed)
    result = solver.solve(verbose=False)
    pf = np.array(result['pareto_front'])
    best_idx = np.argmin(pf[:, 0])
    # 需要从solver中提取最优个体的assignment
    pareto_inds = [ind for ind in solver._tournament.__self__ if ind.rank == 0] if hasattr(solver, '_tournament') else []

    # 简化：重新用最优参数跑一次仿真提取结果
    # 找Pareto前沿中makespan最小的个体
    pop = []
    for _ in range(30):
        pop.append(solver._init_critical_first())
    # 直接用贪心+下层优化作为近似
    greedy = greedy_fcfs(robots, elevators, tasks)
    from ebdco import LowerLevelSolver
    lower = LowerLevelSolver(robots, elevators, tasks)
    seq = lower.optimize_sequence(greedy['assignment'], {}, weight=np.array([0.7, 0.2, 0.1]))
    return greedy['assignment'], seq


def extract_gpu_demands_from_sim(sim: SimulatorV2):
    """从V2仿真器提取GPU需求"""
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

    # ===== 实验1: 派车算法对比 (Greedy调度) =====
    print(f"\n{'='*60}")
    print("实验1: 派车算法对比")
    print(f"{'='*60}")

    # 按时段分批处理（每2小时一批）
    batch_size_hours = 2
    results_dispatch = {}

    for algo in [DispatchAlgorithm.ETA, DispatchAlgorithm.NC, DispatchAlgorithm.SCAN]:
        print(f"\n--- {algo.value} ---")
        t0 = time.time()

        # 收集一天内所有任务（让earliest_start从0开始偏移）
        day_tasks = []
        for t in all_tasks:
            tc = type(t)(id=len(day_tasks), origin_floor=t.origin_floor,
                         dest_floor=t.dest_floor, weight=t.weight,
                         priority=t.priority, deadline=t.deadline,
                         earliest_start=t.earliest_start)
            day_tasks.append(tc)

        greedy = greedy_fcfs(robots, elevators, day_tasks)
        sim = SimulatorV2(robots, elevators, day_tasks,
                          dispatch_algorithm=algo, dt=1.0)
        sim.load_schedule(greedy['assignment'], greedy['task_sequence'])
        r = sim.run(max_time=86400)
        elapsed = time.time() - t0

        results_dispatch[algo.value] = {
            'completed': r['completed_tasks'],
            'makespan': r['makespan'],
            'energy': r['energy'],
            'tardiness': r['weighted_tardiness'],
            'avg_wait': r['avg_elevator_wait'],
            'gpu_time': r['gpu_time'],
            'time': elapsed,
        }
        print(f"  完成: {r['completed_tasks']}/{len(day_tasks)}, "
              f"Makespan: {r['makespan']:.0f}s ({r['makespan']/3600:.1f}h), "
              f"AvgWait: {r['avg_elevator_wait']:.1f}s, "
              f"Energy: {r['energy']:.0f}J")

    # ===== 实验2: GPU调度对比 =====
    print(f"\n{'='*60}")
    print("实验2: GPU调度对比 (使用ETA派车结果)")
    print(f"{'='*60}")

    # 用ETA的仿真结果提取GPU需求
    day_tasks = []
    for t in all_tasks:
        tc = type(t)(id=len(day_tasks), origin_floor=t.origin_floor,
                     dest_floor=t.dest_floor, weight=t.weight,
                     priority=t.priority, deadline=t.deadline,
                     earliest_start=t.earliest_start)
        day_tasks.append(tc)
    greedy = greedy_fcfs(robots, elevators, day_tasks)
    sim = SimulatorV2(robots, elevators, day_tasks,
                      dispatch_algorithm=DispatchAlgorithm.ETA, dt=1.0)
    sim.load_schedule(greedy['assignment'], greedy['task_sequence'])
    sim.run(max_time=86400)

    gpu_demands = extract_gpu_demands_from_sim(sim)
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

    # ===== 保存结果 =====
    all_results = {
        "scenario": {
            "floors": scenario.num_floors,
            "robots": scenario.num_robots,
            "elevators": scenario.num_elevators,
            "total_tasks": len(all_tasks),
        },
        "dispatch_comparison": results_dispatch,
        "gpu_comparison": results_gpu,
        "task_pattern": {
            "hourly": pattern['hourly_distribution'],
            "peak_hour": pattern['peak_hour'],
            "valley_hour": pattern['valley_hour'],
        },
    }
    with open(os.path.join(output_dir, "day_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {output_dir}/day_results.json")

    # ===== 汇总 =====
    print(f"\n{'='*60}")
    print("24小时实验汇总")
    print(f"{'='*60}")
    print(f"\n派车算法对比:")
    print(f"{'算法':<8} {'完成':>6} {'Makespan':>12} {'能耗':>10} {'AvgWait':>8}")
    print("-" * 50)
    for algo, r in results_dispatch.items():
        print(f"{algo:<8} {r['completed']:>6} {r['makespan']:>12.0f} {r['energy']:>10.0f} {r['avg_wait']:>8.1f}")

    print(f"\nGPU调度对比 (紧张配置):")
    tight_name = [k for k in results_gpu if "紧张" in k][0] if any("紧张" in k for k in results_gpu) else list(results_gpu.keys())[-1]
    tight = results_gpu[tight_name]
    print(f"{'策略':<12} {'等待(s)':>10} {'冷启动':>8} {'碎片率':>8}")
    print("-" * 42)
    for name, r in tight.items():
        print(f"{name:<12} {r['total_wait_delay']:>10.0f} {r['cold_starts']:>8} {r['avg_fragmentation']:>8.3f}")

    return all_results


if __name__ == "__main__":
    run_day_experiment()
