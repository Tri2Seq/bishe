"""模块2实验主入口

读取模块1输出的GPU需求时间表，运行不同调度策略并对比。
"""

import sys
import os
import json
import numpy as np

from gpu_config import GPUConfig, GPU_CONFIGS, load_gpu_demands
from gpu_scheduler import (FragAwareScheduler, OnDemandScheduler,
                            FixedScheduler, RoundRobinScheduler)


def run_gpu_experiment(demands_file: str, gpu_config_name: str = "small"):
    """运行GPU调度实验"""
    config = GPU_CONFIGS[gpu_config_name]
    demands = load_gpu_demands(demands_file)

    print(f"\n{'='*60}")
    print(f"GPU调度实验")
    print(f"  GPU配置: {gpu_config_name} ({config.total_gpus} GPUs, "
          f"每GPU {config.max_concurrent_per_gpu} 并发, "
          f"总并发 {config.total_concurrent})")
    print(f"  机器人数: {len(demands)}")
    print(f"{'='*60}")

    # 统计需求
    total_demands = 0
    total_demand_time = 0
    total_idle_time = 0
    max_concurrent = 0

    all_events = []
    for robot_id, segments in demands.items():
        for seg in segments:
            if seg.demand_type == "demand":
                total_demands += 1
                total_demand_time += seg.duration
                all_events.append((seg.start, 1))
                all_events.append((seg.end, -1))
            elif seg.demand_type == "idle":
                total_idle_time += seg.duration

    # 计算最大并发需求
    all_events.sort()
    concurrent = 0
    for t, delta in all_events:
        concurrent += delta
        max_concurrent = max(max_concurrent, concurrent)

    print(f"\nGPU需求统计:")
    print(f"  总需求段数: {total_demands}")
    print(f"  总GPU需求时长: {total_demand_time:.1f}s")
    print(f"  总空闲窗口时长: {total_idle_time:.1f}s")
    print(f"  最大并发GPU需求: {max_concurrent}")
    print(f"  GPU并发上限: {config.total_concurrent}")
    print(f"  峰值利用率: {max_concurrent/config.total_concurrent*100:.1f}%")

    # 运行各调度策略
    schedulers = {
        "FragAware(本文)": lambda: FragAwareScheduler(config),
        "OnDemand(B5)": lambda: OnDemandScheduler(config),
        "Fixed(B6)": lambda: FixedScheduler(config),
        "RoundRobin(B7)": lambda: RoundRobinScheduler(config),
    }

    results = {}
    for name, make_scheduler in schedulers.items():
        scheduler = make_scheduler()
        metrics = scheduler.schedule(demands)
        results[name] = metrics.summary()

    # 打印对比表
    print(f"\n{'='*60}")
    print(f"GPU调度策略对比")
    print(f"{'='*60}")

    header = (f"{'策略':<20} {'需求数':>6} {'等待延迟(s)':>12} {'冷启动':>6} "
              f"{'迁移':>4} {'碎片率':>8} {'GPU时长(s)':>10}")
    print(header)
    print("-" * len(header))

    for name, summary in results.items():
        print(f"{name:<20} {summary['total_demands']:>6} "
              f"{summary['total_wait_delay']:>12.2f} "
              f"{summary['cold_starts']:>6} "
              f"{summary['migrations']:>4} "
              f"{summary['avg_fragmentation']:>8.4f} "
              f"{summary['total_gpu_time']:>10.1f}")

    # 保存结果
    output_dir = os.path.dirname(demands_file)
    scenario = os.path.basename(demands_file).replace("_gpu_demands.json", "")
    result_file = os.path.join(output_dir, f"{scenario}_gpu_scheduling_results.json")
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {result_file}")

    return results


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "module1", "results")

    if len(sys.argv) > 1:
        scenario = sys.argv[1]
    else:
        scenario = "small"

    demands_file = os.path.join(base_dir, f"{scenario}_gpu_demands.json")
    if not os.path.exists(demands_file):
        print(f"文件不存在: {demands_file}")
        print("请先运行模块1生成GPU需求数据")
        sys.exit(1)

    gpu_config = sys.argv[2] if len(sys.argv) > 2 else "small"
    run_gpu_experiment(demands_file, gpu_config)


if __name__ == "__main__":
    main()
