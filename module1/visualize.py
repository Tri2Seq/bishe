"""可视化模块

1. 机器人Gantt图（颜色标注GPU需求阶段）
2. 电梯时空图（纵轴楼层，横轴时间）
3. Pareto前沿对比图
4. 收敛曲线
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from typing import Dict, List, Optional


# 中文支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ACTION_COLORS = {
    "WALK": "#4CAF50",
    "PICKUP": "#2196F3",
    "DELIVER": "#2196F3",
    "APPROACH_ELEVATOR": "#FF5722",   # GPU需求 - 红
    "PRESS_BUTTON": "#F44336",        # GPU需求 - 深红
    "WAIT_ELEVATOR": "#FFC107",       # 等待 - 黄
    "RIDE_ELEVATOR": "#9C27B0",       # 乘梯 - 紫
    "EXIT_ELEVATOR": "#FF9800",       # GPU需求 - 橙
    "IDLE": "#E0E0E0",
}

GPU_ACTIONS = {"APPROACH_ELEVATOR", "PRESS_BUTTON", "EXIT_ELEVATOR"}


def plot_robot_gantt(timelines_file: str, output_file: str = "gantt.png",
                     max_robots: int = 10, figsize: tuple = (16, 8)):
    """机器人Gantt图"""
    with open(timelines_file) as f:
        data = json.load(f)

    robot_ids = sorted(data.keys(), key=lambda x: int(x))[:max_robots]
    active_robots = [rid for rid in robot_ids if data[rid]]

    fig, ax = plt.subplots(figsize=figsize)

    for idx, rid in enumerate(active_robots):
        actions = data[rid]
        for action in actions:
            start = action['start']
            duration = action['end'] - action['start']
            if duration < 0.1:
                continue
            color = ACTION_COLORS.get(action['type'], '#BDBDBD')
            edgecolor = 'red' if action.get('needs_gpu') else 'none'
            linewidth = 1.5 if action.get('needs_gpu') else 0.5
            ax.barh(idx, duration, left=start, height=0.7,
                    color=color, edgecolor=edgecolor, linewidth=linewidth)

    ax.set_yticks(range(len(active_robots)))
    ax.set_yticklabels([f'Robot {rid}' for rid in active_robots])
    ax.set_xlabel('Time (s)')
    ax.set_title('Robot Schedule Gantt Chart (red border = GPU needed)')
    ax.invert_yaxis()

    # 图例
    legend_patches = [
        mpatches.Patch(color="#4CAF50", label="Walk"),
        mpatches.Patch(color="#2196F3", label="Pick/Deliver"),
        mpatches.Patch(color="#FF5722", label="Approach Elev (GPU)"),
        mpatches.Patch(color="#F44336", label="Press Button (GPU)"),
        mpatches.Patch(color="#FFC107", label="Wait Elevator"),
        mpatches.Patch(color="#9C27B0", label="Ride Elevator"),
        mpatches.Patch(color="#FF9800", label="Exit Elev (GPU)"),
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gantt图已保存: {output_file}")


def plot_elevator_spacetime(timelines_file: str, output_file: str = "elevator_spacetime.png",
                             figsize: tuple = (14, 6)):
    """电梯时空图：纵轴楼层，横轴时间，每部电梯一种颜色"""
    with open(timelines_file) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=figsize)
    elev_colors = ['#E91E63', '#3F51B5', '#009688', '#FF9800', '#795548', '#607D8B']

    all_rides = []
    for rid, actions in data.items():
        for action in actions:
            if action['type'] == 'RIDE_ELEVATOR':
                eid = action.get('elevator_id', 0)
                all_rides.append({
                    'robot': rid, 'elevator': eid,
                    'start': action['start'], 'end': action['end'],
                    'from_floor': action['floor'],
                    'to_floor': action.get('dest_floor', action['floor']),
                })

    for ride in all_rides:
        eid = ride['elevator']
        color = elev_colors[eid % len(elev_colors)]
        ax.plot([ride['start'], ride['end']],
                [ride['from_floor'], ride['to_floor']],
                color=color, linewidth=2.5, alpha=0.8)
        ax.scatter([ride['start']], [ride['from_floor']],
                   color=color, s=30, zorder=5)
        ax.scatter([ride['end']], [ride['to_floor']],
                   color=color, s=30, marker='s', zorder=5)
        ax.annotate(f'R{ride["robot"]}', (ride['start'], ride['from_floor']),
                    fontsize=6, alpha=0.7)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Floor')
    ax.set_title('Elevator Space-Time Diagram')
    ax.grid(True, alpha=0.3)

    # 图例
    seen = set()
    patches = []
    for ride in all_rides:
        eid = ride['elevator']
        if eid not in seen:
            patches.append(mpatches.Patch(
                color=elev_colors[eid % len(elev_colors)],
                label=f'Elevator {eid}'))
            seen.add(eid)
    ax.legend(handles=patches, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"电梯时空图已保存: {output_file}")


def plot_pareto_comparison(nsga_pf: np.ndarray, alns_pf: Optional[np.ndarray] = None,
                            greedy_point: Optional[np.ndarray] = None,
                            output_file: str = "pareto.png"):
    """Pareto前沿对比图（2D投影）"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pairs = [(0, 1, 'Makespan (s)', 'Energy (J)'),
             (0, 2, 'Makespan (s)', 'Weighted Tardiness'),
             (1, 2, 'Energy (J)', 'Weighted Tardiness')]

    for ax, (i, j, xlabel, ylabel) in zip(axes, pairs):
        ax.scatter(nsga_pf[:, i], nsga_pf[:, j], c='#2196F3', s=15,
                   alpha=0.6, label='NSGA-II', zorder=3)
        if alns_pf is not None and len(alns_pf) > 0:
            ax.scatter(alns_pf[:, i], alns_pf[:, j], c='#FF5722', s=15,
                       alpha=0.6, label='ALNS', marker='^', zorder=3)
        if greedy_point is not None:
            ax.scatter([greedy_point[i]], [greedy_point[j]], c='#4CAF50',
                       s=100, marker='*', label='Greedy', zorder=4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Pareto Front Comparison (2D Projections)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Pareto对比图已保存: {output_file}")


def plot_convergence(nsga_history: List[Dict], alns_history: List[Dict],
                      output_file: str = "convergence.png"):
    """收敛曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # NSGA-II
    ax = axes[0]
    gens = [h['gen'] for h in nsga_history]
    ms = [h['best_makespan'] for h in nsga_history]
    ps = [h['pareto_size'] for h in nsga_history]
    ax.plot(gens, ms, 'b-', label='Best Makespan')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Makespan (s)', color='b')
    ax2 = ax.twinx()
    ax2.plot(gens, ps, 'r--', alpha=0.5, label='Pareto Size')
    ax2.set_ylabel('Pareto Size', color='r')
    ax.set_title('NSGA-II Convergence')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # ALNS
    ax = axes[1]
    iters = [h['iter'] for h in alns_history[::10]]  # 每10步采样
    best = [h['best_obj'] for h in alns_history[::10]]
    curr = [h['current_obj'] for h in alns_history[::10]]
    ax.plot(iters, best, 'b-', label='Best', linewidth=2)
    ax.plot(iters, curr, 'gray', alpha=0.3, label='Current')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Weighted Objective')
    ax.set_title('ALNS Convergence')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"收敛曲线已保存: {output_file}")


if __name__ == "__main__":
    import os
    os.makedirs("results/figures", exist_ok=True)

    # 生成Gantt图
    if os.path.exists("results/sim_small_timelines.json"):
        plot_robot_gantt("results/sim_small_timelines.json",
                          "results/figures/gantt_small.png")
        plot_elevator_spacetime("results/sim_small_timelines.json",
                                 "results/figures/elevator_small.png")

    if os.path.exists("results/small_timelines.json"):
        plot_robot_gantt("results/small_timelines.json",
                          "results/figures/gantt_milp_small.png")

    # Pareto + 收敛（需要先运行NSGA-II/ALNS）
    from config import ScenarioConfig, generate_scenario
    from nsga2 import NSGA2Solver
    from alns import ALNSSolver
    from simulator import Simulator
    from baselines import greedy_fcfs

    cfg = ScenarioConfig(num_floors=10, num_robots=10, num_elevators=4, num_tasks=30, seed=42)
    robots, elevators, tasks = generate_scenario(cfg)

    nsga = NSGA2Solver(robots, elevators, tasks, pop_size=80, max_gen=100, seed=42)
    nsga_result = nsga.solve(verbose=False)

    alns = ALNSSolver(robots, elevators, tasks, max_iter=5000, seed=42)
    alns_result = alns.solve(verbose=False)

    greedy = greedy_fcfs(robots, elevators, tasks)
    sim = Simulator(robots, elevators, tasks)
    sim.load_schedule(greedy['assignment'], greedy['task_sequence'], greedy['elevator_assignment'])
    greedy_sim = sim.run()
    greedy_point = np.array([greedy_sim['makespan'], greedy_sim['energy'],
                             greedy_sim['weighted_tardiness']])

    nsga_pf = np.array(nsga_result['pareto_front'])
    alns_pf = np.array(alns_result['pareto_front'])

    plot_pareto_comparison(nsga_pf, alns_pf, greedy_point,
                            "results/figures/pareto_medium.png")
    plot_convergence(nsga_result['gen_history'], alns_result['history'],
                      "results/figures/convergence_medium.png")

    print("\n所有图表已生成!")
