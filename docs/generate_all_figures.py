"""生成文档所需的全部图表"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'module1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'module2'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

OUT = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})


# ===================================================================
# 图1: 24小时任务分布
# ===================================================================
def fig_daily_pattern():
    from dynamic_scenario import DayScenario, generate_day_tasks, analyze_day_pattern
    scenario = DayScenario(num_floors=15, num_robots=20, num_elevators=4,
                           num_gpu_servers=2, gpus_per_server=4)
    tasks = generate_day_tasks(scenario, seed=42)
    pattern = analyze_day_pattern(tasks)

    hours = range(24)
    total = pattern['hourly_distribution']
    cf = pattern['hourly_crossfloor']

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(hours, total, color='#42A5F5', label='Total tasks', alpha=0.8)
    ax.bar(hours, cf, color='#EF5350', label='Cross-floor tasks', alpha=0.8)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Number of Tasks')
    ax.set_title('24-Hour Task Arrival Pattern')
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 标注高峰低谷
    peak_h = pattern['peak_hour']
    valley_h = pattern['valley_hour']
    ax.annotate(f'Peak: {int(total[peak_h])} tasks',
                xy=(peak_h, total[peak_h]), xytext=(peak_h+1, total[peak_h]+3),
                arrowprops=dict(arrowstyle='->', color='red'), color='red', fontweight='bold')
    ax.annotate(f'Valley: {int(total[valley_h])}',
                xy=(valley_h, total[valley_h]), xytext=(valley_h-3, total[valley_h]+8),
                arrowprops=dict(arrowstyle='->', color='gray'), color='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'daily_pattern.png'), bbox_inches='tight')
    plt.close()
    print('  ✓ daily_pattern.png')


# ===================================================================
# 图2: Gantt图（V2仿真，带GPU标注）
# ===================================================================
def fig_gantt_v2():
    from dynamic_scenario import DayScenario, generate_day_tasks, generate_day_robots, generate_day_elevators
    from simulator_v2 import SimulatorV2, RobotStrategy
    from elevator_group_control import DispatchAlgorithm
    from baselines import greedy_fcfs

    scenario = DayScenario(num_floors=10, num_robots=8, num_elevators=3,
                           num_gpu_servers=1, gpus_per_server=4)
    tasks = generate_day_tasks(scenario, seed=42)[:20]
    robots = generate_day_robots(scenario, seed=42)[:8]
    elevators = generate_day_elevators(scenario, seed=42)[:3]
    for i, t in enumerate(tasks):
        t.id = i
        t.earliest_start = 0

    greedy = greedy_fcfs(robots, elevators, tasks)
    sim = SimulatorV2(robots, elevators, tasks, dispatch_algorithm=DispatchAlgorithm.ETA, dt=0.5)
    sim.load_schedule(greedy['assignment'], greedy['task_sequence'])
    sim.run(max_time=3000)

    COLORS = {
        'WALK_TO_ELEVATOR': '#66BB6A',
        'APPROACH_ELEVATOR': '#FF7043',
        'PRESS_BUTTON': '#EF5350',
        'WAIT_ELEVATOR': '#FFA726',
        'RIDE_ELEVATOR': '#AB47BC',
        'EXIT_ELEVATOR': '#FF8A65',
        'PICKUP': '#42A5F5',
        'DELIVER': '#29B6F6',
        'BATCH_PICKUP': '#26C6DA',
        'START_TASK': None,
        'EXIT_DONE': None,
    }

    fig, ax = plt.subplots(figsize=(14, 5))
    active_robots = []
    for rid in range(len(robots)):
        actions = [a for a in sim.action_log if a.robot_id == rid and COLORS.get(a.action) is not None]
        if actions:
            active_robots.append(rid)

    for idx, rid in enumerate(active_robots[:8]):
        actions = [a for a in sim.action_log if a.robot_id == rid and COLORS.get(a.action) is not None]
        for a in actions:
            dur = a.end_time - a.start_time
            if dur < 0.5:
                dur = 2.0
            color = COLORS.get(a.action, '#BDBDBD')
            if color is None:
                continue
            ec = 'red' if a.needs_gpu else 'none'
            lw = 2.0 if a.needs_gpu else 0.3
            ax.barh(idx, dur, left=a.start_time, height=0.65,
                    color=color, edgecolor=ec, linewidth=lw)

    ax.set_yticks(range(len(active_robots[:8])))
    ax.set_yticklabels([f'Robot {r}' for r in active_robots[:8]])
    ax.set_xlabel('Time (s)')
    ax.set_title('Robot Schedule Gantt Chart (red border = GPU required)')
    ax.invert_yaxis()

    legend_items = [
        mpatches.Patch(color='#42A5F5', label='Pick / Deliver'),
        mpatches.Patch(color='#66BB6A', label='Walk'),
        mpatches.Patch(color='#FF7043', label='Approach Elev (GPU)'),
        mpatches.Patch(color='#EF5350', label='Press Button (GPU)'),
        mpatches.Patch(color='#FFA726', label='Wait Elevator'),
        mpatches.Patch(color='#AB47BC', label='Ride Elevator'),
        mpatches.Patch(color='#FF8A65', label='Exit Elev (GPU)'),
    ]
    ax.legend(handles=legend_items, loc='upper right', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'gantt_v2.png'), bbox_inches='tight')
    plt.close()
    print('  ✓ gantt_v2.png')


# ===================================================================
# 图3: 电梯时空图
# ===================================================================
def fig_elevator_spacetime():
    from dynamic_scenario import DayScenario, generate_day_tasks, generate_day_robots, generate_day_elevators
    from simulator_v2 import SimulatorV2
    from elevator_group_control import DispatchAlgorithm
    from baselines import greedy_fcfs

    scenario = DayScenario(num_floors=10, num_robots=8, num_elevators=3,
                           num_gpu_servers=1, gpus_per_server=4)
    tasks = generate_day_tasks(scenario, seed=42)[:20]
    robots = generate_day_robots(scenario, seed=42)[:8]
    elevators = generate_day_elevators(scenario, seed=42)[:3]
    for i, t in enumerate(tasks):
        t.id = i; t.earliest_start = 0

    greedy = greedy_fcfs(robots, elevators, tasks)
    sim = SimulatorV2(robots, elevators, tasks, dispatch_algorithm=DispatchAlgorithm.ETA, dt=0.5)
    sim.load_schedule(greedy['assignment'], greedy['task_sequence'])
    sim.run(max_time=3000)

    elev_colors = ['#E91E63', '#3F51B5', '#4CAF50', '#FF9800']
    fig, ax = plt.subplots(figsize=(12, 5))

    for a in sim.action_log:
        if a.action == 'RIDE_ELEVATOR' and a.end_time > a.start_time:
            eid = a.elevator_id if a.elevator_id >= 0 else 0
            # 找对应的dest_floor
            dest = a.floor  # 默认
            for call in sim.egcs.call_history:
                if call.robot_id == a.robot_id and abs(call.call_time - a.start_time) < 60:
                    dest = call.dest_floor
                    break
            color = elev_colors[eid % len(elev_colors)]
            ax.plot([a.start_time, a.end_time], [a.floor, dest],
                    color=color, linewidth=2.5, alpha=0.8)
            ax.scatter([a.start_time], [a.floor], color=color, s=25, zorder=5)
            ax.scatter([a.end_time], [dest], color=color, s=25, marker='s', zorder=5)
            ax.annotate(f'R{a.robot_id}', (a.start_time, a.floor), fontsize=7, alpha=0.7)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Floor')
    ax.set_title('Elevator Space-Time Diagram')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    patches = [mpatches.Patch(color=elev_colors[i], label=f'Elevator {i}') for i in range(3)]
    ax.legend(handles=patches, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'elevator_spacetime.png'), bbox_inches='tight')
    plt.close()
    print('  ✓ elevator_spacetime.png')


# ===================================================================
# 图4: 算法对比柱状图
# ===================================================================
def fig_algorithm_comparison():
    # 使用之前的实验结果
    methods = ['Greedy', 'NSGA-II', 'ALNS', 'NSGA-II\n-EBO']
    makespan = [640, 479, 753, 439]
    energy = [17682, 16114, 15258, 15690]
    tardiness = [613, 40, 972, 0]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, data, title, color in [
        (axes[0], makespan, 'Makespan (s)', '#42A5F5'),
        (axes[1], energy, 'Energy (J)', '#66BB6A'),
        (axes[2], tardiness, 'Weighted Tardiness', '#EF5350'),
    ]:
        bars = ax.bar(methods, data, color=color, alpha=0.85, edgecolor='white', linewidth=1.5)
        ax.set_title(title, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        # 标注最优值
        best_idx = np.argmin(data)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(2.5)
        for i, v in enumerate(data):
            ax.text(i, v + max(data)*0.02, str(v), ha='center', fontsize=9)

    fig.suptitle('Algorithm Comparison (50 tasks, 15 floors, 15 robots, 4 elevators)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'algorithm_comparison.png'), bbox_inches='tight')
    plt.close()
    print('  ✓ algorithm_comparison.png')


# ===================================================================
# 图5: Pareto前沿对比
# ===================================================================
def fig_pareto():
    from config import ScenarioConfig, generate_scenario
    from nsga2 import NSGA2Solver, compute_hypervolume
    from nsga2_ebo import NSGA2EBOSolver
    from simulator_v2 import SimulatorV2
    from baselines import greedy_fcfs
    from elevator_group_control import DispatchAlgorithm

    cfg = ScenarioConfig(num_floors=10, num_robots=10, num_elevators=4, num_tasks=30, seed=42)
    robots, elevators, tasks = generate_scenario(cfg)

    # NSGA-II (V1)
    nsga = NSGA2Solver(robots, elevators, tasks, pop_size=50, max_gen=50, seed=42)
    nr = nsga.solve(verbose=False)
    npf = np.array(nr['pareto_front'])

    # NSGA-II-EBO (V2)
    ebo = NSGA2EBOSolver(robots, elevators, tasks, pop_size=30, max_gen=30,
                          dispatch_algorithm=DispatchAlgorithm.ETA, seed=42)
    er = ebo.solve(verbose=False)
    epf = np.array(er['pareto_front'])

    # Greedy
    greedy = greedy_fcfs(robots, elevators, tasks)
    sim = SimulatorV2(robots, elevators, tasks, dispatch_algorithm=DispatchAlgorithm.ETA)
    sim.load_schedule(greedy['assignment'], greedy['task_sequence'])
    gr = sim.run(max_time=5000)
    gp = np.array([gr['makespan'], gr['energy'], gr['weighted_tardiness']])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (i, j, xl, yl) in zip(axes, [
        (0, 1, 'Makespan (s)', 'Energy (J)'),
        (0, 2, 'Makespan (s)', 'Weighted Tardiness'),
    ]):
        ax.scatter(npf[:, i], npf[:, j], c='#42A5F5', s=20, alpha=0.6, label='NSGA-II')
        ax.scatter(epf[:, i], epf[:, j], c='#FF7043', s=25, alpha=0.7, label='NSGA-II-EBO', marker='^')
        ax.scatter([gp[i]], [gp[j]], c='#4CAF50', s=120, marker='*', label='Greedy', zorder=5)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Pareto Front Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'pareto_comparison.png'), bbox_inches='tight')
    plt.close()
    print('  ✓ pareto_comparison.png')


# ===================================================================
# 图6: GPU调度对比
# ===================================================================
def fig_gpu_comparison():
    methods = ['Predictive', 'FragAware', 'OnDemand']

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # 等待延迟
    wait = [0, 0, 2688]
    axes[0].bar(methods, wait, color=['#4CAF50', '#42A5F5', '#EF5350'], alpha=0.85)
    axes[0].set_title('Wait Delay (s)', fontweight='bold')
    axes[0].set_ylabel('Seconds')
    for i, v in enumerate(wait):
        axes[0].text(i, v + 50, str(v), ha='center', fontsize=10)

    # 冷启动
    cold = [18, 821, 337]
    axes[1].bar(methods, cold, color=['#4CAF50', '#42A5F5', '#EF5350'], alpha=0.85)
    axes[1].set_title('Cold Starts', fontweight='bold')
    for i, v in enumerate(cold):
        axes[1].text(i, v + 15, str(v), ha='center', fontsize=10)

    # 碎片率
    frag = [0.102, 0.009, 0.922]
    axes[2].bar(methods, frag, color=['#4CAF50', '#42A5F5', '#EF5350'], alpha=0.85)
    axes[2].set_title('Avg Fragmentation', fontweight='bold')
    for i, v in enumerate(frag):
        axes[2].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

    fig.suptitle('GPU Scheduling Comparison (Tight: 2 GPUs, 4 concurrent, 1233 demands)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'gpu_comparison.png'), bbox_inches='tight')
    plt.close()
    print('  ✓ gpu_comparison.png')


# ===================================================================
# 图7: 派车算法对比
# ===================================================================
def fig_dispatch_comparison():
    algos = ['ETA', 'NC', 'SCAN']
    completed = [270, 238, 238]
    avg_wait = [9.6, 9.9, 10.4]
    energy = [266868, 224527, 226686]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].bar(algos, completed, color=['#4CAF50', '#42A5F5', '#FF9800'], alpha=0.85)
    axes[0].set_title('Completed Tasks / 319', fontweight='bold')
    axes[0].set_ylim(200, 290)
    for i, v in enumerate(completed):
        axes[0].text(i, v + 2, str(v), ha='center', fontsize=11, fontweight='bold')

    axes[1].bar(algos, avg_wait, color=['#4CAF50', '#42A5F5', '#FF9800'], alpha=0.85)
    axes[1].set_title('Avg Elevator Wait (s)', fontweight='bold')
    axes[1].set_ylim(8, 12)
    for i, v in enumerate(avg_wait):
        axes[1].text(i, v + 0.1, f'{v:.1f}', ha='center', fontsize=11)

    axes[2].bar(algos, [e/1000 for e in energy], color=['#4CAF50', '#42A5F5', '#FF9800'], alpha=0.85)
    axes[2].set_title('Total Energy (kJ)', fontweight='bold')
    for i, v in enumerate(energy):
        axes[2].text(i, v/1000 + 3, f'{v/1000:.0f}', ha='center', fontsize=11)

    fig.suptitle('Elevator Dispatch Algorithm Comparison (24h, 319 tasks)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'dispatch_comparison.png'), bbox_inches='tight')
    plt.close()
    print('  ✓ dispatch_comparison.png')


# ===================================================================
# 主函数
# ===================================================================
if __name__ == '__main__':
    print('生成文档图表...')
    fig_daily_pattern()
    fig_gantt_v2()
    fig_elevator_spacetime()
    fig_algorithm_comparison()
    fig_gpu_comparison()
    fig_dispatch_comparison()
    fig_pareto()
    print(f'\n全部图表已保存到 {OUT}/')
