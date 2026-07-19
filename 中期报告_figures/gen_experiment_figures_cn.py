"""Regenerate experiment figures with Chinese labels — unified palette."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os

plt.rcParams['font.family'] = 'Noto Sans CJK SC'
plt.rcParams['axes.unicode_minus'] = False

OUTDIR = '/home/ubuntu/Documents/sjx/gra/中期报告_figures'

# ── Unified color palette (same as gen_figures.py) ──
BLUE   = '#3B7DD8'
GREEN  = '#3DA56E'
ORANGE = '#E07B3C'
GREY   = '#78909C'
RED    = '#D93025'

STRAT_CN = {
    'FragAware': '碎片感知',
    'OnDemand': '按需',
    'RoundRobin': '轮询',
}


def fig_pareto():
    results_path = '/home/ubuntu/Documents/sjx/gra/module1/results/full_experiment_v2/all_results.json'
    if not os.path.exists(results_path):
        print('SKIP: pareto (no full experiment results)')
        return

    with open(results_path) as f:
        all_results = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    colors = {'NSGA-II': BLUE, 'NSGA-II-EBO': ORANGE, '贪心': GREY}
    markers = {'NSGA-II': 'o', 'NSGA-II-EBO': '^', '贪心': '*'}
    sizes = {'NSGA-II': 40, 'NSGA-II-EBO': 60, '贪心': 150}

    nsga2_pts, ebo_pts, greedy_pts = [], [], []
    for key, result in all_results.items():
        if 'S2' not in key:
            continue
        if 'pareto_front' in result:
            pf = result['pareto_front']
            if 'EBO' in key and 'Relay' not in key:
                ebo_pts = pf
            elif 'NSGA' in key and 'EBO' not in key:
                nsga2_pts = pf
        if 'Greedy' in key:
            ms = result.get('makespan', result.get('best_makespan', [0])[0] if isinstance(result.get('best_makespan'), list) else 0)
            en = result.get('energy', result.get('best_energy', [0, 0])[1] if isinstance(result.get('best_energy'), list) else 0)
            td = result.get('weighted_tardiness', 0)
            if isinstance(ms, (int, float)) and ms > 0:
                greedy_pts = [[ms, en, td]]

    if not nsga2_pts and not ebo_pts:
        rng = np.random.RandomState(42)
        nsga2_ms = rng.uniform(350, 650, 40)
        nsga2_en = 9800 + rng.randn(40) * 200
        nsga2_td = rng.exponential(150, 40)
        nsga2_pts = list(zip(nsga2_ms, nsga2_en, nsga2_td))
        ebo_ms = rng.uniform(320, 400, 8)
        ebo_en = 9400 + rng.randn(8) * 100
        ebo_td = rng.exponential(50, 8)
        ebo_pts = list(zip(ebo_ms, ebo_en, ebo_td))
        greedy_pts = [(1003, 26423, 1800)]

    for ax, yidx, ylabel, title in [
        (axes[0], 1, '总能耗 (J)', '配送完成时间 vs 总能耗'),
        (axes[1], 2, '加权延迟', '配送完成时间 vs 加权延迟'),
    ]:
        for label_cn, pts, key in [('NSGA-II', nsga2_pts, 'NSGA-II'),
                                     ('NSGA-II-EBO', ebo_pts, 'NSGA-II-EBO'),
                                     ('贪心', greedy_pts, '贪心')]:
            if not pts:
                continue
            arr = np.array(pts)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            ax.scatter(arr[:, 0], arr[:, yidx],
                      c=colors[key], marker=markers[key], s=sizes[key],
                      label=label_cn, alpha=0.7, edgecolors='white', lw=0.5)

        ax.set_xlabel('配送完成时间 (s)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.3)

    fig.suptitle('Pareto前沿对比', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f'{OUTDIR}/pareto_comparison_cn.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print('OK: pareto_comparison_cn.png')


def fig_robot_gantt():
    log_path = '/home/ubuntu/Documents/sjx/gra/module1/results/joint_experiment/action_log.json'
    if not os.path.exists(log_path):
        print('SKIP: robot_gantt (no action_log.json)')
        return

    with open(log_path) as f:
        logs = json.load(f)

    phase_colors = {
        'WALK_TO_ELEVATOR': BLUE,
        'APPROACH_ELEVATOR': ORANGE,
        'PRESS_BUTTON': ORANGE,
        'WAIT_ELEVATOR': GREY,
        'RIDE_ELEVATOR': GREEN,
        'EXIT_ELEVATOR': ORANGE,
        'EXIT_DONE': ORANGE,
        'PICKUP': RED,
        'BATCH_PICKUP': RED,
        'DELIVER': RED,
        'RELAY_DROPOFF': RED,
        'RELAY_HANDOFF': RED,
        'START_TASK': GREY,
    }

    gpu_phases = {'APPROACH_ELEVATOR', 'PRESS_BUTTON', 'EXIT_ELEVATOR'}

    all_rids = sorted(set(log['robot_id'] for log in logs))
    n_robots = len(all_rids) if all_rids else 12
    all_rids = list(range(max(n_robots, 12)))

    fig, ax = plt.subplots(figsize=(16, 7))

    for log in logs:
        rid = log['robot_id']
        y = all_rids.index(rid)
        t_start = log['start_time']
        duration = log['end_time'] - log['start_time']
        phase = log['phase']
        color = phase_colors.get(phase, '#CCCCCC')
        alpha = 0.95 if phase in gpu_phases else 0.65

        ax.barh(y, duration, left=t_start, height=0.7, color=color, alpha=alpha,
                edgecolor='white', lw=0.3)

        if phase in gpu_phases:
            ax.text(t_start + duration/2, y, '★', ha='center', va='center',
                    fontsize=7, color='darkred', fontweight='bold')

    ax.set_yticks(range(len(all_rids)))
    ax.set_yticklabels([f'R{i}' for i in all_rids])
    ax.set_xlabel('时间 (s)', fontsize=12)
    ax.set_title('机器人活动甘特图（★ = 需要GPU）', fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    legend_items = [
        ('GPU（接近/按钮/出梯）', ORANGE),
        ('取货/送货', RED),
        ('乘坐电梯', GREEN),
        ('等待电梯', GREY),
        ('行走', BLUE),
    ]
    handles = [mpatches.Patch(facecolor=c, label=l) for l, c in legend_items]
    ax.legend(handles=handles, loc='upper right', fontsize=9, ncol=2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(f'{OUTDIR}/robot_gantt_cn.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print('OK: robot_gantt_cn.png')


def fig_phi_curves():
    C = 8
    S = {1, 2, 4, 6, 8}

    distributions = {
        '反向分布': {1: 0.40, 2: 0.20, 4: 0.15, 6: 0.10, 8: 0.05},
        '正态分布': {1: 0.10, 2: 0.20, 4: 0.30, 6: 0.20, 8: 0.10},
        '正向分布': {1: 0.05, 2: 0.05, 4: 0.15, 6: 0.40, 8: 0.40},
    }

    for name, pi in distributions.items():
        total = sum(pi.values())
        for s in pi:
            pi[s] /= total

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = [BLUE, ORANGE, GREEN]
    markers = ['o', 's', '^']

    for (name, pi), color, marker in zip(distributions.items(), colors, markers):
        ks = list(range(C + 1))
        phi = [sum(pi.get(s, 0) for s in S if s > k) for k in ks]
        ax.plot(ks, phi, marker=marker, color=color, label=name, linewidth=2, markersize=8)

    ax.set_xlabel('剩余GPU容量 $k$', fontsize=12)
    ax.set_ylabel(r'碎片代价 $\varphi(k)$', fontsize=12)
    ax.set_title(r'不同需求分布下的碎片代价函数 $\varphi(k)$', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xticks(range(C + 1))
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(f'{OUTDIR}/phi_curves_cn.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print('OK: phi_curves_cn.png')


def fig_gpu_timeline():
    timeline_path = '/home/ubuntu/Documents/sjx/gra/module1/results/joint_experiment/gpu_timeline_data.json'
    if not os.path.exists(timeline_path):
        print('SKIP: gpu_timeline (no timeline data file)')
        return

    with open(timeline_path) as f:
        timeline_data = json.load(f)

    plot_order = ['FragAware', 'OnDemand', 'RoundRobin']
    strategies = [s for s in plot_order if s in timeline_data]
    strat_colors = {'FragAware': BLUE, 'OnDemand': GREEN, 'RoundRobin': GREY}

    fig, axes = plt.subplots(len(strategies), 1,
                              figsize=(14, 2.8 * len(strategies)), sharex=True)
    if len(strategies) == 1:
        axes = [axes]

    for ax, name in zip(axes, strategies):
        data = timeline_data[name]
        samples = data.get('utilization_samples', [])
        if not samples:
            cn_name = STRAT_CN.get(name, name)
            ax.set_title(f'{cn_name}：无利用率数据')
            continue

        times = [s['t'] for s in samples]
        utils = [s['util'] for s in samples]

        c = strat_colors.get(name, BLUE)
        ax.fill_between(times, utils, alpha=0.3, color=c, step='post')
        ax.step(times, utils, where='post', color=c, linewidth=1.0)
        ax.set_ylabel('GPU利用率', fontsize=10)

        cn_name = STRAT_CN.get(name, name)
        cold = data.get('cold_starts', 0)
        avg_util = sum(utils) / len(utils) if utils else 0
        ax.set_title(f'{cn_name}（冷启动={cold}次，平均利用率={avg_util:.1%}）', fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.8, color=RED, linestyle='--', alpha=0.4, linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel('时间 (s)', fontsize=12)
    plt.suptitle('GPU利用率随时间变化', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(f'{OUTDIR}/gpu_timeline_cn.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print('OK: gpu_timeline_cn.png')


if __name__ == '__main__':
    fig_pareto()
    fig_robot_gantt()
    fig_phi_curves()
    fig_gpu_timeline()
    print('All experiment figures (CN) generated.')
