"""Generate figures and tables for the GPU scheduling thesis chapter."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

from frag_cluster import DISTRIBUTIONS, DEMAND_SIZES, phi_table

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['legend.fontsize'] = 10
rcParams['figure.dpi'] = 150

RESULT_DIR = "results/thesis_gpu"
FIG_DIR = "results/thesis_gpu/figures"

METHOD_LABELS = {
    'Ours': 'FragMig (Ours)',
    'BF': 'Best Fit (BF)',
    'SC': 'Server Consolidation (SC)',
    'MismatchPred': 'Inaccurate Prediction',
}

METHOD_COLORS = {
    'Ours': '#2166ac',
    'BF': '#92c5de',
    'SC': '#f4a582',
    'MismatchPred': '#b2182b',
}

DIST_LABELS = {'inv': 'Inverse', 'norm': 'Normal', 'dir': 'Direct'}


def fig1_phi_curves():
    """Figure: φ(k) fragmentation cost curves for three distributions."""
    fig, ax = plt.subplots(figsize=(6, 4))
    C = 8
    ks = list(range(C + 1))

    for dist_name, style in [('inv', '-o'), ('norm', '-s'), ('dir', '-^')]:
        vals = phi_table(DISTRIBUTIONS[dist_name], C)
        ax.plot(ks, vals, style, label=DIST_LABELS[dist_name],
                markersize=6, linewidth=1.5)

    ax.set_xlabel('Remaining GPU capacity $k$')
    ax.set_ylabel('Fragmentation cost $\\phi(k)$')
    ax.set_xticks(ks)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "phi_curves.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "phi_curves.png"))
    plt.close(fig)
    print("  phi_curves done")


def fig2_demand_distributions():
    """Figure: Three demand distributions as bar charts."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)

    for ax, (dist_name, probs) in zip(axes, DISTRIBUTIONS.items()):
        bars = ax.bar([str(s) for s in DEMAND_SIZES], probs,
                      color=['#4393c3', '#92c5de', '#fddbc7', '#f4a582', '#d6604d'],
                      edgecolor='black', linewidth=0.5)
        ax.set_title(DIST_LABELS[dist_name])
        ax.set_xlabel('GPU demand size $s$')
        if ax == axes[0]:
            ax.set_ylabel('Probability $\\pi_s$')
        ax.set_ylim(0, 0.5)
        for bar, p in zip(bars, probs):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{p:.2f}', ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "demand_distributions.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "demand_distributions.png"))
    plt.close(fig)
    print("  demand_distributions done")


def _load_exp1():
    with open(os.path.join(RESULT_DIR, "exp1_main.json")) as f:
        return json.load(f)


def fig3_waiting_time_bars():
    """Figure: Average waiting time comparison across methods and distributions."""
    data = _load_exp1()
    methods = ['Ours', 'BF', 'SC', 'MismatchPred']
    dists = ['inv', 'norm', 'dir']

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=False)
    x = np.arange(len(methods))
    width = 0.6

    for ax, dist_name in zip(axes, dists):
        means = []
        stds = []
        for m in methods:
            vals = [r['avg_wait'] for r in data[dist_name][m]]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        bars = ax.bar(x, means, width, yerr=stds, capsize=3,
                      color=[METHOD_COLORS[m] for m in methods],
                      edgecolor='black', linewidth=0.5)
        ax.set_title(f'{DIST_LABELS[dist_name]}')
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m].split('(')[0].strip() for m in methods],
                           rotation=20, ha='right', fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel('Average waiting time $\\bar{w}$ (h)')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "waiting_time.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "waiting_time.png"))
    plt.close(fig)
    print("  waiting_time done")


def fig4_queue_cardtime_bars():
    """Figure: Average queue card-time comparison."""
    data = _load_exp1()
    methods = ['Ours', 'BF', 'SC', 'MismatchPred']
    dists = ['inv', 'norm', 'dir']

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=False)
    x = np.arange(len(methods))
    width = 0.6

    for ax, dist_name in zip(axes, dists):
        means = []
        stds = []
        for m in methods:
            vals = [r['avg_queue_cardtime'] for r in data[dist_name][m]]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        ax.bar(x, means, width, yerr=stds, capsize=3,
               color=[METHOD_COLORS[m] for m in methods],
               edgecolor='black', linewidth=0.5)
        ax.set_title(f'{DIST_LABELS[dist_name]}')
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m].split('(')[0].strip() for m in methods],
                           rotation=20, ha='right', fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel('Average queue card-time $\\bar{Q}$ (GPU·h)')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "queue_cardtime.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "queue_cardtime.png"))
    plt.close(fig)
    print("  queue_cardtime done")


def fig5_utilization_bars():
    """Figure: Average GPU utilization comparison."""
    data = _load_exp1()
    methods = ['Ours', 'BF', 'SC', 'MismatchPred']
    dists = ['inv', 'norm', 'dir']

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    x = np.arange(len(methods))
    width = 0.6

    for ax, dist_name in zip(axes, dists):
        means = []
        stds = []
        for m in methods:
            vals = [r['avg_utilization'] * 100 for r in data[dist_name][m]]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        ax.bar(x, means, width, yerr=stds, capsize=3,
               color=[METHOD_COLORS[m] for m in methods],
               edgecolor='black', linewidth=0.5)
        ax.set_title(f'{DIST_LABELS[dist_name]}')
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m].split('(')[0].strip() for m in methods],
                           rotation=20, ha='right', fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel('Average utilization $U$ (%)')
        ax.set_ylim(50, 100)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "utilization.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "utilization.png"))
    plt.close(fig)
    print("  utilization done")


def fig6_makespan_bars():
    """Figure: Average makespan comparison."""
    data = _load_exp1()
    methods = ['Ours', 'BF', 'SC', 'MismatchPred']
    dists = ['inv', 'norm', 'dir']

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=False)
    x = np.arange(len(methods))
    width = 0.6

    for ax, dist_name in zip(axes, dists):
        means = []
        stds = []
        for m in methods:
            vals = [r['makespan'] for r in data[dist_name][m]]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        ax.bar(x, means, width, yerr=stds, capsize=3,
               color=[METHOD_COLORS[m] for m in methods],
               edgecolor='black', linewidth=0.5)
        ax.set_title(f'{DIST_LABELS[dist_name]}')
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m].split('(')[0].strip() for m in methods],
                           rotation=20, ha='right', fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel('Makespan (h)')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "makespan.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "makespan.png"))
    plt.close(fig)
    print("  makespan done")


def fig7_scale_sensitivity():
    """Figure: Performance vs. cluster scale."""
    with open(os.path.join(RESULT_DIR, "exp2_scale.json")) as f:
        data = json.load(f)

    server_counts = sorted(int(k) for k in data.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for method, style in [('Ours', '-o'), ('BF', '-s'), ('SC', '-^')]:
        waits = [np.mean([r['avg_wait'] for r in data[str(p)][method]])
                 for p in server_counts]
        utils = [np.mean([r['avg_utilization'] * 100 for r in data[str(p)][method]])
                 for p in server_counts]
        ax1.plot(server_counts, waits, style, label=METHOD_LABELS[method],
                 color=METHOD_COLORS[method], linewidth=1.5, markersize=6)
        ax2.plot(server_counts, utils, style, label=METHOD_LABELS[method],
                 color=METHOD_COLORS[method], linewidth=1.5, markersize=6)

    ax1.set_xlabel('Number of servers $|\\mathcal{P}|$')
    ax1.set_ylabel('Average waiting time $\\bar{w}$ (h)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Number of servers $|\\mathcal{P}|$')
    ax2.set_ylabel('Average utilization $U$ (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "scale_sensitivity.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "scale_sensitivity.png"))
    plt.close(fig)
    print("  scale_sensitivity done")


def fig8_epsilon_ablation():
    """Figure: Migration penalty ε ablation."""
    with open(os.path.join(RESULT_DIR, "exp3_ablation.json")) as f:
        data = json.load(f)

    eps_data = data['epsilon']
    epsilons = sorted(float(k) for k in eps_data.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    waits = [np.mean([r['avg_wait'] for r in eps_data[str(e)]]) for e in epsilons]
    migs = [np.mean([r['migrations'] for r in eps_data[str(e)]]) for e in epsilons]

    ax1.plot(range(len(epsilons)), waits, '-o', color='#2166ac', linewidth=1.5, markersize=6)
    ax1.set_xticks(range(len(epsilons)))
    ax1.set_xticklabels([str(e) for e in epsilons])
    ax1.set_xlabel('Migration penalty $\\epsilon$')
    ax1.set_ylabel('Average waiting time $\\bar{w}$ (h)')
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(len(epsilons)), migs, '-s', color='#d6604d', linewidth=1.5, markersize=6)
    ax2.set_xticks(range(len(epsilons)))
    ax2.set_xticklabels([str(e) for e in epsilons])
    ax2.set_xlabel('Migration penalty $\\epsilon$')
    ax2.set_ylabel('Total migrations')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "epsilon_ablation.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "epsilon_ablation.png"))
    plt.close(fig)
    print("  epsilon_ablation done")


def fig9_checkpoint_ablation():
    """Figure: Checkpoint frequency ablation."""
    with open(os.path.join(RESULT_DIR, "exp3_ablation.json")) as f:
        data = json.load(f)

    ck_data = data['checkpoint']
    intervals = sorted(float(k) for k in ck_data.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    waits = [np.mean([r['avg_wait'] for r in ck_data[str(i)]]) for i in intervals]
    migs = [np.mean([r['migrations'] for r in ck_data[str(i)]]) for i in intervals]

    ax1.plot(intervals, waits, '-o', color='#2166ac', linewidth=1.5, markersize=6)
    ax1.set_xlabel('Checkpoint interval (h)')
    ax1.set_ylabel('Average waiting time $\\bar{w}$ (h)')
    ax1.grid(True, alpha=0.3)

    ax2.plot(intervals, migs, '-s', color='#d6604d', linewidth=1.5, markersize=6)
    ax2.set_xlabel('Checkpoint interval (h)')
    ax2.set_ylabel('Total migrations')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "checkpoint_ablation.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "checkpoint_ablation.png"))
    plt.close(fig)
    print("  checkpoint_ablation done")


def table1_main_results():
    """Generate LaTeX table of main results."""
    data = _load_exp1()
    methods = ['Ours', 'BF', 'SC', 'MismatchPred']
    dists = ['inv', 'norm', 'dir']
    metrics = ['avg_wait', 'avg_queue_cardtime', 'avg_utilization', 'makespan', 'migrations']
    metric_names = ['$\\bar{w}$ (h)', '$\\bar{Q}$ (GPU$\\cdot$h)', '$U$ (\\%)', 'Makespan (h)', 'Migrations']

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Main comparison results (50 servers $\\times$ 8 GPUs, 10 seeds)}")
    lines.append("\\label{tab:gpu_main}")
    lines.append("\\begin{tabular}{ll" + "r" * len(methods) + "}")
    lines.append("\\toprule")
    lines.append("Dist. & Metric & " + " & ".join(METHOD_LABELS[m] for m in methods) + " \\\\")
    lines.append("\\midrule")

    for dist_name in dists:
        for mi, metric in enumerate(metrics):
            prefix = DIST_LABELS[dist_name] if mi == 0 else ""
            vals = []
            for m in methods:
                raw = [r[metric] for r in data[dist_name][m]]
                if metric == 'avg_utilization':
                    raw = [v * 100 for v in raw]
                mean = np.mean(raw)
                std = np.std(raw)
                if metric == 'migrations':
                    vals.append(f"{mean:.0f}$\\pm${std:.0f}")
                else:
                    vals.append(f"{mean:.2f}$\\pm${std:.2f}")
            line = f"{prefix} & {metric_names[mi]} & " + " & ".join(vals) + " \\\\"
            lines.append(line)
        if dist_name != 'dir':
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tex = "\n".join(lines)
    with open(os.path.join(FIG_DIR, "table_main.tex"), "w") as f:
        f.write(tex)
    print("  table_main.tex done")
    return tex


def fig0_counterproductive_consolidation():
    """Figure: Counterproductive effect of server consolidation (Fig 1 in paper)."""
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))

    colors = {'existing': '#4393c3', 'remaining': '#d9d9d9',
              'migration': '#f4a582', 'new': '#66c2a5'}

    def draw_server(ax, servers, title, annotations=None):
        ax.set_title(title, fontsize=10)
        n = len(servers)
        bar_w = 0.6
        for i, (existing, remaining, mig, new_t) in enumerate(servers):
            bottom = 0
            if existing > 0:
                ax.bar(i, existing, bar_w, bottom=bottom, color=colors['existing'],
                       edgecolor='black', linewidth=0.5)
                bottom += existing
            if mig > 0:
                ax.bar(i, mig, bar_w, bottom=bottom, color=colors['migration'],
                       edgecolor='black', linewidth=0.5)
                bottom += mig
            if new_t > 0:
                ax.bar(i, new_t, bar_w, bottom=bottom, color=colors['new'],
                       edgecolor='black', linewidth=0.5)
                bottom += new_t
            if remaining > 0:
                ax.bar(i, remaining, bar_w, bottom=bottom, color=colors['remaining'],
                       edgecolor='black', linewidth=0.5, alpha=0.5)
        ax.set_xticks(range(n))
        ax.set_xticklabels([f'PM{i+1}' for i in range(n)])
        ax.set_ylim(0, 9)
        ax.set_ylabel('GPUs' if i == 0 else '')
        ax.set_yticks(range(0, 9))
        if annotations:
            for txt, xy, xytext in annotations:
                ax.annotate(txt, xy=xy, xytext=xytext, fontsize=8,
                           arrowprops=dict(arrowstyle='->', color='red'),
                           color='red')

    # Scenario: new jobs of size 5 and 6 arrive
    # Original (rem 4,5,7): PM2 fits 5, PM3 fits 6 — both placed
    # SC (rem 4,4,8): PM3 fits 5 (rem 3), but no server fits 6 — blocked!
    draw_server(axes[0], [(4, 4, 0, 0), (3, 5, 0, 0), (1, 7, 0, 0)],
                '(a) Original\n(4, 3, 1)')
    draw_server(axes[1], [(4, 4, 0, 0), (4, 4, 0, 0), (0, 8, 0, 0)],
                '(b) After SC\n(4, 4, 0)')
    draw_server(axes[2], [(4, 4, 0, 0), (3, 0, 0, 5), (1, 1, 0, 6)],
                '(c) Original + Arrivals\nsize-5, size-6 both fit')
    draw_server(axes[3], [(4, 4, 0, 0), (4, 4, 0, 0), (0, 3, 0, 5)],
                '(d) SC + Arrivals\nsize-6 blocked!',
                annotations=[('size-6\nrejected!', (1.5, 5), (1.5, 7.5))])

    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=colors['existing'], label='Existing Tasks'),
                      mpatches.Patch(color=colors['remaining'], label='Remaining'),
                      mpatches.Patch(color=colors['new'], label='New Tasks')]
    fig.legend(handles=legend_patches, loc='upper center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(FIG_DIR, "counterproductive_consolidation.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, "counterproductive_consolidation.png"), bbox_inches='tight')
    plt.close(fig)
    print("  counterproductive_consolidation done")


def fig_algorithm_flowchart():
    """Figure: Algorithm 1 flowchart — Rolling checkpoint-triggered migration."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    box_style = dict(boxstyle='round,pad=0.4', facecolor='#e6f2ff', edgecolor='#2166ac', linewidth=1.5)
    decision_style = dict(boxstyle='round,pad=0.3', facecolor='#fff3e6', edgecolor='#d6604d', linewidth=1.5)
    action_style = dict(boxstyle='round,pad=0.3', facecolor='#e6ffe6', edgecolor='#4393c3', linewidth=1.5)

    elements = [
        (5, 11, 'Initialize: $t \\leftarrow t_{now}$', box_style),
        (5, 9.5, '$t \\in \\mathcal{T}$ ?\n(checkpoint time?)', decision_style),
        (5, 7.5, 'Identify eligible jobs:\n$\\mathcal{E} = \\{j : t \\in CK_j\\}$', action_style),
        (5, 5.5, 'Solve MILP:\n$\\min \\sum_p \\sum_k \\phi(k) f_{p,k} + \\epsilon \\sum x_{j,p}$', action_style),
        (5, 3.5, 'Apply migrations $X^*(t)$\nUpdate cluster state', action_style),
        (5, 1.5, '$t \\leftarrow t + \\Delta t$', box_style),
    ]

    for x, y, text, style in elements:
        ax.text(x, y, text, ha='center', va='center', fontsize=10, bbox=style)

    arrows = [(5, 10.6, 0, -0.7), (5, 9.0, 0, -0.7), (5, 6.9, 0, -0.7),
              (5, 5.0, 0, -0.7), (5, 3.0, 0, -0.7)]
    for x, y, dx, dy in arrows:
        ax.annotate('', xy=(x+dx, y+dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    ax.annotate('', xy=(1, 9.5), xytext=(3.2, 9.5),
                arrowprops=dict(arrowstyle='->', color='#d6604d', lw=1.5))
    ax.text(2, 9.9, 'No', fontsize=9, color='#d6604d', ha='center')
    ax.annotate('', xy=(1, 1.5), xytext=(1, 9.5),
                arrowprops=dict(arrowstyle='->', color='#d6604d', lw=1.5))

    ax.text(5.5, 8.9, 'Yes', fontsize=9, color='#4393c3')

    ax.annotate('', xy=(5, 11.4), xytext=(9, 1.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1, ls='--',
                               connectionstyle='arc3,rad=0.3'))
    ax.text(9.2, 6.5, 'Loop', fontsize=9, color='gray', rotation=90)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "algorithm_flowchart.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, "algorithm_flowchart.png"), bbox_inches='tight')
    plt.close(fig)
    print("  algorithm_flowchart done")


def fig10_utilization_timeseries():
    """Figure: GPU utilization over a 24-hour window under different methods."""
    from frag_cluster import (ClusterSimulator, BestFitNoMigration,
                              ServerConsolidation, InaccuratePrediction,
                              generate_workload, GPUJob, DISTRIBUTIONS)

    def copy(jobs):
        return [GPUJob(id=j.id, gpu_req=j.gpu_req, duration=j.duration,
                       arrival=j.arrival, checkpoints=list(j.checkpoints)) for j in jobs]

    dist = DISTRIBUTIONS['norm']
    wrong = DISTRIBUTIONS['dir']
    jobs = generate_workload('norm', total_time=100.0, arrival_rate=10.0, C=8, seed=42)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    dist_names = ['inv', 'norm', 'dir']

    for ax, dn in zip(axes, dist_names):
        d = DISTRIBUTIONS[dn]
        wd = DISTRIBUTIONS[{'inv': 'norm', 'norm': 'dir', 'dir': 'inv'}[dn]]
        js = generate_workload(dn, 100.0, 10.0, 8, seed=42)

        for name, cls, kw, color in [
            ('FragMig', ClusterSimulator, dict(num_servers=50, gpu_per_server=8,
             dist=d, epsilon=0.01, milp_time_limit=2), '#2166ac'),
            ('BF', BestFitNoMigration, dict(num_servers=50, gpu_per_server=8, dist=d), '#92c5de'),
            ('SC', ServerConsolidation, dict(num_servers=50, gpu_per_server=8, dist=d), '#f4a582'),
            ('Mismatch', InaccuratePrediction, dict(num_servers=50, gpu_per_server=8,
             actual_dist=d, wrong_dist=wd, milp_time_limit=2), '#b2182b'),
        ]:
            sim = cls(**kw)
            m = sim.run(copy(js), 100.0)
            ts = [t for t, u in m.utilization_samples if 5 <= t <= 90]
            us = [u * 100 for t, u in m.utilization_samples if 5 <= t <= 90]
            if ts:
                ax.plot(ts, us, label=name, color=color, linewidth=1.0, alpha=0.8)

        ax.set_title(f'{DIST_LABELS[dn]}')
        ax.set_xlabel('Time (h)')
        if ax == axes[0]:
            ax.set_ylabel('Utilization (%)')
        ax.set_ylim(40, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "utilization_timeseries.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "utilization_timeseries.png"))
    plt.close(fig)
    print("  utilization_timeseries done")


def generate_all():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("Generating figures...")

    fig0_counterproductive_consolidation()
    fig1_phi_curves()
    fig2_demand_distributions()

    fig10_utilization_timeseries()

    if os.path.exists(os.path.join(RESULT_DIR, "exp1_main.json")):
        fig3_waiting_time_bars()
        fig4_queue_cardtime_bars()
        fig5_utilization_bars()
        fig6_makespan_bars()
        table1_main_results()
    else:
        print("  [SKIP] exp1_main.json not found")

    if os.path.exists(os.path.join(RESULT_DIR, "exp2_scale.json")):
        fig7_scale_sensitivity()
    else:
        print("  [SKIP] exp2_scale.json not found")

    if os.path.exists(os.path.join(RESULT_DIR, "exp3_ablation.json")):
        fig8_epsilon_ablation()
        fig9_checkpoint_ablation()
    else:
        print("  [SKIP] exp3_ablation.json not found")

    print("All figures generated in", FIG_DIR)


if __name__ == "__main__":
    generate_all()
