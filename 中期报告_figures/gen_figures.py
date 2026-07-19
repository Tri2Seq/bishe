"""Generate figures for the mid-term report — all text in Chinese, unified palette."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import json, os

plt.rcParams['font.family'] = 'Noto Sans CJK SC'
plt.rcParams['axes.unicode_minus'] = False

OUTDIR = '/home/ubuntu/Documents/sjx/gra/中期报告_figures'

# ── Unified color palette (ALL figures use these) ──
C1     = '#3B7DD8'   # primary blue — our method / module 1
C2     = '#E07B3C'   # orange — module 3 / GPU / accent
C3     = '#3DA56E'   # green — module 2 / perception
C4     = '#78909C'   # grey — baseline / neutral
C5     = '#D93025'   # red — highlight / deadline
C1_L   = '#D6E4F0'   # blue light bg
C2_L   = '#FDEADC'   # orange light bg
C3_L   = '#D9EFDF'   # green light bg


def fig_architecture():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')

    ax.text(7, 8.5, '多机器人多楼层配送系统三层架构',
            ha='center', va='center', fontsize=17, fontweight='bold')

    layers = [
        ('第一层：配送任务调度', C1, C1_L, 5.6, 2.3, '模块一',
         [(1.2, 'NSGA-II-EBO', '多目标进化优化'),
          (4.4, 'EBO', '电梯瓶颈优化器'),
          (7.6, '接力配送', '任务拆分与交接'),
          (10.5, '仿真器V2', '时间驱动仿真')]),
        ('第二层：视觉感知与深度估计', C3, C3_L, 3.0, 2.0, '模块二',
         [(1.2, '按钮检测', '电梯按钮数据集+网络'),
          (4.4, '单目深度', 'VAE+DiT 架构'),
          (7.6, '双目/红外深度', '立体视觉与ToF适配')]),
        ('第三层：GPU推理资源调度', C2, C2_L, 0.4, 2.0, '模块三',
         [(1.2, r'$\varphi(k)$碎片率', '阻塞风险量化'),
          (4.4, 'MILP迁移', '独热编码线性化'),
          (7.6, '预测式预加载', '时间表驱动调度'),
          (10.5, 'GPU集群', '2 GPU / 4槽并发')]),
    ]

    for title, color, bg, y0, h, mod_label, boxes in layers:
        ax.add_patch(FancyBboxPatch((0.5, y0), 12.5, h, boxstyle="round,pad=0.15",
                                     facecolor=bg, edgecolor=color, lw=2))
        ax.text(0.85, y0 + h - 0.3, title, fontsize=13, fontweight='bold', color=color)
        ax.text(13.5, y0 + h/2, mod_label, ha='center', va='center',
                fontsize=12, color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=bg, edgecolor=color, lw=1.5))

        for x, label, detail in boxes:
            bw = 2.6 if len(boxes) <= 3 else 2.3
            ax.add_patch(FancyBboxPatch((x, y0 + 0.3), bw, 1.3, boxstyle="round,pad=0.1",
                                         facecolor='white', edgecolor=color, lw=1.5))
            ax.text(x + bw/2, y0 + 1.15, label, ha='center', va='center',
                    fontsize=10, fontweight='bold')
            ax.text(x + bw/2, y0 + 0.65, detail, ha='center', va='center',
                    fontsize=8, color='#555')

    # GPU demand link box
    ax.add_patch(FancyBboxPatch((10.5, 3.3), 2.3, 1.3, boxstyle="round,pad=0.1",
                                 facecolor=C2_L, edgecolor=C2, lw=1.5, linestyle='--'))
    ax.text(11.65, 4.15, 'GPU需求', ha='center', va='center',
            fontsize=10, fontweight='bold', color=C2)
    ax.text(11.65, 3.65, r'$g_j(t)\in\{0,1\}$', ha='center', va='center',
            fontsize=9, color=C2)

    for xp in [2.5, 5.7, 8.9]:
        ax.annotate('', xy=(xp, 5.15), xytext=(xp, 5.6),
                    arrowprops=dict(arrowstyle='<->', color=C4, lw=1.8))
        ax.annotate('', xy=(xp, 2.55), xytext=(xp, 3.0),
                    arrowprops=dict(arrowstyle='<->', color=C4, lw=1.8))

    plt.tight_layout()
    fig.savefig(f'{OUTDIR}/system_architecture.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print('OK: system_architecture.png')


def fig_multiscale_comparison():
    scenarios = ['S1\n(5层/5机/10任务)', 'S2\n(10层/15机/40任务)', 'S3\n(15层/25机/80任务)']
    methods = ['贪心', 'EBO', 'EBO+接力']

    makespan = np.array([
        [651, 192, 178], [685, 221, 214], [1003, 342, 235],
    ])
    energy = np.array([
        [10470, 4443, 4083], [26423, 5855, 5614], [44207, 13137, 10633],
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(scenarios))
    width = 0.25
    colors_bar = [C4, C1, C2]

    for ax, data, ylabel, title in [
        (axes[0], makespan, '配送完成时间 (s)', '配送完成时间对比'),
        (axes[1], energy, '总能耗 (J)', '总能耗对比'),
    ]:
        for i, (method, color) in enumerate(zip(methods, colors_bar)):
            bars = ax.bar(x + (i - 1) * width, data[:, i], width, label=method,
                         color=color, edgecolor='white', lw=0.5)
            for bar, val in zip(bars, data[:, i]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(data.flatten())*0.01,
                        str(val), ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(f'{OUTDIR}/multiscale_comparison.png', dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('OK: multiscale_comparison.png')


def fig_gpu_strategy():
    COLD_START_TIME = 3.0
    data_path = '/home/ubuntu/Documents/sjx/gra/module1/results/joint_experiment/gpu_timeline_data.json'
    log_path = '/home/ubuntu/Documents/sjx/gra/module1/results/joint_experiment/action_log.json'

    base_makespan = 330.0
    num_robots = 9
    if os.path.exists(log_path):
        with open(log_path) as f:
            logs = json.load(f)
        base_makespan = max(e['end_time'] for e in logs)
        num_robots = len(set(e['robot_id'] for e in logs))

    if os.path.exists(data_path):
        with open(data_path) as f:
            raw = json.load(f)
        order = ['FragAware', 'OnDemand', 'RoundRobin']
        cn = {'FragAware': '碎片感知\n(本文)', 'OnDemand': '按需', 'RoundRobin': '轮询'}
        strategies = [cn[k] for k in order if k in raw]
        cold = [raw[k]['cold_starts'] for k in order if k in raw]
        wait = [raw[k].get('total_wait_delay', 0) for k in order if k in raw]
        eff_makespan = [base_makespan + c * COLD_START_TIME / num_robots for c in cold]
        gpu_wait = [c * COLD_START_TIME + w for c, w in zip(cold, wait)]
        utils_pct = []
        for k in order:
            if k not in raw: continue
            s = raw[k]['utilization_samples']
            u = [x['util'] for x in s]
            utils_pct.append(100 * sum(u) / len(u))
    else:
        strategies = ['碎片感知\n(本文)', '按需', '轮询']
        eff_makespan = [332.3, 346.7, 346.7]
        utils_pct = [51.5, 29.6, 29.6]
        gpu_wait = [21.0, 150.0, 150.0]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    colors_s = [C1, C3, C4]

    metrics = [
        (axes[0], eff_makespan, '等效配送时间 (s)', '等效配送时间', '{:.1f}'),
        (axes[1], utils_pct, '平均GPU利用率 (%)', '平均GPU利用率', '{:.1f}%'),
        (axes[2], gpu_wait, 'GPU等待时间 (s)', 'GPU等待时间', '{:.1f}'),
    ]

    for ax, vals, ylabel, title, fmt in metrics:
        bars = ax.bar(strategies, vals, color=colors_s, edgecolor='white', lw=0.5, width=0.6)
        vmin, vmax = min(vals), max(vals)
        if vmax - vmin < vmax * 0.3:
            ax.set_ylim(vmin * 0.9, vmax * 1.15)
        else:
            ax.set_ylim(0, vmax * 1.25 if vmax > 0 else 1)
        ymax_plot = ax.get_ylim()[1]
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (ymax_plot - ax.get_ylim()[0])*0.02,
                    fmt.format(val), ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('GPU调度策略性能对比（2 GPU / 4并发槽）', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f'{OUTDIR}/gpu_strategy_comparison.png', dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('OK: gpu_strategy_comparison.png')


def fig_progress_timeline():
    fig, ax = plt.subplots(figsize=(16, 6.5))

    tasks = [
        ('文献调研', '2026-01', '2026-02', C1, True),
        ('NSGA-II-EBO算法', '2026-02', '2026-05', C1, True),
        ('接力配送机制', '2026-04', '2026-05', C1, True),
        ('仿真平台V2', '2026-03', '2026-05', C1, True),
        ('IFAC论文（碎片率+MILP）', '2026-02', '2026-04', C2, True),
        ('GPU调度器（碎片感知+预测）', '2026-04', '2026-07', C2, True),
        ('按钮数据集+检测网络', '2026-05', '2026-06', C3, True),
        ('联合实验流水线', '2026-06', '2026-07', C4, True),
        ('多传感器深度估计适配', '2026-07', '2026-08', C3, False),
        ('实体机械臂平台验证', '2026-09', '2026-10', C3, False),
        ('实际电梯场景测试', '2026-09', '2026-10', C3, False),
        ('学位论文撰写', '2026-11', '2027-05', C4, False),
    ]

    months = ['2026-01', '2026-02', '2026-03', '2026-04', '2026-05', '2026-06',
              '2026-07', '2026-08', '2026-09', '2026-10', '2026-11', '2026-12',
              '2027-01', '2027-02', '2027-03', '2027-04', '2027-05']
    month_idx = {m: i for i, m in enumerate(months)}
    month_labels = ['1月', '2月', '3月', '4月', '5月', '6月',
                    '7月', '8月', '9月', '10月', '11月', '12月',
                    '1月', '2月', '3月', '4月', '5月']

    for i, (name, start, end, color, done) in enumerate(reversed(tasks)):
        y = i
        x_start = month_idx[start]
        x_end = month_idx[end]
        duration = x_end - x_start
        alpha = 1.0 if done else 0.45
        hatch = '' if done else '///'
        ax.barh(y, duration, left=x_start, height=0.6,
                color=color, alpha=alpha, edgecolor='white', lw=1, hatch=hatch)
        if duration >= 3:
            ax.text(x_start + duration / 2, y, name,
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    color='white' if done else '#333')
        else:
            ax.text(x_end + 0.15, y, name,
                    ha='left', va='center', fontsize=9, fontweight='bold',
                    color='#333')

    now_idx = month_idx['2026-07'] + 0.5
    ax.axvline(x=now_idx, color=C5, lw=2, linestyle='--', alpha=0.7)
    ax.text(now_idx + 0.1, len(tasks) - 0.5, '当前（2026年7月）', fontsize=10,
            color=C5, fontweight='bold', va='top', ha='left')

    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(month_labels, fontsize=9)
    ax.set_yticks([])

    ax.axvline(x=11.5, color='#999', lw=1, linestyle=':')
    ax.text(5.5, -1.2, '2026年', ha='center', fontsize=12, fontweight='bold', color='#666')
    ax.text(14, -1.2, '2027年', ha='center', fontsize=12, fontweight='bold', color='#666')

    completed_patch = mpatches.Patch(facecolor=C1, label='已完成')
    planned_patch = mpatches.Patch(facecolor=C3, alpha=0.45, hatch='///', label='计划中')
    ax.legend(handles=[completed_patch, planned_patch], loc='lower right', fontsize=10)

    ax.set_title('研究进度时间线', fontsize=15, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    fig.savefig(f'{OUTDIR}/progress_timeline.png', dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('OK: progress_timeline.png')


def fig_faster_rcnn():
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7)
    ax.axis('off')

    ax.text(7.5, 6.6, 'Faster R-CNN 电梯按钮检测网络结构', ha='center', va='center',
            fontsize=16, fontweight='bold')

    def box(x, y, w, h, label, sub, color, bg=None):
        if bg is None:
            bg = color + '20'
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                     facecolor=bg, edgecolor=color, lw=2))
        ax.text(x + w/2, y + h/2 + 0.15, label, ha='center', va='center',
                fontsize=11, fontweight='bold', color=color)
        if sub:
            ax.text(x + w/2, y + h/2 - 0.2, sub, ha='center', va='center',
                    fontsize=8, color='#555')

    def arrow(x1, y1, x2, y2, color=C4):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))

    # Input
    box(0.3, 2.5, 2.0, 1.6, '输入图像', '电梯面板\n(224×224×3)', C4, '#ECEFF1')

    # Backbone
    box(3.0, 2.5, 2.2, 1.6, '骨干网络', 'ResNet-50\n预训练权重', C1, C1_L)

    # Shared feature map
    box(5.9, 2.5, 2.0, 1.6, '共享特征图', 'Conv特征\n(H/16×W/16×2048)', C1, C1_L)

    # RPN branch (top)
    box(8.6, 4.6, 2.2, 1.6, '区域候选网络', 'RPN\n3×3卷积+锚框', C3, C3_L)

    # Proposals
    ax.text(11.5, 5.4, '候选区域', ha='center', va='center', fontsize=10,
            fontweight='bold', color=C3,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=C3_L, edgecolor=C3, lw=1.5))

    # ROI Pooling
    box(8.6, 2.5, 2.2, 1.6, 'ROI 池化', '自适应池化\n(7×7)', C2, C2_L)

    # FC layers
    box(11.5, 2.5, 1.8, 1.6, '全连接层', 'FC 1024\n+ FC 1024', C2, C2_L)

    # Classification head
    box(11.0, 0.3, 1.5, 1.2, '分类头', 'Softmax', C1, C1_L)

    # Regression head
    box(12.8, 0.3, 1.5, 1.2, '回归头', '边界框', C2, C2_L)

    # Output labels
    ax.text(11.75, -0.3, '按钮类别\n(楼层/开门/关门/报警)', ha='center', va='center',
            fontsize=9, color=C1, fontweight='bold')
    ax.text(13.55, -0.3, '位置坐标\n(x, y, w, h)', ha='center', va='center',
            fontsize=9, color=C2, fontweight='bold')

    # Arrows: main flow
    arrow(2.3, 3.3, 3.0, 3.3, C4)
    arrow(5.2, 3.3, 5.9, 3.3, C1)
    arrow(7.9, 3.3, 8.6, 3.3, C2)
    arrow(10.8, 3.3, 11.5, 3.3, C2)

    # Arrow: feature map to RPN (up)
    arrow(6.9, 4.1, 8.6, 5.2, C3)

    # Arrow: RPN output to ROI pooling
    arrow(11.5, 4.8, 11.5, 4.3, C3)
    ax.annotate('', xy=(10.8, 3.8), xytext=(11.3, 4.3),
                arrowprops=dict(arrowstyle='->', color=C3, lw=1.5, connectionstyle='arc3,rad=-0.3'))

    # Arrows: FC to heads
    arrow(12.0, 2.5, 11.75, 1.5, C1)
    arrow(12.8, 2.5, 13.55, 1.5, C2)

    # Arrow: heads to output
    arrow(11.75, 0.3, 11.75, 0.0, C1)
    arrow(13.55, 0.3, 13.55, 0.0, C2)

    # Legend
    legend_items = [
        ('特征提取', C1),
        ('区域候选', C3),
        ('检测输出', C2),
    ]
    for i, (label, color) in enumerate(legend_items):
        ax.add_patch(FancyBboxPatch((0.5 + i*2.5, 0.15), 0.4, 0.4, boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor=color, alpha=0.3))
        ax.text(1.1 + i*2.5, 0.35, label, va='center', fontsize=9, color=color, fontweight='bold')

    plt.tight_layout()
    fig.savefig(f'{OUTDIR}/faster_rcnn_architecture.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print('OK: faster_rcnn_architecture.png')


if __name__ == '__main__':
    fig_architecture()
    fig_multiscale_comparison()
    fig_gpu_strategy()
    fig_faster_rcnn()
    fig_progress_timeline()
    print('All figures generated.')
