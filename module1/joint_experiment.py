"""Module1 + Module2 Joint Experiment

Run NSGA-II-EBO → SimulatorV2 → GPU Scheduling pipeline with detailed
per-robot action logging and comprehensive chart generation.
"""

import sys
import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'module2'))

from config import ScenarioConfig, generate_scenario, RelayPlan
from simulator_v2 import SimulatorV2, RobotPhase, GPU_PHASES, ActionLog, RobotStrategy
from nsga2_ebo import NSGA2EBOSolver
from gpu_config import GPUConfig, GPUDemand
from gpu_scheduler import (FragAwareScheduler, PredictiveScheduler,
                            OnDemandScheduler, FixedScheduler, RoundRobinScheduler)


def run_joint_experiment(seed=42, output_dir="results/joint_experiment"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    cfg = ScenarioConfig(
        num_floors=10, num_robots=12, num_elevators=4,
        num_tasks=40, seed=seed,
    )
    robots, elevators, tasks = generate_scenario(cfg)

    print(f"Scene: {cfg.num_floors}F, {cfg.num_robots}R, {cfg.num_elevators}E, {cfg.num_tasks}T")
    cross_floor = sum(1 for t in tasks if t.is_cross_floor)
    print(f"  Cross-floor tasks: {cross_floor}/{len(tasks)}")

    # Phase 1: NSGA-II-EBO optimization
    print("\n=== Phase 1: NSGA-II-EBO Optimization ===")
    t0 = time.time()
    nsga = NSGA2EBOSolver(robots, elevators, tasks, pop_size=50, max_gen=30, seed=seed)
    nsga_result = nsga.solve(verbose=True)
    opt_time = time.time() - t0

    assignment, task_seq, relay_plans, strategy = nsga.get_best_schedule(objective_idx=0)

    print(f"\nOptimization time: {opt_time:.1f}s")
    print(f"Best makespan: {nsga_result['best_makespan'][0]:.0f}s")
    print(f"Relay tasks: {nsga_result['relay_active_best']}")

    # Phase 2: Detailed simulation
    print("\n=== Phase 2: Detailed Simulation ===")
    sim = SimulatorV2(robots, elevators, tasks, dt=0.5, strategy=strategy)
    sim.load_schedule(assignment, task_seq, relay_plans=relay_plans)
    sim_result = sim.run(max_time=20000)
    sim.print_summary(sim_result)

    # Phase 3: Extract GPU demands from action log
    print("\n=== Phase 3: Extract GPU Demands ===")
    demands = extract_gpu_demands(sim)

    total_gpu_segs = sum(1 for segs in demands.values()
                         for s in segs if s.demand_type == "demand")
    print(f"  Robots with GPU needs: {len(demands)}")
    print(f"  Total GPU demand segments: {total_gpu_segs}")

    # Phase 4: Run GPU schedulers
    print("\n=== Phase 4: GPU Scheduling Comparison ===")
    gpu_cfg = GPUConfig(num_servers=1, gpus_per_server=2,
                        vram_per_gpu=16.0, depth_model_vram=6.0)
    print(f"  GPU config: {gpu_cfg.total_gpus} GPUs, "
          f"{gpu_cfg.total_concurrent} max concurrent")

    scheduler_factories = {
        "FragAware": lambda: FragAwareScheduler(gpu_cfg),
        "Predictive": lambda: PredictiveScheduler(gpu_cfg),
        "OnDemand": lambda: OnDemandScheduler(gpu_cfg),
        "Fixed": lambda: FixedScheduler(gpu_cfg),
        "RoundRobin": lambda: RoundRobinScheduler(gpu_cfg),
    }

    gpu_results = {}
    for name, factory in scheduler_factories.items():
        sched = factory()
        metrics = sched.schedule(demands)
        summary = metrics.summary()
        gpu_results[name] = summary
        gpu_results[name]["allocations"] = [
            {"robot_id": a.robot_id, "gpu_id": a.gpu_id,
             "start": a.start, "end": a.end, "type": a.alloc_type}
            for a in sched.allocations
        ]
        gpu_results[name]["utilization_samples"] = [
            {"t": t, "util": u} for t, u in metrics.gpu_utilization_samples
        ]

    print(f"\n{'Strategy':<15} {'Demands':>8} {'Wait(s)':>10} {'Cold':>6} "
          f"{'Migrate':>8} {'Frag':>8}")
    print("-" * 60)
    for name, s in gpu_results.items():
        print(f"{name:<15} {s['total_demands']:>8} {s['total_wait_delay']:>10.2f} "
              f"{s['cold_starts']:>6} {s['migrations']:>8} "
              f"{s['avg_fragmentation']:>8.4f}")

    # Phase 5: Build detailed robot timeline
    print("\n=== Phase 5: Robot Timeline Analysis ===")
    robot_timeline = build_robot_timeline(sim, tasks, assignment, relay_plans)
    for rid, info in sorted(robot_timeline.items()):
        task_ids = info["task_ids"]
        if not task_ids:
            continue
        relay_note = ""
        if info["relay_tasks"]:
            relay_note = f" (relay: {info['relay_tasks']})"
        elev_note = ""
        if info["elevators_used"]:
            elev_note = f", elevators={info['elevators_used']}"
        print(f"  R{rid}: {len(task_ids)} tasks{relay_note}, "
              f"GPU={info['gpu_time']:.1f}s, "
              f"GPU_segs={info['gpu_segments']}{elev_note}")

    # Phase 6: Generate figures
    print("\n=== Phase 6: Generating Figures ===")
    fig_dir = os.path.join(output_dir, "figures")
    plot_robot_gantt(sim, tasks, relay_plans, fig_dir)
    plot_gpu_comparison_bar(gpu_results, fig_dir)
    plot_gpu_timeline(gpu_results, gpu_cfg, fig_dir)
    plot_elevator_usage(sim, fig_dir)
    plot_robot_gpu_demand_density(demands, fig_dir)
    plot_fragmentation_over_time(gpu_results, fig_dir)

    # Save results
    save_data = {
        "scenario": {
            "floors": cfg.num_floors, "robots": cfg.num_robots,
            "elevators": cfg.num_elevators, "tasks": cfg.num_tasks, "seed": seed,
        },
        "optimization": {
            "time": opt_time,
            "pareto_size": nsga_result["pareto_size"],
            "best_makespan": nsga_result["best_makespan"],
            "best_energy": nsga_result["best_energy"],
            "relay_active": nsga_result["relay_active_best"],
        },
        "simulation": {
            k: v for k, v in sim_result.items()
            if k not in ("elevator_stats", "robot_stats")
        },
        "simulation_robot_stats": sim_result.get("robot_stats", {}),
        "gpu_scheduling": {
            name: {k: v for k, v in s.items()
                   if k not in ("allocations", "utilization_samples")}
            for name, s in gpu_results.items()
        },
        "robot_timeline": {
            str(rid): {k: v for k, v in info.items() if k != "actions"}
            for rid, info in robot_timeline.items()
        },
    }

    result_path = os.path.join(output_dir, "joint_results.json")
    with open(result_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {result_path}")
    print(f"Figures saved to {fig_dir}/")

    return save_data


def extract_gpu_demands(sim: SimulatorV2):
    """Extract GPU demand segments from simulator action log."""
    robot_actions = defaultdict(list)
    for log in sim.action_log:
        robot_actions[log.robot_id].append(log)

    demands = {}
    for rid in sorted(robot_actions.keys()):
        actions = sorted(robot_actions[rid], key=lambda a: a.start_time)
        segs = []
        i = 0
        while i < len(actions):
            a = actions[i]
            if a.needs_gpu:
                gpu_action_map = {
                    "APPROACH_ELEVATOR": 8.0,
                    "PRESS_BUTTON": 7.0,
                    "EXIT_ELEVATOR": 5.0,
                }
                duration = gpu_action_map.get(a.action, 5.0)
                segs.append(GPUDemand(
                    robot_id=rid, start=a.start_time,
                    end=a.start_time + duration,
                    demand_type="demand",
                    elevator_id=a.elevator_id, floor=a.floor,
                ))
            elif a.action == "RIDE_ELEVATOR":
                # Riding = idle (no GPU needed, potential checkpoint)
                ride_end = a.start_time + 20.0  # estimate
                for j in range(i + 1, len(actions)):
                    if actions[j].action in ("EXIT_ELEVATOR", "EXIT_DONE"):
                        ride_end = actions[j].start_time
                        break
                segs.append(GPUDemand(
                    robot_id=rid, start=a.start_time, end=ride_end,
                    demand_type="idle", reason="RIDING",
                ))
            i += 1
        if segs:
            demands[rid] = segs
    return demands


def build_robot_timeline(sim, tasks, assignment, relay_plans):
    """Build per-robot timeline with task/elevator/GPU details."""
    timeline = {}
    for rs in sim.robot_states:
        rid = rs.id
        assigned_tasks = [tid for tid, r in assignment.items() if r == rid]
        relay_info = []
        for tid in assigned_tasks:
            if tid in relay_plans:
                rp = relay_plans[tid]
                if rp.robot_1 == rid:
                    relay_info.append(f"T{tid}:seg1→F{rp.relay_floor}")
                elif rp.robot_2 == rid:
                    relay_info.append(f"T{tid}:seg2")

        robot_actions = [a for a in sim.action_log if a.robot_id == rid]
        gpu_actions = [a for a in robot_actions if a.needs_gpu]
        elevator_ids = set(a.elevator_id for a in robot_actions
                           if a.elevator_id >= 0)

        timeline[rid] = {
            "task_ids": assigned_tasks,
            "relay_tasks": relay_info,
            "completed": rs.tasks_completed,
            "gpu_time": rs.total_gpu_time,
            "gpu_segments": len(gpu_actions),
            "elevators_used": sorted(elevator_ids),
            "distance": rs.total_distance,
            "energy": rs.total_energy,
            "idle_time": rs.total_idle_time,
        }
    return timeline


# ---- Figure generation ----

plt.rcParams.update({
    'font.size': 10,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def plot_robot_gantt(sim, tasks, relay_plans, fig_dir):
    """Robot Gantt chart with GPU annotations."""
    robot_actions = defaultdict(list)
    for log in sim.action_log:
        robot_actions[log.robot_id].append(log)

    all_rids = sorted(set(rs.id for rs in sim.robot_states))
    n_robots = len(all_rids)
    if n_robots == 0:
        return

    fig, ax = plt.subplots(figsize=(16, max(4, n_robots * 0.6)))

    color_map = {
        "WALK_TO_ELEVATOR": "#6baed6",
        "APPROACH_ELEVATOR": "#e6550d",
        "PRESS_BUTTON": "#fd8d3c",
        "WAIT_ELEVATOR": "#969696",
        "RIDE_ELEVATOR": "#31a354",
        "EXIT_ELEVATOR": "#e6550d",
        "PICKUP": "#756bb1",
        "BATCH_PICKUP": "#9e9ac8",
        "DELIVER": "#2171b5",
        "RELAY_DROPOFF": "#d94801",
        "RELAY_HANDOFF": "#d94801",
        "START_TASK": "#bdbdbd",
    }

    y_map = {rid: i for i, rid in enumerate(all_rids)}

    for rid in all_rids:
        if rid not in robot_actions or not robot_actions[rid]:
            continue
        actions = sorted(robot_actions[rid], key=lambda a: a.start_time)
        y = y_map[rid]
        for i, a in enumerate(actions):
            duration_map = {
                "APPROACH_ELEVATOR": 8.0, "PRESS_BUTTON": 7.0,
                "EXIT_ELEVATOR": 5.0, "PICKUP": 10.0,
                "BATCH_PICKUP": 5.0, "DELIVER": 10.0,
                "RELAY_DROPOFF": 5.0, "WALK_TO_ELEVATOR": 4.0,
            }
            dur = duration_map.get(a.action, 2.0)

            # Estimate ride duration from log
            if a.action == "RIDE_ELEVATOR":
                for j in range(i + 1, len(actions)):
                    if actions[j].action in ("EXIT_ELEVATOR", "EXIT_DONE"):
                        dur = actions[j].start_time - a.start_time
                        break

            color = color_map.get(a.action, "#bdbdbd")
            alpha = 1.0 if a.needs_gpu else 0.7
            ax.barh(y, dur, left=a.start_time, height=0.6,
                    color=color, alpha=alpha, edgecolor='white', linewidth=0.3)

            if a.needs_gpu:
                ax.plot(a.start_time + dur / 2, y, marker='*', color='red',
                        markersize=6, zorder=5)

    ax.set_yticks(range(len(all_rids)))
    ax.set_yticklabels([f"R{rid}" for rid in all_rids])
    ax.set_xlabel("Time (s)")
    ax.set_title("Robot Activity Gantt Chart (★ = GPU required)")
    ax.invert_yaxis()

    # Legend
    legend_items = [
        ("GPU (approach/press/exit)", "#e6550d"),
        ("Ride elevator", "#31a354"),
        ("Wait elevator", "#969696"),
        ("Pickup", "#756bb1"),
        ("Deliver", "#2171b5"),
        ("Walk", "#6baed6"),
    ]
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=c, label=l) for l, c in legend_items]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=8, ncol=2)

    plt.savefig(os.path.join(fig_dir, "robot_gantt.png"))
    plt.close()
    print("  [1/6] robot_gantt.png")


def plot_gpu_comparison_bar(gpu_results, fig_dir):
    """Bar chart comparing GPU scheduling strategies (exclude Fixed from wait chart)."""
    names = list(gpu_results.keys())
    colors_map = {
        'FragAware': '#1f77b4', 'Predictive': '#ff7f0e', 'OnDemand': '#2ca02c',
        'Fixed': '#d62728', 'RoundRobin': '#9467bd',
    }

    metrics_to_plot = [
        ("total_wait_delay", "Total Wait Delay (s)", True),
        ("cold_starts", "Cold Starts", False),
        ("avg_fragmentation", "Avg Fragmentation", False),
        ("avg_wait_delay", "Avg Wait per Demand (s)", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for ax, (metric, label, exclude_fixed) in zip(axes.flat, metrics_to_plot):
        plot_names = [n for n in names if not (exclude_fixed and n == "Fixed")]
        vals = [gpu_results[n].get(metric, 0) for n in plot_names]
        bar_colors = [colors_map.get(n, '#888888') for n in plot_names]
        bars = ax.bar(plot_names, vals, color=bar_colors, edgecolor='white', width=0.6)
        ax.set_ylabel(label)
        ax.set_title(label)
        for bar, v in zip(bars, vals):
            fmt = f"{v:.2f}" if isinstance(v, float) else str(v)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    fmt, ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.tick_params(axis='x', rotation=15)
        ax.set_ylim(0, max(vals) * 1.2 if max(vals) > 0 else 1)

    if "Fixed" in gpu_results:
        fixed_wait = gpu_results["Fixed"].get("total_wait_delay", 0)
        if fixed_wait > 1000:
            axes[0, 0].annotate(
                f"Fixed: {fixed_wait:.0f}s\n(off scale, excluded)",
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffcccc', alpha=0.8))

    plt.suptitle("GPU Scheduling Strategy Comparison\n"
                 "(2 GPU / 4 concurrent slots, 9 active robots)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "gpu_comparison.png"))
    plt.close()
    print("  [2/6] gpu_comparison.png")


def plot_gpu_timeline(gpu_results, gpu_cfg, fig_dir):
    """GPU utilization over time for each strategy."""
    fig, axes = plt.subplots(len(gpu_results), 1,
                              figsize=(14, 3 * len(gpu_results)), sharex=True)
    if len(gpu_results) == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, gpu_results.items()):
        samples = data.get("utilization_samples", [])
        if not samples:
            ax.set_title(f"{name}: No utilization data")
            continue
        times = [s["t"] for s in samples]
        utils = [s["util"] for s in samples]
        ax.fill_between(times, utils, alpha=0.4, color='#1f77b4')
        ax.plot(times, utils, color='#1f77b4', linewidth=0.8)
        ax.set_ylabel("GPU Util.")
        ax.set_title(f"{name} (wait={data['total_wait_delay']:.1f}s, "
                     f"cold={data['cold_starts']})")
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, linewidth=0.8)

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle("GPU Utilization Over Time", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "gpu_timeline.png"))
    plt.close()
    print("  [3/6] gpu_timeline.png")


def plot_elevator_usage(sim, fig_dir):
    """Elevator usage timeline and statistics."""
    egcs_stats = sim.egcs.get_statistics()
    per_elev = egcs_stats['per_elevator']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart of elevator statistics
    eids = sorted(per_elev.keys(), key=int)
    trips = [per_elev[e]['trips'] for e in eids]
    utils = [per_elev[e]['utilization'] * 100 for e in eids]

    x = np.arange(len(eids))
    w = 0.35
    bars1 = ax1.bar(x - w / 2, trips, w, label='Trips', color='#2171b5')
    ax1_twin = ax1.twinx()
    bars2 = ax1_twin.bar(x + w / 2, utils, w, label='Utilization %',
                          color='#fd8d3c', alpha=0.8)
    ax1.set_xlabel("Elevator ID")
    ax1.set_ylabel("Trips")
    ax1_twin.set_ylabel("Utilization (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"E{e}" for e in eids])
    ax1.set_title("Elevator Usage")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Wait time distribution
    wait_times = []
    for call in sim.egcs.call_history:
        if call.pickup_time > 0:
            wait_times.append(call.pickup_time - call.call_time)

    if wait_times:
        ax2.hist(wait_times, bins=20, color='#756bb1', edgecolor='white', alpha=0.8)
        ax2.axvline(np.mean(wait_times), color='red', linestyle='--',
                    label=f'Mean={np.mean(wait_times):.1f}s')
        ax2.set_xlabel("Wait Time (s)")
        ax2.set_ylabel("Count")
        ax2.set_title("Elevator Wait Time Distribution")
        ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "elevator_usage.png"))
    plt.close()
    print("  [4/6] elevator_usage.png")


def plot_robot_gpu_demand_density(demands, fig_dir):
    """Heatmap of GPU demand density over time per robot."""
    if not demands:
        return

    max_time = max(s.end for segs in demands.values()
                   for s in segs if s.demand_type == "demand")
    bin_size = max(max_time / 50, 5.0)
    n_bins = int(np.ceil(max_time / bin_size))
    rids = sorted(demands.keys())

    matrix = np.zeros((len(rids), n_bins))
    for i, rid in enumerate(rids):
        for seg in demands[rid]:
            if seg.demand_type == "demand":
                start_bin = int(seg.start / bin_size)
                end_bin = min(int(seg.end / bin_size), n_bins - 1)
                for b in range(start_bin, end_bin + 1):
                    matrix[i, b] += 1

    fig, ax = plt.subplots(figsize=(14, max(4, len(rids) * 0.5)))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_yticks(range(len(rids)))
    ax.set_yticklabels([f"R{r}" for r in rids])

    n_xticks = min(10, n_bins)
    xtick_pos = np.linspace(0, n_bins - 1, n_xticks, dtype=int)
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels([f"{int(p * bin_size)}s" for p in xtick_pos])
    ax.set_xlabel("Time")
    ax.set_title("GPU Demand Density per Robot")
    plt.colorbar(im, ax=ax, label="Demand count")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "gpu_demand_density.png"))
    plt.close()
    print("  [5/6] gpu_demand_density.png")


def plot_fragmentation_over_time(gpu_results, fig_dir):
    """Fragmentation comparison across strategies."""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for idx, (name, data) in enumerate(gpu_results.items()):
        samples = data.get("utilization_samples", [])
        if not samples:
            continue
        times = [s["t"] for s in samples]
        # Use utilization as proxy since fragmentation_samples are internal
        utils = [s["util"] for s in samples]
        ax.plot(times, utils, label=name, color=colors[idx % len(colors)],
                linewidth=1.2, alpha=0.8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("GPU Utilization")
    ax.set_title("GPU Utilization Comparison Across Strategies")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fragmentation_comparison.png"))
    plt.close()
    print("  [6/6] fragmentation_comparison.png")


if __name__ == "__main__":
    run_joint_experiment()
