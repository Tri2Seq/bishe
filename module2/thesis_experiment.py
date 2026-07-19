"""Thesis Chapter Experiments for GPU Cluster Scheduling

Experiment 1: Main comparison (3 distributions × 4 methods × 20 seeds)
Experiment 2: Scale sensitivity (vary server count)
Experiment 3: Parameter sensitivity (ε, checkpoint frequency)
Experiment 4: Robot inference integration (24h scenario from module1)
"""

import sys
import os
import json
import time
import numpy as np
from typing import Dict, List

sys.path.insert(0, os.path.dirname(__file__))

from frag_cluster import (
    ClusterSimulator, BestFitNoMigration, ServerConsolidation,
    InaccuratePrediction, generate_workload, SimMetrics,
    DISTRIBUTIONS, phi_table
)


def run_single(sim_cls, jobs, total_time, **kwargs) -> Dict:
    sim = sim_cls(**kwargs)
    t0 = time.time()
    m = sim.run(jobs, total_time)
    dt = time.time() - t0
    return {
        'avg_wait': round(m.avg_wait, 4),
        'avg_queue_cardtime': round(m.avg_queue_cardtime, 4),
        'avg_utilization': round(m.avg_utilization, 4),
        'makespan': round(m.makespan, 2),
        'migrations': m.total_migrations,
        'completed': m.completed_jobs,
        'total': m.total_jobs,
        'solve_time': round(dt, 2),
    }


def experiment1_main_comparison(output_dir: str, n_seeds: int = 20,
                                 num_servers: int = 50, C: int = 8,
                                 total_time: float = 100.0,
                                 arrival_rate: float = 5.0):
    """Main comparison: 3 distributions × 4 methods × n_seeds."""
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    mismatch_map = {'inv': 'norm', 'norm': 'dir', 'dir': 'inv'}

    for dist_name in ['inv', 'norm', 'dir']:
        dist = DISTRIBUTIONS[dist_name]
        wrong_dist = DISTRIBUTIONS[mismatch_map[dist_name]]
        print(f"\n{'='*60}")
        print(f"Distribution: {dist_name}")
        print(f"{'='*60}")

        dist_results = {m: [] for m in ['Ours', 'BF', 'SC', 'MismatchPred']}

        for seed in range(n_seeds):
            jobs = generate_workload(dist_name, total_time, arrival_rate, C, seed=seed)
            print(f"  Seed {seed}: {len(jobs)} jobs", end="", flush=True)

            r = run_single(ClusterSimulator, [_copy_job(j) for j in jobs], total_time,
                           num_servers=num_servers, gpu_per_server=C,
                           dist=dist, epsilon=0.01, milp_time_limit=10)
            dist_results['Ours'].append(r)
            print(f" | Ours w={r['avg_wait']:.2f}", end="", flush=True)

            r = run_single(BestFitNoMigration, [_copy_job(j) for j in jobs], total_time,
                           num_servers=num_servers, gpu_per_server=C,
                           dist=dist, epsilon=0.01)
            dist_results['BF'].append(r)
            print(f" BF={r['avg_wait']:.2f}", end="", flush=True)

            r = run_single(ServerConsolidation, [_copy_job(j) for j in jobs], total_time,
                           num_servers=num_servers, gpu_per_server=C,
                           dist=dist, epsilon=0.01)
            dist_results['SC'].append(r)
            print(f" SC={r['avg_wait']:.2f}", end="", flush=True)

            r = run_single(InaccuratePrediction, [_copy_job(j) for j in jobs], total_time,
                           num_servers=num_servers, gpu_per_server=C,
                           actual_dist=dist, wrong_dist=wrong_dist,
                           epsilon=0.01)
            dist_results['MismatchPred'].append(r)
            print(f" MP={r['avg_wait']:.2f}")

        results[dist_name] = dist_results

    _print_summary(results)

    with open(os.path.join(output_dir, "exp1_main.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_dir}/exp1_main.json")
    return results


def experiment2_scale(output_dir: str, n_seeds: int = 5,
                      total_time: float = 100.0, arrival_rate: float = 5.0):
    """Scale sensitivity: vary server count."""
    os.makedirs(output_dir, exist_ok=True)
    C = 8
    dist = DISTRIBUTIONS['norm']
    server_counts = [10, 25, 50, 100]

    results = {}
    for P in server_counts:
        print(f"\n--- {P} servers ---")
        run_results = {m: [] for m in ['Ours', 'BF', 'SC']}
        for seed in range(n_seeds):
            jobs = generate_workload('norm', total_time, arrival_rate, C, seed=seed)
            for method, cls in [('Ours', ClusterSimulator),
                                ('BF', BestFitNoMigration),
                                ('SC', ServerConsolidation)]:
                r = run_single(cls, [_copy_job(j) for j in jobs], total_time,
                               num_servers=P, gpu_per_server=C,
                               dist=dist, epsilon=0.01, milp_time_limit=10)
                run_results[method].append(r)
            print(f"  Seed {seed}: Ours w={run_results['Ours'][-1]['avg_wait']:.2f} "
                  f"BF={run_results['BF'][-1]['avg_wait']:.2f}")
        results[str(P)] = run_results

    with open(os.path.join(output_dir, "exp2_scale.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_dir}/exp2_scale.json")
    return results


def experiment3_ablation(output_dir: str, n_seeds: int = 5,
                         total_time: float = 100.0, arrival_rate: float = 5.0):
    """Ablation: ε sensitivity and checkpoint frequency."""
    os.makedirs(output_dir, exist_ok=True)
    C = 8
    P = 50
    dist = DISTRIBUTIONS['norm']

    epsilon_results = {}
    for eps in [0.0, 0.001, 0.01, 0.1, 1.0]:
        runs = []
        for seed in range(n_seeds):
            jobs = generate_workload('norm', total_time, arrival_rate, C, seed=seed)
            r = run_single(ClusterSimulator, [_copy_job(j) for j in jobs], total_time,
                           num_servers=P, gpu_per_server=C,
                           dist=dist, epsilon=eps, milp_time_limit=10)
            runs.append(r)
        epsilon_results[str(eps)] = runs
        avg_w = np.mean([r['avg_wait'] for r in runs])
        avg_m = np.mean([r['migrations'] for r in runs])
        print(f"  ε={eps}: avg_wait={avg_w:.3f}, migrations={avg_m:.1f}")

    ckpt_results = {}
    for ckpt_interval in [0.5, 1.0, 2.0, 4.0]:
        runs = []
        for seed in range(n_seeds):
            jobs = generate_workload('norm', total_time, arrival_rate, C, seed=seed)
            for j in jobs:
                n_ck = max(1, int(j.duration / ckpt_interval))
                rng = np.random.RandomState(seed + j.id)
                j.checkpoints = sorted([j.arrival + rng.uniform(0, j.duration)
                                        for _ in range(n_ck)])
            r = run_single(ClusterSimulator, [_copy_job(j) for j in jobs], total_time,
                           num_servers=P, gpu_per_server=C,
                           dist=dist, epsilon=0.01, milp_time_limit=10)
            runs.append(r)
        ckpt_results[str(ckpt_interval)] = runs
        avg_w = np.mean([r['avg_wait'] for r in runs])
        print(f"  ckpt_interval={ckpt_interval}h: avg_wait={avg_w:.3f}")

    all_results = {'epsilon': epsilon_results, 'checkpoint': ckpt_results}
    with open(os.path.join(output_dir, "exp3_ablation.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_dir}/exp3_ablation.json")
    return all_results


def _copy_job(job):
    from frag_cluster import GPUJob
    return GPUJob(id=job.id, gpu_req=job.gpu_req, duration=job.duration,
                  arrival=job.arrival, checkpoints=list(job.checkpoints))


def _print_summary(results: Dict):
    print(f"\n{'='*80}")
    print("Summary: mean ± std across seeds")
    print(f"{'='*80}")
    for dist_name, dist_results in results.items():
        print(f"\n--- {dist_name} ---")
        print(f"{'Method':<15} {'Wait':>10} {'QueueCard':>12} {'Util':>8} "
              f"{'Makespan':>10} {'Migrations':>10}")
        print("-" * 70)
        for method, runs in dist_results.items():
            w = [r['avg_wait'] for r in runs]
            q = [r['avg_queue_cardtime'] for r in runs]
            u = [r['avg_utilization'] for r in runs]
            ms = [r['makespan'] for r in runs]
            mg = [r['migrations'] for r in runs]
            print(f"{method:<15} {np.mean(w):>7.3f}±{np.std(w):.3f} "
                  f"{np.mean(q):>9.3f}±{np.std(q):.3f} "
                  f"{np.mean(u):>5.3f}±{np.std(u):.3f} "
                  f"{np.mean(ms):>7.1f}±{np.std(ms):.1f} "
                  f"{np.mean(mg):>7.1f}±{np.std(mg):.1f}")


if __name__ == "__main__":
    out = "results/thesis_gpu"
    print("Experiment 1: Main Comparison")
    experiment1_main_comparison(out, n_seeds=10, arrival_rate=10.0)
    print("\n\nExperiment 2: Scale Sensitivity")
    experiment2_scale(out, n_seeds=5, arrival_rate=10.0)
    print("\n\nExperiment 3: Ablation Study")
    experiment3_ablation(out, n_seeds=5, arrival_rate=10.0)
