"""GPU Cluster Scheduling via Checkpoint-based Live Migration

Faithful implementation of the IFAC paper's core algorithm:
- φ(k) fragmentation cost function based on demand distribution
- MILP with one-hot linearization for migration decisions
- Rolling checkpoint-triggered optimization (Algorithm 1)
- Baselines: Best Fit (BF), Server Consolidation (SC), Inaccurate Prediction
"""

import heapq
import numpy as np
import pulp
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


DEMAND_SIZES = [1, 2, 4, 6, 8]

DISTRIBUTIONS = {
    'inv':  [0.40, 0.25, 0.20, 0.10, 0.05],
    'norm': [0.10, 0.20, 0.40, 0.20, 0.10],
    'dir':  [0.05, 0.10, 0.20, 0.25, 0.40],
}


def phi(k: int, dist: List[float], C: int = 8) -> float:
    if k <= 0 or k >= C:
        return 0.0
    return sum(p for s, p in zip(DEMAND_SIZES, dist) if s > k)


def phi_table(dist: List[float], C: int = 8) -> List[float]:
    return [phi(k, dist, C) for k in range(C + 1)]


@dataclass
class GPUJob:
    id: int
    gpu_req: int
    duration: float
    arrival: float
    checkpoints: List[float] = field(default_factory=list)
    server: int = -1
    start_time: float = -1.0
    end_time: float = -1.0


class EventType(Enum):
    ARRIVAL = 0
    CHECKPOINT = 1
    COMPLETION = 2


@dataclass(order=True)
class Event:
    time: float
    priority: int = field(compare=True)
    etype: EventType = field(compare=False)
    job_id: int = field(compare=False, default=-1)


def generate_workload(dist_name: str, total_time: float = 100.0,
                      arrival_rate: float = 5.0, C: int = 8,
                      seed: int = 42) -> List[GPUJob]:
    rng = np.random.RandomState(seed)
    dist = DISTRIBUTIONS[dist_name]
    jobs = []
    t = 0.0
    jid = 0
    while t < total_time:
        t += rng.exponential(1.0 / arrival_rate)
        if t >= total_time:
            break
        gpu_req = rng.choice(DEMAND_SIZES, p=dist)
        duration = rng.uniform(2, 20)
        ckpt_interval = rng.uniform(1, 3)
        n_ckpts = max(1, int(duration / ckpt_interval))
        ckpts = sorted(rng.uniform(0, duration, n_ckpts))
        abs_ckpts = [t + c for c in ckpts]
        jobs.append(GPUJob(id=jid, gpu_req=gpu_req, duration=duration,
                           arrival=t, checkpoints=abs_ckpts))
        jid += 1
    return jobs


@dataclass
class ServerState:
    id: int
    capacity: int
    tasks: Dict[int, int] = field(default_factory=dict)

    @property
    def used(self) -> int:
        return sum(self.tasks.values())

    @property
    def remaining(self) -> int:
        return self.capacity - self.used

    def can_fit(self, gpu_req: int) -> bool:
        return self.remaining >= gpu_req

    def place(self, job_id: int, gpu_req: int):
        self.tasks[job_id] = gpu_req

    def remove(self, job_id: int):
        self.tasks.pop(job_id, None)


@dataclass
class SimMetrics:
    total_jobs: int = 0
    completed_jobs: int = 0
    total_wait: float = 0.0
    total_queue_cardtime: float = 0.0
    total_migrations: int = 0
    utilization_samples: List[Tuple[float, float]] = field(default_factory=list)
    makespan: float = 0.0

    @property
    def avg_wait(self) -> float:
        return self.total_wait / max(self.completed_jobs, 1)

    @property
    def avg_queue_cardtime(self) -> float:
        return self.total_queue_cardtime / max(self.completed_jobs, 1)

    @property
    def avg_utilization(self) -> float:
        if not self.utilization_samples:
            return 0.0
        return np.mean([u for _, u in self.utilization_samples])


def solve_migration_milp(servers: List[ServerState], eligible: List[GPUJob],
                         phi_tab: List[float], C: int, epsilon: float = 0.01,
                         time_limit: int = 5) -> Dict[int, int]:
    if not eligible or len(eligible) > 20:
        return {}

    eligible_ids = {j.id for j in eligible}
    involved_sids = {j.server for j in eligible}
    src_servers = [s for s in servers if s.id in involved_sids]
    target_servers = sorted([s for s in servers if s.id not in involved_sids and s.remaining > 0],
                            key=lambda s: s.remaining, reverse=True)[:10]
    relevant = src_servers + target_servers
    if len(relevant) < 2:
        return {}

    P = len(relevant)
    sid_to_idx = {s.id: i for i, s in enumerate(relevant)}

    prob = pulp.LpProblem("FragMig", pulp.LpMinimize)

    x = {}
    for j_idx, job in enumerate(eligible):
        src = sid_to_idx.get(job.server)
        for p_idx in range(P):
            if p_idx != src and relevant[p_idx].remaining >= job.gpu_req:
                x[j_idx, p_idx] = pulp.LpVariable(f"x_{j_idx}_{p_idx}", cat="Binary")

    if not x:
        return {}

    f = {}
    for p in range(P):
        for k in range(C + 1):
            f[p, k] = pulp.LpVariable(f"f_{p}_{k}", cat="Binary")

    for p in range(P):
        prob += pulp.lpSum(f[p, k] for k in range(C + 1)) == 1, f"oh_{p}"

    fixed = {}
    for p_idx, srv in enumerate(relevant):
        fixed[p_idx] = sum(g for jid, g in srv.tasks.items() if jid not in eligible_ids)

    y = {}
    for j_idx, job in enumerate(eligible):
        out = [x[j_idx, p] for p in range(P) if (j_idx, p) in x]
        if out:
            prob += pulp.lpSum(out) <= 1, f"sm_{j_idx}"
            y[j_idx] = pulp.lpSum(out)
        else:
            y[j_idx] = 0

    for p_idx, srv in enumerate(relevant):
        L_expr = C - fixed[p_idx]
        for j_idx, job in enumerate(eligible):
            src = sid_to_idx.get(job.server)
            if src == p_idx:
                L_expr = L_expr - job.gpu_req * (1 - y[j_idx])
            if (j_idx, p_idx) in x:
                L_expr = L_expr - job.gpu_req * x[j_idx, p_idx]
        prob += pulp.lpSum(k * f[p_idx, k] for k in range(C + 1)) == L_expr, f"lk_{p_idx}"

    obj = pulp.lpSum(phi_tab[k] * f[p, k] for p in range(P) for k in range(C + 1))
    obj += epsilon * pulp.lpSum(x[key] for key in x)
    prob += obj, "obj"

    solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=0)
    prob.solve(solver)

    if prob.status != 1:
        return {}

    migrations = {}
    for j_idx, job in enumerate(eligible):
        for p_idx in range(P):
            if (j_idx, p_idx) in x:
                v = pulp.value(x[j_idx, p_idx])
                if v is not None and v > 0.5:
                    migrations[job.id] = relevant[p_idx].id
    return migrations


class ClusterSimulator:
    """Event-driven GPU cluster simulator with checkpoint-triggered migration."""

    def __init__(self, num_servers: int, gpu_per_server: int,
                 dist: List[float], epsilon: float = 0.01,
                 milp_time_limit: int = 5):
        self.num_servers = num_servers
        self.C = gpu_per_server
        self.dist = dist
        self.epsilon = epsilon
        self.milp_time_limit = milp_time_limit
        self.phi_tab = phi_table(dist, gpu_per_server)

    def _reset(self):
        self.servers = [ServerState(id=p, capacity=self.C)
                        for p in range(self.num_servers)]
        self.jobs: Dict[int, GPUJob] = {}
        self.queue: List[GPUJob] = []
        self.metrics = SimMetrics()
        self._active_jobs: Dict[int, GPUJob] = {}

    def _best_fit_place(self, job: GPUJob) -> Optional[int]:
        best_srv = None
        best_remaining = self.C + 1
        for srv in self.servers:
            if srv.can_fit(job.gpu_req):
                r = srv.remaining - job.gpu_req
                if r < best_remaining:
                    best_remaining = r
                    best_srv = srv.id
        return best_srv

    def _sample_utilization(self, t: float):
        total_used = sum(s.used for s in self.servers)
        total_cap = self.num_servers * self.C
        self.metrics.utilization_samples.append((t, total_used / total_cap))

    def _do_place(self, job: GPUJob, server_id: int, t: float):
        self.servers[server_id].place(job.id, job.gpu_req)
        job.server = server_id
        job.start_time = t
        job.end_time = t + job.duration
        wait = t - job.arrival
        self.metrics.total_wait += wait
        self.metrics.total_queue_cardtime += wait * job.gpu_req
        self.metrics.completed_jobs += 1
        self._active_jobs[job.id] = job

    def _try_place_queued(self, t: float, events):
        placed = []
        for i, job in enumerate(self.queue):
            sid = self._best_fit_place(job)
            if sid is not None:
                self._do_place(job, sid, t)
                heapq.heappush(events,
                               Event(job.end_time, 2, EventType.COMPLETION, job.id))
                placed.append(i)
        for i in reversed(placed):
            self.queue.pop(i)

    def _get_eligible(self, t: float, tol: float = 0.25) -> List[GPUJob]:
        return [j for j in self._active_jobs.values()
                if j.server >= 0 and any(abs(ck - t) < tol for ck in j.checkpoints)]

    def _handle_checkpoint(self, eligible: List[GPUJob], t: float):
        mig = solve_migration_milp(
            self.servers, eligible, self.phi_tab, self.C,
            self.epsilon, self.milp_time_limit)
        for jid, target_sid in mig.items():
            job = self._active_jobs.get(jid)
            if job is None or job.server < 0:
                continue
            self.servers[job.server].remove(jid)
            self.servers[target_sid].place(jid, job.gpu_req)
            job.server = target_sid
            self.metrics.total_migrations += 1

    def run(self, jobs: List[GPUJob], total_time: float = 100.0) -> SimMetrics:
        self._reset()
        self.metrics.total_jobs = len(jobs)
        self.jobs = {j.id: j for j in jobs}

        events = []
        ckpt_times = set()
        for job in jobs:
            heapq.heappush(events, Event(job.arrival, 0, EventType.ARRIVAL, job.id))
            for ck in job.checkpoints:
                ckpt_times.add(round(ck, 1))

        ckpt_step = 0.5
        for ck_t in np.arange(0, total_time + ckpt_step, ckpt_step):
            heapq.heappush(events, Event(float(ck_t), 1, EventType.CHECKPOINT))

        sample_interval = max(1.0, total_time / 200)
        last_sample = -sample_interval

        while events:
            ev = heapq.heappop(events)

            if ev.etype == EventType.ARRIVAL:
                job = self.jobs[ev.job_id]
                sid = self._best_fit_place(job)
                if sid is not None:
                    self._do_place(job, sid, ev.time)
                    heapq.heappush(events,
                                   Event(job.end_time, 2, EventType.COMPLETION, job.id))
                else:
                    self.queue.append(job)

            elif ev.etype == EventType.CHECKPOINT:
                eligible = self._get_eligible(ev.time)
                if eligible:
                    self._handle_checkpoint(eligible, ev.time)
                    self._try_place_queued(ev.time, events)

            elif ev.etype == EventType.COMPLETION:
                job = self._active_jobs.pop(ev.job_id, None)
                if job and job.server >= 0:
                    self.servers[job.server].remove(job.id)
                    job.server = -1
                    self.metrics.makespan = max(self.metrics.makespan, ev.time)
                self._try_place_queued(ev.time, events)

            if ev.time - last_sample >= sample_interval:
                self._sample_utilization(ev.time)
                last_sample = ev.time

        for job in self.queue:
            self.metrics.total_wait += total_time - job.arrival
            self.metrics.total_queue_cardtime += (total_time - job.arrival) * job.gpu_req

        return self.metrics


class BestFitNoMigration(ClusterSimulator):
    """Baseline: Best-Fit placement, no migration at checkpoints."""

    def _handle_checkpoint(self, eligible: List[GPUJob], t: float):
        pass


class ServerConsolidation(ClusterSimulator):
    """Baseline: Migrate to consolidate tasks onto fewer servers."""

    def _handle_checkpoint(self, eligible: List[GPUJob], t: float):
        sorted_servers = sorted(self.servers, key=lambda s: s.used, reverse=True)
        for job in sorted(eligible, key=lambda j: j.gpu_req):
            for srv in sorted_servers:
                if srv.id != job.server and srv.can_fit(job.gpu_req):
                    self.servers[job.server].remove(job.id)
                    srv.place(job.id, job.gpu_req)
                    job.server = srv.id
                    self.metrics.total_migrations += 1
                    break


class InaccuratePrediction(ClusterSimulator):
    """Baseline: Uses wrong demand distribution for φ(k)."""

    def __init__(self, num_servers: int, gpu_per_server: int,
                 actual_dist: List[float], wrong_dist: List[float],
                 epsilon: float = 0.01, milp_time_limit: int = 5):
        super().__init__(num_servers, gpu_per_server, actual_dist,
                         epsilon, milp_time_limit)
        self.phi_tab = phi_table(wrong_dist, gpu_per_server)
