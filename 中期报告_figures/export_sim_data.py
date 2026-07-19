"""Quick re-run of joint simulation to export action_log and GPU timeline data.

The simulator's action_log records events as points (start_time == end_time).
We convert them to intervals by computing each action's duration from the gap
to the next event for the same robot, or using fixed estimates as fallback.
"""
import sys, os, json
from collections import defaultdict

sys.path.insert(0, '/home/ubuntu/Documents/sjx/gra/module1')
sys.path.insert(0, '/home/ubuntu/Documents/sjx/gra/module2')

from config import ScenarioConfig, generate_scenario
from simulator_v2 import SimulatorV2, RobotPhase, GPU_PHASES, RobotStrategy
from nsga2_ebo import NSGA2EBOSolver
from gpu_config import GPUConfig, GPUDemand
from gpu_scheduler import (FragAwareScheduler, PredictiveScheduler,
                            OnDemandScheduler, FixedScheduler, RoundRobinScheduler)

OUTDIR = '/home/ubuntu/Documents/sjx/gra/module1/results/joint_experiment'

DURATION_MAP = {
    "APPROACH_ELEVATOR": 8.0,
    "PRESS_BUTTON": 7.0,
    "EXIT_ELEVATOR": 5.0,
    "EXIT_DONE": 3.0,
    "PICKUP": 10.0,
    "BATCH_PICKUP": 5.0,
    "DELIVER": 10.0,
    "RELAY_DROPOFF": 5.0,
    "RELAY_HANDOFF": 5.0,
    "WALK_TO_ELEVATOR": 4.0,
    "START_TASK": 1.0,
    "WAIT_ELEVATOR": 5.0,
    "RIDE_ELEVATOR": 10.0,
}

GPU_ACTIONS = {"APPROACH_ELEVATOR", "PRESS_BUTTON", "EXIT_ELEVATOR"}

cfg = ScenarioConfig(num_floors=10, num_robots=12, num_elevators=4, num_tasks=40, seed=42)
robots, elevators, tasks = generate_scenario(cfg)

print("Running NSGA-II-EBO (50 gen for speed)...")
solver = NSGA2EBOSolver(robots, elevators, tasks, pop_size=50, max_gen=50, seed=42)
solver.solve(verbose=False)
assignment, task_seq, relay_plans, strategy = solver.get_best_schedule(objective_idx=0)

print("Running simulation...")
sim = SimulatorV2(robots, elevators, tasks, dt=0.5, strategy=strategy)
sim.load_schedule(assignment, task_seq, relay_plans=relay_plans)
result = sim.run(max_time=20000)

# Group raw events by robot and sort by time
robot_events = defaultdict(list)
for log in sim.action_log:
    robot_events[log.robot_id].append(log)
for rid in robot_events:
    robot_events[rid].sort(key=lambda a: a.start_time)

# Convert point events to intervals
action_log = []
for rid in sorted(robot_events.keys()):
    events = robot_events[rid]
    for i, log in enumerate(events):
        # Duration: gap to next event for same robot, or use fallback
        if i + 1 < len(events):
            gap = events[i + 1].start_time - log.start_time
            if gap > 0:
                dur = min(gap, DURATION_MAP.get(log.action, 2.0))
            else:
                dur = DURATION_MAP.get(log.action, 2.0)
        else:
            dur = DURATION_MAP.get(log.action, 2.0)

        # Special: RIDE_ELEVATOR duration from gap to EXIT
        if log.action == "RIDE_ELEVATOR":
            for j in range(i + 1, len(events)):
                if events[j].action in ("EXIT_ELEVATOR", "EXIT_DONE"):
                    dur = events[j].start_time - log.start_time
                    break

        # Special: WAIT_ELEVATOR duration from gap to RIDE
        if log.action == "WAIT_ELEVATOR":
            for j in range(i + 1, len(events)):
                if events[j].action == "RIDE_ELEVATOR":
                    dur = events[j].start_time - log.start_time
                    break

        action_log.append({
            'robot_id': rid,
            'phase': log.action,
            'start_time': log.start_time,
            'end_time': log.start_time + dur,
            'task_id': log.task_id,
            'needs_gpu': log.needs_gpu,
        })

with open(os.path.join(OUTDIR, 'action_log.json'), 'w') as f:
    json.dump(action_log, f)
print(f"Exported {len(action_log)} action log entries")

# Build GPU demands with real durations
gpu_cfg = GPUConfig(num_servers=1, gpus_per_server=2, vram_per_gpu=16.0, depth_model_vram=6.0)
demands_by_robot = defaultdict(list)

for entry in action_log:
    d = GPUDemand(
        robot_id=entry['robot_id'],
        start=entry['start_time'],
        end=entry['end_time'],
        demand_type="demand" if entry['needs_gpu'] else "idle",
        reason=entry['phase'],
    )
    demands_by_robot[entry['robot_id']].append(d)

for rid in demands_by_robot:
    demands_by_robot[rid].sort(key=lambda x: x.start)
demands_dict = dict(demands_by_robot)

gpu_demands = sum(1 for e in action_log if e['needs_gpu'])
print(f"Extracted {gpu_demands} GPU demands across {len(demands_dict)} robots")

# Run GPU schedulers
timeline_data = {}
schedulers = {
    'FragAware': FragAwareScheduler(gpu_cfg),
    'OnDemand': OnDemandScheduler(gpu_cfg),
    'RoundRobin': RoundRobinScheduler(gpu_cfg),
}

for name, sched in schedulers.items():
    metrics = sched.schedule(demands_dict)
    samples = [{'t': t, 'util': u} for t, u in metrics.gpu_utilization_samples]
    timeline_data[name] = {
        'total_wait_delay': metrics.total_wait_delay,
        'cold_starts': metrics.total_cold_starts,
        'avg_fragmentation': metrics.avg_fragmentation,
        'utilization_samples': samples,
    }
    print(f"  {name}: wait={metrics.total_wait_delay:.1f}s, cold={metrics.total_cold_starts}, samples={len(samples)}")

with open(os.path.join(OUTDIR, 'gpu_timeline_data.json'), 'w') as f:
    json.dump(timeline_data, f)
print("Done. Data exported.")
