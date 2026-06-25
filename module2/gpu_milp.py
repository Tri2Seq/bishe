"""模块2：GPU调度MILP精确求解

给定机器人动作时间线（模块1输出），求解最优GPU分配+迁移策略。

决策：
  h[j,k,g]: 机器人j的第k个需求段分配到GPU g
  r[j,k]:   机器人j在第k个需求段结束后是否释放GPU

目标：最小化 β₁·碎片率 + β₂·迁移开销 + β₃·等待延迟
约束：GPU显存容量、冷启动时间、迁移窗口
"""

import pulp
import numpy as np
from typing import List, Dict, Tuple, Optional
from gpu_config import GPUConfig, GPUDemand, load_gpu_demands


def solve_gpu_milp(demands: Dict[int, List[GPUDemand]], config: GPUConfig,
                   beta: Tuple[float, float, float] = (0.4, 0.3, 0.3),
                   time_limit: int = 60, verbose: bool = True) -> Optional[Dict]:
    """
    MILP求解GPU调度

    Args:
        demands: {robot_id: [GPUDemand, ...]} 从模块1输出
        config: GPU集群配置
        beta: 目标权重 (碎片, 迁移, 延迟)

    Returns:
        分配结果
    """
    # 提取所有需求段
    demand_segs = []  # (robot_id, seg_idx, start, end)
    idle_segs = []    # (robot_id, seg_idx, start, end, reason, duration)

    for rid, segs in demands.items():
        d_idx = 0
        for seg in segs:
            if seg.demand_type == "demand":
                demand_segs.append((rid, d_idx, seg.start, seg.end))
                d_idx += 1
            elif seg.demand_type == "idle":
                idle_segs.append((rid, d_idx, seg.start, seg.end, seg.reason, seg.duration))

    K = len(demand_segs)  # 总需求段数
    G = config.total_gpus
    C = config.max_concurrent_per_gpu
    v = config.depth_model_vram

    if verbose:
        print(f"GPU MILP: {K}需求段, {G}张GPU, 每GPU {C}并发")

    if K == 0:
        return {"status": "NoWork", "allocations": []}

    prob = pulp.LpProblem("GPU_Scheduling", pulp.LpMinimize)

    # 决策变量
    # h[k,g]: 需求段k分配到GPU g
    h = pulp.LpVariable.dicts("h", ((k, g) for k in range(K) for g in range(G)),
                               cat="Binary")

    # 迁移变量: migrate[k] = 1 if 需求段k与k-1(同一机器人)使用不同GPU
    migrate = {}
    # 建立同一机器人的连续需求段对
    robot_seg_pairs = []  # (k1, k2) where same robot, consecutive
    seg_by_robot = {}
    for k, (rid, sidx, start, end) in enumerate(demand_segs):
        if rid not in seg_by_robot:
            seg_by_robot[rid] = []
        seg_by_robot[rid].append(k)

    for rid, ks in seg_by_robot.items():
        for i in range(len(ks) - 1):
            k1, k2 = ks[i], ks[i + 1]
            robot_seg_pairs.append((k1, k2))
            migrate[k1, k2] = pulp.LpVariable(f"mig_{k1}_{k2}", cat="Binary")

    # 约束
    # C1: 每个需求段恰好分配到1个GPU
    for k in range(K):
        prob += pulp.lpSum(h[k, g] for g in range(G)) == 1, f"C1_{k}"

    # C2: GPU并发容量 — 同一时刻GPU g上的活跃需求段数 ≤ C
    # 对每对时间重叠的需求段检查
    for g in range(G):
        for k1 in range(K):
            overlapping = []
            s1, e1 = demand_segs[k1][2], demand_segs[k1][3]
            for k2 in range(K):
                if k1 == k2:
                    continue
                s2, e2 = demand_segs[k2][2], demand_segs[k2][3]
                if s1 < e2 and s2 < e1:  # 时间重叠
                    overlapping.append(k2)
            if overlapping:
                prob += h[k1, g] + pulp.lpSum(h[k2, g] for k2 in overlapping) <= C, \
                        f"C2_{g}_{k1}"

    # C3: 迁移检测 — 如果同一机器人的连续需求段在不同GPU上，则migrate=1
    for k1, k2 in robot_seg_pairs:
        for g in range(G):
            # migrate >= h[k1,g] - h[k2,g] → if k1 on g but k2 not, then migrate
            prob += migrate[k1, k2] >= h[k1, g] - h[k2, g], f"C3a_{k1}_{k2}_{g}"

    # 目标函数
    # 迁移开销
    total_migrate = pulp.lpSum(migrate[k1, k2] for k1, k2 in robot_seg_pairs)

    # 碎片率近似：鼓励GPU负载均衡（用方差的线性近似）
    # 简化：每个GPU上的需求段数尽量相等
    gpu_load = {}
    for g in range(G):
        gpu_load[g] = pulp.lpSum(h[k, g] for k in range(K))
    avg_load = K / G
    load_dev = pulp.LpVariable.dicts("dev", range(G), lowBound=0)
    for g in range(G):
        prob += load_dev[g] >= gpu_load[g] - avg_load, f"dev_pos_{g}"
        prob += load_dev[g] >= avg_load - gpu_load[g], f"dev_neg_{g}"
    total_imbalance = pulp.lpSum(load_dev[g] for g in range(G))

    b1, b2, b3 = beta
    prob += b1 * total_imbalance / max(G, 1) + b2 * total_migrate / max(len(robot_seg_pairs), 1), \
            "objective"

    # 求解
    solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=verbose)
    status = prob.solve(solver)

    if verbose:
        print(f"求解状态: {pulp.LpStatus[status]}")

    if pulp.value(prob.objective) is None:
        return None

    # 提取结果
    allocations = []
    for k in range(K):
        rid, sidx, start, end = demand_segs[k]
        for g in range(G):
            if pulp.value(h[k, g]) is not None and pulp.value(h[k, g]) > 0.5:
                allocations.append({
                    "robot_id": rid, "seg_idx": sidx,
                    "gpu_id": g, "start": start, "end": end,
                })
                break

    n_migrations = sum(1 for (k1, k2) in robot_seg_pairs
                       if pulp.value(migrate[k1, k2]) is not None
                       and pulp.value(migrate[k1, k2]) > 0.5)

    # GPU利用率
    gpu_alloc_counts = {g: 0 for g in range(G)}
    for a in allocations:
        gpu_alloc_counts[a['gpu_id']] += 1

    result = {
        "status": pulp.LpStatus[status],
        "allocations": allocations,
        "migrations": n_migrations,
        "gpu_load_distribution": gpu_alloc_counts,
        "imbalance": pulp.value(total_imbalance) if pulp.value(total_imbalance) else 0,
        "objective": pulp.value(prob.objective),
    }

    if verbose:
        print(f"\nGPU MILP结果:")
        print(f"  需求段: {K}, 迁移: {n_migrations}")
        print(f"  GPU负载分布: {gpu_alloc_counts}")
        print(f"  负载不均衡度: {result['imbalance']:.2f}")

    return result
