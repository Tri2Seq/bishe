"""模块2：GPU资源调度器

基于机器人动作时间线，优化GPU分配、迁移和碎片整理。

调度策略：
  1. FragAware  — 碎片感知分配 + 乘梯窗口迁移（本文方法）
  2. OnDemand   — 按需分配，用完释放，不迁移（基线B5）
  3. Fixed      — 每台机器人绑定一张GPU（基线B6）
  4. RoundRobin — 轮询分配（基线B7）
"""

import heapq
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np

from gpu_config import GPUConfig, GPUDemand, GPUAllocation, GPUState


@dataclass
class SchedulerMetrics:
    """调度器性能指标"""
    total_demands: int = 0
    total_gpu_time: float = 0.0
    total_wait_delay: float = 0.0    # GPU未就绪导致的等待
    total_cold_starts: int = 0
    total_migrations: int = 0
    total_migration_time: float = 0.0
    fragmentation_samples: List[float] = field(default_factory=list)
    gpu_utilization_samples: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def avg_fragmentation(self) -> float:
        return np.mean(self.fragmentation_samples) if self.fragmentation_samples else 0

    @property
    def max_fragmentation(self) -> float:
        return max(self.fragmentation_samples) if self.fragmentation_samples else 0

    @property
    def avg_wait_delay(self) -> float:
        return self.total_wait_delay / max(self.total_demands, 1)

    def summary(self) -> Dict:
        return {
            "total_demands": self.total_demands,
            "total_gpu_time": round(self.total_gpu_time, 1),
            "total_wait_delay": round(self.total_wait_delay, 2),
            "avg_wait_delay": round(self.avg_wait_delay, 2),
            "cold_starts": self.total_cold_starts,
            "migrations": self.total_migrations,
            "migration_time": round(self.total_migration_time, 2),
            "avg_fragmentation": round(self.avg_fragmentation, 4),
            "max_fragmentation": round(self.max_fragmentation, 4),
        }


def compute_fragmentation(gpu_states: List[GPUState], config: GPUConfig,
                          demand_distribution: Optional[List[float]] = None) -> float:
    """
    计算集群总显存碎片率

    碎片率定义（来自选题报告）：
    φ(r) = Σ_{s∈S: s > r} π_s
    即：剩余显存r无法容纳的未来任务比例

    默认假设所有任务都需要 depth_model_vram，则：
    φ(r) = 1 if r < depth_model_vram else 0
    """
    if demand_distribution is None:
        # 默认：所有任务需要相同显存
        total_frag = 0.0
        for gpu in gpu_states:
            free = gpu.vram_free
            if free < config.depth_model_vram and free > 0:
                total_frag += 1.0  # 有剩余空间但装不下一个任务
        return total_frag / len(gpu_states)
    else:
        total_frag = 0.0
        for gpu in gpu_states:
            free = gpu.vram_free
            # φ(r) = sum of π_s for s > r
            frag = sum(p for s, p in demand_distribution if s > free)
            total_frag += frag
        return total_frag / len(gpu_states)


class BaseScheduler:
    """调度器基类"""

    def __init__(self, config: GPUConfig):
        self.config = config
        self.gpu_states = [
            GPUState(
                gpu_id=g,
                server_id=g // config.gpus_per_server,
                vram_total=config.vram_per_gpu,
            )
            for g in range(config.total_gpus)
        ]
        self.metrics = SchedulerMetrics()
        self.allocations: List[GPUAllocation] = []
        # robot_id -> gpu_id (当前分配)
        self.robot_gpu_map: Dict[int, int] = {}

    def _allocate(self, robot_id: int, gpu_id: int, start: float, end: float,
                  alloc_type: str = "active"):
        gpu = self.gpu_states[gpu_id]
        gpu.vram_used += self.config.depth_model_vram
        gpu.active_robots[robot_id] = start
        self.robot_gpu_map[robot_id] = gpu_id
        self.allocations.append(GPUAllocation(
            robot_id=robot_id, gpu_id=gpu_id,
            start=start, end=end, alloc_type=alloc_type,
        ))

    def _release(self, robot_id: int, gpu_id: int):
        gpu = self.gpu_states[gpu_id]
        gpu.vram_used = max(0, gpu.vram_used - self.config.depth_model_vram)
        gpu.active_robots.pop(robot_id, None)
        self.robot_gpu_map.pop(robot_id, None)

    def _find_best_gpu(self) -> Optional[int]:
        """找碎片率最低（空闲显存最大）的GPU"""
        best = None
        best_free = -1
        for gpu in self.gpu_states:
            if gpu.vram_free >= self.config.depth_model_vram:
                if gpu.vram_free > best_free:
                    best_free = gpu.vram_free
                    best = gpu.gpu_id
        return best

    def _sample_metrics(self, t: float):
        frag = compute_fragmentation(self.gpu_states, self.config)
        self.metrics.fragmentation_samples.append(frag)
        total_used = sum(g.vram_used for g in self.gpu_states)
        total_cap = sum(g.vram_total for g in self.gpu_states)
        self.metrics.gpu_utilization_samples.append((t, total_used / total_cap))

    def schedule(self, all_demands: Dict[int, List[GPUDemand]]) -> SchedulerMetrics:
        raise NotImplementedError


class FragAwareScheduler(BaseScheduler):
    """
    碎片感知调度器（本文方法）

    核心策略：
    1. 需求到来时：分配空闲显存最多的GPU（降低碎片）
    2. 需求结束时：查看同一机器人的下一个需求
       - 间隔短(< release_threshold)且GPU不紧张 → 保持（省冷启动）
       - 间隔长 或 GPU紧张 → 立即释放
    3. 利用已知的未来需求时刻表做预加载
    """

    def __init__(self, config: GPUConfig, release_threshold: float = 8.0):
        super().__init__(config)
        self.release_threshold = release_threshold

    def schedule(self, all_demands: Dict[int, List[GPUDemand]]) -> SchedulerMetrics:
        # 构建每台机器人的需求段列表（仅demand类型）
        robot_demand_segs: Dict[int, List[GPUDemand]] = {}
        for robot_id, segments in all_demands.items():
            robot_demand_segs[robot_id] = [s for s in segments if s.demand_type == "demand"]

        # 构建下一个需求的查找表
        next_demand: Dict[Tuple[int, int], Optional[GPUDemand]] = {}
        for robot_id, segs in robot_demand_segs.items():
            for idx, seg in enumerate(segs):
                if idx + 1 < len(segs):
                    next_demand[(robot_id, idx)] = segs[idx + 1]
                else:
                    next_demand[(robot_id, idx)] = None

        # 为每个需求段编号
        demand_index: Dict[int, int] = {}  # robot_id -> current demand index
        for rid in robot_demand_segs:
            demand_index[rid] = 0

        # 收集所有需求事件
        events = []
        for robot_id, segs in robot_demand_segs.items():
            for idx, seg in enumerate(segs):
                events.append((seg.start, 0, "start", robot_id, seg, idx))
                events.append((seg.end, 1, "end", robot_id, seg, idx))
        events.sort(key=lambda e: (e[0], e[1]))

        for t, _, event_type, robot_id, seg, seg_idx in events:
            if event_type == "start":
                self.metrics.total_demands += 1

                if robot_id in self.robot_gpu_map:
                    # 已有分配（保持策略生效）→ 无冷启动
                    self.metrics.total_gpu_time += seg.duration
                else:
                    # 需要新分配
                    gpu_id = self._find_best_gpu()
                    if gpu_id is not None:
                        self._allocate(robot_id, gpu_id, t, seg.end)
                        self.metrics.total_gpu_time += seg.duration
                        self.metrics.total_cold_starts += 1
                    else:
                        # GPU全满 → 回收一个idle状态的分配
                        reclaimed = self._reclaim_idle(t, all_demands, robot_demand_segs)
                        if reclaimed is not None:
                            self._allocate(robot_id, reclaimed, t, seg.end)
                            self.metrics.total_gpu_time += seg.duration
                            self.metrics.total_cold_starts += 1
                        else:
                            # 真的没有可用空间
                            self.metrics.total_wait_delay += self.config.cold_start_time

                self._sample_metrics(t)

            elif event_type == "end":
                if robot_id not in self.robot_gpu_map:
                    continue

                gpu_id = self.robot_gpu_map[robot_id]
                nxt = next_demand.get((robot_id, seg_idx))

                # 决策：保持还是释放
                should_keep = False
                if nxt is not None:
                    gap = nxt.start - seg.end
                    gpu_pressure = self._gpu_pressure(t)
                    if gap <= self.release_threshold and gpu_pressure < 0.8:
                        should_keep = True

                if not should_keep:
                    self._release(robot_id, gpu_id)

                self._sample_metrics(t)

        return self.metrics

    def _gpu_pressure(self, t: float) -> float:
        """当前GPU压力：已用槽位/总槽位"""
        used = sum(len(g.active_robots) for g in self.gpu_states)
        total = self.config.total_concurrent
        return used / total if total > 0 else 1.0

    def _reclaim_idle(self, t: float, all_demands: Dict[int, List[GPUDemand]],
                      robot_demand_segs: Dict[int, List[GPUDemand]]) -> Optional[int]:
        """回收当前不在demand阶段的机器人的GPU"""
        for gpu in self.gpu_states:
            for rid in list(gpu.active_robots.keys()):
                # 检查该机器人当前是否在demand阶段
                is_active = False
                for seg in robot_demand_segs.get(rid, []):
                    if seg.start <= t < seg.end:
                        is_active = True
                        break
                if not is_active:
                    self._release(rid, gpu.gpu_id)
                    self.metrics.total_migrations += 1
                    return gpu.gpu_id
        return None


class OnDemandScheduler(BaseScheduler):
    """B5: 按需分配，用完释放，不迁移"""

    def schedule(self, all_demands: Dict[int, List[GPUDemand]]) -> SchedulerMetrics:
        events = []
        for robot_id, segments in all_demands.items():
            for seg in segments:
                if seg.demand_type == "demand":
                    events.append((seg.start, "start", robot_id, seg))
                    events.append((seg.end, "end", robot_id, seg))

        events.sort(key=lambda e: (e[0], 0 if e[1] == "end" else 1))

        for t, event_type, robot_id, seg in events:
            if event_type == "start":
                self.metrics.total_demands += 1
                gpu_id = self._find_best_gpu()
                if gpu_id is not None:
                    self._allocate(robot_id, gpu_id, t, seg.end)
                    self.metrics.total_gpu_time += seg.duration
                    self.metrics.total_cold_starts += 1
                else:
                    self.metrics.total_wait_delay += self.config.cold_start_time
                self._sample_metrics(t)

            elif event_type == "end":
                if robot_id in self.robot_gpu_map:
                    gpu_id = self.robot_gpu_map[robot_id]
                    self._release(robot_id, gpu_id)
                self._sample_metrics(t)

        return self.metrics


class FixedScheduler(BaseScheduler):
    """B6: 每台机器人固定绑定一张GPU"""

    def schedule(self, all_demands: Dict[int, List[GPUDemand]]) -> SchedulerMetrics:
        robot_ids = sorted(all_demands.keys())
        max_slots = self.config.total_gpus * self.config.max_concurrent_per_gpu

        # 预分配
        for idx, robot_id in enumerate(robot_ids):
            gpu_id = idx % self.config.total_gpus
            if self.gpu_states[gpu_id].vram_free >= self.config.depth_model_vram:
                self._allocate(robot_id, gpu_id, 0, float("inf"))
            else:
                self.metrics.total_wait_delay += 1e6  # 无法分配

        # 统计
        for robot_id, segments in all_demands.items():
            for seg in segments:
                if seg.demand_type == "demand":
                    self.metrics.total_demands += 1
                    if robot_id in self.robot_gpu_map:
                        self.metrics.total_gpu_time += seg.duration
                    else:
                        self.metrics.total_wait_delay += seg.duration

        self._sample_metrics(0)
        return self.metrics


class PredictiveScheduler(BaseScheduler):
    """
    预测性GPU调度器：利用已知的24小时任务周期性提前预测GPU需求。
    预加载（高峰前加载模型）+ 智能保持（短间隔+高未来密度）+ 预释放（低谷前释放）
    """

    def __init__(self, config: GPUConfig,
                 preload_lead: float = 10.0,
                 release_threshold: float = 30.0,
                 prediction_horizon: float = 60.0):
        super().__init__(config)
        self.preload_lead = preload_lead
        self.release_threshold = release_threshold
        self.prediction_horizon = prediction_horizon

    def schedule(self, all_demands: Dict[int, List[GPUDemand]]) -> SchedulerMetrics:
        all_events = []
        robot_demand_segs: Dict[int, List[GPUDemand]] = {}
        for rid, segs in all_demands.items():
            d_segs = [s for s in segs if s.demand_type == "demand"]
            robot_demand_segs[rid] = d_segs
            for idx, seg in enumerate(d_segs):
                all_events.append((seg.start, 0, "start", rid, seg, idx))
                all_events.append((seg.end, 1, "end", rid, seg, idx))
                preload_time = max(0, seg.start - self.preload_lead)
                all_events.append((preload_time, -1, "preload", rid, seg, idx))
        all_events.sort(key=lambda e: (e[0], e[1]))

        next_demand = {}
        for rid, segs in robot_demand_segs.items():
            for idx in range(len(segs)):
                next_demand[(rid, idx)] = segs[idx + 1] if idx + 1 < len(segs) else None

        for t, _, event_type, rid, seg, seg_idx in all_events:
            if event_type == "preload":
                if rid in self.robot_gpu_map:
                    continue
                gpu_id = self._find_best_gpu()
                if gpu_id is not None:
                    self._allocate(rid, gpu_id, t, seg.end, "preloading")
                self._sample_metrics(t)

            elif event_type == "start":
                self.metrics.total_demands += 1
                if rid in self.robot_gpu_map:
                    self.metrics.total_gpu_time += seg.duration
                else:
                    gpu_id = self._find_best_gpu()
                    if gpu_id is not None:
                        self._allocate(rid, gpu_id, t, seg.end)
                        self.metrics.total_gpu_time += seg.duration
                        self.metrics.total_cold_starts += 1
                    else:
                        # 回收空闲
                        for gpu in self.gpu_states:
                            for r in list(gpu.active_robots.keys()):
                                active = any(s.start <= t < s.end
                                             for s in robot_demand_segs.get(r, []))
                                if not active:
                                    self._release(r, gpu.gpu_id)
                                    self.metrics.total_migrations += 1
                                    self._allocate(rid, gpu.gpu_id, t, seg.end)
                                    self.metrics.total_gpu_time += seg.duration
                                    self.metrics.total_cold_starts += 1
                                    break
                            else:
                                continue
                            break
                        else:
                            self.metrics.total_wait_delay += self.config.cold_start_time
                            self.metrics.total_gpu_time += seg.duration  # 需求存在，只是延迟
                self._sample_metrics(t)

            elif event_type == "end":
                if rid not in self.robot_gpu_map:
                    continue
                gpu_id = self.robot_gpu_map[rid]
                nxt = next_demand.get((rid, seg_idx))
                should_keep = False
                if nxt is not None:
                    gap = nxt.start - seg.end
                    pressure = sum(len(g.active_robots) for g in self.gpu_states) / max(self.config.total_concurrent, 1)
                    if gap <= self.release_threshold and pressure < 0.8:
                        should_keep = True
                    future = sum(1 for s in robot_demand_segs.get(rid, [])
                                 if t < s.start < t + self.prediction_horizon)
                    if future >= 2 and pressure < 0.7:
                        should_keep = True
                if not should_keep:
                    self._release(rid, gpu_id)
                self._sample_metrics(t)

        return self.metrics


class RoundRobinScheduler(BaseScheduler):
    """B7: 轮询分配"""

    def schedule(self, all_demands: Dict[int, List[GPUDemand]]) -> SchedulerMetrics:
        events = []
        for robot_id, segments in all_demands.items():
            for seg in segments:
                if seg.demand_type == "demand":
                    events.append((seg.start, "start", robot_id, seg))
                    events.append((seg.end, "end", robot_id, seg))

        events.sort(key=lambda e: (e[0], 0 if e[1] == "end" else 1))
        rr_counter = 0

        for t, event_type, robot_id, seg in events:
            if event_type == "start":
                self.metrics.total_demands += 1
                # 从rr_counter开始找可用GPU
                allocated = False
                for offset in range(self.config.total_gpus):
                    gpu_id = (rr_counter + offset) % self.config.total_gpus
                    if self.gpu_states[gpu_id].vram_free >= self.config.depth_model_vram:
                        self._allocate(robot_id, gpu_id, t, seg.end)
                        self.metrics.total_gpu_time += seg.duration
                        self.metrics.total_cold_starts += 1
                        rr_counter = (gpu_id + 1) % self.config.total_gpus
                        allocated = True
                        break
                if not allocated:
                    self.metrics.total_wait_delay += self.config.cold_start_time
                self._sample_metrics(t)

            elif event_type == "end":
                if robot_id in self.robot_gpu_map:
                    gpu_id = self.robot_gpu_map[robot_id]
                    self._release(robot_id, gpu_id)
                self._sample_metrics(t)

        return self.metrics
