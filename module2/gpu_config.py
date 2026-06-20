"""模块2：GPU集群配置与数据结构"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import json


@dataclass
class GPUConfig:
    num_servers: int = 2
    gpus_per_server: int = 4
    vram_per_gpu: float = 24.0       # GB
    depth_model_vram: float = 4.0    # GB per inference instance
    cold_start_time: float = 3.0     # seconds to load model
    migrate_time: float = 1.0        # seconds for live migration
    network_latency: float = 0.02    # seconds (20ms)

    @property
    def total_gpus(self) -> int:
        return self.num_servers * self.gpus_per_server

    @property
    def max_concurrent_per_gpu(self) -> int:
        return int(self.vram_per_gpu // self.depth_model_vram)

    @property
    def total_concurrent(self) -> int:
        return self.total_gpus * self.max_concurrent_per_gpu


GPU_CONFIGS = {
    "tight":  GPUConfig(num_servers=1, gpus_per_server=2, vram_per_gpu=16.0, depth_model_vram=6.0),  # 2 GPU, 每GPU 2并发=4总
    "small":  GPUConfig(num_servers=1, gpus_per_server=4),   # 4 GPU, 24并发
    "medium": GPUConfig(num_servers=2, gpus_per_server=4),   # 8 GPU, 48并发
    "large":  GPUConfig(num_servers=2, gpus_per_server=8),   # 16 GPU, 96并发
    "xlarge": GPUConfig(num_servers=4, gpus_per_server=8),   # 32 GPU, 192并发
}


@dataclass
class GPUDemand:
    """单个GPU需求段"""
    robot_id: int
    start: float
    end: float
    demand_type: str  # "demand" or "idle"
    reason: str = ""  # for idle: "RIDING", "WALKING", etc.
    elevator_id: int = -1
    floor: int = -1

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class GPUAllocation:
    """GPU分配决策"""
    robot_id: int
    gpu_id: int
    start: float
    end: float
    alloc_type: str = "active"  # "active", "preloading", "migrating"


@dataclass
class GPUState:
    """单张GPU的实时状态"""
    gpu_id: int
    server_id: int
    vram_total: float
    vram_used: float = 0.0
    active_robots: Dict[int, float] = field(default_factory=dict)  # robot_id -> alloc_start

    @property
    def vram_free(self) -> float:
        return self.vram_total - self.vram_used

    @property
    def num_active(self) -> int:
        return len(self.active_robots)

    @property
    def utilization(self) -> float:
        return self.vram_used / self.vram_total if self.vram_total > 0 else 0


def load_gpu_demands(filepath: str) -> Dict[int, List[GPUDemand]]:
    """从模块1输出的JSON文件加载GPU需求"""
    with open(filepath) as f:
        data = json.load(f)

    demands = {}
    for robot_id_str, segments in data.items():
        robot_id = int(robot_id_str)
        robot_demands = []
        for seg in segments:
            robot_demands.append(GPUDemand(
                robot_id=robot_id,
                start=seg["start"],
                end=seg["end"],
                demand_type=seg["type"],
                reason=seg.get("reason", ""),
                elevator_id=seg.get("elevator_id", -1),
                floor=seg.get("floor", -1),
            ))
        demands[robot_id] = robot_demands

    return demands
