"""GPU resource allocation for parallel container execution.

Supports three allocation strategies:
- Explicit: user passes --gpus with exact specs
- Multi-GPU grouping: fewer jobs than GPUs → group GPUs into fat slots
- MIG-aware: more jobs than GPUs → use existing MIG instances, keeping
  each slot's MIGs on the same physical GPU (cross-GPU MIG doesn't work)
"""

import subprocess
from dataclasses import dataclass
from queue import Queue
from typing import Any


@dataclass
class GpuSlot:
    """A GPU resource slot assignable to a container."""
    device_ids: list[str]  # ["0"], ["MIG-xxx"], or ["0", "1"] for multi-GPU
    is_mig: bool

    @property
    def device_id(self) -> str:
        """Primary device ID (for display purposes)."""
        return self.device_ids[0]

    def label(self) -> str:
        if len(self.device_ids) == 1:
            return self.device_ids[0]
        return "+".join(self.device_ids)


class GpuAllocator:
    """Thread-safe GPU slot pool with acquire/release semantics."""

    def __init__(self, slots: list[GpuSlot]):
        self._slots: list[GpuSlot] = list(slots)
        self._queue: Queue[GpuSlot] = Queue()
        for slot in self._slots:
            self._queue.put(slot)

    @classmethod
    def from_specs(cls, gpu_specs: list[str]) -> "GpuAllocator":
        """Build an allocator from user-provided spec strings.

        Supports "+" syntax for multi-GPU slots: "0+1" means GPUs 0 and 1.
        """
        slots = []
        for spec in gpu_specs:
            device_ids = spec.split("+")
            is_mig = any(d.startswith("MIG-") for d in device_ids)
            slots.append(GpuSlot(device_ids=device_ids, is_mig=is_mig))
        return cls(slots)

    @property
    def num_slots(self) -> int:
        return len(self._slots)

    def acquire(self) -> GpuSlot:
        """Block until a GPU slot is available, then return it."""
        return self._queue.get()

    def release(self, slot: GpuSlot) -> None:
        """Return a GPU slot to the pool."""
        self._queue.put(slot)

    def __repr__(self) -> str:
        labels = [s.label() for s in self._slots]
        return f"GpuAllocator(slots={labels})"


@dataclass
class GpuInfo:
    """Structured GPU detection result."""
    gpu_indices: list[str]                    # ["0", "1", "2", "3"]
    mig_by_gpu: dict[str, list[str]]          # {"0": ["MIG-aaa", "MIG-bbb"], "1": [...]}
    non_mig_gpus: list[str]                   # GPUs without MIG enabled

    @property
    def total_gpus(self) -> int:
        return len(self.gpu_indices)

    @property
    def total_migs(self) -> int:
        return sum(len(v) for v in self.mig_by_gpu.values())

    @property
    def has_mig(self) -> bool:
        return self.total_migs > 0


def detect_gpu_info() -> GpuInfo:
    """Detect available GPUs and MIG instances via nvidia-smi.

    Returns structured GpuInfo with per-GPU MIG mapping so that allocation
    can keep MIG instances on the same physical GPU together.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True, text=True, timeout=10
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return GpuInfo(gpu_indices=[], mig_by_gpu={}, non_mig_gpus=[])

    if result.returncode != 0:
        return GpuInfo(gpu_indices=[], mig_by_gpu={}, non_mig_gpus=[])

    gpu_indices: list[str] = []
    mig_by_gpu: dict[str, list[str]] = {}
    current_gpu: str | None = None

    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("GPU "):
            # "GPU 0: NVIDIA A100 (UUID: GPU-xxxx)"
            idx = stripped.split(":")[0].replace("GPU ", "").strip()
            gpu_indices.append(idx)
            current_gpu = idx
        elif stripped.startswith("MIG ") and "UUID: " in stripped:
            # "  MIG 1g.10gb  Device  0: (UUID: MIG-xxxx)"
            uuid = stripped.split("UUID: ")[1].rstrip(")")
            if current_gpu is not None:
                mig_by_gpu.setdefault(current_gpu, []).append(uuid)

    non_mig_gpus = [idx for idx in gpu_indices if idx not in mig_by_gpu]
    return GpuInfo(gpu_indices=gpu_indices, mig_by_gpu=mig_by_gpu,
                   non_mig_gpus=non_mig_gpus)


def auto_allocate_slots(info: GpuInfo, num_jobs: int) -> list[GpuSlot]:
    """Intelligently allocate GPU slots based on available hardware and job count.

    Strategy:
    1. No MIG anywhere, jobs <= GPUs: group whole GPUs into multi-GPU slots
       e.g. 4 GPUs, 2 jobs → slots ["0+1", "2+3"]
    2. No MIG anywhere, jobs > GPUs: one GPU per slot, excess jobs queue
    3. MIG exists, jobs <= non-MIG GPUs: use only whole GPUs
    4. MIG exists, jobs > non-MIG GPUs: combine whole-GPU slots for non-MIG
       GPUs + MIG-based slots for MIG GPUs. MIG instances within each slot
       always come from the SAME physical GPU (cross-GPU MIG doesn't work).

    Examples:
        4 GPUs (no MIG), 2 jobs → [GPU 0+1, GPU 2+3]
        4 GPUs (no MIG), 8 jobs → [GPU 0, GPU 1, GPU 2, GPU 3] (4 queue)
        2 whole + 2 MIG GPUs (2 MIGs each), 6 jobs → [GPU 0, GPU 1, MIG-2a, MIG-2b, MIG-3a, MIG-3b]
        4 MIG GPUs (2 MIGs each), 4 jobs → [MIG-0a+0b, MIG-1a+1b, MIG-2a+2b, MIG-3a+3b]
    """
    if not info.gpu_indices:
        return []

    # No MIG at all — straightforward whole-GPU allocation
    if not info.has_mig:
        return _allocate_whole_gpu_slots(info.gpu_indices, num_jobs)

    # MIG exists. If we have enough non-MIG GPUs, just use those.
    if info.non_mig_gpus and num_jobs <= len(info.non_mig_gpus):
        return _allocate_whole_gpu_slots(info.non_mig_gpus, num_jobs)

    # Need more slots than non-MIG GPUs provide. Combine:
    # - 1 slot per non-MIG GPU (whole GPU)
    # - Remaining slots come from MIG instances
    slots: list[GpuSlot] = []
    for gpu_idx in info.non_mig_gpus:
        slots.append(GpuSlot(device_ids=[gpu_idx], is_mig=False))

    remaining_jobs = num_jobs - len(slots)
    if remaining_jobs > 0:
        mig_slots = _allocate_mig_slots(info, remaining_jobs)
        slots.extend(mig_slots)

    return slots


def _allocate_whole_gpu_slots(gpu_indices: list[str],
                               num_jobs: int) -> list[GpuSlot]:
    """Group whole GPUs into slots, distributing evenly across jobs.

    Examples:
        4 GPUs, 2 jobs → [["0","1"], ["2","3"]]
        4 GPUs, 4 jobs → [["0"], ["1"], ["2"], ["3"]]
        4 GPUs, 6 jobs → [["0"], ["1"], ["2"], ["3"]]  (4 slots, 2 jobs queue)
        3 GPUs, 2 jobs → [["0","1"], ["2"]]  (best-effort even split)
    """
    num_gpus = len(gpu_indices)
    num_slots = min(num_jobs, num_gpus)

    slots: list[GpuSlot] = []
    # Distribute GPUs as evenly as possible across slots
    base_size = num_gpus // num_slots
    remainder = num_gpus % num_slots
    idx = 0
    for i in range(num_slots):
        size = base_size + (1 if i < remainder else 0)
        device_ids = gpu_indices[idx:idx + size]
        slots.append(GpuSlot(device_ids=device_ids, is_mig=False))
        idx += size

    return slots


def _allocate_mig_slots(info: GpuInfo, num_jobs: int) -> list[GpuSlot]:
    """Allocate MIG-based slots, keeping each slot's MIGs on the same GPU.

    Never mixes MIG instances from different physical GPUs in one slot,
    since cross-GPU MIG doesn't work.

    Examples (4 GPUs, 2 MIGs each = 8 MIGs):
        8 jobs → 8 slots, 1 MIG each
        4 jobs → 4 slots, 2 MIGs each (paired by GPU)
        2 jobs → 2 slots, 4 MIGs each (all MIGs from 2 GPUs per slot)
    """
    # Collect MIG lists per GPU, sorted by GPU index
    gpu_mig_pairs: list[tuple[str, list[str]]] = [
        (gpu_idx, info.mig_by_gpu[gpu_idx])
        for gpu_idx in info.gpu_indices
        if gpu_idx in info.mig_by_gpu
    ]

    total_migs = sum(len(migs) for _, migs in gpu_mig_pairs)
    num_mig_gpus = len(gpu_mig_pairs)

    if total_migs == 0:
        return []

    # How many MIGs per slot? We want at most num_jobs slots.
    # Each slot's MIGs must come from the same GPU.
    if num_jobs >= total_migs:
        # One MIG per slot (most granular)
        slots: list[GpuSlot] = []
        for _, migs in gpu_mig_pairs:
            for mig_uuid in migs:
                slots.append(GpuSlot(device_ids=[mig_uuid], is_mig=True))
        return slots

    # Need to group MIGs. Figure out how many MIGs per slot.
    # Constraint: all MIGs in a slot must be from the same GPU.
    # Strategy: divide each GPU's MIGs into chunks, round-robin across GPUs.
    #
    # Target: num_jobs slots total, spread across num_mig_gpus GPUs.
    # slots_per_gpu[i] = how many slots GPU i contributes
    slots_per_gpu = _distribute_evenly(num_jobs, num_mig_gpus)

    slots = []
    for (gpu_idx, migs), n_slots in zip(gpu_mig_pairs, slots_per_gpu):
        if n_slots == 0:
            # Assign all this GPU's MIGs to one slot
            slots.append(GpuSlot(device_ids=list(migs), is_mig=True))
            continue

        # Split this GPU's MIGs into n_slots groups
        chunk_sizes = _distribute_evenly(len(migs), n_slots)
        idx = 0
        for size in chunk_sizes:
            chunk = migs[idx:idx + size]
            if chunk:
                slots.append(GpuSlot(device_ids=chunk, is_mig=True))
            idx += size

    return slots


def _distribute_evenly(total: int, buckets: int) -> list[int]:
    """Distribute `total` items into `buckets` as evenly as possible.

    Returns a list of length `buckets` where each element is the count.
    """
    if buckets <= 0:
        return []
    base = total // buckets
    remainder = total % buckets
    return [base + (1 if i < remainder else 0) for i in range(buckets)]


def allocate_shared_slots(info: GpuInfo, num_jobs: int) -> list[GpuSlot]:
    """Allocate slots by round-robining jobs across GPUs.

    Multiple containers may share the same GPU — each container only sees
    one GPU, but there's no hardware isolation between containers on the
    same GPU. Use this on GPUs that don't support MIG.

    Examples (4 GPUs):
        4 jobs → [GPU 0, GPU 1, GPU 2, GPU 3]
        8 jobs → [GPU 0, GPU 1, GPU 2, GPU 3, GPU 0, GPU 1, GPU 2, GPU 3]
        6 jobs → [GPU 0, GPU 1, GPU 2, GPU 3, GPU 0, GPU 1]
    """
    available = info.non_mig_gpus if info.has_mig else info.gpu_indices
    if not available:
        return []

    slots: list[GpuSlot] = []
    for i in range(num_jobs):
        gpu_idx = available[i % len(available)]
        slots.append(GpuSlot(device_ids=[gpu_idx], is_mig=False))
    return slots


def parse_gpu_specs(specs_str: str) -> list[str]:
    """Parse comma-separated GPU specs string into a list.

    Examples:
        "0,1,2,3" -> ["0", "1", "2", "3"]
        "MIG-xxx,MIG-yyy" -> ["MIG-xxx", "MIG-yyy"]
        "0+1,2+3" -> ["0+1", "2+3"]  (multi-GPU slots)
    """
    return [s.strip() for s in specs_str.split(",") if s.strip()]


def build_device_request(slot: GpuSlot) -> dict[str, Any]:
    """Build Docker device_requests entry for a GPU slot.

    Returns kwargs suitable for docker.types.DeviceRequest().
    """
    return {
        "device_ids": slot.device_ids,
        "capabilities": [["gpu"]],
    }
