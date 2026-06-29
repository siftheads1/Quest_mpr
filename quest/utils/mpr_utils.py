"""MPR helpers: CrossLayerCPUBackupStore and MPRBlockRegistry.

CrossLayerCPUBackupStore wraps SemanticCPUBackupStore to handle the cross-layer
KV blocks that Quest's shared KvPool produces.  Quest's KvPool stores all layers
in a single buffer: [num_layers, capacity, 2, block_len, num_kv_heads, head_dim].
One pool slot index therefore covers the same block position across every layer.
Eviction and recall always transfer all layers simultaneously.

MPRBlockRegistry tracks which logical (absolute) blocks are GPU-resident vs.
CPU-backed and manages the mapping to physical GPU pool slots.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Sequence

import torch

from quest.utils.mpr import (
    SemanticCPUBackupStore,
    CPUBackupKey,
    CPUBackupStats,
    FP16BackupCodec,
    INT8BackupCodec,
    INT4BackupCodec,
    PrecisionTier,
)
from quest.utils.mpr.backup_codec import (
    FP16_BACKUP_FORMAT,
    INT8_BACKUP_FORMAT,
    INT4_BACKUP_FORMAT,
)


_CROSS_LAYER_KEY = "cross_layer"


class PageResidency(enum.Enum):
    GPU_RESIDENT = "gpu_resident"
    CPU_BACKED_UP = "cpu_backed_up"


@dataclass
class PageRecord:
    """State for one logical block in the growing sequence."""
    gpu_slot: int | None       # current physical pool slot; None when evicted
    residency: PageResidency
    last_score: float = 0.0    # updated each decode step; used for eviction victim selection


class CrossLayerCPUBackupStore:
    """CPU backup store for cross-layer KV blocks.

    Quest's KvPool shape: [num_layers, capacity, 2, block_len, num_kv_heads, head_dim].
    One `abs_block_id` maps to one pool slot shared across all layers, so backup
    and recall operate on the full [num_layers, 2, block_len, nkv, d] slice.
    """

    def __init__(self) -> None:
        self._store = SemanticCPUBackupStore()
        self._fp16_codec = FP16BackupCodec()
        self._int8_codec = INT8BackupCodec()
        self._int4_codec = INT4BackupCodec()

    def put(self, abs_block_id: int, kv_all_layers: torch.Tensor) -> None:
        """Back up a cross-layer KV block to CPU at FP16 + INT8 + INT4.

        Args:
            abs_block_id: Logical block id (sequence_offset // page_size).
            kv_all_layers: [num_layers, 2, block_len, num_kv_heads, head_dim] GPU tensor.
        """
        # SemanticCPUBackupStore expects [2, block_len, num_kv_heads, head_dim] per call.
        # We concatenate all layers along dim=0: [num_layers*2, block_len, nkv, d]
        # then treat it as a 4-D block by temporarily viewing the first dim as 2.
        # Simpler: just store the whole thing reshaped as [2, num_layers*block_len, nkv, d].
        nl, two, bl, nkv, d = kv_all_layers.shape
        assert two == 2
        # Reshape to [2, nl*bl, nkv, d] so the codec sees a valid [2, *, nkv, d] block.
        block_cpu = kv_all_layers.permute(1, 0, 2, 3, 4).reshape(2, nl * bl, nkv, d)
        key = CPUBackupKey(layer_name=_CROSS_LAYER_KEY, physical_block_id=abs_block_id)
        self._store.put(
            layer_name=_CROSS_LAYER_KEY,
            physical_block_id=abs_block_id,
            kv_block=block_cpu,
            backup_storage_mode="eager_fp16_int8_int4",
        )

    def recall(
        self,
        abs_block_id: int,
        precision: PrecisionTier,
        num_layers: int,
        block_len: int,
        num_kv_heads: int,
        head_dim: int,
        target_dtype: torch.dtype,
        target_device: torch.device,
    ) -> torch.Tensor:
        """Recall a cross-layer KV block from CPU at the requested precision.

        Returns:
            [num_layers, 2, block_len, num_kv_heads, head_dim] tensor on target_device.
        """
        key = CPUBackupKey(layer_name=_CROSS_LAYER_KEY, physical_block_id=abs_block_id)
        if precision is PrecisionTier.FP16:
            payload = self._store.get_payload(key, FP16_BACKUP_FORMAT)
            materialized = self._fp16_codec.materialize(
                payload, target_dtype=target_dtype, target_device=target_device
            )
        elif precision is PrecisionTier.INT8:
            payload = self._store.get_payload(key, INT8_BACKUP_FORMAT)
            materialized = self._int8_codec.materialize(
                payload, target_dtype=target_dtype, target_device=target_device
            )
        elif precision is PrecisionTier.INT4:
            payload = self._store.get_payload(key, INT4_BACKUP_FORMAT)
            materialized = self._int4_codec.materialize(
                payload, target_dtype=target_dtype, target_device=target_device
            )
        else:
            raise ValueError(f"Cannot recall with precision {precision}")

        # Undo the [2, nl*bl, nkv, d] → [nl, 2, bl, nkv, d] reshape
        result = materialized.reshape(2, num_layers, block_len, num_kv_heads, head_dim)
        return result.permute(1, 0, 2, 3, 4).contiguous()

    def release(self, abs_block_id: int) -> None:
        self._store.release_blocks({abs_block_id})

    def clear(self) -> None:
        self._store.clear()

    def stats(self) -> CPUBackupStats:
        return self._store.stats()


class MPRBlockRegistry:
    """Tracks residency and physical slot mapping for logical KV blocks.

    Logical block ids (abs_block_id) are monotonically assigned: the k-th full
    page in the growing sequence gets abs_block_id = k.  Physical GPU slot
    indices are recycled from a free set that mirrors KvPool._free.

    This class is the Python-side complement to Quest's KvPool.  It does NOT
    directly manipulate KvPool._free; instead it keeps its own `_free_slots`
    set in sync and lets the caller write into pool_buf.
    """

    def __init__(self, gpu_capacity: int) -> None:
        self.gpu_capacity = gpu_capacity
        self._records: dict[int, PageRecord] = {}
        self._free_slots: set[int] = set(range(gpu_capacity))
        self._next_abs_id: int = 0

    # ------------------------------------------------------------------ #
    #  Allocation                                                           #
    # ------------------------------------------------------------------ #

    @property
    def num_free_slots(self) -> int:
        return len(self._free_slots)

    def alloc_slot(self) -> int:
        """Pop and return a free GPU slot (caller must check / pre-evict)."""
        if not self._free_slots:
            raise RuntimeError("No free GPU slots; call pre_evict() first.")
        return self._free_slots.pop()

    def register_new_block(self, gpu_slot: int) -> int:
        """Register a newly allocated KV block and return its abs_block_id."""
        abs_id = self._next_abs_id
        self._next_abs_id += 1
        self._records[abs_id] = PageRecord(
            gpu_slot=gpu_slot,
            residency=PageResidency.GPU_RESIDENT,
        )
        return abs_id

    # ------------------------------------------------------------------ #
    #  Eviction                                                             #
    # ------------------------------------------------------------------ #

    def evict_to_cpu(
        self,
        abs_block_id: int,
        pool_buf: torch.Tensor,
        backup_store: CrossLayerCPUBackupStore,
    ) -> int:
        """Evict a GPU-resident block to CPU and return its freed slot."""
        rec = self._records[abs_block_id]
        assert rec.residency is PageResidency.GPU_RESIDENT
        slot = rec.gpu_slot
        # pool_buf: [num_layers, capacity, 2, block_len, nkv, d]
        kv_all_layers = pool_buf[:, slot, :, :, :, :].clone()
        backup_store.put(abs_block_id, kv_all_layers)
        rec.residency = PageResidency.CPU_BACKED_UP
        rec.gpu_slot = None
        self._free_slots.add(slot)
        return slot

    def evict_victim(
        self,
        protect_ids: set[int],
        pool_buf: torch.Tensor,
        backup_store: CrossLayerCPUBackupStore,
    ) -> int:
        """Evict the lowest-score GPU-resident block (not in protect_ids)."""
        candidates = [
            (abs_id, rec)
            for abs_id, rec in self._records.items()
            if rec.residency is PageResidency.GPU_RESIDENT and abs_id not in protect_ids
        ]
        if not candidates:
            raise RuntimeError(
                "No evictable GPU-resident block found. "
                "Increase gpu_capacity or reduce sequence length."
            )
        victim_id = min(candidates, key=lambda x: x[1].last_score)[0]
        return self.evict_to_cpu(victim_id, pool_buf, backup_store)

    # ------------------------------------------------------------------ #
    #  Recall                                                               #
    # ------------------------------------------------------------------ #

    def recall_from_cpu(
        self,
        abs_block_id: int,
        precision: PrecisionTier,
        pool_buf: torch.Tensor,
        backup_store: CrossLayerCPUBackupStore,
    ) -> int:
        """Recall a CPU-backed block to a free GPU slot.

        Returns the GPU slot index it was placed in.
        """
        rec = self._records[abs_block_id]
        assert rec.residency is PageResidency.CPU_BACKED_UP
        slot = self.alloc_slot()
        nl, cap, two, bl, nkv, d = pool_buf.shape
        recalled = backup_store.recall(
            abs_block_id=abs_block_id,
            precision=precision,
            num_layers=nl,
            block_len=bl,
            num_kv_heads=nkv,
            head_dim=d,
            target_dtype=pool_buf.dtype,
            target_device=pool_buf.device,
        )
        pool_buf[:, slot, :, :, :, :].copy_(recalled)
        rec.gpu_slot = slot
        rec.residency = PageResidency.GPU_RESIDENT
        return slot

    # ------------------------------------------------------------------ #
    #  Queries                                                              #
    # ------------------------------------------------------------------ #

    def is_gpu_resident(self, abs_block_id: int) -> bool:
        return self._records[abs_block_id].residency is PageResidency.GPU_RESIDENT

    def get_slot(self, abs_block_id: int) -> int:
        return self._records[abs_block_id].gpu_slot

    def update_score(self, abs_block_id: int, score: float) -> None:
        self._records[abs_block_id].last_score = score

    def finalized_abs_ids(self) -> list[int]:
        """Return all abs_block_ids that have been fully written (all except current)."""
        return list(self._records.keys())

    def resident_slots_ordered(self, abs_ids: Sequence[int]) -> list[int]:
        """Return GPU slot indices for the given abs_ids that are GPU-resident."""
        return [
            self._records[aid].gpu_slot
            for aid in abs_ids
            if self._records[aid].residency is PageResidency.GPU_RESIDENT
        ]

    def reset(self) -> None:
        self._records.clear()
        self._free_slots = set(range(self.gpu_capacity))
        self._next_abs_id = 0
