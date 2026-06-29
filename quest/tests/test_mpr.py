"""Unit tests for MPR components (no CUDA kernels required).

Run with:
    pytest quest/tests/test_mpr.py -v
"""

import pytest
import torch

from quest.utils.mpr import (
    PrecisionTier,
    ThresholdPrecisionPolicy,
    TopRatioPrecisionPolicy,
    SemanticCPUBackupStore,
    CPUBackupKey,
)
from quest.utils.mpr_utils import (
    CrossLayerCPUBackupStore,
    MPRBlockRegistry,
    PageResidency,
)


# ------------------------------------------------------------------ #
#  CrossLayerCPUBackupStore                                           #
# ------------------------------------------------------------------ #

def _make_kv(num_layers=2, block_len=4, nkv=2, d=8, dtype=torch.float32):
    return torch.randn(num_layers, 2, block_len, nkv, d, dtype=dtype)


class TestCrossLayerCPUBackupStore:
    def setup_method(self):
        self.store = CrossLayerCPUBackupStore()

    def _recall(self, abs_id, precision, ref):
        nl, two, bl, nkv, d = ref.shape
        return self.store.recall(
            abs_id,
            precision=precision,
            num_layers=nl,
            block_len=bl,
            num_kv_heads=nkv,
            head_dim=d,
            target_dtype=ref.dtype,
            target_device=ref.device,
        )

    def test_fp16_recall_exact(self):
        # FP16 codec stores in FP16; use FP16 input to verify exact round-trip.
        kv = _make_kv(dtype=torch.float16)
        self.store.put(0, kv)
        recalled = self._recall(0, PrecisionTier.FP16, kv)
        assert recalled.shape == kv.shape
        assert torch.allclose(recalled.float(), kv.float(), atol=0.0)

    def test_int8_recall_bounded_error(self):
        kv = _make_kv()
        self.store.put(0, kv)
        recalled = self._recall(0, PrecisionTier.INT8, kv)
        max_err = (recalled - kv).abs().max().item()
        assert max_err < 0.5, f"INT8 max error {max_err:.4f} exceeds bound"

    def test_int4_recall_bounded_error(self):
        kv = _make_kv()
        self.store.put(0, kv)
        recalled = self._recall(0, PrecisionTier.INT4, kv)
        max_err = (recalled - kv).abs().max().item()
        assert max_err < 2.0, f"INT4 max error {max_err:.4f} exceeds bound"

    def test_recall_shape_preserved(self):
        nl, bl, nkv, d = 4, 8, 4, 16
        kv = _make_kv(num_layers=nl, block_len=bl, nkv=nkv, d=d)
        self.store.put(1, kv)
        for precision in (PrecisionTier.FP16, PrecisionTier.INT8, PrecisionTier.INT4):
            recalled = self._recall(1, precision, kv)
            assert recalled.shape == kv.shape, f"{precision}: shape mismatch"

    def test_recall_target_dtype(self):
        kv = _make_kv().to(torch.float32)
        self.store.put(0, kv)
        recalled = self.store.recall(
            0, PrecisionTier.INT8, num_layers=2, block_len=4,
            num_kv_heads=2, head_dim=8,
            target_dtype=torch.float16, target_device=kv.device,
        )
        assert recalled.dtype == torch.float16

    def test_release_frees_memory(self):
        kv = _make_kv()
        self.store.put(0, kv)
        assert self.store.stats().block_count == 1
        self.store.release(0)
        assert self.store.stats().block_count == 0

    def test_stats_block_count(self):
        for i in range(3):
            self.store.put(i, _make_kv())
        assert self.store.stats().block_count == 3

    def test_zero_block_exact_recall(self):
        """All-zero KV block must survive encode/recall exactly (scale placeholder)."""
        kv = torch.zeros(2, 2, 4, 2, 8)
        self.store.put(0, kv)
        for precision in (PrecisionTier.FP16, PrecisionTier.INT8, PrecisionTier.INT4):
            recalled = self._recall(0, precision, kv)
            assert torch.allclose(recalled, kv, atol=0.0), f"{precision}: non-zero recalled from zero"


# ------------------------------------------------------------------ #
#  MPRBlockRegistry                                                    #
# ------------------------------------------------------------------ #

def _make_pool_buf(gpu_capacity=4, num_layers=2, block_len=4, nkv=2, d=8):
    return torch.randn(num_layers, gpu_capacity, 2, block_len, nkv, d)


class TestMPRBlockRegistry:
    def setup_method(self):
        self.gpu_cap = 4
        self.reg = MPRBlockRegistry(gpu_capacity=self.gpu_cap)

    def test_alloc_slot_returns_valid_index(self):
        slot = self.reg.alloc_slot()
        assert 0 <= slot < self.gpu_cap

    def test_register_new_block_increments_abs_id(self):
        for expected in range(3):
            slot = self.reg.alloc_slot()
            abs_id = self.reg.register_new_block(slot)
            assert abs_id == expected

    def test_is_gpu_resident_after_register(self):
        slot = self.reg.alloc_slot()
        abs_id = self.reg.register_new_block(slot)
        assert self.reg.is_gpu_resident(abs_id)

    def test_evict_changes_residency(self):
        pool = _make_pool_buf(self.gpu_cap)
        backup = CrossLayerCPUBackupStore()
        slot = self.reg.alloc_slot()
        abs_id = self.reg.register_new_block(slot)
        freed = self.reg.evict_to_cpu(abs_id, pool, backup)
        assert freed == slot
        assert not self.reg.is_gpu_resident(abs_id)
        assert self.reg._records[abs_id].residency is PageResidency.CPU_BACKED_UP

    def test_recall_restores_residency(self):
        pool = _make_pool_buf(self.gpu_cap)
        backup = CrossLayerCPUBackupStore()
        slot = self.reg.alloc_slot()
        abs_id = self.reg.register_new_block(slot)
        self.reg.evict_to_cpu(abs_id, pool, backup)
        new_slot = self.reg.recall_from_cpu(abs_id, PrecisionTier.FP16, pool, backup)
        assert self.reg.is_gpu_resident(abs_id)
        assert self.reg.get_slot(abs_id) == new_slot

    def test_free_slots_decrease_on_alloc(self):
        initial = self.reg.num_free_slots
        self.reg.alloc_slot()
        assert self.reg.num_free_slots == initial - 1

    def test_evict_victim_selects_lowest_score(self):
        pool = _make_pool_buf(self.gpu_cap)
        backup = CrossLayerCPUBackupStore()
        abs_ids = []
        for _ in range(3):
            slot = self.reg.alloc_slot()
            abs_id = self.reg.register_new_block(slot)
            abs_ids.append(abs_id)
        # Set scores: 0→10.0, 1→1.0 (lowest), 2→5.0
        self.reg.update_score(abs_ids[0], 10.0)
        self.reg.update_score(abs_ids[1], 1.0)
        self.reg.update_score(abs_ids[2], 5.0)
        self.reg.evict_victim(protect_ids=set(), pool_buf=pool, backup_store=backup)
        assert not self.reg.is_gpu_resident(abs_ids[1])  # lowest score evicted
        assert self.reg.is_gpu_resident(abs_ids[0])
        assert self.reg.is_gpu_resident(abs_ids[2])

    def test_evict_victim_respects_protect_ids(self):
        pool = _make_pool_buf(self.gpu_cap)
        backup = CrossLayerCPUBackupStore()
        abs_ids = []
        for _ in range(2):
            slot = self.reg.alloc_slot()
            abs_id = self.reg.register_new_block(slot)
            abs_ids.append(abs_id)
        self.reg.update_score(abs_ids[0], 0.0)  # would be victim
        self.reg.update_score(abs_ids[1], 9.0)
        # Protect abs_ids[0]; abs_ids[1] must be evicted instead.
        self.reg.evict_victim(protect_ids={abs_ids[0]}, pool_buf=pool, backup_store=backup)
        assert self.reg.is_gpu_resident(abs_ids[0])
        assert not self.reg.is_gpu_resident(abs_ids[1])

    def test_resident_slots_ordered_skips_cpu_blocks(self):
        pool = _make_pool_buf(self.gpu_cap)
        backup = CrossLayerCPUBackupStore()
        slots, abs_ids = [], []
        for _ in range(3):
            slot = self.reg.alloc_slot()
            abs_id = self.reg.register_new_block(slot)
            slots.append(slot)
            abs_ids.append(abs_id)
        # Evict middle block.
        self.reg.evict_to_cpu(abs_ids[1], pool, backup)
        result = self.reg.resident_slots_ordered(abs_ids)
        assert len(result) == 2
        assert slots[1] not in result

    def test_reset_clears_state(self):
        slot = self.reg.alloc_slot()
        self.reg.register_new_block(slot)
        self.reg.reset()
        assert self.reg._next_abs_id == 0
        assert len(self.reg._records) == 0
        assert self.reg.num_free_slots == self.gpu_cap


# ------------------------------------------------------------------ #
#  ThresholdPrecisionPolicy                                            #
# ------------------------------------------------------------------ #

class TestThresholdPrecisionPolicy:
    def test_tier_assignment_basic(self):
        policy = ThresholdPrecisionPolicy(high_threshold=5.0, mid_threshold=2.0, low_threshold=1.0)
        scores = torch.tensor([6.0, 3.0, 1.5, 0.5])
        ids = [0, 1, 2, 3]
        result = policy.assign_tiers(block_scores=scores, physical_block_ids=ids)
        assert result.fp16_block_ids == [0]
        assert result.int8_block_ids == [1]
        assert result.int4_block_ids == [2]
        assert result.skipped_block_ids == [3]

    def test_all_blocks_assigned(self):
        policy = ThresholdPrecisionPolicy(high_threshold=5.0, mid_threshold=2.0, low_threshold=1.0)
        scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        ids = list(range(6))
        result = policy.assign_tiers(block_scores=scores, physical_block_ids=ids)
        assert len(result.all_block_ids) == 6

    def test_all_skip_when_thresholds_very_high(self):
        policy = ThresholdPrecisionPolicy(high_threshold=100.0, mid_threshold=50.0, low_threshold=10.0)
        scores = torch.tensor([1.0, 2.0])
        result = policy.assign_tiers(block_scores=scores, physical_block_ids=[0, 1])
        assert len(result.skipped_block_ids) == 2
        assert len(result.fp16_block_ids) == 0
