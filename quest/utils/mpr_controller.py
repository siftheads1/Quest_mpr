"""MPRController: InferenceController extended with CPU backup and tiered recall.

Key differences from InferenceController:
- kv_cache KvPool capacity is limited to gpu_capacity (physical GPU slots).
- metadata_cache KvPool capacity covers the full max_seq_len (never evicts).
- MPRBlockRegistry tracks abs_block_id → gpu_slot mappings and residency.
- CrossLayerCPUBackupStore holds FP16/INT8/INT4 backups for evicted blocks.
- mpr_step_select() runs once per decode step (guarded by _step_executed) and:
    1. Collapses per-head scores to per-block scores (GQA max).
    2. Assigns precision tiers via ThresholdPrecisionPolicy.
    3. Pre-evicts non-needed GPU blocks to make room.
    4. Recalls needed CPU blocks at their assigned precision.
    5. Builds and returns the [num_kv_heads, num_selected] GPU index tensor.
    6. Re-initialises the FlashInfer decode handler for the actual page count.
"""

from __future__ import annotations

import time
from typing import Optional

import torch

from quest.utils.controller import InferenceController
from quest.utils.kv_cache import KvCache
from quest.utils.mpr import (
    PrecisionTier,
    ThresholdPrecisionPolicy,
    TopRatioPrecisionPolicy,
    TierAssignment,
)
from quest.utils.mpr_utils import (
    CrossLayerCPUBackupStore,
    MPRBlockRegistry,
    PageResidency,
)


class MPRController(InferenceController):
    """InferenceController with GPU-capacity limit, CPU backup, and tiered recall."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,          # num_q_heads (for buffers)
        num_kv_heads: int,       # needed for GQA collapse
        head_dim: int,
        page_size: int,
        page_budget: int,        # attention budget in pages (used for scoring guard)
        gpu_capacity: int,       # max physical GPU page slots
        max_seq_len: int,        # used for metadata_cache (full sequence length)
        tier_high: float,
        tier_mid: float,
        tier_low: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        # We call the parent __init__ but override kv_cache after.
        # Parent computes kv_cache capacity from max_seq_len; we want gpu_capacity instead.
        # Pass a fake max_seq_len for kv_cache sizing, then rebuild below.
        super().__init__(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            page_size=page_size,
            page_budget=page_budget,
            max_seq_len=gpu_capacity * page_size,  # limits kv_cache to gpu_capacity slots
            dtype=dtype,
            device=device,
        )

        # Replace metadata_cache with a full-size one (never evicts).
        max_meta_pages = (max_seq_len + page_size - 1) // page_size
        self.metadata_cache = KvCache(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            max_seq_len=max_meta_pages * page_size,
            page_size=page_size,
            dtype=dtype,
            device=device,
        )

        self.num_kv_heads = num_kv_heads
        self.gqa_group_size = num_heads // num_kv_heads

        self.gpu_capacity = gpu_capacity
        self.tier_policy = ThresholdPrecisionPolicy(
            high_threshold=tier_high,
            mid_threshold=tier_mid,
            low_threshold=tier_low,
        )

        self.backup_store = CrossLayerCPUBackupStore()
        self.registry = MPRBlockRegistry(gpu_capacity=gpu_capacity)

        # Step-level cache so mpr_step_select runs only once per decode step.
        self._step_gpu_indices: Optional[torch.Tensor] = None
        self._step_executed: bool = False
        self._handler_active: bool = False

        # Timing breakdown (accumulated, reset by clear_timing())
        self.recall_wall_seconds: float = 0.0
        self.evict_wall_seconds: float = 0.0

    # ------------------------------------------------------------------ #
    #  Prepare / begin / end                                               #
    # ------------------------------------------------------------------ #

    def prepare_metadata(self, seq_len: int) -> None:
        """Reserve slots for new tokens, evicting GPU blocks if the pool is full."""
        if seq_len <= 0:
            return

        # Count how many new KV pages will be needed (ceiling arithmetic).
        block_len = self.kv_cache.pool.block_len
        cur_seqlen = self.kv_cache.seqlen
        pages_now = (cur_seqlen + block_len - 1) // block_len if cur_seqlen > 0 else 0
        pages_after = (cur_seqlen + seq_len + block_len - 1) // block_len
        new_pages_needed = pages_after - pages_now

        if new_pages_needed > self.gpu_capacity:
            raise RuntimeError(
                f"Sequence requires {new_pages_needed} pages but gpu_capacity={self.gpu_capacity}. "
                f"Increase gpu_capacity to at least {new_pages_needed} "
                f"(i.e. at least {new_pages_needed * block_len} tokens)."
            )

        # Pre-evict until we have enough free GPU slots.
        protect_ids: set[int] = set()
        # Protect the current (partially filled) page if it exists.
        n_registered = self.registry._next_abs_id
        if n_registered > 0:
            protect_ids.add(n_registered - 1)

        t0 = time.perf_counter()
        while self.registry.num_free_slots < new_pages_needed:
            freed_slot = self.registry.evict_victim(
                protect_ids=protect_ids,
                pool_buf=self.kv_cache.pool.buf,
                backup_store=self.backup_store,
            )
            self.kv_cache.pool.free_block(freed_slot)
        self.evict_wall_seconds += time.perf_counter() - t0

        # Now let parent alloc pages (pool has enough free slots).
        old_kv_pages = len(self.kv_cache.indicies)
        appended_new_pages = self.kv_cache.append_seq(seq_len)
        _ = self.metadata_cache.append_seq(appended_new_pages)

        # Register newly allocated KV pages in the registry.
        new_kv_pages = len(self.kv_cache.indicies)
        for page_pos in range(old_kv_pages, new_kv_pages):
            gpu_slot = self.kv_cache.indicies[page_pos]
            # Sync free_slots: parent allocated this slot from pool._free,
            # so remove it from registry._free_slots if still there.
            self.registry._free_slots.discard(gpu_slot)
            abs_id = self.registry.register_new_block(gpu_slot)

    def begin_forward(self, seq_len: int, updateTensor: bool = True) -> None:
        self._step_executed = False
        self._step_gpu_indices = None

        if updateTensor:
            self.kv_indptr_for_append = torch.tensor(
                [0, len(self.kv_cache.indicies)], dtype=torch.int32, device=self.device
            )
            self.metadata_indptr_for_append = torch.tensor(
                [0, len(self.metadata_cache.indicies)], dtype=torch.int32, device=self.device
            )
            self.kv_last_page_idx = self.kv_cache.indicies[-1]
            self.metadata_last_page_idx = self.metadata_cache.indicies[-1]

        if seq_len > 1:
            if updateTensor:
                self.kv_indices_with_last = torch.tensor(
                    self.kv_cache.indicies, dtype=torch.int32, device=self.device
                )
                self.metadata_indices = torch.tensor(
                    self.metadata_cache.indicies, dtype=torch.int32, device=self.device
                )
        else:
            cur_page_nums = len(self.kv_cache.indicies)
            assert cur_page_nums > 1

            if updateTensor:
                self.kv_indices_with_last = torch.tensor(
                    self.kv_cache.indicies, dtype=torch.int32, device=self.device
                )
                # kv_indices_without_last: per-head, only GPU-resident finalized pages.
                # Will be rebuilt after recall in mpr_step_select; initialise with all.
                resident_slots = self.registry.resident_slots_ordered(
                    range(self.registry._next_abs_id - 1)  # all but current page
                )
                if resident_slots:
                    self.kv_indices_without_last = torch.tensor(
                        resident_slots, dtype=torch.int32, device=self.device
                    ).unsqueeze(0).expand(self.num_heads, -1).contiguous()
                else:
                    self.kv_indices_without_last = torch.zeros(
                        (self.num_heads, 0), dtype=torch.int32, device=self.device
                    )
                self.metadata_indices = torch.tensor(
                    self.metadata_cache.indicies, dtype=torch.int32, device=self.device
                )

            self.inference_page_budget = min(self._page_budget, cur_page_nums)
            self.kv_indptr_for_approx_decode = torch.tensor(
                [0, self.inference_page_budget - 1], dtype=torch.int32, device=self.device
            )
            self.topk_dout_buffer = torch.zeros(
                (self.num_heads, self.inference_page_budget - 1), dtype=self.dtype, device=self.device
            )
            self.topk_dindices_buffer = torch.zeros(
                (self.num_heads, self.inference_page_budget - 1), dtype=torch.int32, device=self.device
            )
            self.topk_buf = torch.zeros(
                (self.num_heads, 8192 * 2 * (2 + 4) // 2 // 48), dtype=self.dtype, device=self.device
            )
            # Handler is initialised inside mpr_step_select with the actual page count.
            # Store a sentinel to detect whether it needs initialising.
            self._handler_active = False

    def end_forward(self) -> None:
        if self._handler_active:
            self._decode_handler.end_forward()
            self._handler_active = False

    # ------------------------------------------------------------------ #
    #  Core MPR selection (runs once per decode step)                      #
    # ------------------------------------------------------------------ #

    def mpr_step_select(
        self,
        estimated_attn_score: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Score → tier → recall → return GPU attention index tensor.

        Args:
            estimated_attn_score: [num_q_heads, N_finalized_blocks] from decode_estimate.
            layer_idx: The calling layer index (used for pool_buf access; not scored per-layer).

        Returns:
            gpu_indices: [num_kv_heads, num_selected] int32 GPU slot indices.
        """
        if self._step_executed:
            return self._step_gpu_indices

        # ---- 1. GQA collapse: [num_q_heads, N] → [N] ----
        num_q_heads, N = estimated_attn_score.shape
        if N == 0:
            self._step_gpu_indices = torch.zeros(
                (self.num_kv_heads, 0), dtype=torch.int32, device=self.device
            )
            self._reinit_handler(0)
            self._step_executed = True
            return self._step_gpu_indices

        scores = estimated_attn_score  # [num_q_heads, N]
        # Reshape to [num_kv_heads, group_size, N], max over group_size
        scores_kv = scores.reshape(
            self.num_kv_heads, self.gqa_group_size, N
        ).max(dim=1).values  # [num_kv_heads, N]
        block_scores = scores_kv.max(dim=0).values  # [N]

        # The finalized abs_block_ids (all except the currently-filling page).
        n_total = self.registry._next_abs_id
        # Last abs_id (n_total-1) is the current page; finalized = [0, n_total-2].
        # N should equal n_total - 1 (metadata_cache.seqlen - 1 from decode_estimate).
        finalized_abs_ids = list(range(n_total - 1))

        # ---- 2. Update scores in registry ----
        for i, abs_id in enumerate(finalized_abs_ids):
            self.registry.update_score(abs_id, float(block_scores[i]))

        # ---- 3. Tier assignment ----
        assignment: TierAssignment = self.tier_policy.assign_tiers(
            block_scores=block_scores,
            physical_block_ids=finalized_abs_ids,
        )

        # ---- 4. Budget cap ----
        # Reserve 1 GPU slot for the current page being filled.
        max_attendable = max(0, self.gpu_capacity - 1)
        fp16_sel = assignment.fp16_block_ids[:max_attendable]
        rem = max_attendable - len(fp16_sel)
        int8_sel = assignment.int8_block_ids[:rem]
        rem -= len(int8_sel)
        int4_sel = assignment.int4_block_ids[:rem]

        attend_set: set[int] = set(fp16_sel) | set(int8_sel) | set(int4_sel)
        attended_ordered = [aid for aid in finalized_abs_ids if aid in attend_set]

        # Which CPU-backed blocks need to be recalled?
        needed_recalls: list[tuple[int, PrecisionTier]] = [
            (aid, PrecisionTier.FP16) for aid in fp16_sel
            if not self.registry.is_gpu_resident(aid)
        ] + [
            (aid, PrecisionTier.INT8) for aid in int8_sel
            if not self.registry.is_gpu_resident(aid)
        ] + [
            (aid, PrecisionTier.INT4) for aid in int4_sel
            if not self.registry.is_gpu_resident(aid)
        ]

        # ---- 5. Phase 1: pre-evict non-needed GPU blocks to make room ----
        current_abs_id = n_total - 1
        protect_ids = attend_set | {current_abs_id}

        # Slots needed: one per recall, plus one if current page will trigger a new page alloc.
        slots_needed = len(needed_recalls)

        t0 = time.perf_counter()
        while self.registry.num_free_slots < slots_needed:
            candidates = [
                (aid, rec)
                for aid, rec in self.registry._records.items()
                if rec.residency is PageResidency.GPU_RESIDENT and aid not in protect_ids
            ]
            if not candidates:
                # Can't free more — cap recalls to available free slots.
                avail = self.registry.num_free_slots
                needed_recalls = needed_recalls[:avail]
                recalled_set = {aid for aid, _ in needed_recalls}
                already_resident = {
                    aid for aid in attend_set if self.registry.is_gpu_resident(aid)
                }
                attended_ordered = [
                    aid for aid in attended_ordered
                    if aid in recalled_set or aid in already_resident
                ]
                break
            victim_id = min(candidates, key=lambda x: x[1].last_score)[0]
            freed_slot = self.registry.evict_to_cpu(
                victim_id, self.kv_cache.pool.buf, self.backup_store
            )
            self.kv_cache.pool.free_block(freed_slot)
        self.evict_wall_seconds += time.perf_counter() - t0

        # ---- 6. Phase 2: recall ----
        t0 = time.perf_counter()
        for abs_id, precision in needed_recalls:
            recalled_slot = self.registry.recall_from_cpu(
                abs_id, precision, self.kv_cache.pool.buf, self.backup_store
            )
            self.kv_cache.pool._free.discard(recalled_slot)
        self.recall_wall_seconds += time.perf_counter() - t0

        # ---- 7. Build GPU index tensor [num_kv_heads, num_selected] ----
        final_slots = self.registry.resident_slots_ordered(attended_ordered)
        num_selected = len(final_slots)

        if num_selected == 0:
            gpu_indices = torch.zeros(
                (self.num_kv_heads, 0), dtype=torch.int32, device=self.device
            )
        else:
            gpu_indices = (
                torch.tensor(final_slots, dtype=torch.int32, device=self.device)
                .unsqueeze(0)
                .expand(self.num_kv_heads, -1)
                .contiguous()
            )

        # ---- 8. Re-init FlashInfer handler with actual page count ----
        self._reinit_handler(num_selected)

        self._step_gpu_indices = gpu_indices
        self._step_executed = True
        return gpu_indices

    def _reinit_handler(self, num_selected: int) -> None:
        if self._handler_active:
            self._decode_handler.end_forward()
        self.kv_indptr_for_approx_decode = torch.tensor(
            [0, num_selected], dtype=torch.int32, device=self.device
        )
        if num_selected > 0:
            self._decode_handler.begin_forward(
                self.kv_indptr_for_approx_decode,
                self.num_kv_heads,
                self.num_kv_heads,
                self.head_dim,
                self.page_size,
                self.dtype,
            )
            self._handler_active = True
        else:
            self._handler_active = False

    # ------------------------------------------------------------------ #
    #  State management                                                     #
    # ------------------------------------------------------------------ #

    def clean_states(self) -> None:
        self.kv_cache.release()
        self.metadata_cache.release()
        self.backup_store.clear()
        self.registry.reset()
        self._step_executed = False
        self._step_gpu_indices = None
        self._handler_active = False

    def clear_timing(self) -> None:
        self.recall_wall_seconds = 0.0
        self.evict_wall_seconds = 0.0

    def timing_summary(self) -> dict:
        return {
            "recall_wall_seconds": self.recall_wall_seconds,
            "evict_wall_seconds": self.evict_wall_seconds,
        }
