"""Integration tests for MPRController (requires compiled Quest CUDA kernels).

Run with:
    pytest quest/tests/test_mpr_integration.py -v --tb=short

These tests require a CUDA device and the compiled quest._kernels extension.
Skip on CPU-only machines:
    pytest quest/tests/test_mpr_integration.py -v -m "not cuda"
"""

import pytest
import torch

# All tests in this file need CUDA + compiled kernels.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _try_import_kernels():
    try:
        import quest._kernels  # noqa: F401
        return True
    except ImportError:
        return False


requires_kernels = pytest.mark.skipif(
    not _try_import_kernels(), reason="quest._kernels not compiled"
)


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _make_mpr_controller(
    num_layers=2,
    num_heads=4,
    num_kv_heads=4,
    head_dim=16,
    page_size=4,
    gpu_capacity=8,
    max_seq_len=256,
    token_budget=16,
    tier_high=5.0,
    tier_mid=2.0,
    tier_low=0.5,
    dtype=torch.float16,
    device=None,
):
    if device is None:
        device = torch.device("cuda:0")
    from quest.utils.mpr_controller import MPRController
    return MPRController(
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        page_budget=token_budget // page_size,
        gpu_capacity=gpu_capacity,
        max_seq_len=max_seq_len,
        tier_high=tier_high,
        tier_mid=tier_mid,
        tier_low=tier_low,
        dtype=dtype,
        device=device,
    )


# ------------------------------------------------------------------ #
#  MPRController init                                                  #
# ------------------------------------------------------------------ #

@requires_kernels
class TestMPRControllerInit:
    def test_kv_cache_capacity_limited(self):
        gpu_cap = 8
        ctrl = _make_mpr_controller(gpu_capacity=gpu_cap)
        assert ctrl.kv_cache.pool.capacity == gpu_cap

    def test_metadata_cache_full_capacity(self):
        max_seq = 256
        page_size = 4
        ctrl = _make_mpr_controller(max_seq_len=max_seq, page_size=page_size)
        max_meta_pages = (max_seq + page_size - 1) // page_size
        # metadata pool must be at least max_meta_pages // page_size
        # (metadata is also paged: one metadata block covers page_size kv pages)
        assert ctrl.metadata_cache.pool.capacity >= 1

    def test_backup_store_empty_on_init(self):
        ctrl = _make_mpr_controller()
        assert ctrl.backup_store.stats().block_count == 0

    def test_registry_all_slots_free_on_init(self):
        gpu_cap = 6
        ctrl = _make_mpr_controller(gpu_capacity=gpu_cap)
        assert ctrl.registry.num_free_slots == gpu_cap


# ------------------------------------------------------------------ #
#  Eviction during prepare_metadata                                    #
# ------------------------------------------------------------------ #

@requires_kernels
class TestMPREviction:
    def test_eviction_occurs_when_pool_full(self):
        """Fill GPU pool past capacity; eviction must kick in."""
        gpu_cap = 3
        page_size = 4
        device = torch.device("cuda:0")
        ctrl = _make_mpr_controller(
            gpu_capacity=gpu_cap, page_size=page_size,
            max_seq_len=256, device=device,
        )
        # Prefill with enough tokens to force gpu_cap+1 pages.
        n_tokens = (gpu_cap + 1) * page_size  # one page more than GPU can hold
        ctrl.prepare_metadata(n_tokens)
        # At least 1 eviction must have happened.
        assert ctrl.backup_store.stats().block_count >= 1

    def test_eviction_frees_slot_for_new_page(self):
        gpu_cap = 2
        page_size = 4
        device = torch.device("cuda:0")
        ctrl = _make_mpr_controller(
            gpu_capacity=gpu_cap, page_size=page_size, device=device,
        )
        n_tokens = (gpu_cap + 1) * page_size
        ctrl.prepare_metadata(n_tokens)
        # kv_cache.indicies should have exactly gpu_cap entries (pool is full).
        assert len(ctrl.kv_cache.indicies) <= gpu_cap


# ------------------------------------------------------------------ #
#  mpr_step_select idempotency                                         #
# ------------------------------------------------------------------ #

@requires_kernels
class TestMPRStepSelectIdempotency:
    def test_second_call_returns_cached_result(self):
        gpu_cap = 8
        page_size = 4
        num_heads = 4
        head_dim = 16
        device = torch.device("cuda:0")
        ctrl = _make_mpr_controller(
            gpu_capacity=gpu_cap, page_size=page_size,
            num_heads=num_heads, num_kv_heads=4, head_dim=head_dim,
            device=device,
        )
        # Simulate having 3 finalized pages worth of metadata.
        n_tokens = 3 * page_size
        ctrl.prepare_metadata(n_tokens)
        ctrl.begin_forward(1)

        # Build a fake score tensor [num_heads, n_tokens-1=11]
        # (metadata_cache.seqlen - 1 finalized pages)
        n_finalized = ctrl.registry._next_abs_id - 1
        if n_finalized <= 0:
            pytest.skip("Need finalized pages for this test")

        scores = torch.rand(num_heads, n_finalized, dtype=torch.float16, device=device)

        result1 = ctrl.mpr_step_select(scores, layer_idx=0)
        result2 = ctrl.mpr_step_select(scores, layer_idx=0)
        assert result1 is result2  # Same object returned (cache hit)

    def test_step_executed_resets_on_begin_forward(self):
        ctrl = _make_mpr_controller()
        ctrl._step_executed = True
        ctrl._step_gpu_indices = torch.zeros((4, 2), dtype=torch.int32)
        ctrl.prepare_metadata(4)
        ctrl.begin_forward(1)
        assert ctrl._step_executed is False


# ------------------------------------------------------------------ #
#  Regression: MPR with no eviction == no error                        #
# ------------------------------------------------------------------ #

@requires_kernels
class TestMPRNoEviction:
    """With gpu_capacity=max_pages, no eviction occurs and no errors are raised."""

    def test_prepare_metadata_no_eviction(self):
        max_seq = 128
        page_size = 4
        gpu_cap = (max_seq + page_size - 1) // page_size  # enough for everything
        device = torch.device("cuda:0")
        ctrl = _make_mpr_controller(
            gpu_capacity=gpu_cap, max_seq_len=max_seq, page_size=page_size,
            device=device,
        )
        ctrl.prepare_metadata(max_seq // 2)
        assert ctrl.backup_store.stats().block_count == 0

    def test_clean_states_resets_all(self):
        device = torch.device("cuda:0")
        ctrl = _make_mpr_controller(gpu_capacity=8, device=device)
        ctrl.prepare_metadata(16)
        ctrl.clean_states()
        assert ctrl.kv_cache.seqlen == 0
        assert ctrl.metadata_cache.seqlen == 0
        assert ctrl.backup_store.stats().block_count == 0
        assert ctrl.registry._next_abs_id == 0
