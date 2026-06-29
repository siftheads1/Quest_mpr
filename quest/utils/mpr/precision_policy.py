# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Score-to-precision tier assignment helpers for MPR."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import torch


class PrecisionTier(str, Enum):
    """Precision tiers used by MPR mixed-precision recovery."""

    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    SKIP = "skip"


@dataclass(frozen=True)
class TierAssignment:
    """Physical KV block ids grouped by their assigned recovery tier."""

    fp16_block_ids: list[int]
    int8_block_ids: list[int]
    skipped_block_ids: list[int]
    int4_block_ids: list[int] = field(default_factory=list)

    @property
    def all_block_ids(self) -> list[int]:
        """Return all assigned block ids in tier order."""
        return [
            *self.fp16_block_ids,
            *self.int8_block_ids,
            *self.int4_block_ids,
            *self.skipped_block_ids,
        ]


class PrecisionPolicy(Protocol):
    """Interface for score-to-tier assignment policies."""

    def assign_tiers(
        self,
        *,
        block_scores: torch.Tensor,
        physical_block_ids: list[int],
    ) -> TierAssignment:
        ...


@dataclass(frozen=True)
class TopRatioPrecisionPolicy:
    """Assign highest-scoring blocks to fp16/int8/int4 tiers by ratio.

    Scores are sorted descending with a stable tie-break, so equal scores keep
    their original candidate order. Tier counts use ceil rounding and are
    clamped so fp16 plus int8 plus int4 never exceeds the candidate count. Any
    remaining candidates are assigned to the skip tier.
    """

    fp16_ratio: float
    int8_ratio: float
    int4_ratio: float = 0.0

    def __post_init__(self) -> None:
        _validate_ratio("fp16_ratio", self.fp16_ratio)
        _validate_ratio("int8_ratio", self.int8_ratio)
        _validate_ratio("int4_ratio", self.int4_ratio)
        ratio_sum = self.fp16_ratio + self.int8_ratio + self.int4_ratio
        if ratio_sum > 1.0:
            raise ValueError(
                "fp16_ratio + int8_ratio + int4_ratio must be <= 1, got "
                f"{ratio_sum}."
            )

    def assign_tiers(
        self,
        *,
        block_scores: torch.Tensor,
        physical_block_ids: list[int],
    ) -> TierAssignment:
        """Assign tiers using stable descending score order."""
        _validate_policy_inputs(block_scores, physical_block_ids)
        num_candidates = int(block_scores.numel())
        if num_candidates == 0:
            return TierAssignment(
                fp16_block_ids=[],
                int8_block_ids=[],
                int4_block_ids=[],
                skipped_block_ids=[],
            )

        ordered_indices = torch.argsort(
            block_scores,
            descending=True,
            stable=True,
        ).detach().cpu()
        ordered_block_ids = [
            int(physical_block_ids[int(index)]) for index in ordered_indices
        ]
        fp16_count = min(
            num_candidates,
            math.ceil(num_candidates * self.fp16_ratio),
        )
        remaining_count = num_candidates - fp16_count
        int8_count = min(
            remaining_count,
            math.ceil(num_candidates * self.int8_ratio),
        )
        remaining_count -= int8_count
        int4_count = min(
            remaining_count,
            math.ceil(num_candidates * self.int4_ratio),
        )
        int4_start = fp16_count + int8_count
        skip_start = int4_start + int4_count
        assignment = TierAssignment(
            fp16_block_ids=ordered_block_ids[:fp16_count],
            int8_block_ids=ordered_block_ids[fp16_count:int4_start],
            int4_block_ids=ordered_block_ids[int4_start:skip_start],
            skipped_block_ids=ordered_block_ids[skip_start:],
        )
        _validate_assignment(assignment, physical_block_ids)
        return assignment


@dataclass(frozen=True)
class ThresholdPrecisionPolicy:
    """Assign blocks to fp16/int8/int4/skip tiers by score thresholds."""

    high_threshold: float
    mid_threshold: float
    low_threshold: float

    def __post_init__(self) -> None:
        if self.high_threshold < self.mid_threshold:
            raise ValueError(
                "high_threshold must be >= mid_threshold, got "
                f"{self.high_threshold} < {self.mid_threshold}."
            )
        if self.mid_threshold < self.low_threshold:
            raise ValueError(
                "mid_threshold must be >= low_threshold, got "
                f"{self.mid_threshold} < {self.low_threshold}."
            )

    def assign_tiers(
        self,
        *,
        block_scores: torch.Tensor,
        physical_block_ids: list[int],
    ) -> TierAssignment:
        """Assign tiers using candidate order and inclusive thresholds."""
        _validate_policy_inputs(block_scores, physical_block_ids)
        fp16_block_ids: list[int] = []
        int8_block_ids: list[int] = []
        int4_block_ids: list[int] = []
        skipped_block_ids: list[int] = []

        for index, score in enumerate(block_scores.detach().cpu()):
            block_id = int(physical_block_ids[index])
            score_value = float(score)
            if score_value >= self.high_threshold:
                fp16_block_ids.append(block_id)
            elif score_value >= self.mid_threshold:
                int8_block_ids.append(block_id)
            elif score_value >= self.low_threshold:
                int4_block_ids.append(block_id)
            else:
                skipped_block_ids.append(block_id)

        assignment = TierAssignment(
            fp16_block_ids=fp16_block_ids,
            int8_block_ids=int8_block_ids,
            int4_block_ids=int4_block_ids,
            skipped_block_ids=skipped_block_ids,
        )
        _validate_assignment(assignment, physical_block_ids)
        return assignment


def _validate_ratio(name: str, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")


def _validate_policy_inputs(
    block_scores: torch.Tensor,
    physical_block_ids: list[int],
) -> None:
    if block_scores.ndim != 1:
        raise ValueError(
            "MPR precision policy expects block_scores shaped [num_blocks], "
            f"got {tuple(block_scores.shape)}."
        )
    if len(physical_block_ids) != int(block_scores.numel()):
        raise ValueError(
            "MPR precision policy requires physical_block_ids length to match "
            f"block_scores, got {len(physical_block_ids)} ids and "
            f"{int(block_scores.numel())} scores."
        )
    if len(set(physical_block_ids)) != len(physical_block_ids):
        raise ValueError(
            "MPR precision policy requires unique physical block ids."
        )


def _validate_assignment(
    assignment: TierAssignment,
    physical_block_ids: list[int],
) -> None:
    assigned_ids = assignment.all_block_ids
    if len(assigned_ids) != len(physical_block_ids):
        raise ValueError(
            "MPR precision policy must assign every candidate block exactly "
            f"once, got {len(assigned_ids)} assignments for "
            f"{len(physical_block_ids)} candidates."
        )
    if set(assigned_ids) != set(physical_block_ids):
        raise ValueError(
            "MPR precision policy assignment must contain exactly the input "
            "physical block ids."
        )
