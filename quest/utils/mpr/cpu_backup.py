# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU KV backup stores for the MPR sidecar."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import torch

from .backup_codec import (
    FP16_BACKUP_FORMAT,
    FP16BackupCodec,
    FP16BackupPayload,
    INT8_BACKUP_FORMAT,
    INT8BackupCodec,
    INT8BackupPayload,
    INT4_BACKUP_FORMAT,
    INT4BackupCodec,
    INT4BackupPayload,
)


@dataclass(frozen=True)
class CPUBackupKey:
    """Internal key for one MPR CPU backup entry."""

    layer_name: str
    physical_block_id: int
    request_id: str | None = None
    logical_block_idx: int | None = None
    generation: int | None = None


@dataclass(frozen=True)
class CPUBackupPutResult:
    key: CPUBackupKey
    shape: tuple[int, ...]
    dtype: torch.dtype
    num_bytes: int
    fp16_payload_bytes: int
    int8_payload_bytes: int
    int8_scale_bytes: int
    int4_payload_bytes: int
    int4_scale_bytes: int
    total_actual_backup_bytes: int
    copy_wall_seconds: float


@dataclass(frozen=True)
class CPUBackupReleaseResult:
    released_entries: int
    released_bytes: int
    fp16_payload_bytes: int
    int8_payload_bytes: int
    int8_scale_bytes: int
    int4_payload_bytes: int
    int4_scale_bytes: int
    total_actual_backup_bytes: int


@dataclass(frozen=True)
class CPUBackupStats:
    block_count: int
    total_bytes: int
    put_count: int
    release_count: int
    total_copy_wall_seconds: float
    fp16_payload_bytes: int
    int8_payload_bytes: int
    int8_scale_bytes: int
    int4_payload_bytes: int
    int4_scale_bytes: int
    total_actual_backup_bytes: int


@dataclass(frozen=True)
class _CPUBackupEntry:
    fp16_payload: FP16BackupPayload
    int8_payload: INT8BackupPayload | None = None
    int4_payload: INT4BackupPayload | None = None

    @property
    def fp16_payload_bytes(self) -> int:
        return self.fp16_payload.payload_nbytes

    @property
    def int8_payload_bytes(self) -> int:
        return 0 if self.int8_payload is None else _tensor_nbytes(self.int8_payload.quantized)

    @property
    def int8_scale_bytes(self) -> int:
        return 0 if self.int8_payload is None else _tensor_nbytes(self.int8_payload.scale)

    @property
    def int4_payload_bytes(self) -> int:
        return 0 if self.int4_payload is None else _tensor_nbytes(self.int4_payload.packed)

    @property
    def int4_scale_bytes(self) -> int:
        return 0 if self.int4_payload is None else _tensor_nbytes(self.int4_payload.scale)

    @property
    def total_actual_backup_bytes(self) -> int:
        return (
            self.fp16_payload_bytes
            + self.int8_payload_bytes
            + self.int8_scale_bytes
            + self.int4_payload_bytes
            + self.int4_scale_bytes
        )


class SemanticCPUBackupStore:
    """Semantic CPU tensor backup store for MPR recovery."""

    def __init__(self) -> None:
        self._entries: dict[CPUBackupKey, _CPUBackupEntry] = {}
        self._fp16_payload_bytes = 0
        self._int8_payload_bytes = 0
        self._int8_scale_bytes = 0
        self._int4_payload_bytes = 0
        self._int4_scale_bytes = 0
        self._put_count = 0
        self._release_count = 0
        self._total_copy_wall_seconds = 0.0
        self._fp16_codec = FP16BackupCodec()
        self._int8_codec = INT8BackupCodec()
        self._int4_codec = INT4BackupCodec()

    def put(
        self,
        *,
        layer_name: str,
        physical_block_id: int,
        kv_block: torch.Tensor,
        request_id: str | None = None,
        logical_block_idx: int | None = None,
        generation: int | None = None,
        backup_storage_mode: str = "fp16_only",
    ) -> CPUBackupPutResult:
        _validate_backup_storage_mode(backup_storage_mode)
        key = CPUBackupKey(
            layer_name=layer_name,
            physical_block_id=int(physical_block_id),
            request_id=request_id,
            logical_block_idx=logical_block_idx,
            generation=generation,
        )

        start = time.perf_counter()
        fp16_payload = self._fp16_codec.encode(kv_block)
        int8_payload = (
            self._int8_codec.encode(fp16_payload.tensor)
            if backup_storage_mode in {"eager_fp16_int8", "eager_fp16_int8_int4"}
            else None
        )
        int4_payload = (
            self._int4_codec.encode(fp16_payload.tensor)
            if backup_storage_mode == "eager_fp16_int8_int4"
            else None
        )
        entry = _CPUBackupEntry(
            fp16_payload=fp16_payload,
            int8_payload=int8_payload,
            int4_payload=int4_payload,
        )
        elapsed = time.perf_counter() - start

        old = self._entries.get(key)
        if old is not None:
            self._subtract_entry_bytes(old)

        self._entries[key] = entry
        self._add_entry_bytes(entry)
        self._put_count += 1
        self._total_copy_wall_seconds += elapsed
        return CPUBackupPutResult(
            key=key,
            shape=tuple(fp16_payload.tensor.shape),
            dtype=fp16_payload.tensor.dtype,
            num_bytes=entry.fp16_payload_bytes,
            fp16_payload_bytes=entry.fp16_payload_bytes,
            int8_payload_bytes=entry.int8_payload_bytes,
            int8_scale_bytes=entry.int8_scale_bytes,
            int4_payload_bytes=entry.int4_payload_bytes,
            int4_scale_bytes=entry.int4_scale_bytes,
            total_actual_backup_bytes=entry.total_actual_backup_bytes,
            copy_wall_seconds=elapsed,
        )

    def get_payload(
        self,
        key: CPUBackupKey,
        backup_format: str,
    ) -> FP16BackupPayload | INT8BackupPayload | INT4BackupPayload | None:
        if backup_format not in {FP16_BACKUP_FORMAT, INT8_BACKUP_FORMAT, INT4_BACKUP_FORMAT}:
            raise ValueError(f"backup_format must be 'fp16', 'int8', or 'int4', got {backup_format!r}.")
        entry = self._entries.get(key)
        if entry is None:
            return None
        if backup_format == FP16_BACKUP_FORMAT:
            return entry.fp16_payload
        if backup_format == INT8_BACKUP_FORMAT:
            return entry.int8_payload
        return entry.int4_payload

    def release_blocks(self, physical_block_ids: set[int]) -> CPUBackupReleaseResult:
        if not physical_block_ids:
            return CPUBackupReleaseResult(
                released_entries=0, released_bytes=0,
                fp16_payload_bytes=0, int8_payload_bytes=0, int8_scale_bytes=0,
                int4_payload_bytes=0, int4_scale_bytes=0, total_actual_backup_bytes=0,
            )
        released_entries = 0
        rel_fp16 = rel_int8 = rel_int8s = rel_int4 = rel_int4s = 0
        for key in list(self._entries):
            if key.physical_block_id not in physical_block_ids:
                continue
            entry = self._entries.pop(key)
            self._subtract_entry_bytes(entry)
            released_entries += 1
            rel_fp16 += entry.fp16_payload_bytes
            rel_int8 += entry.int8_payload_bytes
            rel_int8s += entry.int8_scale_bytes
            rel_int4 += entry.int4_payload_bytes
            rel_int4s += entry.int4_scale_bytes
            del entry
        if released_entries:
            self._release_count += released_entries
        total = rel_fp16 + rel_int8 + rel_int8s + rel_int4 + rel_int4s
        return CPUBackupReleaseResult(
            released_entries=released_entries, released_bytes=total,
            fp16_payload_bytes=rel_fp16, int8_payload_bytes=rel_int8,
            int8_scale_bytes=rel_int8s, int4_payload_bytes=rel_int4,
            int4_scale_bytes=rel_int4s, total_actual_backup_bytes=total,
        )

    def clear(self) -> None:
        self._entries.clear()
        self._fp16_payload_bytes = 0
        self._int8_payload_bytes = 0
        self._int8_scale_bytes = 0
        self._int4_payload_bytes = 0
        self._int4_scale_bytes = 0

    def stats(self) -> CPUBackupStats:
        total = self._total_actual_backup_bytes()
        return CPUBackupStats(
            block_count=len(self._entries),
            total_bytes=total,
            put_count=self._put_count,
            release_count=self._release_count,
            total_copy_wall_seconds=self._total_copy_wall_seconds,
            fp16_payload_bytes=self._fp16_payload_bytes,
            int8_payload_bytes=self._int8_payload_bytes,
            int8_scale_bytes=self._int8_scale_bytes,
            int4_payload_bytes=self._int4_payload_bytes,
            int4_scale_bytes=self._int4_scale_bytes,
            total_actual_backup_bytes=total,
        )

    def _add_entry_bytes(self, entry: _CPUBackupEntry) -> None:
        self._fp16_payload_bytes += entry.fp16_payload_bytes
        self._int8_payload_bytes += entry.int8_payload_bytes
        self._int8_scale_bytes += entry.int8_scale_bytes
        self._int4_payload_bytes += entry.int4_payload_bytes
        self._int4_scale_bytes += entry.int4_scale_bytes

    def _subtract_entry_bytes(self, entry: _CPUBackupEntry) -> None:
        self._fp16_payload_bytes -= entry.fp16_payload_bytes
        self._int8_payload_bytes -= entry.int8_payload_bytes
        self._int8_scale_bytes -= entry.int8_scale_bytes
        self._int4_payload_bytes -= entry.int4_payload_bytes
        self._int4_scale_bytes -= entry.int4_scale_bytes

    def _total_actual_backup_bytes(self) -> int:
        return (
            self._fp16_payload_bytes + self._int8_payload_bytes + self._int8_scale_bytes
            + self._int4_payload_bytes + self._int4_scale_bytes
        )


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _validate_backup_storage_mode(backup_storage_mode: str) -> None:
    if backup_storage_mode not in {"fp16_only", "eager_fp16_int8", "eager_fp16_int8_int4"}:
        raise ValueError(
            "backup_storage_mode must be 'fp16_only', 'eager_fp16_int8', or "
            f"'eager_fp16_int8_int4', got {backup_storage_mode!r}."
        )
