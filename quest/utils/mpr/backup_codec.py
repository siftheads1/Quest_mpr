# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Precision-specific KV backup payload codecs for MPR.

Backup codecs define logical payload formats and reference encode/materialize
behavior. They do not define how payloads are stored in CPU memory; CPU backup
layout and lifecycle are handled by the backup store in later M4 steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


FP16_BACKUP_FORMAT = "fp16"
INT8_BACKUP_FORMAT = "int8"
INT4_BACKUP_FORMAT = "int4"
PER_TOKEN_PER_KV_HEAD_SCALE = "per_token_per_kv_head"
INT8_QMAX = 127.0
INT4_QMAX = 7.0
INT4_QMIN = -7.0


@dataclass(frozen=True)
class FP16BackupPayload:
    """Logical fp16 backup payload for one semantic K/V block."""

    tensor: torch.Tensor
    original_shape: tuple[int, ...]
    format: str = FP16_BACKUP_FORMAT

    @property
    def payload_nbytes(self) -> int:
        """Return payload tensor storage bytes."""
        return self.tensor.numel() * self.tensor.element_size()


@dataclass(frozen=True)
class INT8BackupPayload:
    """Logical int8 backup payload for one semantic K/V block.

    ``quantized`` stores signed symmetric int8 values. ``scale`` stores one
    scale for each K/V, token, and KV head vector. The actual CPU store may
    keep these tensors in any layout as long as this logical contract is
    preserved for materialization.
    """

    quantized: torch.Tensor
    scale: torch.Tensor
    original_shape: tuple[int, ...]
    scale_granularity: str = PER_TOKEN_PER_KV_HEAD_SCALE
    format: str = INT8_BACKUP_FORMAT

    @property
    def payload_nbytes(self) -> int:
        """Return quantized data plus scale tensor storage bytes."""
        return (
            self.quantized.numel() * self.quantized.element_size()
            + self.scale.numel() * self.scale.element_size()
        )


@dataclass(frozen=True)
class INT4BackupPayload:
    """Logical packed int4 backup payload for one semantic K/V block.

    ``packed`` stores two signed symmetric INT4 values per uint8 byte. The
    low nibble holds the even head-dim element and the high nibble holds the
    odd element. Signed values are encoded as 4-bit two's-complement bit
    patterns, i.e. ``q & 0xF``. The actual CPU store may keep these tensors in
    any layout as long as this logical contract is preserved for
    materialization.
    """

    packed: torch.Tensor
    scale: torch.Tensor
    original_shape: tuple[int, ...]
    scale_granularity: str = PER_TOKEN_PER_KV_HEAD_SCALE
    quantized_range: tuple[int, int] = (int(INT4_QMIN), int(INT4_QMAX))
    format: str = INT4_BACKUP_FORMAT

    @property
    def payload_nbytes(self) -> int:
        """Return packed data plus scale tensor storage bytes."""
        return (
            self.packed.numel() * self.packed.element_size()
            + self.scale.numel() * self.scale.element_size()
        )


class BackupCodec(Protocol):
    """Encode and materialize one precision-specific KV backup payload."""

    def encode(self, kv_block: torch.Tensor):
        """Encode a semantic K/V block into a logical backup payload."""
        ...

    def materialize(
        self,
        payload,
        *,
        target_dtype: torch.dtype,
        target_device: torch.device | str,
    ) -> torch.Tensor:
        """Materialize a payload into the target dtype/device."""
        ...


class FP16BackupCodec:
    """Reference codec for semantic fp16 K/V backup payloads."""

    def encode(self, kv_block: torch.Tensor) -> FP16BackupPayload:
        """Copy one semantic K/V block to a CPU fp16 payload."""
        _validate_kv_block(kv_block)
        copied = kv_block.detach().to(
            device="cpu",
            dtype=torch.float16,
            copy=True,
        )
        return FP16BackupPayload(
            tensor=copied,
            original_shape=tuple(kv_block.shape),
        )

    def materialize(
        self,
        payload: FP16BackupPayload,
        *,
        target_dtype: torch.dtype,
        target_device: torch.device | str,
    ) -> torch.Tensor:
        """Materialize an fp16 payload into the target dtype/device."""
        return payload.tensor.to(
            device=target_device,
            dtype=target_dtype,
            copy=False,
        )


class INT8BackupCodec:
    """Reference codec for per-token-per-kv-head int8 K/V payloads.

    All-zero vectors use scale ``1.0`` as a placeholder to avoid division by
    zero. Their quantized values remain zero, so dequantization still exactly
    reconstructs the original zero vector.
    """

    def encode(self, kv_block: torch.Tensor) -> INT8BackupPayload:
        """Quantize one semantic K/V block into a CPU int8 payload."""
        _validate_kv_block(kv_block)
        source = kv_block.detach().to(
            device="cpu",
            dtype=torch.float32,
            copy=True,
        )
        max_abs = source.abs().amax(dim=-1)
        scale = max_abs / INT8_QMAX
        scale = torch.where(
            max_abs == 0,
            torch.ones_like(scale),
            scale,
        )
        quantized = torch.round(source / scale.unsqueeze(-1)).clamp(
            -INT8_QMAX,
            INT8_QMAX,
        ).to(torch.int8)
        return INT8BackupPayload(
            quantized=quantized,
            scale=scale,
            original_shape=tuple(kv_block.shape),
        )

    def materialize(
        self,
        payload: INT8BackupPayload,
        *,
        target_dtype: torch.dtype,
        target_device: torch.device | str,
    ) -> torch.Tensor:
        """Dequantize an int8 payload into the target dtype/device."""
        quantized = payload.quantized.to(device=target_device, copy=False)
        scale = payload.scale.to(device=target_device, copy=False)
        materialized = quantized.to(torch.float32) * scale.unsqueeze(-1)
        return materialized.to(dtype=target_dtype)


class INT4BackupCodec:
    """Reference codec for packed per-token-per-kv-head int4 K/V payloads.

    All-zero vectors use scale ``1.0`` as a placeholder to avoid division by
    zero. Their packed values remain zero, so dequantization still exactly
    reconstructs the original zero vector. This reference implementation keeps
    packing and unpacking in PyTorch; future kernel work should target the
    broader recovery materialization path rather than this codec alone.
    """

    def encode(self, kv_block: torch.Tensor) -> INT4BackupPayload:
        """Quantize and pack one semantic K/V block into a CPU int4 payload."""
        _validate_kv_block(kv_block)
        source = kv_block.detach().to(
            device="cpu",
            dtype=torch.float32,
            copy=True,
        )
        max_abs = source.abs().amax(dim=-1)
        scale = max_abs / INT4_QMAX
        scale = torch.where(
            max_abs == 0,
            torch.ones_like(scale),
            scale,
        )
        quantized = torch.round(source / scale.unsqueeze(-1)).clamp(
            INT4_QMIN,
            INT4_QMAX,
        ).to(torch.int8)
        low_nibbles = torch.bitwise_and(
            quantized[..., 0::2],
            0xF,
        ).to(torch.uint8)
        high_nibbles = torch.zeros_like(low_nibbles)
        high_values = quantized[..., 1::2]
        if high_values.shape[-1] > 0:
            high_nibbles[..., : high_values.shape[-1]] = torch.bitwise_and(
                high_values,
                0xF,
            ).to(torch.uint8)
        packed = torch.bitwise_or(
            low_nibbles,
            torch.bitwise_left_shift(high_nibbles, 4),
        )
        return INT4BackupPayload(
            packed=packed,
            scale=scale,
            original_shape=tuple(kv_block.shape),
        )

    def materialize(
        self,
        payload: INT4BackupPayload,
        *,
        target_dtype: torch.dtype,
        target_device: torch.device | str,
    ) -> torch.Tensor:
        """Unpack and dequantize an int4 payload into target dtype/device."""
        _validate_int4_payload(payload)
        packed = payload.packed.to(device=target_device, copy=False)
        scale = payload.scale.to(device=target_device, copy=False)

        low_nibbles = torch.bitwise_and(packed, 0xF)
        high_nibbles = torch.bitwise_right_shift(packed, 4)
        low_values = _sign_extend_int4_nibbles(low_nibbles)
        high_values = _sign_extend_int4_nibbles(high_nibbles)
        unpacked = torch.stack((low_values, high_values), dim=-1).reshape(
            *packed.shape[:-1],
            int(packed.shape[-1]) * 2,
        )
        head_dim = int(payload.original_shape[-1])
        unpacked = unpacked[..., :head_dim]
        materialized = unpacked.to(torch.float32) * scale.unsqueeze(-1)
        return materialized.to(dtype=target_dtype)


def _validate_kv_block(kv_block: torch.Tensor) -> None:
    if kv_block.ndim != 4 or int(kv_block.shape[0]) != 2:
        raise ValueError(
            "MPR backup codec expects a semantic K/V block shaped "
            "[2, block_size, num_kv_heads, head_dim], got "
            f"{tuple(kv_block.shape)}."
        )


def _validate_int4_payload(payload: INT4BackupPayload) -> None:
    original_shape = tuple(payload.original_shape)
    if len(original_shape) != 4 or int(original_shape[0]) != 2:
        raise ValueError(
            "MPR int4 payload original_shape must be "
            "[2, block_size, num_kv_heads, head_dim], got "
            f"{original_shape}."
        )
    expected_packed_shape = (
        *original_shape[:-1],
        (int(original_shape[-1]) + 1) // 2,
    )
    if tuple(payload.packed.shape) != expected_packed_shape:
        raise ValueError(
            "MPR int4 packed payload shape does not match original_shape, "
            f"expected {expected_packed_shape}, got "
            f"{tuple(payload.packed.shape)}."
        )
    if tuple(payload.scale.shape) != original_shape[:-1]:
        raise ValueError(
            "MPR int4 scale shape must match original_shape without head_dim, "
            f"expected {original_shape[:-1]}, got {tuple(payload.scale.shape)}."
        )


def _sign_extend_int4_nibbles(nibbles: torch.Tensor) -> torch.Tensor:
    values = nibbles.to(torch.int16)
    return torch.where(values >= 8, values - 16, values)
