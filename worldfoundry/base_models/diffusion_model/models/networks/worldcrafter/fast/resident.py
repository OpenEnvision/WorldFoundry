# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
"""Lossless GPU-resident BF16 branch storage.

A materialized BF16 model + reversible packed integer bit-pattern differences.
LoRA and UCPE tensors remain independent. No CPU copies, decompression rounding,
or new floating-point additions enter a denoiser forward. A stage switch updates
shared storage on the current CUDA stream; concurrent branch forwards are not
supported. Both complete checkpoints remain represented on GPU at all times.
"""

from dataclasses import dataclass
import gc
import logging
import time
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

logger = logging.getLogger(__name__)


@triton.jit
def _switch_dense(
    W,
    P,
    MASK,
    PREFIX,
    VALUES,
    N: tl.constexpr,
    B: tl.constexpr,
    BITMAP: tl.constexpr,
    SIGN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    if B == 16:
        d = tl.load(P + i, valid, other=0).to(tl.int32)
    else:
        per = 8 // B
        byte = tl.load(P + i // per, valid, other=0).to(tl.int32)
        code = (byte >> ((i % per) * B)) & ((1 << B) - 1)
        d = (code ^ (1 << (B - 1))) - (1 << (B - 1))
        if BITMAP:
            flags = tl.load(MASK + i // 32, valid, other=0).to(tl.uint32)
            lower = (tl.full((BLOCK,), 1, tl.uint32) << (i % 32)) - 1
            exceptional = ((flags >> (i % 32)) & 1) != 0
            prefix = tl.load(PREFIX + i // 32, valid, other=0).to(tl.int32)
            rank = libdevice.popc((flags & lower).to(tl.int32)).to(tl.int32)
            exc = tl.load(VALUES + prefix + rank, valid & exceptional, other=0).to(
                tl.int32
            )
            d = tl.where(exceptional, exc, d)
    before = tl.load(W + i, valid, other=0).to(tl.int32)
    tl.store(W + i, (before + SIGN * d).to(tl.int16), valid)


@triton.jit
def _switch_sparse(
    W, INDEX, VALUES, N: tl.constexpr, SIGN: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    idx = tl.load(INDEX + i, i < N, other=0).to(tl.int32)
    d = tl.load(VALUES + i, i < N, other=0).to(tl.int32)
    before = tl.load(W + idx, i < N, other=0).to(tl.int32)
    tl.store(W + idx, (before + SIGN * d).to(tl.int16), i < N)


def pack_codes(codes, bits):
    per = 8 // bits
    n = codes.numel()
    pad = (-n) % per
    if pad:
        codes = F.pad(codes, (0, pad))
    shifts = torch.arange(per, device=codes.device, dtype=torch.int32) * bits
    return (codes.reshape(-1, per).int() << shifts).sum(1).to(torch.uint8).contiguous()


def choose_format(d):
    n = d.numel()
    choices = [(n * 2, 16, "dense")]
    if bool((d == 0).all()):
        return (0, 0, "shared")
    for b in [1, 2, 4, 8]:
        count = int(((d < -(1 << (b - 1))) | (d >= (1 << (b - 1)))).sum())
        dense = (n * b + 7) // 8
        choices += [
            (dense + count * 6, b, "sparse"),
            (dense + ((n + 31) // 32) * 8 + count * 2, b, "bitmap"),
        ]
    return min(choices)


@dataclass
class PackedDelta:
    weight: torch.Tensor
    bits: int
    mode: str
    packed: torch.Tensor
    indices: torch.Tensor
    values: torch.Tensor
    bitmap: torch.Tensor
    prefix: torch.Tensor
    name: str = ""

    @property
    def encoded_bytes(self):
        return sum(
            t.numel() * t.element_size()
            for t in [self.packed, self.indices, self.values, self.bitmap, self.prefix]
        )

    def apply(self, sign):
        if self.bits == 0:
            return
        n = self.weight.numel()
        _switch_dense[(triton.cdiv(n, 2048),)](
            self.weight,
            self.packed,
            self.bitmap,
            self.prefix,
            self.values,
            n,
            self.bits,
            self.mode == "bitmap",
            sign,
            2048,
        )
        if self.mode == "sparse" and self.indices.numel():
            _switch_sparse[(triton.cdiv(self.indices.numel(), 256),)](
                self.weight, self.indices, self.values, self.indices.numel(), sign, 256
            )


def encode(left, right, name=""):
    if not (left.dtype == right.dtype == torch.bfloat16 and left.shape == right.shape):
        raise ValueError("Invalid branch weight layout")
    if not (left.is_cuda and left.is_contiguous() and right.is_contiguous()):
        raise ValueError("Invalid branch weight layout")
    right = right.to(left.device)
    d = (
        (
            right.view(torch.int16).flatten().int()
            - left.view(torch.int16).flatten().int()
        )
        .to(torch.int16)
        .int()
    )
    expected_bytes, bits, mode = choose_format(d)
    empty = torch.empty(0, device=left.device, dtype=torch.int32)
    p = PackedDelta(
        left.view(torch.int16).flatten(),
        bits,
        mode,
        empty,
        empty,
        empty,
        empty,
        empty,
        name,
    )
    if bits == 0:
        if not (bool((d == 0).all())):
            raise ValueError("Invalid branch weight layout")
        return p
    if bits == 16:
        p.packed = d.to(torch.int16)
        return p
    exception = (d < -(1 << (bits - 1))) | (d >= (1 << (bits - 1)))
    indices = torch.nonzero(exception).flatten()
    p.values = d[indices].to(torch.int16)
    p.packed = pack_codes(torch.where(exception, 0, d & ((1 << bits) - 1)), bits)
    if mode == "sparse":
        p.indices = indices.to(torch.int32)
    elif mode == "bitmap":
        flags = exception.int()
        pad = (-flags.numel()) % 32
        if pad:
            flags = F.pad(flags, (0, pad))
        flags = flags.reshape(-1, 32)
        shifts = torch.arange(32, device=left.device, dtype=torch.int64)
        p.bitmap = (flags.to(torch.int64) << shifts).sum(1).to(torch.int32)
        counts = flags.sum(1, dtype=torch.int32)
        p.prefix = (counts.cumsum(0) - counts).to(torch.int32)
    else:
        raise ValueError(mode)
    if not (p.encoded_bytes == expected_bytes):
        raise ValueError((name, p.encoded_bytes, expected_bytes))
    return p


class ResidentBranches:
    def __init__(self, early, late, use_graph=True):
        start = time.perf_counter()
        self.plans = []
        self.active = "equal"
        self.switches = 0
        self.graphs = {}
        self.device = next(early.parameters()).device
        early_params = dict(early.named_parameters())
        late_params = dict(late.named_parameters())
        if not (set(early_params) == set(late_params)):
            raise ValueError("Invalid branch weight layout")
        before = torch.cuda.memory_allocated()
        shared_bytes = 0
        independent = []
        for name, left in early_params.items():
            right = late_params[name]
            if (
                ".lora_" in name
                or ".cam_self_attn." in name
                or left.dtype != torch.bfloat16
            ):
                independent.append(name)
                continue
            if not (left.dtype == right.dtype and left.shape == right.shape):
                raise ValueError(name)
            plan = encode(left.detach(), right.detach(), name)
            # Verify reconstruction before sharing the dense parameter storage.
            plan.apply(1)
            if not (
                torch.equal(
                    left.detach().view(torch.int16),
                    right.detach().to(left.device).view(torch.int16),
                )
            ):
                raise ValueError(name)
            plan.apply(-1)
            right.data = (
                left.data
            )  # Both module objects reference one materialized BF16 tensor.
            self.plans.append(plan)
            shared_bytes += left.numel() * left.element_size()
            if len(self.plans) % 100 == 0:
                logger.debug("Packed %d parameter tensors", len(self.plans))
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Compile/warm both directions; exact round trip restores the initial branch.
        self._apply(1)
        self._apply(-1)
        torch.cuda.synchronize()
        if use_graph:
            for sign in [1, -1]:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self._apply(sign)
                self.graphs[sign] = graph
            torch.cuda.synchronize()
        self.report = dict(
            shared_dense_bytes=shared_bytes,
            delta_bytes=sum(p.encoded_bytes for p in self.plans),
            base_parameter_tensors=len(self.plans),
            independent_parameter_tensors=len(independent),
            format_counts={
                mode: sum(p.mode == mode for p in self.plans)
                for mode in ["shared", "dense", "sparse", "bitmap"]
            },
            gpu_allocated_before=before,
            gpu_allocated_after=torch.cuda.memory_allocated(),
            gpu_reserved_after=torch.cuda.memory_reserved(),
            setup_seconds=time.perf_counter() - start,
            gpu_only_switch=True,
            lossless_bf16_bit_patterns=True,
            cuda_graph=use_graph,
            all_reconstructed_base_tensors_bitwise_verified=True,
        )
        logger.info("Prepared resident branches in %.1fs", self.report["setup_seconds"])

    def _apply(self, sign):
        for plan in self.plans:
            plan.apply(sign)

    def switch(self, branch):
        if branch == self.active:
            return
        if branch not in ["equal", "old"]:
            raise ValueError(branch)
        sign = 1 if branch == "old" else -1
        if self.graphs:
            self.graphs[sign].replay()
        else:
            self._apply(sign)
        self.active = branch
        self.switches += 1
