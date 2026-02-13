
"""HyperGLU reference implementation (dense + chunked) with variant support.

This file is intended for research reproduction and benchmarking.

Key additions vs the minimal backbone:
- Configurable attention family: "glu" (HyperGLU) or "relu" (HyperMLP-style).
- Configurable attention normalization: "l2", "out_scale", or "none".
- Optional KV causal depthwise conv, optional RoPE, optional QK/VO gates.
- Supports both offset (skewed) layout and identity (left/torch.tril-style) layout.
"""

import argparse
import copy
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


class CausalConv1d(nn.Module):
    def __init__(self, channels, kernel_size, activation=None, layout="BLC", bias=False, init_std=0.02):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.layout = layout
        self.activation = activation
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, groups=channels, bias=bias)

        nn.init.normal_(self.conv.weight, mean=0.0, std=init_std)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layout != "BLC":
            raise ValueError("This minimal CausalConv1d only supports layout='BLC'.")
        x = x.transpose(1, 2)                 # (B,C,T)
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv(x)                      # (B,C,T)
        x = x.transpose(1, 2)                 # (B,T,C)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SimpleRoPE(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # q,k: (B,T,H,d)
        B, T, H, d = q.size()
        dev = q.device
        t = torch.arange(T, device=dev, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("t,f->tf", t, self.inv_freq)  # (T,d/2)
        emb = torch.cat((freqs, freqs), dim=-1)            # (T,d)

        cos = emb.cos()[None, :, None, :]  # (1,T,1,d)
        sin = emb.sin()[None, :, None, :]

        def rot(x):
            x1, x2 = x[..., ::2], x[..., 1::2]
            xr = torch.stack((-x2, x1), dim=-1).reshape_as(x)
            return xr

        q_out = q * cos + rot(q) * sin
        k_out = k * cos + rot(k) * sin
        return q_out, k_out


class RowOffsetAligner(nn.Module):
    """Causal left<->offset alignment using pad + as_strided.

    Offset layout (skewed) stores only the causal triangle in a right-aligned form.
    """

    def __init__(self, max_L: int, impl_train: str = "padview", impl_eval: str = "padview"):
        super().__init__()
        self.L = int(max_L)

        self.impl_train = str(impl_train)
        self.impl_eval = str(impl_eval)
        if self.impl_train != "padview" or self.impl_eval != "padview":
            raise ValueError("RowOffsetAligner: only impl_train/impl_eval='padview' is supported.")

        sqrt_idx = torch.sqrt(torch.arange(1, self.L + 1, dtype=torch.float32)).view(1, 1, self.L, 1)
        self.register_buffer("sqrt_idx", sqrt_idx, persistent=False)

    def to_offset_rows(self, X_left_rows: torch.Tensor, *, row_start: int) -> torch.Tensor:
        # X_left_rows: (B,h,M,T)
        B, h, M, T = X_left_rows.shape
        assert 0 <= row_start <= T
        assert row_start + M <= T
        assert T <= self.L

        X_pad = F.pad(X_left_rows, (T - 1, 0))  # (B,h,M,2T-1)
        s0, s1, s2, s3 = X_pad.stride()
        storage_offset = row_start * s3
        Y = X_pad.as_strided(
            size=(B, h, M, T),
            stride=(s0, s1, s2 + s3, s3),
            storage_offset=storage_offset,
        )
        return Y

    def from_offset_rows(self, X_off_rows: torch.Tensor, *, row_start: int) -> torch.Tensor:
        # X_off_rows: (B,h,M,T)
        B, h, M, T = X_off_rows.shape
        assert 0 <= row_start <= T
        assert row_start + M <= T
        assert T <= self.L

        X_pad = F.pad(X_off_rows, (0, T - 1))  # (B,h,M,2T-1)
        base = (T - 1 - row_start)
        s0, s1, s2, s3 = X_pad.stride()
        storage_offset = base * s3
        Y = X_pad.as_strided(
            size=(B, h, M, T),
            stride=(s0, s1, s2 - s3, s3),
            storage_offset=storage_offset,
        )
        return Y

    def to_offset(self, X_left: torch.Tensor) -> torch.Tensor:
        B, h, T, T2 = X_left.shape
        assert T == T2
        return self.to_offset_rows(X_left, row_start=0)

    def from_offset(self, X_off: torch.Tensor) -> torch.Tensor:
        B, h, T, T2 = X_off.shape
        assert T == T2
        return self.from_offset_rows(X_off, row_start=0)

    def sqrt_rows(self, *, row_start: int, M: int) -> torch.Tensor:
        return self.sqrt_idx[:, :, row_start: row_start + M, :]


class IdentityAligner(nn.Module):
    """No-op aligner for left (standard) layout.

    This aligner is also used to implement the "non-offset" variants, where causality is enforced
    by a left-layout causal mask (torch.tril-style).
    """

    def __init__(self, max_L: int):
        super().__init__()
        self.L = int(max_L)

        sqrt_idx = torch.sqrt(torch.arange(1, self.L + 1, dtype=torch.float32)).view(1, 1, self.L, 1)
        self.register_buffer("sqrt_idx", sqrt_idx, persistent=False)

    def to_offset_rows(self, X_left_rows: torch.Tensor, *, row_start: int) -> torch.Tensor:
        B, h, M, T = X_left_rows.shape
        assert T <= self.L
        assert 0 <= row_start
        assert row_start + M <= self.L
        return X_left_rows

    def from_offset_rows(self, X_off_rows: torch.Tensor, *, row_start: int) -> torch.Tensor:
        B, h, M, T = X_off_rows.shape
        assert T <= self.L
        assert 0 <= row_start
        assert row_start + M <= self.L
        return X_off_rows

    def to_offset(self, X_left: torch.Tensor) -> torch.Tensor:
        return X_left

    def from_offset(self, X_off: torch.Tensor) -> torch.Tensor:
        return X_off

    def sqrt_rows(self, *, row_start: int, M: int) -> torch.Tensor:
        return self.sqrt_idx[:, :, row_start: row_start + M, :]


class LowRankRightMix(nn.Module):
    """Stores low-rank parameters (U, Vt, optional d). The forward method is a reference implementation."""

    def __init__(
        self,
        L: int,
        rank: int,
        heads: int,
        per_head: bool = True,
        use_diag: bool = True,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.L = int(L)
        self.rank = int(rank)
        self.heads = int(heads)
        self.per_head = bool(per_head)
        self.use_diag = bool(use_diag)

        if per_head:
            self.U = nn.Parameter(torch.randn(heads, self.L, rank) * init_std)   # (h,L,r)
            self.Vt = nn.Parameter(torch.randn(heads, rank, self.L) * init_std)  # (h,r,L)
            if use_diag:
                self.d = nn.Parameter(torch.zeros(heads, self.L))                # (h,L)
        else:
            # This code path is kept for completeness; the fused path assumes per_head=True.
            self.U = nn.Parameter(torch.randn(self.L, rank) * init_std)
            self.Vt = nn.Parameter(torch.randn(rank, self.L) * init_std)
            if use_diag:
                self.d = nn.Parameter(torch.zeros(self.L))

    def forward(self, X: torch.Tensor, control=None) -> torch.Tensor:
        # Reference implementation (the main path uses the fused autograd Function).
        # X: (B,h,M,T) or (B,h,T,T) depending on caller.
        B, h, M, T = X.shape
        assert h == self.heads
        assert T <= self.L
        s = self.L - T

        if not self.per_head:
            raise NotImplementedError("LowRankRightMix: only per_head=True is supported in the fused path.")

        U = self.U[:, s:, :]      # (h,T,r)
        Vt = self.Vt[:, :, s:]    # (h,r,T)

        # reference: H = X@U, Z = H*control, Y = Z@Vt (+diag)
        H = torch.einsum("bhmt,htr->bhmr", X, U)           # (B,h,M,r)
        if control is not None:
            H = H * control
        Y = torch.einsum("bhmr,hrt->bhmt", H, Vt)          # (B,h,M,T)

        if self.use_diag:
            d = self.d[:, s:]                               # (h,T)
            Y = Y + X * d[None, :, None, :]

        return Y


def _causal_mask_offset_rows(M: int, T: int, row_start: int, device) -> torch.Tensor:
    """
    Offset layout causal mask:
      global row r = row_start + m
      valid <=> t + r >= T - 1
    return: (M,T) bool
    """
    r = (row_start + torch.arange(M, device=device, dtype=torch.int32)).view(M, 1)  # (M,1)
    t = torch.arange(T, device=device, dtype=torch.int32).view(1, T)               # (1,T)
    return (t + r) >= (T - 1)


def _causal_mask_left_rows(M: int, T: int, row_start: int, device) -> torch.Tensor:
    """
    Left (standard) layout causal mask:
      global row r = row_start + m
      valid <=> t <= r
    return: (M,T) bool
    """
    r = (row_start + torch.arange(M, device=device, dtype=torch.int32)).view(M, 1)  # (M,1)
    t = torch.arange(T, device=device, dtype=torch.int32).view(1, T)               # (1,T)
    return t <= r


def _causal_mask_rows(M: int, T: int, row_start: int, device, kind: int) -> torch.Tensor:
    """Unified causal mask for different layouts.

    kind:
      0 -> offset layout mask
      1 -> left layout mask
    """
    if kind == 0:
        return _causal_mask_offset_rows(M, T, row_start, device)
    if kind == 1:
        return _causal_mask_left_rows(M, T, row_start, device)
    raise ValueError(f"Unknown mask kind={kind}. Expected 0(offset) or 1(left).")


class FusedLowRankResidualMixFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        X: torch.Tensor,                   # (B,h,M,T)
        control: Optional[torch.Tensor],   # (B,h,M,r) or None
        U_full: torch.Tensor,              # (h,L,r)
        Vt_full: torch.Tensor,             # (h,r,L)
        d_full: torch.Tensor,              # (h,L) or empty
        row_start: int,
        L: int,
        use_diag: bool,
        shortcut: bool,
        do_masking: bool,
        mask_kind: int,                    # 0=offset, 1=left
    ):
        assert X.dim() == 4
        B, h, M, T = X.shape
        L = int(L)
        s = L - T

        U_s = U_full[:, s:, :]          # (h,T,r)
        Vt_s = Vt_full[:, :, s:]        # (h,r,T)
        d_s = None
        if use_diag and d_full.numel() > 0:
            d_s = d_full[:, s:]         # (h,T)

        H = torch.einsum("bhmt,htr->bhmr", X, U_s)     # (B,h,M,r)
        if control is not None:
            Z = H * control
        else:
            Z = H
        Y_lr = torch.einsum("bhmr,hrt->bhmt", Z, Vt_s)     # (B,h,M,T)

        # epilogue: shortcut + diag + causal mask (pure torch)
        Y = Y_lr
        if shortcut:
            Y = Y + X
        if use_diag and d_s is not None:
            Y = Y + X * d_s[None, :, None, :]
        if do_masking:
            mask = _causal_mask_rows(M, T, int(row_start), device=X.device, kind=int(mask_kind))  # (M,T)
            Y = Y * mask.to(dtype=Y.dtype).view(1, 1, M, T)

        # save for backward (no mask saved)
        ctx.save_for_backward(
            X,
            control if control is not None else torch.tensor([], device=X.device),
            H, Z, U_s, Vt_s,
            d_s if (use_diag and d_s is not None) else torch.tensor([], device=X.device),
        )
        ctx.L = L
        ctx.s = s
        ctx.row_start = int(row_start)
        ctx.use_diag = bool(use_diag)
        ctx.shortcut = bool(shortcut)
        ctx.do_masking = bool(do_masking)
        ctx.has_control = control is not None
        ctx.U_shape = tuple(U_full.shape)
        ctx.Vt_shape = tuple(Vt_full.shape)
        ctx.d_shape = tuple(d_full.shape) if d_full.numel() > 0 else None
        ctx.mask_kind = int(mask_kind)

        return Y

    @staticmethod
    def backward(ctx, grad_Y: torch.Tensor):
        (X,
         control_saved,
         H, Z, U_s, Vt_s,
         d_s_saved) = ctx.saved_tensors

        B, h, M, T = X.shape
        s = ctx.s
        row_start = ctx.row_start
        use_diag = ctx.use_diag
        shortcut = ctx.shortcut
        do_masking = ctx.do_masking
        has_control = ctx.has_control
        mask_kind = int(ctx.mask_kind)

        control = control_saved if has_control else None

        # apply causal mask to upstream grad (recompute; no saved mask tensor)
        if do_masking:
            mask = _causal_mask_rows(M, T, row_start, device=grad_Y.device, kind=mask_kind)  # (M,T)
            dY_core = grad_Y * mask.to(dtype=grad_Y.dtype).view(1, 1, M, T)
        else:
            dY_core = grad_Y

        # shortcut & diag grads
        dX = torch.zeros_like(X)
        dd_s = None
        if shortcut:
            dX = dX + dY_core
        if use_diag and d_s_saved.numel() > 0:
            d_s = d_s_saved                       # (h,T)
            dX = dX + dY_core * d_s[None, :, None, :]
            dd_s = (dY_core * X).sum(dim=(0, 2))   # (h,T)

        dZ = torch.einsum("bhmt,hrt->bhmr", dY_core, Vt_s)    # (B,h,M,r)
        dVt_s = torch.einsum("bhmr,bhmt->hrt", Z, dY_core)    # (h,r,T)

        if control is not None:
            dH = dZ * control
            dcontrol = dZ * H
        else:
            dH = dZ
            dcontrol = None

        dX_lr = torch.einsum("bhmr,htr->bhmt", dH, U_s)       # (B,h,M,T)
        dX = dX + dX_lr

        dU_s = torch.einsum("bhmt,bhmr->htr", X, dH)          # (h,T,r)

        dU_full = torch.zeros(ctx.U_shape, device=X.device, dtype=X.dtype)
        dVt_full = torch.zeros(ctx.Vt_shape, device=X.device, dtype=X.dtype)
        dd_full = None
        if use_diag and (ctx.d_shape is not None) and (dd_s is not None):
            dd_full = torch.zeros(ctx.d_shape, device=X.device, dtype=X.dtype)

        dU_full[:, s:, :] = dU_s
        dVt_full[:, :, s:] = dVt_s
        if dd_full is not None:
            dd_full[:, s:] = dd_s

        dcontrol_out = dcontrol if has_control else None

        # backward outputs must align with forward inputs:
        # (X, control, U_full, Vt_full, d_full, row_start, L, use_diag, shortcut, do_masking, mask_kind)
        return dX, dcontrol_out, dU_full, dVt_full, dd_full, None, None, None, None, None, None


class OffsetLowRankResidualMix(nn.Module):
    """Low-rank mixing with optional residual and masking, applied in a configurable layout.

    mask_kind:
      - "offset": skewed offset layout (matches RowOffsetAligner + offset mask)
      - "left":   standard left layout (torch.tril-style mask)
    """

    def __init__(
        self,
        L: int,
        rank: int,
        heads: int,
        per_head: bool = True,
        use_diag: bool = True,
        init_std: float = 0.02,
        shortcut: bool = True,
        mask_kind: str = "offset",
    ):
        super().__init__()
        self.L = int(L)
        self.mix = LowRankRightMix(
            L=self.L, rank=rank, heads=heads, per_head=per_head,
            use_diag=use_diag, init_std=init_std
        )
        self.shortcut = bool(shortcut)

        mask_kind = str(mask_kind)
        if mask_kind not in ("offset", "left"):
            raise ValueError(f"mask_kind={mask_kind!r} must be 'offset' or 'left'.")
        self.mask_kind = mask_kind
        self.mask_kind_code = 0 if mask_kind == "offset" else 1

    def _forward_ref(self, X: torch.Tensor, control=None, do_masking=True, *, row_start: int = 0) -> torch.Tensor:
        B, h, M, T = X.shape

        Y_core = self.mix(X, control)  # low-rank (+diag inside)
        Y = (X + Y_core) if self.shortcut else Y_core

        if do_masking:
            mask = _causal_mask_rows(M, T, int(row_start), device=X.device, kind=self.mask_kind_code)  # (M,T)
            Y = Y * mask.to(dtype=Y.dtype).view(1, 1, M, T)

        return Y

    def forward(self, X: torch.Tensor, control=None, do_masking=True, *, row_start: int = 0) -> torch.Tensor:
        if not self.mix.per_head:
            return self._forward_ref(X, control, do_masking, row_start=row_start)

        device = X.device
        dtype = X.dtype

        if self.mix.use_diag and hasattr(self.mix, "d"):
            d_full = self.mix.d
        else:
            d_full = torch.empty(0, device=device, dtype=dtype)

        return FusedLowRankResidualMixFn.apply(
            X,
            control,
            self.mix.U,
            self.mix.Vt,
            d_full,
            int(row_start),
            self.L,
            bool(self.mix.use_diag),
            bool(self.shortcut),
            bool(do_masking),
            int(self.mask_kind_code),
        )


@dataclass
class HyperGLUConfig:
    """HyperGLU configuration.

    Extended to cover the ReLU/GLU variants in hypermlp_exps.py (ignoring softmax variants).
    """

    # core dims
    n_embd: int
    n_head: int
    block_size: int
    bias: bool = False
    n_layer: int = 24

    # projection sizes
    # If None: qk_rank defaults to n_embd // 4, vo_rank defaults to n_embd
    qk_rank: Optional[int] = None
    vo_rank: Optional[int] = None

    # conv
    conv_kernel_size: int = 4
    use_kv_conv: bool = True  # True: use depthwise causal conv on K/V inputs

    # RoPE
    use_rope: bool = False
    rope_base: float = 10000.0

    # attention family
    # - "glu": split heads into 2h and apply softplus(x2)*relu(x1)
    # - "relu": l2norm+relu attention (HyperMLP-style)
    attn_kind: str = "glu"  # "glu" | "relu"

    # attention normalization strategy
    # - "l2":        apply F.normalize(..., dim=-1) before activation
    # - "out_scale": compute inv_norm and apply it to the output (backbone behavior)
    # - "none":      no normalization
    attn_norm: str = "out_scale"  # "l2" | "out_scale" | "none"

    # gating / numerics
    gate_act: str = "sigmoid"   # "sigmoid" or "none"
    norm_eps: float = 1e-12

    # optional gates (dtqk/dtvo)
    use_qk_gate: bool = True
    use_vo_gate: bool = True

    # implementation (forward path)
    # One of: dense / chunked
    impl_default: str = "chunked"
    chunk_T: int = 1024

    # P1 (op1)
    use_op1: bool = True
    op1_rank: int = 16
    op1_per_head: bool = True
    op1_use_diag: bool = True
    op1_init_std: float = 0.02
    op1_shortcut: bool = True
    op1_do_masking: bool = True

    # P2 (op2)
    use_op2: bool = True
    op2_rank: int = 16
    op2_per_head: bool = True
    op2_use_diag: bool = True
    op2_init_std: float = 0.02
    op2_shortcut: bool = True
    op2_do_masking: bool = True

    # causal / masking (global)
    # If False: run in non-causal (bidirectional) mode.
    do_masking: bool = True

    # layout kind
    # - "offset":   RowOffsetAligner (skewed offset layout; efficient for causal)
    # - "identity": IdentityAligner (standard left layout; also used for non-offset variants)
    aligner_kind: str = "offset"  # "offset" | "identity"

    # aligner implementations
    # Kept for compatibility; only "padview" is supported here.
    aligner_impl_train: str = "padview"
    aligner_impl_eval: str = "padview"

    def __post_init__(self):
        # default ranks
        if self.qk_rank is None:
            self.qk_rank = self.n_embd // 4
        if self.vo_rank is None:
            self.vo_rank = self.n_embd

        allowed_impl = {"dense", "chunked"}
        if self.impl_default not in allowed_impl:
            raise ValueError(f"impl_default={self.impl_default!r} must be one of {sorted(allowed_impl)}")

        if self.chunk_T <= 0:
            raise ValueError(f"chunk_T must be > 0, got {self.chunk_T}")

        if self.conv_kernel_size <= 0:
            raise ValueError(f"conv_kernel_size must be > 0, got {self.conv_kernel_size}")

        if self.gate_act not in ("sigmoid", "none"):
            raise ValueError(f"gate_act must be 'sigmoid' or 'none', got {self.gate_act!r}")

        if self.attn_kind not in ("glu", "relu"):
            raise ValueError(f"attn_kind must be 'glu' or 'relu', got {self.attn_kind!r}")

        if self.attn_norm not in ("l2", "out_scale", "none"):
            raise ValueError(f"attn_norm must be in {{'l2','out_scale','none'}}, got {self.attn_norm!r}")

        h = int(self.n_head)

        # qk per-head dim depends on attn_kind
        if self.attn_kind == "glu":
            if self.qk_rank % (2 * h) != 0:
                raise ValueError(f"qk_rank={self.qk_rank} must be divisible by 2*n_head={2*h} for attn_kind='glu'.")
        else:  # "relu"
            if self.qk_rank % h != 0:
                raise ValueError(f"qk_rank={self.qk_rank} must be divisible by n_head={h} for attn_kind='relu'.")

        if self.vo_rank % h != 0:
            raise ValueError(f"vo_rank={self.vo_rank} must be divisible by n_head={h}.")

        if self.op1_rank <= 0 or self.op2_rank <= 0:
            raise ValueError(f"op1_rank/op2_rank must be > 0, got {self.op1_rank}/{self.op2_rank}")

        allowed_aligner_kind = {"offset", "identity"}
        if self.aligner_kind not in allowed_aligner_kind:
            raise ValueError(f"aligner_kind={self.aligner_kind} must be in {sorted(allowed_aligner_kind)}")

        # RoPE requires an even per-head dimension
        if self.use_rope:
            qk_heads = (2 * h) if (self.attn_kind == "glu") else h
            d = int(self.qk_rank) // qk_heads
            if d % 2 != 0:
                raise ValueError(f"RoPE requires even per-head dim, got {d} (qk_rank={self.qk_rank}, heads={qk_heads}).")

        if (not self.do_masking) and (self.aligner_kind == "offset"):
            warnings.warn(
                "do_masking=False with aligner_kind='offset' keeps offset alignment (drops the upper triangle / "
                "introduces implicit zeros). This is NOT a true non-causal full-matrix mode. "
                "For non-causal full matrix: use aligner_kind='identity'.",
                RuntimeWarning,
            )

        allowed_aligner_impl = {"padview"}
        if self.aligner_impl_train not in allowed_aligner_impl:
            raise ValueError(
                f"aligner_impl_train={self.aligner_impl_train!r} must be in {sorted(allowed_aligner_impl)}"
            )
        if self.aligner_impl_eval not in allowed_aligner_impl:
            raise ValueError(
                f"aligner_impl_eval={self.aligner_impl_eval!r} must be in {sorted(allowed_aligner_impl)}"
            )


class HyperGLU(nn.Module):
    NORM_EPS: float = 1e-12

    def __init__(self, config: HyperGLUConfig):
        super().__init__()

        self.cfg = config

        self.n_head = int(config.n_head)
        self.n_embd = int(config.n_embd)

        # Variant switches
        self.attn_kind = str(config.attn_kind)     # "glu" | "relu"
        self.attn_norm = str(config.attn_norm)     # "l2" | "out_scale" | "none"
        self.qk_heads = (2 * self.n_head) if (self.attn_kind == "glu") else self.n_head

        self.use_op1 = bool(config.use_op1)
        self.use_op2 = bool(config.use_op2)

        self.chunk_T = int(config.chunk_T)
        self.impl_default = str(config.impl_default)

        # Numerics
        self.NORM_EPS = float(config.norm_eps)

        # Gate activation (avoid string checks inside hot loops)
        if config.gate_act == "sigmoid":
            self._gate_act_kind = 0
        elif config.gate_act == "none":
            self._gate_act_kind = 1
        else:
            raise ValueError(f"Unknown gate_act={config.gate_act!r}")

        # Ranks
        qk_rank = int(config.qk_rank)
        vo_rank = int(config.vo_rank)
        self.qk_rank = qk_rank
        self.vo_rank = vo_rank

        # Projections
        self.q = nn.Linear(self.n_embd, qk_rank, bias=config.bias)
        self.k = nn.Linear(self.n_embd, qk_rank, bias=config.bias)
        self.v = nn.Linear(self.n_embd, vo_rank, bias=config.bias)
        self.c_proj = nn.Linear(vo_rank, self.n_embd, bias=config.bias)

        # Convs (optional)
        k_conv = int(config.conv_kernel_size)
        if bool(config.use_kv_conv):
            self.conv1 = CausalConv1d(channels=self.n_embd, kernel_size=k_conv, activation=None, layout="BLC")
            self.conv2 = CausalConv1d(channels=self.n_embd, kernel_size=k_conv, activation=None, layout="BLC")
        else:
            self.conv1 = nn.Identity()
            self.conv2 = nn.Identity()

        # RoPE (optional)
        d_qk = qk_rank // self.qk_heads
        self.rope = SimpleRoPE(dim=d_qk, base=float(config.rope_base)) if bool(config.use_rope) else None

        # Aligner / layout kind
        if config.aligner_kind == "offset":
            self.aligner = RowOffsetAligner(
                config.block_size,
                impl_train=str(config.aligner_impl_train),
                impl_eval=str(config.aligner_impl_eval),
            )
            mask_kind = "offset"
        elif config.aligner_kind == "identity":
            self.aligner = IdentityAligner(config.block_size)
            mask_kind = "left"
        else:
            raise ValueError(f"Unknown aligner_kind={config.aligner_kind}. Use 'offset' or 'identity'.")

        # P1 (op1): heads depend on attn_kind (glu uses 2h, relu uses h)
        if self.use_op1:
            self.op = OffsetLowRankResidualMix(
                L=config.block_size,
                rank=int(config.op1_rank),
                heads=self.qk_heads,
                per_head=bool(config.op1_per_head),
                use_diag=bool(config.op1_use_diag),
                init_std=float(config.op1_init_std),
                shortcut=bool(config.op1_shortcut),
                mask_kind=mask_kind,
            )
            self.control1 = nn.Linear(self.n_embd, self.qk_heads * int(config.op1_rank), bias=True)
        else:
            self.op = None
            self.control1 = None

        # P2 (op2): always uses h heads (post-nonlinearity)
        if self.use_op2:
            self.op2 = OffsetLowRankResidualMix(
                L=config.block_size,
                rank=int(config.op2_rank),
                heads=self.n_head,
                per_head=bool(config.op2_per_head),
                use_diag=bool(config.op2_use_diag),
                init_std=float(config.op2_init_std),
                shortcut=bool(config.op2_shortcut),
                mask_kind=mask_kind,
            )
            self.control2 = nn.Linear(self.n_embd, self.n_head * int(config.op2_rank), bias=True)
        else:
            self.op2 = None
            self.control2 = None

        # Optional dt gates
        self.left_gate = nn.Linear(self.n_embd, qk_rank, bias=True) if bool(config.use_qk_gate) else None
        self.right_gate = nn.Linear(self.n_embd, vo_rank, bias=True) if bool(config.use_vo_gate) else None

    def reset_parameters(self):
        """Reinitialize only the custom low-rank parameters (U/Vt). Other modules follow the outer initializer."""
        import torch
        import torch.nn as nn

        def _safe_normal_(p: torch.Tensor, std: float):
            try:
                nn.init.normal_(p, mean=0.0, std=float(std))
            except Exception:
                return

        with torch.no_grad():
            if getattr(self, "use_op1", False) and (getattr(self, "op", None) is not None):
                mix = self.op.mix  # LowRankRightMix
                std1 = float(getattr(self.cfg, "op1_init_std", 0.02))
                _safe_normal_(mix.U, std=std1)
                _safe_normal_(mix.Vt, std=std1)

            if getattr(self, "use_op2", False) and (getattr(self, "op2", None) is not None):
                mix = self.op2.mix
                std2 = float(getattr(self.cfg, "op2_init_std", 0.02))
                _safe_normal_(mix.U, std=std2)
                _safe_normal_(mix.Vt, std=std2)

    def act(self, x):
        if x is None:
            return None
        if self._gate_act_kind == 0:
            return torch.sigmoid(x)
        return x

    def _maybe_mask_left_layout(self, X_rows: torch.Tensor, *, row_start: int) -> torch.Tensor:
        """Enforce causality in left layout before any right-mixing.

        Offset layout already represents only the causal triangle, so no pre-mask is needed there.
        """
        if (not self.cfg.do_masking) or (self.cfg.aligner_kind != "identity"):
            return X_rows
        B, h, M, T = X_rows.shape
        mask = _causal_mask_left_rows(M, T, int(row_start), device=X_rows.device)  # (M,T)
        return X_rows * mask.to(dtype=X_rows.dtype).view(1, 1, M, T)

    def _precompute(self, x: torch.Tensor, attention_mask=None):
        B, T, _ = x.shape
        assert T <= self.aligner.L

        # controls
        control1 = (
            self.control1(x).view(B, T, self.qk_heads, -1).transpose(1, 2)
            if self.use_op1 else None
        )
        control2 = (
            self.control2(x).view(B, T, self.n_head, -1).transpose(1, 2)
            if self.use_op2 else None
        )

        # dt gates (optional)
        dtqk = self.left_gate(x).view(B, T, self.qk_heads, -1) if (self.left_gate is not None) else None
        dtvo = self.right_gate(x).view(B, T, self.n_head, -1) if (self.right_gate is not None) else None

        control1, control2, dtqk, dtvo = map(self.act, (control1, control2, dtqk, dtvo))

        # KV conv (optional)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x)

        # q,k
        q = self.q(x).view(B, T, self.qk_heads, -1)
        k = self.k(x_conv1).view(B, T, self.qk_heads, -1)

        # RoPE (optional) is applied before dt gating
        if self.rope is not None:
            q, k = self.rope(q, k)

        if dtqk is not None:
            q = q * dtqk

        # attention_mask support (optional)
        if attention_mask is not None:
            m = attention_mask.to(x.dtype).unsqueeze(-1).unsqueeze(-1)  # (B,T,1,1)
            k = k * m

        k = k.permute(0, 2, 3, 1).contiguous()           # (B,heads,d,T)
        xl = q.permute(0, 2, 1, 3).contiguous()          # (B,heads,T,d)

        # v
        v = self.v(x_conv2).view(B, T, self.n_head, -1)
        if attention_mask is not None:
            v = v * m
        v = v.permute(0, 2, 1, 3).contiguous()           # (B,h,T,dv)

        return control1, control2, dtvo, xl, k, v

    def _apply_attn_glu(self, X: torch.Tensor):
        """GLU nonlinearity on (B,2h,*,T). Returns (base, inv_norm or None)."""
        x1, x2 = torch.chunk(X, 2, dim=1)  # (B,h,*,T) each

        inv_norm = None
        if self.attn_norm == "out_scale":
            den = x1.norm(dim=-1, keepdim=True).clamp_min(self.NORM_EPS)
            inv_norm = den.reciprocal()
            x1_act = F.relu(x1)
        elif self.attn_norm == "l2":
            x1n = F.normalize(x1, dim=-1, eps=self.NORM_EPS)
            x1_act = F.relu(x1n)
        elif self.attn_norm == "none":
            x1_act = F.relu(x1)
        else:
            raise ValueError(f"Unknown attn_norm={self.attn_norm!r}")

        base = F.softplus(x2) * x1_act
        return base, inv_norm

    def _apply_attn_relu(self, X: torch.Tensor):
        """ReLU attention on (B,h,*,T). Returns (Y, inv_norm or None)."""
        inv_norm = None
        if self.attn_norm == "out_scale":
            den = X.norm(dim=-1, keepdim=True).clamp_min(self.NORM_EPS)
            inv_norm = den.reciprocal()
            Y = F.relu(X)
        elif self.attn_norm == "l2":
            Y = F.normalize(X, dim=-1, eps=self.NORM_EPS)
            Y = F.relu(Y)
        elif self.attn_norm == "none":
            Y = F.relu(X)
        else:
            raise ValueError(f"Unknown attn_norm={self.attn_norm!r}")
        return Y, inv_norm

    def _forward_dense(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        B, T, _ = x.size()
        control1, control2, dtvo, xl, k, v = self._precompute(x, attention_mask=attention_mask)

        xlx_left = torch.matmul(xl, k)                         # (B,qk_heads,T,T)
        xlx_left = self._maybe_mask_left_layout(xlx_left, row_start=0)

        X_layout = self.aligner.to_offset(xlx_left)            # (B,qk_heads,T,T) or identity

        if self.use_op1:
            X_layout = self.op(
                X_layout,
                control=control1,
                do_masking=(self.cfg.do_masking and self.cfg.op1_do_masking),
                row_start=0,
            )

        if self.attn_kind == "glu":
            base_layout, inv_norm = self._apply_attn_glu(X_layout)      # (B,h,T,T)
            if self.use_op2:
                base_layout = self.op2(
                    base_layout,
                    control=control2,
                    do_masking=(self.cfg.do_masking and self.cfg.op2_do_masking),
                    row_start=0,
                )
            A_left = self.aligner.from_offset(base_layout)              # (B,h,T,T)
            out = torch.matmul(A_left, v)                               # (B,h,T,dv)
            if inv_norm is not None:
                out = out * inv_norm
        elif self.attn_kind == "relu":
            Y_layout, inv_norm = self._apply_attn_relu(X_layout)        # (B,h,T,T)
            if self.use_op2:
                Y_layout = self.op2(
                    Y_layout,
                    control=control2,
                    do_masking=(self.cfg.do_masking and self.cfg.op2_do_masking),
                    row_start=0,
                )
            A_left = self.aligner.from_offset(Y_layout)
            out = torch.matmul(A_left, v)
            if inv_norm is not None:
                out = out * inv_norm
        else:
            raise ValueError(f"Unknown attn_kind={self.attn_kind!r}")

        out_t = out.transpose(1, 2)                             # (B,T,h,dv)
        if dtvo is not None:
            out_t = out_t * dtvo
        pre = out_t.reshape(B, T, -1)
        y = self.c_proj(pre)
        return y

    def _forward_chunked(self, x: torch.Tensor, *, chunk_T: int, attention_mask=None) -> torch.Tensor:
        B, T, _ = x.size()
        assert chunk_T > 0
        control1, control2, dtvo, xl, k, v = self._precompute(x, attention_mask=attention_mask)

        pre = x.new_empty((B, T, self.vo_rank))

        for i0 in range(0, T, chunk_T):
            i1 = min(T, i0 + chunk_T)
            Mi = i1 - i0

            # Prefix pruning is only valid in offset layout (right-aligned representation).
            # In identity/left layout we keep Ti=T to keep parameter slicing consistent across blocks.
            if self.cfg.do_masking and (self.cfg.aligner_kind == "offset"):
                Ti = i1
            else:
                Ti = T

            xl_block = xl[:, :, i0:i1, :]                   # (B,qk_heads,Mi,dk)
            k_pref = k[:, :, :, :Ti]                        # (B,qk_heads,dk,Ti)
            xlx_left_rows = torch.matmul(xl_block, k_pref)  # (B,qk_heads,Mi,Ti)
            xlx_left_rows = self._maybe_mask_left_layout(xlx_left_rows, row_start=i0)

            X_rows = self.aligner.to_offset_rows(xlx_left_rows, row_start=i0)

            if self.use_op1:
                c1_blk = control1[:, :, i0:i1, :] if control1 is not None else None
                X_rows = self.op(
                    X_rows,
                    control=c1_blk,
                    do_masking=(self.cfg.do_masking and self.cfg.op1_do_masking),
                    row_start=i0,
                )

            if self.attn_kind == "glu":
                base_rows, inv_norm = self._apply_attn_glu(X_rows)          # (B,h,Mi,Ti)
                if self.use_op2:
                    c2_blk = control2[:, :, i0:i1, :] if control2 is not None else None
                    base_rows = self.op2(
                        base_rows,
                        control=c2_blk,
                        do_masking=(self.cfg.do_masking and self.cfg.op2_do_masking),
                        row_start=i0,
                    )
                A_left_rows = self.aligner.from_offset_rows(base_rows, row_start=i0)
                v_pref = v[:, :, :Ti, :]                                    # (B,h,Ti,dv)
                out_blk = torch.matmul(A_left_rows, v_pref)                 # (B,h,Mi,dv)
                if inv_norm is not None:
                    out_blk = out_blk * inv_norm
            elif self.attn_kind == "relu":
                Y_rows, inv_norm = self._apply_attn_relu(X_rows)            # (B,h,Mi,Ti)
                if self.use_op2:
                    c2_blk = control2[:, :, i0:i1, :] if control2 is not None else None
                    Y_rows = self.op2(
                        Y_rows,
                        control=c2_blk,
                        do_masking=(self.cfg.do_masking and self.cfg.op2_do_masking),
                        row_start=i0,
                    )
                A_left_rows = self.aligner.from_offset_rows(Y_rows, row_start=i0)
                v_pref = v[:, :, :Ti, :]
                out_blk = torch.matmul(A_left_rows, v_pref)
                if inv_norm is not None:
                    out_blk = out_blk * inv_norm
            else:
                raise ValueError(f"Unknown attn_kind={self.attn_kind!r}")

            out_blk_t = out_blk.transpose(1, 2)                              # (B,Mi,h,dv)
            if dtvo is not None:
                dtvo_blk = dtvo[:, i0:i1, :, :]                              # (B,Mi,h,dv)
                out_blk_t = out_blk_t * dtvo_blk

            pre[:, i0:i1, :] = out_blk_t.reshape(B, Mi, -1)

        y = self.c_proj(pre)
        return y

    def forward(
        self,
        x: torch.Tensor,
        *,
        impl: Optional[str] = None,
        chunk_T: Optional[int] = None,
        attention_mask=None,
    ) -> torch.Tensor:
        which = self.impl_default if impl is None else str(impl)
        ct = self.chunk_T if chunk_T is None else int(chunk_T)

        if which == "dense":
            return self._forward_dense(x, attention_mask=attention_mask)
        if which == "chunked":
            return self._forward_chunked(x, chunk_T=ct, attention_mask=attention_mask)
        raise ValueError(f"Unknown impl={which}. Use one of: dense / chunked")


# compare utilities
@dataclass
class CompareStats:
    max_abs: float
    max_rel: float
    rmse: float


def compare_tensors(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> CompareStats:
    with torch.no_grad():
        diff = (a - b).abs()
        max_abs = diff.max().item()
        denom = torch.maximum(a.abs(), b.abs())
        max_rel = (diff / (denom + eps)).max().item()
        rmse = torch.sqrt(torch.mean((a - b) ** 2)).item()
    return CompareStats(max_abs=max_abs, max_rel=max_rel, rmse=rmse)


# FLOPs estimator (approx)
def _sum_visible_prefix_area(T: int, chunk_T: int) -> int:
    s = 0
    for i0 in range(0, T, chunk_T):
        i1 = min(T, i0 + chunk_T)
        Mi = i1 - i0
        Ti = i1
        s += Mi * Ti
    return s


def estimate_hyperglu_fwd_flops(
    cfg: HyperGLUConfig,
    *,
    B: int,
    T: int,
    impl: Optional[str] = None,
    chunk_T: Optional[int] = None,
) -> int:
    """
    Rough FLOPs estimator for a HyperGLU forward pass.

    Counts only the major GEMMs and depthwise convs, and ignores:
      - masking / indexing / gathers / pads
      - elementwise ops (sigmoid/softplus/relu/etc.)
      - reductions and misc overhead

    Convention: 1 MAC = 2 FLOPs.

    For impl == "chunked" with cfg.do_masking=True AND cfg.aligner_kind=="offset",
    we use the materialized area after prefix pruning:
        area = sum_i (Mi * Ti) where each query block is [i0, i1), Mi=i1-i0, Ti=i1.

    Otherwise, we use full area = T*T.
    """
    which = cfg.impl_default if impl is None else str(impl)

    h = int(cfg.n_head)
    n_embd = int(cfg.n_embd)

    qk_rank = int(cfg.qk_rank) if cfg.qk_rank is not None else (n_embd // 4)
    vo_rank = int(cfg.vo_rank) if cfg.vo_rank is not None else n_embd

    qk_heads = (2 * h) if (cfg.attn_kind == "glu") else h
    assert qk_rank % qk_heads == 0
    assert vo_rank % h == 0

    d_qk = qk_rank // qk_heads
    d_v = vo_rank // h

    rank1 = int(cfg.op1_rank)
    rank2 = int(cfg.op2_rank)

    BT = B * T

    def flops_linear(in_f: int, out_f: int) -> int:
        return 2 * BT * in_f * out_f

    flops = 0

    # q,k,v
    flops += flops_linear(n_embd, qk_rank)  # q
    flops += flops_linear(n_embd, qk_rank)  # k
    flops += flops_linear(n_embd, vo_rank)  # v

    # gates
    if cfg.use_qk_gate:
        flops += flops_linear(n_embd, qk_rank)  # left_gate
    if cfg.use_vo_gate:
        flops += flops_linear(n_embd, vo_rank)  # right_gate

    # controls
    if cfg.use_op1:
        flops += flops_linear(n_embd, qk_heads * rank1)
    if cfg.use_op2:
        flops += flops_linear(n_embd, h * rank2)

    # convs (depthwise approx)
    k_conv = int(cfg.conv_kernel_size)
    if cfg.use_kv_conv:
        flops += 2 * k_conv * B * T * n_embd  # conv1
        flops += 2 * k_conv * B * T * n_embd  # conv2

    # area
    if not cfg.do_masking:
        area = T * T
    else:
        if which == "dense":
            area = T * T
        elif which == "chunked":
            if cfg.aligner_kind == "offset":
                ct = int(cfg.chunk_T if chunk_T is None else chunk_T)
                area = _sum_visible_prefix_area(T, ct)
            else:
                area = T * T
        else:
            raise ValueError(f"Unknown impl for flops: {which}")

    # xl@k
    flops += 2 * B * qk_heads * area * d_qk

    # op1/op2 (two GEMMs each: X@U and Z@Vt)
    if cfg.use_op1:
        flops += 4 * B * qk_heads * area * rank1
    if cfg.use_op2:
        flops += 4 * B * h * area * rank2

    # out = A_left @ v
    flops += 2 * B * h * area * d_v

    # output projection
    flops += 2 * BT * vo_rank * n_embd

    return int(flops)


@dataclass
class BenchResult:
    ms_per_iter: float
    total_s: float
    tok_per_s: float
    peak_alloc_mb: float
    peak_reserved_mb: float
    extra_alloc_mb: float
    extra_reserved_mb: float
    tflops: float


def run_benchmark(*, name, fn, tokens, flops, iters, warmup) -> BenchResult:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()
    base_reserved = torch.cuda.memory_reserved()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    ms_per_iter = total_ms / iters

    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    extra_alloc = peak_alloc - base_alloc
    extra_reserved = peak_reserved - base_reserved

    tok_per_s = tokens / (ms_per_iter / 1000.0)
    tflops = (flops / (ms_per_iter / 1000.0)) / 1e12

    return BenchResult(
        ms_per_iter=ms_per_iter,
        total_s=total_ms / 1000.0,
        tok_per_s=tok_per_s,
        peak_alloc_mb=peak_alloc / (1024 ** 2),
        peak_reserved_mb=peak_reserved / (1024 ** 2),
        extra_alloc_mb=extra_alloc / (1024 ** 2),
        extra_reserved_mb=extra_reserved / (1024 ** 2),
        tflops=tflops,
    )


def print_bench(name: str, r: BenchResult):
    print(
        f"{name:18s} "
        f"{r.ms_per_iter:8.3f} ms/iter  "
        f"tok/s={r.tok_per_s:12.1f}  "
        f"extra_alloc={r.extra_alloc_mb:9.1f} MB  "
        f"peak_alloc={r.peak_alloc_mb:9.1f} MB  "
        f"TFLOPs~{r.tflops:7.2f}"
    )


def numerical_consistency_test(
    *,
    model: HyperGLU,
    x: torch.Tensor,
    chunk_T: int,
    rtol: float,
    atol: float,
    impls: List[str],
) -> None:
    print("=== Numerical consistency (forward+backward) ===")
    model_ref = copy.deepcopy(model).eval()
    with torch.no_grad():
        y_ref = model_ref(x, impl="dense", chunk_T=chunk_T)

    print(f"baseline: dense y shape={tuple(y_ref.shape)} dtype={y_ref.dtype}")

    for impl in impls:
        if impl == "dense":
            continue
        m = copy.deepcopy(model).eval()
        with torch.no_grad():
            y = m(x, impl=impl, chunk_T=chunk_T)
        st = compare_tensors(y_ref, y)
        print(f"[forward] dense vs {impl:12s}: {st}")
        torch.testing.assert_close(y_ref, y, rtol=rtol, atol=atol)

    print("\n--- backward consistency ---")
    grads_ref = {}
    x_ref = x.detach().clone().requires_grad_(True)
    m_ref = copy.deepcopy(model).train()
    y1 = m_ref(x_ref, impl="dense", chunk_T=chunk_T)
    y1.sum().backward()
    xg_ref = x_ref.grad.detach().clone()
    for n, p in m_ref.named_parameters():
        if p.grad is not None:
            grads_ref[n] = p.grad.detach().clone()

    for impl in impls:
        if impl == "dense":
            continue

        m = copy.deepcopy(model).train()
        x2 = x.detach().clone().requires_grad_(True)
        y2 = m(x2, impl=impl, chunk_T=chunk_T)
        y2.sum().backward()

        stx = compare_tensors(xg_ref, x2.grad)
        print(f"[bwd xgrad] dense vs {impl:12s}: {stx}")
        torch.testing.assert_close(xg_ref, x2.grad, rtol=rtol, atol=atol)

        worst = []
        for n, p in m.named_parameters():
            if p.grad is None or n not in grads_ref:
                continue
            stp = compare_tensors(grads_ref[n], p.grad)
            worst.append((n, stp.max_abs))
        worst.sort(key=lambda t: t[1], reverse=True)
        print(f"[bwd param grad] top-5 max_abs dense vs {impl}:")
        for n, v in worst[:5]:
            print(f"  {n:40s} {v:.3e}")

        for n, p in m.named_parameters():
            if p.grad is None or n not in grads_ref:
                continue
            torch.testing.assert_close(grads_ref[n], p.grad, rtol=rtol, atol=atol)

    print("OK: all selected impls match dense within tolerances.")


@torch.no_grad()
def unit_test_aligner_rows(aligner: nn.Module, device="cuda"):
    if isinstance(aligner, IdentityAligner):
        print("[unit_test_aligner_rows] skip: IdentityAligner is a no-op by design")
        return

    torch.manual_seed(0)
    B, h = 2, 3
    T = 64
    for row_start in [0, 1, 7, 31]:
        for M in [1, 5, 16]:
            if row_start + M > T:
                continue
            X = torch.randn(B, h, M, T, device=device, dtype=torch.float32)

            # Reference: explicit gather-based construction (local T coordinates).
            r = (row_start + torch.arange(M, device=device, dtype=torch.long)).view(M, 1)  # (M,1)
            c = torch.arange(T, device=device, dtype=torch.long).view(1, T)               # (1,T)
            idx = c + r - (T - 1)          # (M,T)
            mask = idx >= 0                # (M,T)
            idx = idx.clamp(min=0)

            idx_exp = idx.view(1, 1, M, T).expand(B, h, M, T)
            exp = torch.gather(X, dim=-1, index=idx_exp)
            exp = exp.masked_fill(~mask.view(1, 1, M, T), 0.0)

            y = aligner.to_offset_rows(X, row_start=row_start)

            max_abs = (exp - y).abs().max().item()
            print("to_offset_rows", "row_start", row_start, "M", M, "max_abs", max_abs)
            assert max_abs == 0.0, "aligner.to_offset_rows is not exactly equivalent!"


@torch.no_grad()
def unit_test_op_fused_equiv(op: OffsetLowRankResidualMix, device="cuda"):
    torch.manual_seed(0)
    B, h, M, T = 2, op.mix.heads, 8, 32
    X = torch.randn(B, h, M, T, device=device, dtype=torch.float32)
    control = torch.randn(B, h, M, op.mix.rank, device=device, dtype=torch.float32)

    op_fp32 = copy.deepcopy(op).to(device=device, dtype=torch.float32).eval()

    y_ref = op_fp32._forward_ref(X, control=control, do_masking=True, row_start=0)
    y_fused = op_fp32(X, control=control, do_masking=True, row_start=0)

    max_abs = (y_ref - y_fused).abs().max().item()
    rmse = torch.sqrt(((y_ref - y_fused) ** 2).mean()).item()
    print("op fused vs ref:", "max_abs", max_abs, "rmse", rmse)
    assert max_abs < 1e-5


def unit_test_op_fused_bwd_equiv(op: "OffsetLowRankResidualMix", device="cuda"):
    """Verify that the fused custom backward matches the reference autograd path in fp32."""
    torch.manual_seed(0)
    print("== unit_test_op_fused_bwd_equiv ==")

    op_fp32 = copy.deepcopy(op).to(device=device, dtype=torch.float32).train()

    B = 2
    h = op_fp32.mix.heads
    M = 8
    T = 32
    r = op_fp32.mix.rank

    X = torch.randn(B, h, M, T, device=device, dtype=torch.float32, requires_grad=True)
    C = torch.randn(B, h, M, r, device=device, dtype=torch.float32, requires_grad=True)

    W = torch.randn(B, h, M, T, device=device, dtype=torch.float32)

    op_fp32.zero_grad(set_to_none=True)
    if X.grad is not None:
        X.grad = None
    if C.grad is not None:
        C.grad = None

    y_ref = op_fp32._forward_ref(X, control=C, do_masking=True, row_start=0)
    loss_ref = (y_ref * W).sum()
    loss_ref.backward()

    gX_ref = X.grad.detach().clone()
    gC_ref = C.grad.detach().clone()
    gU_ref = op_fp32.mix.U.grad.detach().clone()
    gV_ref = op_fp32.mix.Vt.grad.detach().clone()
    gd_ref = op_fp32.mix.d.grad.detach().clone() if getattr(op_fp32.mix, "use_diag", False) else None

    op_fp32.zero_grad(set_to_none=True)
    X.grad = None
    C.grad = None

    y_fused = op_fp32(X, control=C, do_masking=True, row_start=0)
    fwd_max_abs = (y_ref - y_fused).abs().max().item()
    print(f"  forward(max_abs)={fwd_max_abs:.3e}")
    assert fwd_max_abs < 1e-6

    loss_fused = (y_fused * W).sum()
    loss_fused.backward()

    gX = X.grad.detach()
    gC = C.grad.detach()
    gU = op_fp32.mix.U.grad.detach()
    gV = op_fp32.mix.Vt.grad.detach()
    gd = (
        op_fp32.mix.d.grad.detach()
        if getattr(op_fp32.mix, "use_diag", False) and op_fp32.mix.d.grad is not None
        else None
    )

    def _stat(a, b, name):
        diff = (a - b).abs()
        print(f"  {name:10s} max_abs={diff.max().item():.3e} rmse={torch.sqrt((diff**2).mean()).item():.3e}")

    _stat(gX, gX_ref, "dX")
    _stat(gC, gC_ref, "dC")
    _stat(gU, gU_ref, "dU")
    _stat(gV, gV_ref, "dVt")
    if gd_ref is not None and gd is not None:
        _stat(gd, gd_ref, "dd")

    torch.testing.assert_close(gX, gX_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(gC, gC_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(gU, gU_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(gV, gV_ref, rtol=1e-5, atol=1e-6)
    if gd_ref is not None and gd is not None:
        torch.testing.assert_close(gd, gd_ref, rtol=1e-5, atol=1e-6)

    print("  OK: fused backward matches reference in fp32.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--T", type=int, default=4096)
    parser.add_argument("--n_embd", type=int, default=1024)
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--block_size", type=int, default=4096)
    parser.add_argument("--bias", action="store_true")

    # variant knobs (ReLU/GLU variants)
    parser.add_argument("--attn_kind", type=str, default="glu", choices=["glu", "relu"])
    parser.add_argument("--attn_norm", type=str, default="out_scale", choices=["l2", "out_scale", "none"])
    parser.add_argument("--qk_rank", type=int, default=None)
    parser.add_argument("--vo_rank", type=int, default=None)

    parser.add_argument("--no_kv_conv", action="store_true")
    parser.add_argument("--rope", action="store_true")
    parser.add_argument("--rope_base", type=float, default=10000.0)

    parser.add_argument("--no_qk_gate", action="store_true")
    parser.add_argument("--no_vo_gate", action="store_true")

    parser.add_argument("--no_op1", action="store_true")
    parser.add_argument("--no_op2", action="store_true")

    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--chunk_T", type=int, default=1536)
    parser.add_argument(
        "--aligner_kind",
        type=str,
        default="offset",
        choices=["offset", "identity"],
        help="Aligner kind: offset (skewed causal) or identity (left layout; torch.tril-style)",
    )
    parser.add_argument("--no_masking", action="store_true",
                        help="Disable causal masking/prefix pruning (non-causal full-matrix mode).")
    parser.add_argument("--non_causal", action="store_true",
                        help="Convenience flag: equivalent to --aligner_kind identity --no_masking")

    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=100)

    parser.add_argument("--mode", type=str, default="all", choices=["compare", "bench", "all"])
    parser.add_argument("--rtol", type=float, default=3e-2)
    parser.add_argument("--atol", type=float, default=3e-2)

    parser.add_argument("--compile", action="store_true", help="compile both inference and training with torch.compile")
    parser.add_argument("--with_optim", action="store_true", help="training bench include optimizer.step()")

    parser.add_argument("--compile_infer", action="store_true", help="torch.compile inference runners")
    parser.add_argument("--compile_train", action="store_true", help="torch.compile training runners")

    parser.add_argument(
        "--run_unit_tests",
        action="store_true",
        help="run internal unit tests (aligner/op_fused) and exit",
    )

    parser.add_argument(
        "--impls",
        type=str,
        default="dense,chunked",
        help="comma-separated impls to run in compare/bench. Options: dense,chunked",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")

    device = "cuda"
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    torch.manual_seed(0)

    if args.non_causal:
        args.aligner_kind = "identity"
        args.no_masking = True

    cfg = HyperGLUConfig(
        n_embd=args.n_embd,
        n_head=args.n_head,
        block_size=args.block_size,
        bias=args.bias,

        qk_rank=args.qk_rank,
        vo_rank=args.vo_rank,

        chunk_T=args.chunk_T,
        aligner_kind=args.aligner_kind,
        do_masking=(not args.no_masking),

        attn_kind=args.attn_kind,
        attn_norm=args.attn_norm,

        use_kv_conv=(not args.no_kv_conv),

        use_rope=bool(args.rope),
        rope_base=float(args.rope_base),

        use_qk_gate=(not args.no_qk_gate),
        use_vo_gate=(not args.no_vo_gate),

        use_op1=(not args.no_op1),
        use_op2=(not args.no_op2),
    )

    base_model = HyperGLU(cfg).to(device=device, dtype=dtype)

    B, T = args.B, args.T
    x = torch.randn((B, T, args.n_embd), device=device, dtype=dtype)

    impls = [s.strip() for s in args.impls.split(",") if s.strip()]
    tokens = B * T

    for impl in impls:
        if impl not in ("dense", "chunked"):
            raise ValueError(f"Unknown impl in --impls: {impl}. Allowed: dense,chunked")

    if args.run_unit_tests:
        print("Running unit tests ...")
        model_for_test = copy.deepcopy(base_model).to(device=device, dtype=torch.float32)

        unit_test_aligner_rows(model_for_test.aligner, device=device)

        if model_for_test.use_op1 and model_for_test.op is not None:
            unit_test_op_fused_equiv(model_for_test.op.to(device), device=device)
            unit_test_op_fused_bwd_equiv(model_for_test.op.to(device), device=device)

        if model_for_test.use_op2 and model_for_test.op2 is not None:
            unit_test_op_fused_bwd_equiv(model_for_test.op2.to(device), device=device)

        print("All unit tests finished.")
        return

    if args.mode in ("compare", "all"):
        try:
            numerical_consistency_test(
                model=copy.deepcopy(base_model).to(device=device, dtype=dtype),
                x=x,
                chunk_T=args.chunk_T,
                rtol=args.rtol,
                atol=args.atol,
                impls=impls,
            )
        except Exception as e:
            print(f"[compare failed] {e}")
            torch.cuda.empty_cache()

    if args.mode in ("bench", "all"):
        do_compile_infer = bool(args.compile or args.compile_infer)
        do_compile_train = bool(args.compile or args.compile_train)

        print("\n=== Benchmark: inference (forward-only) ===")
        print(f"B={B} T={T} n_embd={args.n_embd} n_head={args.n_head} dtype={dtype}")
        print(f"iters={args.iters} warmup={args.warmup} chunk_T={args.chunk_T}")
        print(f"attn_kind={args.attn_kind} attn_norm={args.attn_norm} aligner_kind={args.aligner_kind}")
        print(f"kv_conv={not args.no_kv_conv} rope={args.rope} qk_gate={not args.no_qk_gate} vo_gate={not args.no_vo_gate}")
        print(f"op1={not args.no_op1} op2={not args.no_op2} do_masking={not args.no_masking}")
        print(f"compile_infer={do_compile_infer}")
        print(f"compile_train={do_compile_train}\n")

        infer_models: Dict[str, nn.Module] = {}
        for impl in impls:
            m = copy.deepcopy(base_model).to(device=device, dtype=dtype).eval()
            m.impl_default = impl
            m.chunk_T = args.chunk_T
            m.cfg.impl_default = impl
            m.cfg.chunk_T = args.chunk_T
            if do_compile_infer:
                m = torch.compile(m)
            infer_models[impl] = m

        def make_infer_fn(impl: str):
            m = infer_models[impl]

            def _fn():
                with torch.no_grad():
                    return m(x)
            return _fn

        for impl in impls:
            try:
                make_infer_fn(impl)()
            except Exception as e:
                print(f"[infer warmup compile failed] impl={impl}: {e}")

        for impl in impls:
            fl = estimate_hyperglu_fwd_flops(
                base_model.cfg,
                B=B,
                T=T,
                impl=impl,
                chunk_T=args.chunk_T,
            )
            try:
                r = run_benchmark(
                    name=f"{impl}-infer",
                    fn=make_infer_fn(impl),
                    tokens=tokens,
                    flops=fl,
                    iters=args.iters,
                    warmup=args.warmup,
                )
                print_bench(f"{impl}-infer", r)
            except torch.cuda.OutOfMemoryError:
                print(f"[OOM] {impl}-infer OOM. Reduce --T or skip {impl}.")
                torch.cuda.empty_cache()

        print("\n=== Benchmark: training (forward+backward) ===")
        print(f"with_optim={args.with_optim}\n")

        train_models: Dict[str, nn.Module] = {}
        train_opts: Dict[str, torch.optim.Optimizer] = {}

        for impl in impls:
            m = copy.deepcopy(base_model).to(device=device, dtype=dtype).train()
            m.impl_default = impl
            m.chunk_T = args.chunk_T
            m.cfg.impl_default = impl
            m.cfg.chunk_T = args.chunk_T
            if do_compile_train:
                m = torch.compile(m)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
            train_models[impl] = m
            train_opts[impl] = opt

        def make_train_fn(impl: str):
            m = train_models[impl]
            opt = train_opts[impl]

            def _fn():
                opt.zero_grad(set_to_none=True)
                y = m(x)
                loss = y.sum()
                loss.backward()
                if args.with_optim:
                    opt.step()
                return loss
            return _fn

        for impl in impls:
            try:
                make_train_fn(impl)()
            except Exception as e:
                print(f"[train warmup compile failed] impl={impl}: {e}")

        for impl in impls:
            fl_fwd = estimate_hyperglu_fwd_flops(
                base_model.cfg,
                B=B,
                T=T,
                impl=impl,
                chunk_T=args.chunk_T,
            )
            fl_train = 3 * fl_fwd

            try:
                r = run_benchmark(
                    name=f"{impl}-train",
                    fn=make_train_fn(impl),
                    tokens=tokens,
                    flops=fl_train,
                    iters=args.iters,
                    warmup=args.warmup,
                )
                print_bench(f"{impl}-train", r)
            except torch.cuda.OutOfMemoryError:
                print(f"[OOM] {impl}-train OOM. Reduce --T or skip {impl}.")
                torch.cuda.empty_cache()

        print("\n[Notes]")
        print(
            "- Compile inference and training separately (and use one model per impl) "
            "to avoid recompiles when switching modes."
        )
        print("- This script keeps only the dense and chunked implementations.")

if __name__ == "__main__":
    main()
