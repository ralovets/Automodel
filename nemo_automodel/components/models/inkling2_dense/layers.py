# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Layer primitives of the dense Inkling2 model.

These compute what the Inkling2 reference modules compute, arranged for training
throughput: activations stay in the model dtype with FP32 only inside reductions,
completed depth blocks are normalized once, and the short convolution is a few
shifted multiply-adds.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import ParamSpec

import torch
from torch import nn
from torch.nn import functional as F

from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module

from .config import Inkling2DenseConfig


@dataclass(frozen=True)
class TokenLayout:
    """Token validity and document layout of one forward call, shared by every layer.

    Attributes:
        positions: Integer tensor of shape [batch, sequence]: position within the token's
            document, restarting at zero where a new document begins.
        valid: Tensor of shape [batch, sequence, 1] in the model dtype, one for real
            tokens and zero for padding; None when every token is real.
        lag_valid: Tensor of shape [batch, sequence, lags] in the model dtype; entry
            ``lag - 1`` is one where the token ``lag`` positions back belongs to the same
            document. None when every row is a single unpadded document, where a shift
            past the row start reads zeros.
    """

    positions: torch.Tensor
    valid: torch.Tensor | None
    lag_valid: torch.Tensor | None


@dataclass(frozen=True)
class DepthState:
    """Residual state between layers.

    Attributes:
        blocks: Completed depth blocks, each already RMS-normalized over hidden, stacked
            as [batch, sequence, blocks, hidden]; None before the first block completes.
        partial: Current incomplete block of shape [batch, sequence, hidden], not normalized.
    """

    blocks: torch.Tensor | None
    partial: torch.Tensor


_Arguments = ParamSpec("_Arguments")


def _fused(function: Callable[_Arguments, torch.Tensor]) -> Callable[_Arguments, torch.Tensor]:
    """Compile an elementwise chain into one forward and one backward kernel on CUDA.

    The compiled function is shared by every layer, so it compiles once per distinct
    input shape. CPU calls run the plain function.
    """
    compiled = torch.compile(function, dynamic=False)

    def dispatch(*args: _Arguments.args, **kwargs: _Arguments.kwargs) -> torch.Tensor:
        first = args[0]
        return (compiled if isinstance(first, torch.Tensor) and first.is_cuda else function)(*args, **kwargs)

    return dispatch


@_fused
def rms_scale(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Divide by the root mean square over the last axis, reduced in FP32.

    Args:
        x: Tensor of shape [..., features], with arbitrary leading dimensions.
        eps: Added to the mean square.

    Returns:
        Tensor of x's shape and dtype.
    """
    mean_square = torch.linalg.vector_norm(x, dim=-1, keepdim=True, dtype=torch.float32).square() / x.shape[-1]
    return x * torch.rsqrt(mean_square + eps).to(x.dtype)


@_fused
def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Scale x of shape [..., features] to unit root mean square, then by weight of shape [features]."""
    mean_square = torch.linalg.vector_norm(x, dim=-1, keepdim=True, dtype=torch.float32).square() / x.shape[-1]
    return x * torch.rsqrt(mean_square + eps).to(x.dtype) * weight


@_fused
def l2_clip(x: torch.Tensor, scale: torch.Tensor | None, threshold: float) -> torch.Tensor:
    """Scale each vector, then shrink it to at most ``threshold`` in L2 norm, reduced in FP32.

    Args:
        x: Tensor of shape [batch, sequence, heads, width].
        scale: Tensor of shape [heads, width] applied before clipping, or None.
        threshold: Largest L2 norm over the last axis after clipping.

    Returns:
        Tensor of x's shape and dtype.
    """
    value = x if scale is None else x * scale.to(x.dtype)
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True, dtype=torch.float32)
    return value * (threshold / (norm + 1e-6)).clamp(max=1).to(x.dtype)


def mix_depth(blocks: torch.Tensor | None, partial: torch.Tensor, queries: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Mix depth sources independently at each token, once per query.

    Args:
        blocks: RMS-normalized completed blocks of shape [batch, sequence, blocks, hidden], or None.
        partial: RMS-normalized current block of shape [batch, sequence, hidden].
        queries: FP32 tensor of shape [queries, hidden].

    Returns:
        One tensor of shape [batch, sequence, hidden] per query, in partial's dtype: the
        softmax-weighted sum of the blocks and the partial block, weighted by each
        source's scaled dot product with the query. The softmax runs in FP32.
    """
    if blocks is None:
        # A softmax over one source is one.
        return (partial,) * queries.shape[0]
    queries = queries.to(partial.dtype)
    sources = torch.cat((blocks, partial.unsqueeze(2)), dim=2)
    scores = sources @ queries.t()
    weights = (scores.float() * queries.shape[-1] ** -0.5).softmax(2).to(partial.dtype)
    return (weights.transpose(2, 3) @ sources).unbind(2)


class RMSNorm(nn.Module):
    """RMS normalization with a learned scale; the reduction runs in FP32."""

    def __init__(self, features: int, *, eps: float, dtype: torch.dtype) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(features, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize over the last axis.

        Args:
            x: Tensor of shape [..., features], with arbitrary leading dimensions.

        Returns:
            Tensor of x's shape and dtype.
        """
        return _rms_norm(x, self.weight, self.eps)


class L2Clip(nn.Module):
    """Per-head L2 clipping, optionally preceded by learned FP32 per-channel scales."""

    def __init__(self, heads: int, width: int, threshold: float, *, learned_scale: bool) -> None:
        super().__init__()
        self.threshold = threshold
        self.scale = nn.Parameter(torch.ones(heads, width, dtype=torch.float32)) if learned_scale else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Clip head vectors.

        Args:
            x: Tensor of shape [batch, sequence, heads, width].

        Returns:
            Tensor of x's shape and dtype.
        """
        return l2_clip(x, self.scale, self.threshold)


@_fused
def capped_swiglu(gate_up: torch.Tensor, gate_softcap: float, up_softcap: float) -> torch.Tensor:
    """Apply Inkling2's softcapped SwiGLU to a fused gate/up projection.

    Args:
        gate_up: Tensor of shape [..., 2 * intermediate]: gate rows, then up rows.
        gate_softcap: Positive tanh cap of the gate's linear factor; the sigmoid sees the raw gate.
        up_softcap: Positive tanh cap of the up branch.

    Returns:
        Tensor of shape [..., intermediate], in gate_up's dtype.
    """
    gate, up = gate_up.chunk(2, dim=-1)
    capped = torch.tanh(gate / gate_softcap) * torch.sigmoid(gate) * torch.tanh(up / up_softcap)
    return capped * (gate_softcap * up_softcap)


@_fused
def _static_taps(x: torch.Tensor, taps: torch.Tensor, lag_valid: torch.Tensor | None) -> torch.Tensor:
    """Add weighted preceding positions with one tap per channel and lag.

    Args:
        x: Tensor of shape [batch, sequence, channels], zero at padding.
        taps: Tensor of shape [channels, width] in conv1d order: oldest lag first, current position last.
        lag_valid: Tensor of shape [batch, sequence, width - 1], or None; see ``TokenLayout``.

    Returns:
        Tensor of shape [batch, sequence, channels]: (x + taps applied to x) scaled by (width + 1) ** -0.5.
    """
    width, sequence = taps.shape[1], x.shape[1]
    history = F.pad(x, (0, 0, width - 1, 0))
    result = x * (1 + taps[:, width - 1])
    for lag in range(1, width):
        shifted = history[:, width - 1 - lag : width - 1 - lag + sequence]
        if lag_valid is not None:
            shifted = shifted * lag_valid[..., lag - 1 : lag]
        result = result + shifted * taps[:, width - 1 - lag]
    return result * (width + 1) ** -0.5


@_fused
def _dynamic_taps(
    x: torch.Tensor, generated: torch.Tensor, static: torch.Tensor, lag_valid: torch.Tensor | None
) -> torch.Tensor:
    """Add weighted preceding positions with static per-channel plus generated per-head taps.

    Args:
        x: Tensor of shape [batch, sequence, channels], zero at padding.
        generated: Tensor of shape [batch, sequence, heads, width], lag-indexed with the current position first.
        static: Tensor of shape [width, heads, channels / heads], lag-indexed.
        lag_valid: Tensor of shape [batch, sequence, width - 1], or None; see ``TokenLayout``.

    Returns:
        Tensor of shape [batch, sequence, channels]: (x + taps applied to x) scaled by (2 * width + 1) ** -0.5.
    """
    batch, sequence, channels = x.shape
    width, heads = static.shape[0], static.shape[1]
    history = F.pad(x, (0, 0, width - 1, 0))
    result = x.view(batch, sequence, heads, -1) * (1 + static[0] + generated[..., 0, None])
    for lag in range(1, width):
        shifted = history[:, width - 1 - lag : width - 1 - lag + sequence]
        if lag_valid is not None:
            shifted = shifted * lag_valid[..., lag - 1 : lag]
        result = result + shifted.view(batch, sequence, heads, -1) * (static[lag] + generated[..., lag, None])
    return result.reshape(batch, sequence, channels) * (2 * width + 1) ** -0.5


class CappedSwiGLU(nn.Module):
    """Dense feed-forward with Inkling2's softcapped SwiGLU activation.

    Both projections come from the configured linear backend, so a Transformer Engine
    low-precision recipe (``backend.te_fp8`` or ``backend.te_nvfp4``) applies to them.
    """

    def __init__(self, config: Inkling2DenseConfig, backend: BackendConfig, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.gate_softcap = config.mlp_gate_softcap
        self.up_softcap = config.mlp_up_softcap
        # One projection yields the gate rows followed by the up rows.
        self.gate_up = initialize_linear_module(
            backend.linear, config.hidden_size, 2 * config.intermediate_size, bias=False, dtype=dtype
        )
        self.down = initialize_linear_module(
            backend.linear, config.intermediate_size, config.hidden_size, bias=False, dtype=dtype
        )

    @torch.no_grad()
    def reset_parameters(self, init_std: float) -> None:
        """Initialize both projections without checkpoint data."""
        nn.init.normal_(self.gate_up.weight, std=init_std)
        nn.init.normal_(self.down.weight, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward.

        Args:
            x: Tensor of shape [batch, sequence, hidden].

        Returns:
            Tensor of shape [batch, sequence, hidden], in x's dtype.
        """
        return self.down(capped_swiglu(self.gate_up(x), self.gate_softcap, self.up_softcap))


class ShortConvolution(nn.Module):
    """Causal width-W depthwise convolution with static or token-generated per-head taps.

    Parameter names and layouts follow the Inkling2 reference module.
    """

    def __init__(
        self,
        channels: int,
        width: int,
        *,
        dtype: torch.dtype,
        dynamic_input_size: int = 0,
        heads: int = 1,
    ) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("Short convolution heads must divide channels")
        self.channels, self.width, self.heads = channels, width, heads
        if dynamic_input_size:
            self.weight = None
            # Lag-indexed taps, current position first: [width, channels].
            self.static_w = nn.Parameter(torch.zeros(width, channels, dtype=dtype))
            self.weight_proj = nn.Linear(dynamic_input_size, heads * width, bias=False, dtype=dtype)
        else:
            # conv1d order, oldest tap first and current position last: [channels, 1, width].
            self.weight = nn.Parameter(torch.zeros(channels, 1, width, dtype=dtype))
            self.static_w = None
            self.weight_proj = None

    @torch.no_grad()
    def reset_parameters(self, init_std: float) -> None:
        """Initialize every static and dynamic coefficient without checkpoint data."""
        if self.weight is not None:
            nn.init.normal_(self.weight, std=init_std)
        else:
            assert self.static_w is not None and self.weight_proj is not None
            nn.init.normal_(self.static_w, std=init_std)
            nn.init.normal_(self.weight_proj.weight, std=init_std)

    def forward(
        self, x: torch.Tensor, layout: TokenLayout, *, dynamic_input: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Add the weighted current and preceding positions of the same document.

        Args:
            x: Tensor of shape [batch, sequence, channels].
            layout: Token validity and document layout of this forward call.
            dynamic_input: Tensor of shape [batch, sequence, dynamic_input_size] that
                generates the per-head taps; required exactly when the taps are dynamic.

        Returns:
            Tensor of shape [batch, sequence, channels], in x's dtype; padding is zero.
        """
        if (dynamic_input is None) != (self.weight_proj is None):
            raise ValueError("dynamic_input is required exactly for a convolution with dynamic taps")
        if layout.valid is not None:
            x = x * layout.valid
        if self.weight is not None:
            result = _static_taps(x, self.weight[:, 0], layout.lag_valid)
        else:
            assert self.static_w is not None and self.weight_proj is not None and dynamic_input is not None
            generated = self.weight_proj(dynamic_input).view(*x.shape[:2], self.heads, self.width)
            static = self.static_w.view(self.width, self.heads, -1)
            result = _dynamic_taps(x, generated, static, layout.lag_valid)
        return result if layout.valid is None else result * layout.valid
