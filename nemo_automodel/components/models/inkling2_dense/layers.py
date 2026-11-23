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

from dataclasses import dataclass

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
    """Residual state between layers; every tensor has shape [batch, sequence, hidden].

    Attributes:
        blocks: Completed depth blocks, each already RMS-normalized over hidden.
        partial: Current incomplete block, not normalized.
    """

    blocks: tuple[torch.Tensor, ...]
    partial: torch.Tensor


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


def mix_depth(sources: tuple[torch.Tensor, ...], queries: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Mix depth sources independently at each token, once per query.

    Args:
        sources: RMS-normalized tensors of shape [batch, sequence, hidden], one per
            completed block followed by the current partial block.
        queries: FP32 tensor of shape [queries, hidden].

    Returns:
        One tensor of shape [batch, sequence, hidden] per query, in the sources' dtype:
        the softmax-weighted sum of the sources, weighted by each source's scaled dot
        product with the query. The softmax runs in FP32.
    """
    first = sources[0]
    queries = queries.to(first.dtype)
    scores = torch.stack([F.linear(source, queries) for source in sources], dim=-1)
    weights = (scores.float() * queries.shape[-1] ** -0.5).softmax(-1).to(first.dtype)
    mixed = []
    for query in range(queries.shape[0]):
        total = first * weights[..., query, 0:1]
        for index in range(1, len(sources)):
            total = torch.addcmul(total, sources[index], weights[..., query, index : index + 1])
        mixed.append(total)
    return tuple(mixed)


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
        return rms_scale(x, self.eps) * self.weight


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
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        # The tanh cap bounds the gate's linear factor; the sigmoid still sees the raw gate.
        capped = torch.tanh(gate / self.gate_softcap) * torch.sigmoid(gate) * torch.tanh(up / self.up_softcap)
        return self.down(capped * (self.gate_softcap * self.up_softcap))


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
        batch, sequence, _ = x.shape
        history = F.pad(x, (0, 0, self.width - 1, 0))

        def lagged(lag: int) -> torch.Tensor:
            """Return x shifted back by ``lag`` positions, [batch, sequence, channels], zero across documents."""
            shifted = history[:, self.width - 1 - lag : self.width - 1 - lag + sequence]
            return shifted if layout.lag_valid is None else shifted * layout.lag_valid[..., lag - 1 : lag]

        if self.weight is not None:
            taps = self.weight[:, 0]
            result = x * (1 + taps[:, self.width - 1])
            for lag in range(1, self.width):
                result = torch.addcmul(result, lagged(lag), taps[:, self.width - 1 - lag])
            scale = (self.width + 1) ** -0.5
        else:
            assert self.static_w is not None and self.weight_proj is not None and dynamic_input is not None
            generated = self.weight_proj(dynamic_input).view(batch, sequence, self.heads, self.width)
            static = self.static_w.view(self.width, self.heads, -1)
            result = x.view(batch, sequence, self.heads, -1) * (1 + static[0] + generated[..., 0, None])
            for lag in range(1, self.width):
                per_head = lagged(lag).view(batch, sequence, self.heads, -1)
                result = torch.addcmul(result, per_head, static[lag] + generated[..., lag, None])
            result = result.reshape(batch, sequence, self.channels)
            scale = (2 * self.width + 1) ** -0.5
        result = result * scale
        return result if layout.valid is None else result * layout.valid
