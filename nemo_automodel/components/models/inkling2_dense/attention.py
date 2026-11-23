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

"""Learned-relative grouped-query attention of the dense Inkling2 model."""

from typing import TYPE_CHECKING, NamedTuple, cast

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from nemo_automodel.shared.import_utils import safe_import_from

from .config import Inkling2DenseConfig
from .layers import L2Clip, ShortConvolution, TokenLayout, rms_scale

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import AuxOutput

# Keep the eager model importable with older PyTorch installations.
_HAVE_AUX, AuxRequest = safe_import_from("torch.nn.attention.flex_attention", "AuxRequest")


class AttentionSources(NamedTuple):
    """Query, key and value inputs, each [batch, sequence, hidden], after its norm."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor


class AttentionMasks(NamedTuple):
    """FlexAttention block masks of one forward call: sliding-window and full causal."""

    local: BlockMask
    full: BlockMask


class TokenLayoutBuilder(nn.Module):
    """Builds the per-forward token layout and, for FlexAttention, its block masks.

    A batch of unpadded single-document rows is the common pretraining case: its layout
    and masks depend only on the batch shape and are kept for the module's lifetime, one
    entry per distinct (batch, sequence, device).
    """

    def __init__(self, config: Inkling2DenseConfig, *, flex: bool) -> None:
        super().__init__()
        if flex and not _HAVE_AUX:
            raise ImportError("Flex attention requires PyTorch with differentiable FlexAttention auxiliary outputs")
        self.window = config.sliding_window_size
        self.lags = config.sconv_kernel_size - 1
        self.flex = flex
        # Compilation is essential: eager create_block_mask builds a dense token mask.
        self._make_local = torch.compile(create_block_mask, fullgraph=True, dynamic=False) if flex else None
        self._make_full = torch.compile(create_block_mask, fullgraph=True, dynamic=False) if flex else None
        self._plain: dict[tuple[int, int, torch.device], tuple[TokenLayout, AttentionMasks | None]] = {}

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        valid: torch.Tensor | None,
        positions: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> tuple[TokenLayout, AttentionMasks | None]:
        """Describe which tokens are real and where documents start.

        Args:
            input_ids: Integer tensor of shape [batch, sequence]; only its shape and device are read.
            valid: Boolean tensor of shape [batch, sequence], true for real tokens; None when all are.
            positions: Integer tensor of shape [batch, sequence] with the position inside each
                token's document; None when every row is one document.
            dtype: Model dtype of the multiplicative masks.

        Returns:
            The layout and, with FlexAttention, the block masks; None with eager attention.
        """
        batch, sequence = input_ids.shape
        device = input_ids.device
        index = torch.arange(sequence, device=device).expand(batch, sequence)
        plain = (valid is None or bool(valid.all())) and (positions is None or bool((positions == index).all()))
        key = (batch, sequence, device)
        if plain and key in self._plain:
            return self._plain[key]
        if plain:
            layout = TokenLayout(positions=index, valid=None, lag_valid=None)
            mask_valid = torch.ones(batch, sequence, dtype=torch.bool, device=device)
        else:
            mask_valid = torch.ones(batch, sequence, dtype=torch.bool, device=device) if valid is None else valid
            positions = (mask_valid.long().cumsum(-1) - 1).clamp_min(0) if positions is None else positions
            lags = torch.arange(1, self.lags + 1, device=device)
            layout = TokenLayout(
                positions=positions,
                valid=None if valid is None else valid.unsqueeze(-1).to(dtype),
                lag_valid=(positions.unsqueeze(-1) >= lags).to(dtype),
            )
        masks = self._block_masks(mask_valid, layout.positions) if self.flex else None
        if plain:
            self._plain[key] = (layout, masks)
        return layout, masks

    def _block_masks(self, valid: torch.Tensor, positions: torch.Tensor) -> AttentionMasks:
        """Build the sliding-window and full causal block masks.

        Args:
            valid: Boolean tensor of shape [batch, sequence], true for real tokens.
            positions: Integer tensor of shape [batch, sequence], restarting at each document.

        Returns:
            Masks that allow a real query to see earlier real keys of its own document.
        """
        batch, sequence = valid.shape
        window = self.window

        def full(b: torch.Tensor, h: torch.Tensor, qi: torch.Tensor, ki: torch.Tensor) -> torch.Tensor:
            """Allow causal pairs of real tokens in one document.

            Args:
                b: Scalar batch index.
                h: Scalar head index, unused because masks are shared across heads.
                qi: Scalar query index, including kernel padding.
                ki: Scalar key index, including kernel padding.

            Returns:
                Scalar Boolean indicating an allowed query/key pair.
            """
            query, key = qi.clamp(max=sequence - 1), ki.clamp(max=sequence - 1)
            allowed = (qi < sequence) & (ki < sequence) & (ki <= qi) & valid[b, query] & valid[b, key]
            return allowed & (qi - positions[b, query] == ki - positions[b, key])

        def local(b: torch.Tensor, h: torch.Tensor, qi: torch.Tensor, ki: torch.Tensor) -> torch.Tensor:
            """Restrict ``full`` to the sliding window; same arguments and return."""
            return full(b, h, qi, ki) & (qi - ki < window)

        assert self._make_local is not None and self._make_full is not None
        return AttentionMasks(
            local=self._make_local(local, batch, None, sequence, sequence, device=valid.device),
            full=self._make_full(full, batch, None, sequence, sequence, device=valid.device),
        )


class Inkling2DenseAttention(nn.Module):
    """Local or full grouped-query attention with relative features, convolutions, gate and sink.

    One fused projection holds the rows [Q, R, gate, sink, K, V]: query, relative features,
    output gate and sink logit read the query input; K and V read their own inputs in
    full-attention layers and in later loop passes.
    """

    def __init__(self, config: Inkling2DenseConfig, layer_idx: int, *, dtype: torch.dtype, flex: bool) -> None:
        super().__init__()
        self.is_local = layer_idx in config.local_layer_ids
        self.heads = config.swa_num_attention_heads if self.is_local else config.num_attention_heads
        self.kv_heads = config.swa_num_key_value_heads if self.is_local else config.num_key_value_heads
        self.head_dim = config.swa_head_dim if self.is_local else config.head_dim
        self.d_rel = config.d_rel
        self.rel_extent = config.sliding_window_size if self.is_local else config.rel_extent
        self.window = config.sliding_window_size
        self.value_norm = config.use_value_norm_no_gamma
        self.eps = config.rms_norm_eps
        # The length scale is one for every position below the floor.
        self.length_scale = (
            (config.log_scaling_alpha, config.log_scaling_n_floor)
            if not self.is_local and config.max_position_embeddings > config.log_scaling_n_floor
            else None
        )
        q_size, kv_size = self.heads * self.head_dim, self.kv_heads * self.head_dim
        self.query_sizes = (q_size, self.heads * config.d_rel, q_size, self.heads)
        self.query_rows = sum(self.query_sizes)
        self.kv_size = kv_size
        self.qkvr = nn.Linear(config.hidden_size, self.query_rows + 2 * kv_size, bias=config.q_bias, dtype=dtype)
        self.wo_ud = nn.Linear(q_size, config.hidden_size, bias=config.o_bias, dtype=dtype)
        self.relative_table = nn.Parameter(torch.empty(config.d_rel, self.rel_extent, dtype=dtype))
        self.q_l2_clip = L2Clip(self.heads, self.head_dim, config.qk_l2_clip_threshold, learned_scale=True)
        self.k_l2_clip = L2Clip(self.kv_heads, self.head_dim, config.qk_l2_clip_threshold, learned_scale=False)
        self.rel_l2_clip = L2Clip(self.heads, config.d_rel, config.rel_l2_clip_threshold, learned_scale=True)
        dynamic = 0 if self.is_local else config.hidden_size
        self.k_sconv = ShortConvolution(
            kv_size, config.sconv_kernel_size, dtype=dtype, heads=self.kv_heads, dynamic_input_size=dynamic
        )
        self.v_sconv = ShortConvolution(
            kv_size, config.sconv_kernel_size, dtype=dtype, heads=self.kv_heads, dynamic_input_size=dynamic
        )
        self._flex = torch.compile(flex_attention, fullgraph=True, dynamic=False) if flex else None

    @torch.no_grad()
    def reset_parameters(self, init_std: float) -> None:
        """Initialize projections, relative table, scales and convolution weights."""
        nn.init.normal_(self.qkvr.weight, std=init_std)
        nn.init.normal_(self.wo_ud.weight, std=init_std)
        if self.qkvr.bias is not None:
            self.qkvr.bias.zero_()
        if self.wo_ud.bias is not None:
            self.wo_ud.bias.zero_()
        nn.init.normal_(self.relative_table, std=init_std)
        assert self.q_l2_clip.scale is not None and self.rel_l2_clip.scale is not None
        self.q_l2_clip.scale.fill_(1)
        self.rel_l2_clip.scale.fill_(1)
        self.k_sconv.reset_parameters(init_std)
        self.v_sconv.reset_parameters(init_std)

    def _project(self, source: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """Apply rows [start, end) of the fused projection.

        Args:
            source: Tensor of shape [batch, sequence, hidden].
            start: First projection row.
            end: Row after the last projection row.

        Returns:
            Tensor of shape [batch, sequence, end - start].
        """
        bias = None if self.qkvr.bias is None else self.qkvr.bias[start:end]
        return F.linear(source, self.qkvr.weight[start:end], bias)

    def forward(self, sources: AttentionSources, layout: TokenLayout, masks: AttentionMasks | None) -> torch.Tensor:
        """Attend over earlier tokens of the same document, with a zero-valued virtual sink.

        Args:
            sources: Query, key and value inputs of shape [batch, sequence, hidden].
            layout: Token validity and document layout of this forward call.
            masks: FlexAttention block masks of this forward call; None selects the
                quadratic eager evaluation.

        Returns:
            Tensor of shape [batch, sequence, hidden] in sources.q's dtype; padding is zero.
        """
        batch, sequence, _ = sources.q.shape
        dtype = sources.q.dtype
        k_end = self.query_rows + self.kv_size
        if sources.k is sources.q and sources.v is sources.q:
            projected = self.qkvr(sources.q)
            q, r, gate, sink, k, v = projected.split((*self.query_sizes, self.kv_size, self.kv_size), -1)
        else:
            q, r, gate, sink = self._project(sources.q, 0, self.query_rows).split(self.query_sizes, -1)
            k = self._project(sources.k, self.query_rows, k_end)
            v = self._project(sources.v, k_end, k_end + self.kv_size)
        k = self.k_sconv(k, layout, dynamic_input=None if self.is_local else sources.k)
        v = self.v_sconv(v, layout, dynamic_input=None if self.is_local else sources.v)
        tau = None
        if self.length_scale is not None:
            alpha, floor = self.length_scale
            tau = 1 + alpha * torch.log(((layout.positions.float() + 1) / floor).clamp(min=1))
        q = self.q_l2_clip(q.view(batch, sequence, self.heads, self.head_dim))
        k = self.k_l2_clip(k.view(batch, sequence, self.kv_heads, self.head_dim))
        r = self.rel_l2_clip(r.view(batch, sequence, self.heads, self.d_rel))
        if tau is not None:
            # Query and relative features grow with the logarithm of the position.
            q = q * tau[..., None, None].to(dtype)
            r = r * tau[..., None, None].to(dtype)
        v = v.view(batch, sequence, self.kv_heads, self.head_dim)
        if self.value_norm:
            v = rms_scale(v, self.eps)
        profiles = (r @ self.relative_table).transpose(1, 2)
        if masks is None:
            context = self._eager_attention(q, k, v, profiles=profiles, sink=sink, layout=layout)
        else:
            mask = masks.local if self.is_local else masks.full
            context = self._flex_attention(q, k, v, profiles=profiles, sink=sink, mask=mask)
        attended = context.flatten(2).to(dtype) * torch.sigmoid(gate)
        output = self.wo_ud(attended)
        return output if layout.valid is None else output * layout.valid

    def _flex_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        profiles: torch.Tensor,
        sink: torch.Tensor,
        mask: BlockMask,
    ) -> torch.Tensor:
        """Evaluate attention with the compiled FlexAttention kernel.

        Args:
            q: Tensor of shape [batch, sequence, heads, head_dim].
            k: Tensor of shape [batch, sequence, kv_heads, head_dim].
            v: Tensor of shape [batch, sequence, kv_heads, head_dim].
            profiles: Relative logits of shape [batch, heads, sequence, relative_extent].
            sink: Sink logits of shape [batch, sequence, heads].
            mask: Block mask of this layer's attention pattern.

        Returns:
            FP32 tensor of shape [batch, sequence, heads, head_dim]. Q, K and V enter the
            kernel in their own dtype; scores and the softmax accumulate in FP32.
        """
        if q.device.type != "cuda":
            raise ValueError("Flex attention requires CUDA; use attn='eager' on CPU")
        assert self._flex is not None
        sequence, extent, scale = q.shape[1], profiles.shape[-1], self.head_dim**-0.5

        def score_mod(
            score: torch.Tensor, b: torch.Tensor, h: torch.Tensor, qi: torch.Tensor, ki: torch.Tensor
        ) -> torch.Tensor:
            """Add the learned relative logit after scaling the content score.

            Args:
                score: Scalar FP32 unscaled content score.
                b: Scalar batch index.
                h: Scalar query-head index.
                qi: Scalar query position.
                ki: Scalar key position.

            Returns:
                Scalar FP32 content-plus-relative attention logit.
            """
            distance = qi - ki
            relative = profiles[b, h, qi.clamp(max=sequence - 1), distance.clamp(0, extent - 1)].float()
            return score * scale + torch.where((distance >= 0) & (distance < extent), relative, 0.0)

        # Triton's dot product requires width >= 16; pad tiny test heads.
        pad = max(16, 1 << (q.shape[-1] - 1).bit_length()) - q.shape[-1]
        result = self._flex(
            F.pad(q.transpose(1, 2), (0, pad)),
            F.pad(k.transpose(1, 2), (0, pad)),
            F.pad(v.transpose(1, 2), (0, pad)),
            score_mod=score_mod,
            block_mask=mask,
            scale=1.0,
            enable_gqa=True,
            return_aux=AuxRequest(lse=True),
            kernel_options={"BACKEND": "TRITON"},
        )
        attended, auxiliary = cast("tuple[torch.Tensor, AuxOutput]", result)
        assert auxiliary.lse is not None
        # Softmax over the real keys and a zero-valued sink equals softmax(scores) scaled by
        # sigmoid(logsumexp(scores) - sink); the gradient through the log-sum-exp is kept.
        factor = torch.sigmoid(auxiliary.lse - sink.float().transpose(1, 2))
        return (attended[..., : q.shape[-1]].float() * factor.unsqueeze(-1)).transpose(1, 2)

    def _eager_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        profiles: torch.Tensor,
        sink: torch.Tensor,
        layout: TokenLayout,
    ) -> torch.Tensor:
        """Evaluate the quadratic FP32 reference.

        Args:
            q: Tensor of shape [batch, sequence, heads, head_dim].
            k: Tensor of shape [batch, sequence, kv_heads, head_dim].
            v: Tensor of shape [batch, sequence, kv_heads, head_dim].
            profiles: Relative logits of shape [batch, heads, sequence, relative_extent].
            sink: Sink logits of shape [batch, sequence, heads].
            layout: Token validity and document layout of this forward call.

        Returns:
            FP32 tensor of shape [batch, sequence, heads, head_dim].
        """
        batch, sequence = q.shape[:2]
        positions = layout.positions
        groups = self.heads // self.kv_heads
        k = k[:, :, :, None].expand(-1, -1, -1, groups, -1).reshape(batch, sequence, self.heads, self.head_dim)
        v = v[:, :, :, None].expand(-1, -1, -1, groups, -1).reshape(batch, sequence, self.heads, self.head_dim)
        scores = torch.einsum("bthd,bshd->bhts", q.float(), k.float())
        distance = positions[:, :, None] - positions[:, None, :]
        offsets = distance.clamp(0, self.rel_extent - 1)[:, None].expand(-1, self.heads, -1, -1)
        relative = profiles.float().gather(-1, offsets)
        relative = relative.masked_fill((distance[:, None] < 0) | (distance[:, None] >= self.rel_extent), 0)
        # Only the content score is scaled; the relative logits are already in the softmax domain.
        scores = scores * self.head_dim**-0.5 + relative
        index = torch.arange(sequence, device=q.device)
        allowed = (index[None, :] <= index[:, None])[None]
        origin = index[None] - positions
        allowed = allowed & (origin[:, :, None] == origin[:, None, :])
        if layout.valid is not None:
            allowed = allowed & (layout.valid[:, None, :, 0] > 0)
        if self.is_local:
            allowed = allowed & (distance >= 0) & (distance < self.window)
        scores = scores.masked_fill(~allowed[:, None], float("-inf"))
        with_sink = torch.cat((scores, sink.float().transpose(1, 2).unsqueeze(-1)), -1)
        probability = with_sink.softmax(-1)[..., :-1]
        return torch.einsum("bhts,bshd->bthd", probability, v.float())
