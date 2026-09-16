# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Dense-MLP decoder layer for the Qwen3.8-Flash-Next student.

Reuses Flash-Next Gated Residual and GatedDeltaNet, with full attention or QSA
selected by config and a single dense SwiGLU MLP in place of routed experts.
"""

from __future__ import annotations

import torch
from torch import nn

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_8_flash_next.cp import Qwen3_8_FlashNextCPContext
from nemo_automodel.components.models.qwen3_8_flash_next.layers import (
    Qwen3_8_FlashNextDecoderLayer,
    Qwen3_8_FlashNextGatedDeltaNet,
    Qwen3_8_FlashNextGroupedRMSNorm,
    Qwen3_8_FlashNextHyperConnection,
    Qwen3_8_FlashNextQSAAttention,
)
from nemo_automodel.components.models.qwen3_next.layers import Qwen3NextAttention
from nemo_automodel.components.moe.layers import MLP
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

from .config import Qwen3_8_FlashNextMiniTextConfig


class StandardResidual(nn.Module):
    """Ordinary pre-norm residual using the decoder's read/write interface.

    Args:
        hidden_size: Single-stream hidden width.
        rms_norm_eps: Epsilon for the same RMSNorm used by GR branches.
    """

    def __init__(self, hidden_size: int, rms_norm_eps: float) -> None:
        super().__init__()
        self.norm = Qwen3_8_FlashNextGroupedRMSNorm(hidden_size, group_size=hidden_size, eps=rms_norm_eps)

    def mix(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize the sublayer input while preserving the residual.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].

        Returns:
            Normalized input and unchanged residual, both [batch, sequence, hidden].
        """
        return self.norm(hidden_states), hidden_states

    def combine(self, block_output: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Add the sublayer output to the residual.

        Args:
            block_output: Tensor of shape [batch, sequence, hidden].
            residual: Tensor of shape [batch, sequence, hidden].

        Returns:
            Updated hidden states of shape [batch, sequence, hidden].
        """
        return residual + block_output

    def init_weights(self, init_std: float = 0.02) -> None:
        """Reset normalization; init_std is shared with the GR initializer API."""
        self.norm.reset_parameters()


class Qwen3_8_FlashNextMiniDecoderLayer(Qwen3_8_FlashNextDecoderLayer):
    """One dense-MLP Flash-Next decoder layer with two learned HyperConnections.

    Args:
        layer_idx: Zero-based decoder index.
        config: Dense student text configuration.
        backend: Attention and linear backend configuration.
        ple: Optional Engram/PLE module applied to the HC state before the
            attention read. ``None`` on every layer the config does not list.

    Tensor layout:
        The first layer accepts token embeddings ``[batch, sequence, hidden]``
        and expands them to flattened HC state
        ``[batch, sequence, hc_count * hidden]``. Every layer returns that
        flattened HC layout.
    """

    def __init__(
        self,
        layer_idx: int,
        config: Qwen3_8_FlashNextMiniTextConfig,
        backend: BackendConfig,
        *,
        ple: nn.Module | None = None,
    ) -> None:
        # Skips the parent __init__, which unconditionally allocates a routed MoE
        # block, but reuses the parent forward. Every attribute that forward
        # reads must therefore be set here; keep this list in sync with
        # Qwen3_8_FlashNextDecoderLayer.__init__.
        nn.Module.__init__(self)
        self.layer_idx = layer_idx
        self.layer_type = str(config.layer_types[layer_idx])
        self.hidden_size = config.hidden_size
        self.hc_count = config.residual_streams
        self.use_qsa = config.use_qsa
        self.ple = ple
        # PLE's owner-sharded lookup uses collectives the selective-checkpoint
        # dispatch cannot safely cache.
        self._nemo_disable_activation_checkpointing = ple is not None

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_8_FlashNextGatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            attention_class = Qwen3_8_FlashNextQSAAttention if self.use_qsa else Qwen3NextAttention
            self.self_attn = attention_class(config, layer_idx, backend)
        else:
            raise ValueError(f"Unsupported Qwen3.8-Flash-Next layer type {self.layer_type!r}")

        dtype = get_dtype(config.dtype, torch.bfloat16)
        self.mlp = MLP(
            dim=self.hidden_size,
            inter_dim=config.intermediate_size,
            backend=backend.linear,
            dtype=dtype,
            activation="swiglu",
        )
        hc_kwargs = {
            "hidden_size": self.hidden_size,
            "hc_count": self.hc_count,
            "lowrank_size": config.hc_lowrank,
            "rms_norm_eps": config.rms_norm_eps,
            "backend": backend,
            "dtype": dtype,
        }
        if config.use_gr:
            self.attn_hyper_connection = Qwen3_8_FlashNextHyperConnection(**hc_kwargs)
            self.mlp_hyper_connection = Qwen3_8_FlashNextHyperConnection(**hc_kwargs)
        else:
            self.attn_hyper_connection = StandardResidual(self.hidden_size, config.rms_norm_eps)
            self.mlp_hyper_connection = StandardResidual(self.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cp_context: Qwen3_8_FlashNextCPContext | None = None,
        **attn_kwargs: object,
    ) -> torch.Tensor:
        """Run attention/GDN and the dense MLP through both HyperConnections.

        Args:
            hidden_states: One-stream input ``[batch, sequence, hidden]`` on
                the first layer, otherwise flattened HC streams
                ``[batch, sequence, hc_count * hidden]``.
            input_ids: Raw tokenizer IDs of shape ``[batch, sequence]``, hashed
                by the PLE path when this layer carries an Engram table.
            freqs_cis: Composed rotary values ``[batch, sequence, rotary_dim]``
                whose final axis stores concatenated cosine and sine values.
            attention_mask: Optional token mask of shape ``[batch, sequence]``
                or backend-specific causal attention mask.
            padding_mask: Optional ``[batch, sequence]`` mask where ``True``
                marks padding.
            position_ids: Optional positions of shape ``[batch, sequence]`` or
                ``[axes, batch, sequence]``.
            cp_context: Optional contiguous CP metadata. Tensor-bearing fields
                contain replicated global raw IDs/padding of shape ``[batch,
                global_sequence]`` and identify this rank's local interval.
            **attn_kwargs: Attention backend metadata.

        Returns:
            Flattened HC streams of shape
            ``[batch, sequence, hc_count * hidden]``.
        """
        if attention_mask is not None and padding_mask is None and attention_mask.ndim <= 2:
            padding_mask = attention_mask.bool().logical_not()

        hidden_states = self._expand_initial_streams(hidden_states)
        if self.ple is not None:
            hidden_states = hidden_states + self.ple(hidden_states, input_ids, cp_context=cp_context)

        attn_input, attn_residual = self.attn_hyper_connection.mix(hidden_states)
        if self.layer_type == "linear_attention":
            packed_cu_seqlens = attn_kwargs.get("cu_seqlens")
            if packed_cu_seqlens is None and cp_context is not None:
                packed_cu_seqlens = getattr(cp_context, "global_cu_seqlens", None)
            if isinstance(packed_cu_seqlens, torch.Tensor):
                packed_cu_seqlens = packed_cu_seqlens.to(torch.long)
            attn_output = self.linear_attn(
                hidden_states=attn_input,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cu_seqlens=packed_cu_seqlens,
            )
        else:
            full_attention_mask = attention_mask
            if full_attention_mask is None and padding_mask is not None:
                full_attention_mask = padding_mask.logical_not()
            attn_output = self.self_attn(
                x=attn_input,
                attention_mask=full_attention_mask,
                freqs_cis=freqs_cis,
                cp_context=cp_context,
                **attn_kwargs,
            )
        hidden_states = self.attn_hyper_connection.combine(attn_output, attn_residual)

        mlp_input, mlp_residual = self.mlp_hyper_connection.mix(hidden_states)
        mlp_output = self.mlp(mlp_input)
        return self.mlp_hyper_connection.combine(mlp_output, mlp_residual)
