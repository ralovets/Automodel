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

"""Dense, text-only Inkling2 research model.

Shares the Inkling2 decoder contract -- learned-relative local and full attention,
short convolutions, block attention residuals and the optional loop over the core
layers -- with one dense softcapped SwiGLU per layer in place of routed experts,
tied embeddings, and no vision tower.
"""

from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import nn

from nemo_automodel.components.models.common import (
    BackendConfig,
    initialize_linear_module,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.inkling2.attention import Inkling2Attention
from nemo_automodel.components.models.inkling2.layers import (
    Inkling2LMHead,
    MessageController,
    MessageQueries,
    ResidualQuery,
    ShortConvolution,
)
from nemo_automodel.components.models.inkling2.model import Inkling2DecoderLayer, Inkling2ForCausalLM
from nemo_automodel.shared.utils import dtype_from_str

from .config import Inkling2DenseConfig


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
    def init_weights(self, buffer_device: torch.device, init_std: float) -> None:
        """Initialize both projections without checkpoint data.

        Args:
            buffer_device: Unused; kept for the decoder layer's initialization call.
            init_std: Standard deviation of the normal initialization.
        """
        del buffer_device
        nn.init.normal_(self.gate_up.weight, std=init_std)
        nn.init.normal_(self.down.weight, std=init_std)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the feed-forward, with the activation evaluated in FP32.

        Args:
            x: Tensor of shape [batch, sequence, hidden].
            padding_mask: Boolean tensor of shape [batch, sequence], true for padding.
                Unused: the decoder layer's convolution zeroes padded positions.

        Returns:
            Tensor of shape [batch, sequence, hidden], in x's dtype.
        """
        del padding_mask
        gate, up = self.gate_up(x).float().chunk(2, dim=-1)
        # The tanh cap bounds the gate's linear factor; the sigmoid still sees the raw gate.
        activated = self.gate_softcap * torch.tanh(gate / self.gate_softcap) * torch.sigmoid(gate)
        activated = activated * (self.up_softcap * torch.tanh(up / self.up_softcap))
        return self.down(activated.to(x.dtype))


class Inkling2DenseDecoderLayer(Inkling2DecoderLayer):
    """Inkling2 decoder layer whose feed-forward is dense."""

    def __init__(
        self,
        config: Inkling2DenseConfig,
        layer_idx: int,
        backend: BackendConfig,
        *,
        dtype: torch.dtype,
    ) -> None:
        # Skips the parent constructor, which builds routed experts. The parent's forward
        # and reset_parameters are reused, so every attribute they read is set here.
        nn.Module.__init__(self)
        self.is_local = layer_idx in config.local_layer_ids
        self.is_boundary = layer_idx % config.n_layers_per_attnres_block == 0
        streams = 1 if self.is_local else 3
        self.attn_query = (
            None
            if layer_idx == 0
            else ResidualQuery(config.hidden_size, streams, normalize_values=config.attn_res_normalize_values)
        )
        self.mlp_query = ResidualQuery(config.hidden_size, 1, normalize_values=config.attn_res_normalize_values)
        self.attn_norms = nn.ModuleList(
            [
                initialize_rms_norm_module(backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
                for _ in range(streams)
            ]
        )
        self.mlp_norm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )
        self.attn = Inkling2Attention(
            config, layer_idx, dtype=dtype, backend=cast(Literal["eager", "flex"], backend.attn)
        )
        self.mlp = CappedSwiGLU(config, backend, dtype=dtype)
        self.attn_sconv = ShortConvolution(config.hidden_size, config.sconv_kernel_size, dtype=dtype)
        self.mlp_sconv = ShortConvolution(config.hidden_size, config.sconv_kernel_size, dtype=dtype)
        self.controllers = (
            nn.ModuleDict(
                {
                    name: MessageController(config.hidden_size, config.loop_config["gate_rank"], dtype=dtype)
                    for name in ("attn", "k", "v", "mlp")
                }
            )
            if config.loop_config is not None and layer_idx in config.loop_core_layers
            else None
        )


class Inkling2DenseForCausalLM(Inkling2ForCausalLM):
    """Text-only causal LM: the Inkling2 decoder with dense feed-forwards and a tied head.

    The forward contract is the parent's: unshifted FP32 logits of shape
    [batch, kept_sequence, vocab], softcapped, for token IDs of shape [batch, sequence].
    """

    config_class = Inkling2DenseConfig
    tie_word_embeddings_support: TieSupport = TieSupport.TIED_ONLY
    _tied_weights_keys = ["lm_head.weight"]
    # The parent's sidecar is FSDP2-only and shards routed experts; DDP needs neither.
    parallelizer = None

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Validated with DDP only; no tensor, context, pipeline or expert parallelism."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = False
        supports_thd: bool = False

    def __init__(self, config: Inkling2DenseConfig, *, backend: BackendConfig | None = None) -> None:
        """Build the decoder on the current default device and initialize it.

        Args:
            config: Dense configuration; ``config.dtype`` selects FP32 or BF16 weights.
            backend: Kernel selection; None selects the eager Torch reference.
        """
        reject_unsupported_tie_word_embeddings(type(self), config)
        # Skips the parent constructor (expert backend checks, routed experts, the untied
        # head and the released-checkpoint adapter). The parent's forward is reused, so
        # every attribute it reads is set here.
        nn.Module.__init__(self)
        # Checkpoints use the model's own parameter names, so there is no HF adapter.
        backend = backend or BackendConfig(
            attn="eager", linear="torch", rms_norm="torch", enable_hf_state_dict_adapter=False
        )
        if (
            backend.attn not in ("eager", "flex")
            or backend.linear not in ("torch", "te")
            or backend.rms_norm != "torch"
        ):
            raise ValueError(
                "The dense Inkling2 model requires eager or flex attention, Torch or Transformer Engine "
                "feed-forward projections, and the Torch norm"
            )
        dtype = dtype_from_str(config.dtype, torch.float32)
        if dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("The dense Inkling2 model computes in float32 or bfloat16")
        self.cp_mesh = None
        self.backend = backend
        self.config = config
        self.text_config = config
        self.name_or_path = config.name_or_path
        self.vocab_size = config.vocab_size
        # Shared training paths read `.moe_config`; answer "no experts".
        self.moe_config = None
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.embed_norm = initialize_rms_norm_module("torch", config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.layers = nn.ModuleDict(
            {
                str(i): Inkling2DenseDecoderLayer(config, i, backend, dtype=dtype)
                for i in range(config.num_hidden_layers)
            }
        )
        core_layers = len(config.loop_core_layers)
        self.message_queries = MessageQueries(config.hidden_size, core_layers) if core_layers else None
        self.message_kv_queries = MessageQueries(config.hidden_size, core_layers) if core_layers else None
        self.output_query = ResidualQuery(config.hidden_size, 1, normalize_values=config.attn_res_normalize_values)
        self.norm = initialize_rms_norm_module("torch", config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.lm_head = Inkling2LMHead(config.hidden_size, config.vocab_size, bias=False, dtype=dtype)
        self.tie_weights()
        self.initialize_weights()

    @classmethod
    def from_config(
        cls, config: Inkling2DenseConfig, *, backend: BackendConfig | None = None
    ) -> "Inkling2DenseForCausalLM":
        """Construct the model from its configuration.

        Args:
            config: Dense configuration.
            backend: Kernel selection; None selects the eager Torch reference.

        Returns:
            An initialized model.
        """
        return cls(config, backend=backend)

    def tie_weights(self) -> None:
        """Alias the output projection weight to the token embedding weight."""
        assert self.lm_head is not None and self.embed_tokens is not None
        self.lm_head.weight = self.embed_tokens.weight


ModelClass = Inkling2DenseForCausalLM
