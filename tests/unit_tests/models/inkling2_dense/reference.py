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

"""Reference for the dense model: the upstream Inkling2 forward with a dense feed-forward.

The upstream decoder layer and model are built without their routed experts, and the
dense model's weights are copied into them, so the upstream FP32 forward serves as the
trusted implementation of the layer orchestration, attention, convolutions and loop.
"""

from typing import Literal, cast

import torch
from torch import nn

from nemo_automodel.components.models.common import BackendConfig, initialize_rms_norm_module
from nemo_automodel.components.models.inkling2.attention import Inkling2Attention
from nemo_automodel.components.models.inkling2.layers import (
    Inkling2LMHead,
    MessageController,
    MessageQueries,
    ResidualQuery,
    ShortConvolution,
)
from nemo_automodel.components.models.inkling2.model import Inkling2DecoderLayer, Inkling2ForCausalLM
from nemo_automodel.components.models.inkling2_dense import Inkling2DenseConfig, Inkling2DenseForCausalLM
from nemo_automodel.components.models.inkling2_dense.layers import CappedSwiGLU


class _ReferenceFeedForward(CappedSwiGLU):
    """The dense feed-forward behind the upstream layer's expert call signature."""

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the feed-forward to x of shape [batch, sequence, hidden]; padding_mask [batch, sequence] is unused."""
        return super().forward(x)


class _ReferenceLayer(Inkling2DecoderLayer):
    def __init__(self, config: Inkling2DenseConfig, layer_idx: int, backend: BackendConfig) -> None:
        # Skips the parent constructor, which builds routed experts; its forward is reused.
        nn.Module.__init__(self)
        dtype = torch.float32
        self.is_local = layer_idx in config.local_layer_ids
        self.is_boundary = layer_idx % config.n_layers_per_attnres_block == 0
        streams = 1 if self.is_local else 3
        self.attn_query = None if layer_idx == 0 else ResidualQuery(config.hidden_size, streams, normalize_values=True)
        self.mlp_query = ResidualQuery(config.hidden_size, 1, normalize_values=True)
        self.attn_norms = nn.ModuleList(
            [
                initialize_rms_norm_module("torch", config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
                for _ in range(streams)
            ]
        )
        self.mlp_norm = initialize_rms_norm_module("torch", config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.attn = Inkling2Attention(config, layer_idx, dtype=dtype, backend=cast(Literal["eager"], "eager"))
        self.mlp = _ReferenceFeedForward(config, backend, dtype=dtype)
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


class ReferenceDenseModel(Inkling2ForCausalLM):
    """Upstream FP32 eager forward over dense feed-forwards, holding a dense model's weights."""

    def __init__(self, model: Inkling2DenseForCausalLM) -> None:
        # Skips the parent constructor (routed experts, untied head); its forward is reused.
        nn.Module.__init__(self)
        config = model.config
        backend = BackendConfig(attn="eager", linear="torch", rms_norm="torch", enable_hf_state_dict_adapter=False)
        self.cp_mesh = None
        self.backend = backend
        self.config = self.text_config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=torch.float32)
        self.embed_norm = initialize_rms_norm_module(
            "torch", config.hidden_size, eps=config.rms_norm_eps, dtype=torch.float32
        )
        self.layers = nn.ModuleDict(
            {str(i): _ReferenceLayer(config, i, backend) for i in range(config.num_hidden_layers)}
        )
        core_layers = len(config.loop_core_layers)
        self.message_queries = MessageQueries(config.hidden_size, core_layers) if core_layers else None
        self.message_kv_queries = MessageQueries(config.hidden_size, core_layers) if core_layers else None
        self.output_query = ResidualQuery(config.hidden_size, 1, normalize_values=True)
        self.norm = initialize_rms_norm_module(
            "torch", config.hidden_size, eps=config.rms_norm_eps, dtype=torch.float32
        )
        self.lm_head = Inkling2LMHead(config.hidden_size, config.vocab_size, bias=False, dtype=torch.float32)
        self.lm_head.weight = self.embed_tokens.weight
        self._copy_weights(model)

    @torch.no_grad()
    def _copy_weights(self, model: Inkling2DenseForCausalLM) -> None:
        """Copy every weight; the fused attention projection changes row order.

        The dense model stores rows [Q, R, gate, sink, K, V]; upstream stores
        [Q, K, V, R, gate, sink] followed by zero padding rows.
        """
        state = {name: value.float() for name, value in model.state_dict().items()}
        for index, layer in model.layers.items():
            attention = layer.attn
            q, r, gate, sink = attention.query_sizes
            rows = state.pop(f"layers.{index}.attn.qkvr.weight").split(
                (q, r, gate, sink, attention.kv_size, attention.kv_size)
            )
            upstream = self.layers[index].attn.qkvr.weight
            ordered = torch.cat((rows[0], rows[4], rows[5], rows[1], rows[2], rows[3]))
            upstream.zero_()
            upstream[: ordered.shape[0]] = ordered
        missing, unexpected = self.load_state_dict(state, strict=False)
        assert not unexpected and all(name.endswith("attn.qkvr.weight") for name in missing), (missing, unexpected)
