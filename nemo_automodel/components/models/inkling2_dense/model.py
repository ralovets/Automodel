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

Follows the Inkling2 decoder design -- learned-relative local and full attention,
short convolutions, block attention residuals and the optional loop over the core
layers -- with one dense softcapped SwiGLU per layer in place of routed experts,
tied embeddings, and no vision tower. The layers are this package's own, arranged
for training throughput; see layers.py.
"""

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.inkling2.context_parallel import validate_document_positions
from nemo_automodel.components.models.inkling2.layers import (
    Inkling2LMHead,
    MessageController,
    MessageQueries,
    ResidualQuery,
)
from nemo_automodel.shared.utils import dtype_from_str

from .attention import AttentionMasks, AttentionSources, Inkling2DenseAttention, TokenLayoutBuilder
from .config import Inkling2DenseConfig
from .layers import CappedSwiGLU, DepthState, RMSNorm, ShortConvolution, TokenLayout, mix_depth, rms_scale

# Modules whose parameters stay FP32 when the model is cast to a lower precision.
FP32_MODULES = (
    "attn_query",
    "mlp_query",
    "output_query",
    "message_queries",
    "message_kv_queries",
    "q_l2_clip",
    "rel_l2_clip",
)


class Inkling2DenseDecoderLayer(nn.Module):
    """One attention and feed-forward layer operating on explicit depth state."""

    def __init__(
        self, config: Inkling2DenseConfig, layer_idx: int, backend: BackendConfig, *, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.is_local = layer_idx in config.local_layer_ids
        self.is_boundary = layer_idx % config.n_layers_per_attnres_block == 0
        streams = 1 if self.is_local else 3
        self.attn_query = None if layer_idx == 0 else ResidualQuery(config.hidden_size, streams, normalize_values=True)
        self.mlp_query = ResidualQuery(config.hidden_size, 1, normalize_values=True)
        self.attn_norms = nn.ModuleList(
            [RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype) for _ in range(streams)]
        )
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.attn = Inkling2DenseAttention(config, layer_idx, dtype=dtype, flex=backend.attn == "flex")
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

    @torch.no_grad()
    def reset_parameters(self, init_std: float) -> None:
        """Initialize all layer parameters without checkpoint data."""
        for norm in (*self.attn_norms, self.mlp_norm):
            cast(RMSNorm, norm).weight.fill_(1)
        if self.attn_query is not None:
            self.attn_query.weight.zero_()
        self.mlp_query.weight.zero_()
        self.attn.reset_parameters(init_std)
        self.mlp.reset_parameters(init_std)
        self.attn_sconv.reset_parameters(init_std)
        self.mlp_sconv.reset_parameters(init_std)
        if self.controllers is not None:
            for controller in self.controllers.values():
                cast(MessageController, controller).reset_parameters(init_std)

    def _apply_message(self, name: str, stream: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """Mix a message from the previous loop pass into one layer input.

        Args:
            name: Controller name: "attn", "k", "v" or "mlp".
            stream: Layer input of shape [batch, sequence, hidden].
            message: Message of shape [batch, sequence, hidden].

        Returns:
            Tensor of shape [batch, sequence, hidden] in stream's dtype. The gate runs in FP32.
        """
        assert self.controllers is not None
        return self.controllers[name](stream.float(), message.float()).to(stream.dtype)

    def forward(
        self,
        state: DepthState,
        layout: TokenLayout,
        masks: AttentionMasks | None,
        bank: tuple[torch.Tensor, ...] | None = None,
        message_queries: torch.Tensor | None = None,
    ) -> DepthState:
        """Advance a depth block; a new block begins after a boundary layer's attention.

        Args:
            state: Normalized completed blocks and the partial block, each [batch, sequence, hidden].
            layout: Token validity and document layout of this forward call.
            masks: FlexAttention block masks of this forward call; None with eager attention.
            bank: RMS-normalized depth sources the previous loop pass wrote, each
                [batch, sequence, hidden]; None in the first pass and outside the loop core.
            message_queries: FP32 tensor of shape [4, hidden] with this layer's queries for the
                attention Q, K, V and feed-forward inputs; given together with ``bank``.

        Returns:
            New state with tensors of shape [batch, sequence, hidden]; ``state`` is unchanged.
        """
        normalized_partial = rms_scale(state.partial)
        count = len(self.attn_norms)
        streams = (
            (state.partial,) * count
            if self.attn_query is None
            else mix_depth((*state.blocks, normalized_partial), self.attn_query.weight)
        )
        norms = tuple(self.attn_norms)
        messages = None
        if bank is not None:
            assert message_queries is not None
            messages = mix_depth(bank, message_queries)
            # In later passes a local layer also gets separate Q, K and V inputs, through its one norm.
            streams, norms = (streams * 3, norms * 3) if self.is_local else (streams, norms)
            streams = tuple(
                self._apply_message(name, stream, message)
                for name, stream, message in zip(("attn", "k", "v"), streams, messages)
            )
        normed = tuple(norm(stream) for norm, stream in zip(norms, streams))
        sources = AttentionSources(normed[0], normed[0], normed[0]) if len(normed) == 1 else AttentionSources(*normed)
        attention = self.attn_sconv(self.attn(sources, layout, masks), layout)
        if self.is_boundary:
            blocks, partial = (*state.blocks, normalized_partial), attention
        else:
            blocks, partial = state.blocks, state.partial + attention
        mlp_stream = mix_depth((*blocks, rms_scale(partial)), self.mlp_query.weight)[0]
        if messages is not None:
            mlp_stream = self._apply_message("mlp", mlp_stream, messages[3])
        mlp = self.mlp_sconv(self.mlp(self.mlp_norm(mlp_stream)), layout)
        return DepthState(blocks, partial + mlp)


class Inkling2DenseForCausalLM(HFCheckpointingMixin, nn.Module):
    """Text-only causal LM: the Inkling2 decoder design with dense feed-forwards and a tied head."""

    config_class = Inkling2DenseConfig
    tie_word_embeddings_support: TieSupport = TieSupport.TIED_ONLY
    _tied_weights_keys = ["lm_head.weight"]
    _keep_in_fp32_modules_strict = FP32_MODULES

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Validated with DDP only; no tensor, context, pipeline or expert parallelism."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = False

    def __init__(self, config: Inkling2DenseConfig, *, backend: BackendConfig | None = None) -> None:
        """Build the decoder on the current default device and initialize it.

        Args:
            config: Dense configuration; ``config.dtype`` selects FP32 or BF16 weights.
            backend: Kernel selection; None selects eager attention and Torch projections.
        """
        reject_unsupported_tie_word_embeddings(type(self), config)
        super().__init__()
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
        self.backend = backend
        self.config = config
        self.name_or_path = config.name_or_path
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.embed_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.layout = TokenLayoutBuilder(config, flex=backend.attn == "flex")
        self.layers = nn.ModuleDict(
            {
                str(i): Inkling2DenseDecoderLayer(config, i, backend, dtype=dtype)
                for i in range(config.num_hidden_layers)
            }
        )
        core_layers = len(config.loop_core_layers)
        self.message_queries = MessageQueries(config.hidden_size, core_layers) if core_layers else None
        self.message_kv_queries = MessageQueries(config.hidden_size, core_layers) if core_layers else None
        self.output_query = ResidualQuery(config.hidden_size, 1, normalize_values=True)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
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
            backend: Kernel selection; None selects eager attention and Torch projections.

        Returns:
            An initialized model.
        """
        return cls(config, backend=backend)

    @torch.no_grad()
    def initialize_weights(self) -> None:
        """Initialize a fresh model without a checkpoint."""
        init_std = self.config.initializer_range
        nn.init.normal_(self.embed_tokens.weight, std=init_std)
        self.embed_norm.weight.fill_(1)
        self.norm.weight.fill_(1)
        self.output_query.weight.zero_()
        for queries in (self.message_queries, self.message_kv_queries):
            if queries is not None:
                queries.weight.zero_()
        for layer in self.layers.values():
            cast(Inkling2DenseDecoderLayer, layer).reset_parameters(init_std)

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the token embedding; embed_norm is applied in forward."""
        return self.embed_tokens

    def get_output_embeddings(self) -> Inkling2LMHead:
        """Return the output projection, whose weight is the token embedding."""
        return self.lm_head

    def tie_weights(self) -> None:
        """Alias the output projection weight to the token embedding weight."""
        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_hidden_states: bool = False,
    ) -> torch.Tensor | CausalLMOutputWithPast:
        """Return unshifted text logits; the loss owns the target shift.

        Args:
            input_ids: Integer tensor of shape [batch, sequence].
            attention_mask: Optional tensor of shape [batch, sequence], one/true for real tokens.
            padding_mask: Optional boolean tensor of shape [batch, sequence], true for padding;
                combined with ``attention_mask``.
            position_ids: Optional integer tensor of shape [batch, sequence], increasing by one
                within each document and restarting at zero for a new document. None treats
                every row as one document.
            logits_to_keep: Zero for all positions, a positive tail length, or int64 indices
                of shape [kept_sequence].
            output_hidden_states: Also return the final hidden states, for a loss that
                projects them itself.

        Returns:
            FP32 logits of shape [batch, kept_sequence, vocab], softcapped. When
            ``output_hidden_states`` is true, a ``CausalLMOutputWithPast`` with these logits
            and final ``hidden_states`` of shape [batch, sequence, hidden]. No input is mutated.
        """
        if input_ids.ndim != 2 or min(input_ids.shape) == 0:
            raise ValueError("input_ids must have nonempty [batch, sequence] shape")
        valid = None if attention_mask is None else attention_mask.bool()
        if padding_mask is not None:
            valid = ~padding_mask if valid is None else valid & ~padding_mask
        if valid is not None and valid.shape != input_ids.shape:
            raise ValueError("attention_mask and padding_mask must match input_ids [batch, sequence]")
        if position_ids is not None:
            if position_ids.shape != input_ids.shape or position_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("position_ids must be integer [batch, sequence]")
            real = torch.ones_like(input_ids, dtype=torch.bool) if valid is None else valid
            if torch.any((position_ids >= self.config.max_position_embeddings) & real):
                raise ValueError("position_ids exceed the configured context range")
            validate_document_positions(real, position_ids)
        elif input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("input_ids exceed the configured context range")
        # Storage selects the compute precision; an outer autocast must not change it.
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            embedded = self.embed_tokens(input_ids)
            layout, masks = self.layout(input_ids, valid=valid, positions=position_ids, dtype=embedded.dtype)
            hidden = self.embed_norm(embedded)
            if layout.valid is not None:
                hidden = hidden * layout.valid
            state = self._run_layers(DepthState((), hidden), layout, masks)
            mixed = mix_depth((*state.blocks, rms_scale(state.partial)), self.output_query.weight)[0]
            hidden = self.norm(mixed)
            if isinstance(logits_to_keep, int):
                if logits_to_keep < 0:
                    raise ValueError("logits_to_keep must be nonnegative")
                selected = hidden if logits_to_keep == 0 else hidden[:, -logits_to_keep:]
            else:
                if logits_to_keep.ndim != 1 or logits_to_keep.dtype != torch.int64:
                    raise ValueError("logits_to_keep indices must be int64 [kept_sequence]")
                selected = hidden[:, logits_to_keep]
            cap = self.config.final_logit_softcapping
            logits = cap * torch.tanh(self.lm_head(selected) / cap)
        if output_hidden_states:
            # Automodel's loss contract consumes the final tensor here; HF annotates a
            # tuple of per-layer states.
            return CausalLMOutputWithPast(logits=cast(torch.FloatTensor, logits), hidden_states=hidden)  # ty: ignore[invalid-argument-type]
        return logits

    def _run_layers(self, state: DepthState, layout: TokenLayout, masks: AttentionMasks | None) -> DepthState:
        """Run every layer once, or the loop core once per pass with messages between passes.

        Args:
            state: Depth state entering the first layer, tensors of shape [batch, sequence, hidden].
            layout: Token validity and document layout of this forward call.
            masks: FlexAttention block masks of this forward call; None with eager attention.

        Returns:
            State after the last layer. A later loop pass restarts from the state that entered
            the core and reads the previous pass only through its messages.
        """
        loop = self.config.loop_config
        if loop is None:
            for layer in self.layers.values():
                state = layer(state, layout, masks)
            return state
        core = self.config.loop_core_layers
        assert self.message_queries is not None and self.message_kv_queries is not None
        queries, kv_queries = self.message_queries(), self.message_kv_queries()
        for index in range(core.start):
            state = self.layers[str(index)](state, layout, masks)
        prelude, bank = state, None
        for pass_index in range(loop["num_passes"]):
            if pass_index:
                # The first core layer seals the prelude's partial block; later blocks and the
                # final partial block are what this pass wrote.
                bank = (*state.blocks[len(prelude.blocks) + 1 :], rms_scale(state.partial))
                state = prelude
            for offset, index in enumerate(core):
                # Row 2 * offset holds the attention Q (or K) query, the next row the feed-forward (or V) query.
                first, second = 2 * offset, 2 * offset + 1
                layer_queries = (
                    None
                    if bank is None
                    else torch.stack((queries[first], kv_queries[first], kv_queries[second], queries[second]))
                )
                state = self.layers[str(index)](state, layout, masks, bank, layer_queries)
        for index in range(core.stop, self.config.num_hidden_layers):
            state = self.layers[str(index)](state, layout, masks)
        return state


ModelClass = Inkling2DenseForCausalLM
