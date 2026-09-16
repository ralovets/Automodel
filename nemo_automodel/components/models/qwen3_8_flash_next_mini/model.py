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

"""Trainable dense Qwen3.8-Flash-Next research model.

Shares the Flash-Next decoder contract -- HyperConnections, hybrid
GatedDeltaNet and full attention or QSA, mRoPE -- with a single dense SwiGLU MLP
per layer in place of routed experts, tied embeddings, and optional Engram/PLE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import cast_model_to_dtype, compute_lm_head_logits
from nemo_automodel.components.models.qwen3_5_moe.model import Fp32SafeQwen3_5MoeTextRotaryEmbedding
from nemo_automodel.components.models.qwen3_8_flash_next.cp import Qwen3_8_FlashNextCPContext
from nemo_automodel.components.models.qwen3_8_flash_next.layers import Qwen3_8_FlashNextHyperConnection
from nemo_automodel.components.models.qwen3_8_flash_next.model import (
    Qwen3_8_FlashNextCausalLMOutput,
    Qwen3_8_FlashNextTextModelBackend,
)
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

from .config import Qwen3_8_FlashNextMiniConfig, Qwen3_8_FlashNextMiniTextConfig
from .engram_support import build_ple_layer, neutralize_ple_layer
from .layers import Qwen3_8_FlashNextMiniDecoderLayer, StandardResidual


def _resolve_model_dtype(config: Qwen3_8_FlashNextMiniTextConfig) -> torch.dtype:
    """Return the parameter dtype declared by the text config.

    Args:
        config: Text config whose ``dtype`` names the parameter dtype.

    Returns:
        Resolved parameter dtype, defaulting to ``torch.bfloat16``.
    """
    return get_dtype(config.dtype, torch.bfloat16)


def _mini_backend(config: Qwen3_8_FlashNextMiniTextConfig, backend: BackendConfig | None = None) -> BackendConfig:
    """Return the student's default backend configuration.

    Args:
        config: Text config selecting full attention or QSA.
        backend: Optional caller-supplied backend; returned unchanged when set.

    Returns:
        Backend configuration for attention, linear, and normalization kernels.
    """
    if backend is not None:
        return backend
    # Checkpoints use the model's own parameter names, so there is no HF adapter.
    return BackendConfig(
        attn="flex" if config.use_qsa else "sdpa", linear="torch", rms_norm="torch", enable_hf_state_dict_adapter=False
    )


class Qwen3_8_FlashNextMiniTextModelBackend(Qwen3_8_FlashNextTextModelBackend):
    """Dense Flash-Next text decoder with optional GR streams and PLE memory.

    Args:
        config: Dense student text configuration.
        backend: Attention, linear, and normalization backend configuration.

    Tensor layout:
        Token embeddings start as ``[batch, sequence, hidden_size]``. Decoder
        layers retain flattened HC state
        ``[batch, sequence, hc_count * hidden_size]``. The final learned HC read
        collapses it back to ``[batch, sequence, hidden_size]``.
    """

    def __init__(self, config: Qwen3_8_FlashNextMiniTextConfig, backend: BackendConfig) -> None:
        # Skips the parent __init__ (MoE configuration and production-sized PLE
        # tables) but reuses the parent forward; every attribute that forward
        # reads must be set here.
        nn.Module.__init__(self)
        self.config = config
        self.backend = backend
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.model_dtype = _resolve_model_dtype(config)
        # Shared Flash-Next paths read `.moe_config`; answer "no experts".
        self.moe_config = None

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
            dtype=self.model_dtype,
        )
        self.layers = nn.ModuleDict()
        for layer_idx in range(config.num_hidden_layers):
            ple = None
            # ple_layer_ids is one-based in the checkpoint contract.
            if (layer_idx + 1) in config.ple_layer_ids:
                ple = build_ple_layer(config, backend, self.model_dtype)
                # Start as an exact no-op so Engram can attach to a warm-started
                # checkpoint without shifting its outputs.
                neutralize_ple_layer(ple)
            self.layers[str(layer_idx)] = Qwen3_8_FlashNextMiniDecoderLayer(layer_idx, config, backend, ple=ple)

        if config.use_gr:
            self.hyper_connection_mixer = Qwen3_8_FlashNextHyperConnection(
                hidden_size=config.hidden_size,
                hc_count=config.hc_count,
                lowrank_size=config.hc_lowrank,
                rms_norm_eps=config.rms_norm_eps,
                backend=backend,
                use_combine=False,
                dtype=self.model_dtype,
            )
        else:
            self.hyper_connection_mixer = StandardResidual(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Fp32SafeQwen3_5MoeTextRotaryEmbedding(config=config)


class Qwen3_8_FlashNextMiniModel(nn.Module):
    """Language-only dense-MLP Flash-Next decoder wrapper.

    Args:
        config: Top-level student configuration.
        backend: Attention, linear, and normalization backend configuration.
    """

    def __init__(self, config: Qwen3_8_FlashNextMiniConfig, backend: BackendConfig) -> None:
        super().__init__()
        self.config = config
        self.language_model = Qwen3_8_FlashNextMiniTextModelBackend(config.text_config, backend)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: object | None = None,
        output_hidden_states: bool | None = None,
        _qwen3_8_flash_next_cp_context: Qwen3_8_FlashNextCPContext | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        """Run the language-only dense student decoder.

        Args:
            input_ids: Raw tokenizer IDs of shape ``[batch, sequence]``.
            attention_mask: Optional token-validity mask of shape
                ``[batch, sequence]`` or backend-specific attention mask.
            position_ids: Optional positions of shape ``[batch, sequence]`` or
                ``[axes, batch, sequence]``.
            inputs_embeds: Optional embeddings of shape
                ``[batch, sequence, hidden_size]``.
            past_key_values: Cache state; unsupported for training.
            output_hidden_states: Capture decoder HC states.
            _qwen3_8_flash_next_cp_context: Internal contiguous CP metadata with
                replicated raw-ID/padding tensors of shape
                ``[batch, global_sequence]``.
            **kwargs: Text-attention backend arguments.

        Returns:
            Base-model output whose final text states have shape
            ``[batch, sequence, hidden_size]``.
        """
        return self.language_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            _qwen3_8_flash_next_cp_context=_qwen3_8_flash_next_cp_context,
            **kwargs,
        )


class Qwen3_8_FlashNextMiniForCausalLM(HFCheckpointingMixin, nn.Module):
    """Trainable dense-MLP Flash-Next causal LM with tied embeddings.

    Args:
        config: Top-level student configuration.
        backend: Optional backend override for attention and linear kernels.
    """

    tie_word_embeddings_support: TieSupport = TieSupport.TIED_ONLY
    _tied_weights_keys = ["lm_head.weight"]
    # GatedDeltaNet keeps A_log/dt_bias in an fp32 holder next to bf16 weights.
    _keep_in_fp32_modules = ["_fp32_params"]
    _keep_in_fp32_modules_strict = ["_fp32_params"]

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Validated distributed features for the dense student.

        Context parallelism is off: the Flash-Next CP path needs a model-owned
        contiguous sharder plus QSA/GatedDeltaNet CP setup, none of which this
        student wires yet.
        """

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = False
        supports_mtp_cp: bool = False

    @classmethod
    def from_config(
        cls,
        config: Qwen3_8_FlashNextMiniConfig,
        backend: BackendConfig | None = None,
        **kwargs: Any,
    ) -> Qwen3_8_FlashNextMiniForCausalLM:
        """Construct from a parsed student configuration.

        Args:
            config: Top-level student configuration.
            backend: Optional backend override.
            **kwargs: Forwarded to the constructor.

        Returns:
            An initialized student model.
        """
        return cls(config, backend=backend, **kwargs)

    def __init__(
        self,
        config: Qwen3_8_FlashNextMiniConfig,
        backend: BackendConfig | None = None,
        **kwargs: Any,
    ) -> None:
        reject_unsupported_tie_word_embeddings(type(self), config)
        super().__init__()
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")
        self.config = config
        self.backend = _mini_backend(config.text_config, backend)
        self.model = Qwen3_8_FlashNextMiniModel(config, self.backend)
        dtype = _resolve_model_dtype(config.text_config)
        self.lm_head = initialize_linear_module(
            self.backend.linear,
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
            dtype=dtype,
        )
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.text_config.pad_token_id if config.text_config.pad_token_id is not None else -1
        self.tie_weights()
        if not self.get_input_embeddings().weight.is_meta:
            # DDP builds on a real device and skips the checkpointer's meta
            # initializer. Initialize the text model here as well.
            self.initialize_weights(buffer_device=self.get_input_embeddings().weight.device, dtype=dtype)

    def get_input_embeddings(self) -> nn.Module:
        """Return the text token embedding module."""
        return self.model.language_model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        """Replace the text token embedding module.

        Args:
            value: Module mapping ``[batch, sequence]`` IDs to
                ``[batch, sequence, hidden_size]`` embeddings.
        """
        self.model.language_model.embed_tokens = value
        self.tie_weights()

    def get_output_embeddings(self) -> nn.Module:
        """Return the language-model head."""
        return self.lm_head

    def set_output_embeddings(self, value: nn.Module) -> None:
        """Replace the language-model head.

        Args:
            value: Linear module projecting ``[batch, sequence, hidden_size]``
                to ``[batch, sequence, vocab_size]``.
        """
        self.lm_head = value
        self.tie_weights()

    def tie_weights(self) -> None:
        """Alias the LM head weight to the input embedding weight."""
        self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_hidden_states: bool | None = None,
        _qwen3_8_flash_next_cp_context: Qwen3_8_FlashNextCPContext | None = None,
        **kwargs: Any,
    ) -> Qwen3_8_FlashNextCausalLMOutput:
        """Run the student decoder and project final HC-mixed states.

        Args:
            input_ids: Raw tokenizer IDs of shape ``[batch, sequence]``.
            attention_mask: Optional token-validity mask of shape
                ``[batch, sequence]`` or backend-specific attention mask.
            position_ids: Optional positions of shape ``[batch, sequence]`` or
                ``[axes, batch, sequence]``.
            inputs_embeds: Optional embeddings of shape
                ``[batch, sequence, hidden_size]``.
            labels: Optional labels of shape ``[batch, sequence]``. Accepted for
                recipe compatibility; loss is computed externally.
            past_key_values: Cache state; unsupported.
            use_cache: Cache request; ``True`` is unsupported.
            logits_to_keep: ``0`` for all positions, a positive trailing count,
                or an integer tensor of shape ``[kept_sequence]`` containing
                explicit sequence indices.
            output_hidden_states: Explicit ``True`` returns the embedding, every
                per-layer HC state, and the final state.
            _qwen3_8_flash_next_cp_context: Internal contiguous CP metadata with
                replicated raw-ID/padding tensors of shape
                ``[batch, global_sequence]``.
            **kwargs: Text-attention backend metadata.

        Returns:
            Causal-LM output with logits ``[batch, kept_sequence, vocab_size]``.
        """
        del labels
        if use_cache:
            raise NotImplementedError("Dense Flash-Next student training does not support caches")
        capture_all_hidden_states = output_hidden_states is True
        return_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            output_hidden_states=capture_all_hidden_states,
            _qwen3_8_flash_next_cp_context=_qwen3_8_flash_next_cp_context,
            **kwargs,
        )
        lm_output = compute_lm_head_logits(
            self.lm_head,
            outputs.last_hidden_state,
            logits_to_keep,
            output_hidden_states=return_hidden_states,
        )
        return Qwen3_8_FlashNextCausalLMOutput(
            logits=lm_output.logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states if capture_all_hidden_states else lm_output.hidden_states,
        )

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Initialize checkpoint-free model weights.

        Args:
            buffer_device: Target device for backend initializers.
            dtype: Final parameter dtype, excluding intrinsic fp32 GDN state.
        """
        if buffer_device is None:
            buffer_device = (
                torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
            )
        self.model.language_model.init_weights(buffer_device)
        # The inherited init re-randomizes PLE projections; restore the no-op.
        for layer in self.model.language_model.layers.values():
            if layer.ple is not None:
                neutralize_ple_layer(layer.ple)
                layer.ple.ple_embedding.ngram_embedding.mark_sharding_contract()
        cast_model_to_dtype(self, dtype, skip_modules=("_fp32_params",))
        # Init and the dtype cast both rebind tensors, so re-alias the head.
        self.tie_weights()


ModelClass = Qwen3_8_FlashNextMiniForCausalLM
