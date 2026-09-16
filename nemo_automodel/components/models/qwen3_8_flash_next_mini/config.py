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

"""Configuration classes for the dense Qwen3.8-Flash-Next research model."""

from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PretrainedConfig

from nemo_automodel.components.models.qwen3_8_flash_next.config import (
    Qwen3_8_FlashNextTextConfig,
)


class Qwen3_8_FlashNextMiniTextConfig(Qwen3_8_FlashNextTextConfig):
    """Text config for the dense-MLP Qwen3.8-Flash-Next research model.

    Keeps the Flash-Next decoder contract -- Gated Residuals, hybrid
    GatedDeltaNet/attention layers, mRoPE -- but replaces the routed-MoE block
    with a single dense SwiGLU MLP of width ``intermediate_size``. The defaults
    are the ~1B pretraining geometry used by ``experiment/base.yaml`` (16
    layers, 2048 hidden, cl32k vocabulary, full attention, GR on, no Engram).

    Args:
        intermediate_size: Width of the dense SwiGLU MLP that replaces the MoE
            block. Must be positive.
        use_qsa: Use sparse attention and its indexer at attention layers.
            Set False for the full-attention pretraining baseline.
        use_gr: Use gated, multiple-stream residuals. False selects ordinary
            single-stream pre-norm residuals for the controlled ablation.
        ple_layer_ids: One-based decoder indices carrying an Engram/PLE table.
            Defaults to ``[]`` (no Engram). ``[2]`` matches where the teacher
            puts its own table.
        ngram_table_rows: Approximate global row budget for the Engram table
            when ``ple_layer_ids`` is non-empty. The table holds
            ``padded_rows * (ple_embed_dim // ngram_heads)`` parameters.
        **kwargs: Forwarded to :class:`Qwen3_8_FlashNextTextConfig`.
    """

    model_type = "qwen3_8_flash_next_mini_text"
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size: int = 32768,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        head_dim: int = 256,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 16,
        max_position_embeddings: int = 4096,
        hc_count: int = 4,
        hc_lowrank: int = 256,
        use_qsa: bool = False,
        use_gr: bool = True,
        # SiLU GatedDeltaNet output gate as in Qwen3.5; Flash-Next itself uses a sigmoid.
        output_gate_type: str = "silu",
        tie_word_embeddings: bool = True,
        use_cache: bool = False,
        mtp_num_hidden_layers: int = 0,
        ple_layer_ids: list[int] | None = None,
        ngram_table_rows: int = 600_000,
        rope_parameters: dict[str, Any] | None = None,
        # cl32k reserves IDs 0-4 for EOS/PAD/FIM markers.
        bos_token_id: int = 0,
        eos_token_id: int = 0,
        pad_token_id: int | None = 1,
        **kwargs: Any,
    ) -> None:
        if rope_parameters is None:
            # Qwen3.5 rope: partial rotation over one quarter of head_dim.
            rope_parameters = {
                "rope_type": "default",
                "rope_theta": 10000000.0,
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
            }
        super().__init__(
            rope_parameters=rope_parameters,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            linear_num_key_heads=linear_num_key_heads,
            linear_num_value_heads=linear_num_value_heads,
            max_position_embeddings=max_position_embeddings,
            hc_count=hc_count,
            hc_lowrank=hc_lowrank,
            output_gate_type=output_gate_type,
            tie_word_embeddings=tie_word_embeddings,
            use_cache=use_cache,
            mtp_num_hidden_layers=mtp_num_hidden_layers,
            ple_layer_ids=[] if ple_layer_ids is None else ple_layer_ids,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        if int(self.intermediate_size) <= 0:
            raise ValueError(f"Dense MLP requires a positive intermediate_size, got {self.intermediate_size}.")
        # The inherited Hugging Face DeltaNet reads this list during construction.
        if self.layer_types is None:
            self.layer_types = ["full_attention" if block == "attention" else block for block in self.layers_block_type]
        # Every layer routes through its own MLP, so no downstream reader should
        # see a stale expert count on the inherited MoE fields.
        self.mlp_only_layers = list(range(int(self.num_hidden_layers)))
        self.num_experts = 0
        self.num_experts_per_tok = 0
        if int(ngram_table_rows) <= 0:
            raise ValueError(f"ngram_table_rows must be positive, got {ngram_table_rows}")
        self.ngram_table_rows = int(ngram_table_rows)
        self.use_qsa = use_qsa
        self.use_gr = use_gr

    @property
    def residual_streams(self) -> int:
        """Number of active streams; GR geometry stays serialized for ablations."""
        return self.hc_count if self.use_gr else 1


class Qwen3_8_FlashNextMiniConfig(PretrainedConfig):
    """Top-level config pairing the dense student text backbone with its head.

    Args:
        text_config: Text backbone config, or a mapping used to build one.
        **kwargs: Forwarded to :class:`~transformers.PretrainedConfig`.
    """

    model_type = "qwen3_8_flash_next_mini"
    sub_configs = {"text_config": Qwen3_8_FlashNextMiniTextConfig}

    def __init__(
        self,
        text_config: Qwen3_8_FlashNextMiniTextConfig | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if text_config is None:
            text_config = Qwen3_8_FlashNextMiniTextConfig()
        elif isinstance(text_config, dict):
            text_config = Qwen3_8_FlashNextMiniTextConfig(**text_config)
        self.text_config = text_config
        kwargs.setdefault("tie_word_embeddings", text_config.tie_word_embeddings)
        super().__init__(**kwargs)
