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

"""Configuration of the dense Inkling2 research model."""

from typing import Any

from nemo_automodel.components.models.inkling2.config import Inkling2LoopConfig, Inkling2TextConfig

# Expert routing fields of the parent configuration. The dense model has no experts:
# the parent receives the smallest values its validation accepts, and the fields are
# left out of the serialized configuration.
_EXPERT_FIELDS = ("n_routed_experts", "num_experts_per_tok", "n_shared_experts", "route_scale", "use_gate_bias")


class Inkling2DenseConfig(Inkling2TextConfig):
    """Inkling2 text decoder with one dense feed-forward per layer and tied embeddings.

    The defaults are the 1.64B "micro" geometry: 24 layers of width 2048 with the
    Inkling-2-Mini layer pattern (five local-attention layers then one full-attention
    layer, attention-residual blocks of six) and no loop. Every other field keeps the
    meaning it has in ``Inkling2TextConfig``; ``intermediate_size`` is the width of the
    dense feed-forward.
    """

    model_type = "inkling2_dense"

    def __init__(
        self,
        *,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        vocab_size: int = 32768,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        intermediate_size: int = 8192,
        max_position_embeddings: int = 4096,
        loop_config: Inkling2LoopConfig | None = None,
        tie_word_embeddings: bool = True,
        **kwargs: Any,
    ) -> None:
        if not tie_word_embeddings:
            raise ValueError("The dense Inkling2 model ties its output projection to the token embedding")
        # Hugging Face owns the open-ended serialized metadata at this boundary.
        super().__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            vocab_size=vocab_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            loop_config=loop_config,
            n_routed_experts=2,
            num_experts_per_tok=2,
            tie_word_embeddings=False,
            **kwargs,
        )
        self.tie_word_embeddings = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize without the parent's expert routing fields."""
        output = super().to_dict()
        for name in _EXPERT_FIELDS:
            output.pop(name, None)
        return output
