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

"""Dense Inkling2: forward and gradient contract, tied head, config resolution and initialization."""

import pytest
import torch
from torch.nn import functional as F
from transformers import AutoConfig

from nemo_automodel._transformers.model_init import _resolve_custom_model_cls_for_config
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.inkling2.config import Inkling2LoopConfig
from nemo_automodel.components.models.inkling2.layers import capped_swiglu
from nemo_automodel.components.models.inkling2_dense import Inkling2DenseConfig, Inkling2DenseForCausalLM
from nemo_automodel.components.models.inkling2_dense.model import CappedSwiGLU

pytestmark = pytest.mark.cpu_only

LOOP: Inkling2LoopConfig = {"core_start": 2, "core_end": 6, "num_passes": 2, "gate_rank": 4, "is_varloop": False}
EXPERT_FIELDS = ("n_routed_experts", "num_experts_per_tok", "n_shared_experts", "route_scale", "use_gate_bias")


def tiny_config(loop: Inkling2LoopConfig | None = None, dtype: str = "float32") -> Inkling2DenseConfig:
    """Eight layers in blocks of two; layers 3 and 7 use full attention."""
    return Inkling2DenseConfig(
        hidden_size=32,
        num_hidden_layers=8,
        vocab_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=48,
        max_position_embeddings=64,
        d_rel=3,
        rel_extent=8,
        sliding_window_size=8,
        local_layer_ids=[0, 1, 2, 4, 5, 6],
        n_layers_per_attn_res_block=2,
        log_scaling_n_floor=4,
        loop_config=loop,
        dtype=dtype,
        architectures=["Inkling2DenseForCausalLM"],
    )


@pytest.mark.parametrize("loop", [None, LOOP])
def test_logits_and_gradients_reach_every_parameter(loop: Inkling2LoopConfig | None) -> None:
    torch.manual_seed(3)
    model = Inkling2DenseForCausalLM(tiny_config(loop))
    with torch.no_grad():
        # Queries start at zero, where a softmax over depth sources has no gradient signal to test.
        for name, parameter in model.named_parameters():
            if "quer" in name:
                parameter.normal_(std=0.5)
    ids = torch.randint(0, 64, (2, 16))
    logits = model(ids)
    assert logits.shape == (2, 16, 64) and logits.dtype == torch.float32
    assert logits.abs().max() <= model.config.final_logit_softcapping
    F.cross_entropy(logits[:, :-1].flatten(0, 1), ids[:, 1:].flatten()).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_later_tokens_do_not_change_earlier_logits() -> None:
    torch.manual_seed(5)
    model = Inkling2DenseForCausalLM(tiny_config(LOOP)).eval()
    ids = torch.randint(0, 64, (1, 16))
    changed = ids.clone()
    changed[0, 10:] = (changed[0, 10:] + 1) % 64
    torch.testing.assert_close(model(ids)[:, :10], model(changed)[:, :10], rtol=0, atol=1e-5)


def test_head_is_tied_and_bf16_weights_keep_fp32_logits() -> None:
    model = Inkling2DenseForCausalLM(tiny_config(dtype="bfloat16"))
    assert model.lm_head.weight is model.embed_tokens.weight
    assert model.embed_tokens.weight.dtype == torch.bfloat16
    assert model.output_query.weight.dtype == torch.float32
    names = [name for name, _ in model.named_parameters()]
    assert "embed_tokens.weight" in names and "lm_head.weight" not in names
    assert model(torch.randint(0, 64, (1, 8))).dtype == torch.float32
    restored = Inkling2DenseForCausalLM(tiny_config(dtype="bfloat16"))
    restored.load_state_dict(model.state_dict())
    assert restored.lm_head.weight is restored.embed_tokens.weight
    torch.testing.assert_close(restored.embed_tokens.weight, model.embed_tokens.weight, rtol=0, atol=0)


def test_dense_feed_forward_matches_the_inkling2_activation() -> None:
    """Gate rows then up rows, against the reference activation on its interleaved layout."""
    torch.manual_seed(7)
    config = tiny_config()
    mlp = CappedSwiGLU(config, BackendConfig(linear="torch"), dtype=torch.float32)
    mlp.init_weights(torch.device("cpu"), 1.0)
    x = 4 * torch.randn(2, 5, config.hidden_size)
    gate, up = mlp.gate_up(x).chunk(2, dim=-1)
    interleaved = torch.stack((gate, up), dim=-1).flatten(-2)
    reference = capped_swiglu(
        interleaved.flatten(0, 1),
        torch.ones(10, 1),
        gate_softcap=config.mlp_gate_softcap,
        up_softcap=config.mlp_up_softcap,
    )
    torch.testing.assert_close(mlp(x), mlp.down(reference).view_as(x), rtol=1e-6, atol=1e-6)


def test_checkpoint_free_initialization_is_finite() -> None:
    model = Inkling2DenseForCausalLM(tiny_config(LOOP))
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter).all(), name
    assert torch.count_nonzero(model.output_query.weight) == 0
    assert model.layers["2"].controllers is not None and model.layers["0"].controllers is None
    assert model.layers["3"].mlp.gate_up.weight.std() > 0


def test_config_resolves_from_a_checkpoint_style_file(tmp_path) -> None:
    config = tiny_config(LOOP)
    config.save_pretrained(tmp_path)
    saved = (tmp_path / "config.json").read_text()
    assert not any(f'"{name}"' in saved for name in EXPERT_FIELDS)
    loaded = AutoConfig.from_pretrained(tmp_path)
    assert type(loaded) is Inkling2DenseConfig
    assert loaded.tie_word_embeddings is True and loaded.loop_config == LOOP
    assert loaded.intermediate_size == 48 and loaded.local_layer_ids == [0, 1, 2, 4, 5, 6]
    assert _resolve_custom_model_cls_for_config(loaded) is Inkling2DenseForCausalLM


def test_default_geometry_and_untied_rejection() -> None:
    config = Inkling2DenseConfig()
    assert (config.hidden_size, config.num_hidden_layers, config.intermediate_size) == (2048, 24, 8192)
    assert config.local_layer_ids == [i for i in range(24) if i % 6 != 5]
    assert config.unpadded_vocab_size == config.vocab_size == 32768
    with pytest.raises(ValueError, match="ties its output projection"):
        Inkling2DenseConfig(tie_word_embeddings=False)
