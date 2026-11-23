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

"""Dense Inkling2: parity with the upstream forward, gradients, tied head, config and initialization."""

import pytest
import torch
from torch.nn import functional as F
from transformers import AutoConfig

from nemo_automodel._transformers.model_init import _resolve_custom_model_cls_for_config
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.inkling2.config import Inkling2LoopConfig
from nemo_automodel.components.models.inkling2.layers import capped_swiglu
from nemo_automodel.components.models.inkling2_dense import Inkling2DenseConfig, Inkling2DenseForCausalLM
from nemo_automodel.components.models.inkling2_dense.layers import CappedSwiGLU

from .reference import ReferenceDenseModel

pytestmark = pytest.mark.cpu_only

LOOP: Inkling2LoopConfig = {"core_start": 2, "core_end": 6, "num_passes": 3, "gate_rank": 4, "is_varloop": False}
EXPERT_FIELDS = ("n_routed_experts", "num_experts_per_tok", "n_shared_experts", "route_scale", "use_gate_bias")


def tiny_config(
    loop: Inkling2LoopConfig | None = None,
    dtype: str = "float32",
    *,
    max_position_embeddings: int = 64,
    sliding_window_size: int = 6,
) -> Inkling2DenseConfig:
    """Eight layers in blocks of two; layers 3 and 7 use full attention. Positions pass the length-scale floor."""
    return Inkling2DenseConfig(
        hidden_size=32,
        num_hidden_layers=8,
        vocab_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=48,
        max_position_embeddings=max_position_embeddings,
        d_rel=3,
        rel_extent=8,
        sliding_window_size=sliding_window_size,
        local_layer_ids=[0, 1, 2, 4, 5, 6],
        n_layers_per_attn_res_block=2,
        log_scaling_n_floor=4,
        loop_config=loop,
        dtype=dtype,
        architectures=["Inkling2DenseForCausalLM"],
    )


def randomized_model(config: Inkling2DenseConfig, seed: int) -> Inkling2DenseForCausalLM:
    """Build the model with nonzero queries, scales and gates so every path changes the output."""
    torch.manual_seed(seed)
    model = Inkling2DenseForCausalLM(config)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "quer" in name:
                parameter.normal_(std=0.5)
            elif "controllers" in name or "sconv" in name or "relative_table" in name:
                parameter.normal_(std=0.3)
            elif "l2_clip" in name or "norm" in name:
                parameter.uniform_(0.5, 1.5)
            else:
                parameter.normal_(std=0.2)
    return model


@pytest.mark.parametrize("loop", [None, LOOP])
@pytest.mark.parametrize("packed", [False, True])
def test_logits_match_the_upstream_forward(loop: Inkling2LoopConfig | None, packed: bool) -> None:
    """Plain rows, and rows with left padding, right padding and a second document."""
    model = randomized_model(tiny_config(loop), seed=11).eval()
    reference = ReferenceDenseModel(model).eval()
    ids = torch.randint(0, 64, (2, 20))
    inputs: dict[str, torch.Tensor] = {}
    if packed:
        valid = torch.ones(2, 20, dtype=torch.bool)
        valid[0, :3] = False
        valid[1, 17:] = False
        positions = torch.stack(
            (torch.cat((torch.zeros(3), torch.arange(9), torch.arange(8))), torch.arange(20))
        ).long()
        inputs = {"attention_mask": valid, "position_ids": positions}
    with torch.no_grad():
        expected = reference(ids, **inputs)
        actual = model(ids, **inputs)
    keep = inputs["attention_mask"] if packed else torch.ones(2, 20, dtype=torch.bool)
    torch.testing.assert_close(actual[keep], expected[keep], rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("loop", [None, LOOP])
def test_gradients_match_the_upstream_forward(loop: Inkling2LoopConfig | None) -> None:
    model = randomized_model(tiny_config(loop), seed=13)
    reference = ReferenceDenseModel(model)
    ids = torch.randint(0, 64, (2, 16))
    upstream = torch.randn(2, 16, 64)
    (model(ids) * upstream).sum().backward()
    (reference(ids) * upstream).sum().backward()
    expected = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
        if name.endswith("attn.qkvr.weight"):
            continue  # stored in a different row order; its gradient is covered through the logits
        reference_grad = expected[name].grad
        # Gradients span several orders of magnitude; compare against the largest entry.
        assert (parameter.grad - reference_grad).abs().max() <= 1e-3 * reference_grad.abs().max(), name


def test_later_tokens_do_not_change_earlier_logits() -> None:
    model = randomized_model(tiny_config(LOOP), seed=5).eval()
    ids = torch.randint(0, 64, (1, 16))
    changed = ids.clone()
    changed[0, 10:] = (changed[0, 10:] + 1) % 64
    with torch.no_grad():
        torch.testing.assert_close(model(ids)[:, :10], model(changed)[:, :10], rtol=0, atol=1e-5)


def test_output_selection_and_input_errors() -> None:
    model = Inkling2DenseForCausalLM(tiny_config())
    ids = torch.randint(0, 64, (1, 8))
    full = model(ids)
    assert full.shape == (1, 8, 64) and full.dtype == torch.float32
    assert full.abs().max() <= model.config.final_logit_softcapping
    torch.testing.assert_close(model(ids, logits_to_keep=2), full[:, -2:])
    torch.testing.assert_close(model(ids, logits_to_keep=torch.tensor([0, 2])), full[:, [0, 2]])
    output = model(ids, logits_to_keep=1, output_hidden_states=True)
    assert output.logits.shape == (1, 1, 64) and output.hidden_states.shape == (1, 8, 32)
    with pytest.raises(ValueError, match="nonnegative"):
        model(ids, logits_to_keep=-1)
    with pytest.raises(ValueError, match="context range"):
        model(torch.zeros(1, 65, dtype=torch.long))
    with pytest.raises(ValueError, match="restart at zero"):
        model(ids, position_ids=torch.tensor([[0, 1, 2, 5, 6, 7, 8, 9]]))
    with pytest.raises(ValueError, match="requires CUDA"):
        Inkling2DenseForCausalLM(tiny_config(), backend=BackendConfig(attn="flex", linear="torch", rms_norm="torch"))(
            ids
        )


def test_head_is_tied_and_bf16_weights_keep_fp32_logits() -> None:
    model = Inkling2DenseForCausalLM(tiny_config(dtype="bfloat16"))
    assert model.lm_head.weight is model.embed_tokens.weight
    assert model.embed_tokens.weight.dtype == torch.bfloat16
    assert model.output_query.weight.dtype == torch.float32
    names = [name for name, _ in model.named_parameters()]
    assert "embed_tokens.weight" in names and "lm_head.weight" not in names
    logits = model(torch.randint(0, 64, (1, 8)))
    assert logits.dtype == torch.float32 and torch.isfinite(logits).all()
    restored = Inkling2DenseForCausalLM(tiny_config(dtype="bfloat16"))
    restored.load_state_dict(model.state_dict())
    assert restored.lm_head.weight is restored.embed_tokens.weight
    torch.testing.assert_close(restored.embed_tokens.weight, model.embed_tokens.weight, rtol=0, atol=0)


def test_dense_feed_forward_matches_the_inkling2_activation() -> None:
    """Gate rows then up rows, against the reference activation on its interleaved layout."""
    torch.manual_seed(7)
    config = tiny_config()
    mlp = CappedSwiGLU(config, BackendConfig(linear="torch"), dtype=torch.float32)
    mlp.reset_parameters(1.0)
    x = 4 * torch.randn(2, 5, config.hidden_size)
    gate, up = mlp.gate_up(x).chunk(2, dim=-1)
    interleaved = torch.stack((gate, up), dim=-1).flatten(-2)
    reference = capped_swiglu(
        interleaved.flatten(0, 1),
        torch.ones(10, 1),
        gate_softcap=config.mlp_gate_softcap,
        up_softcap=config.mlp_up_softcap,
    )
    torch.testing.assert_close(mlp(x), mlp.down(reference).view_as(x), rtol=1e-4, atol=1e-4)


def test_checkpoint_free_initialization_is_finite() -> None:
    model = Inkling2DenseForCausalLM(tiny_config(LOOP))
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter).all(), name
    assert torch.count_nonzero(model.output_query.weight) == 0
    assert model.layers["2"].controllers is not None and model.layers["0"].controllers is None
    assert model.layers["3"].mlp.gate_up.weight.std() > 0
    ids = torch.randint(0, 64, (2, 12))
    logits = model(ids)
    F.cross_entropy(logits[:, :-1].flatten(0, 1), ids[:, 1:].flatten()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


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


def test_default_geometry_and_rejected_settings() -> None:
    config = Inkling2DenseConfig()
    assert (config.hidden_size, config.num_hidden_layers, config.intermediate_size) == (2048, 24, 8192)
    assert config.local_layer_ids == [i for i in range(24) if i % 6 != 5]
    assert config.unpadded_vocab_size == config.vocab_size == 32768
    with pytest.raises(ValueError, match="ties its output projection"):
        Inkling2DenseConfig(tie_word_embeddings=False)
    with pytest.raises(ValueError, match="normalized depth blocks"):
        Inkling2DenseConfig(attn_res_normalize_values=False)
    with pytest.raises(ValueError, match="no padded vocabulary rows"):
        Inkling2DenseConfig(unpadded_vocab_size=32000)
