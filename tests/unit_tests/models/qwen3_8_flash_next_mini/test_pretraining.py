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

"""Full-attention pretraining through the existing dense Flash-Next model."""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_model, save_model
from transformers import AutoConfig

from nemo_automodel._transformers.registry import MODEL_ARCH_MAPPING
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_8_flash_next_mini import (
    Qwen3_8_FlashNextMiniConfig,
    Qwen3_8_FlashNextMiniForCausalLM,
    Qwen3_8_FlashNextMiniTextConfig,
)


def _config(use_qsa: bool = False) -> Qwen3_8_FlashNextMiniConfig:
    return Qwen3_8_FlashNextMiniConfig(
        architectures=["Qwen3_8_FlashNextMiniForCausalLM"],
        text_config=Qwen3_8_FlashNextMiniTextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=96,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            hc_count=4,
            hc_lowrank=4,
            use_qsa=use_qsa,
            ple_layer_ids=[],
            indexer_budget=1024,
            indexer_head_dim=8,
            indexer_n_heads=2,
            dtype="float32",
            mtp_num_hidden_layers=0,
            use_cache=False,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=0,
        ),
    )


def _model(use_qsa: bool = False) -> Qwen3_8_FlashNextMiniForCausalLM:
    model = Qwen3_8_FlashNextMiniForCausalLM(
        _config(use_qsa),
        backend=BackendConfig(attn="sdpa", linear="torch", rms_norm="torch"),
    )
    return model


def test_real_device_constructor_initializes_scratch_weights() -> None:
    """DDP uses the constructor without a later meta/checkpoint initializer."""
    model = _model()
    embedding_std = model.get_input_embeddings().weight.float().std().item()
    assert 0.015 < embedding_std < 0.025
    tokens = torch.randint(2, 64, (2, 12))
    logits = model(tokens).logits
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 64), tokens[:, 1:].reshape(-1))
    assert 3 < loss.item() < 6


@pytest.mark.parametrize("padded", [False, True])
def test_full_attention_matches_dense_qsa_forward_and_backward(padded: bool) -> None:
    """Full attention preserves the existing dense-QSA function and gradients."""
    torch.manual_seed(42)
    reference = _model(use_qsa=True)
    actual = _model()
    shared = {k: v for k, v in reference.state_dict().items() if ".indexer." not in k}
    actual.load_state_dict(shared, strict=True)
    tokens = torch.randint(2, 64, (2, 12))
    mask = torch.ones_like(tokens)
    if padded:
        mask[0, -3:] = 0
    expected_logits = reference(tokens, attention_mask=mask).logits
    actual_logits = actual(tokens, attention_mask=mask).logits
    # Padding query outputs are not supervised and may differ by backend.
    valid = mask.bool()
    torch.testing.assert_close(actual_logits[valid], expected_logits[valid], rtol=1e-5, atol=1e-6)
    upstream = torch.randn_like(actual_logits[valid])
    actual_logits[valid].backward(upstream)
    expected_logits[valid].backward(upstream)
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in actual.named_parameters():
        expected = reference_parameters[name].grad
        assert parameter.grad is not None, name
        assert expected is not None, name
        torch.testing.assert_close(parameter.grad, expected, rtol=2e-4, atol=2e-5, msg=name)


def test_meta_initialization_causality_and_learning() -> None:
    """Scratch initialization overwrites all parameters and supports learning."""
    torch.manual_seed(13)
    with torch.device("meta"):
        model = Qwen3_8_FlashNextMiniForCausalLM(_config())
    model.to_empty(device="cpu")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(float("nan"))
        for buffer in model.buffers():
            if buffer.is_floating_point():
                buffer.fill_(float("nan"))
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter).all(), name
    assert model.lm_head.weight is model.get_input_embeddings().weight
    assert model.backend.attn == "sdpa"
    rotary = model.model.language_model.rotary_emb
    expected_frequencies = _model().model.language_model.rotary_emb.inv_freq
    torch.testing.assert_close(rotary.inv_freq, expected_frequencies, rtol=0, atol=0)
    rotary.to(dtype=torch.bfloat16)
    assert rotary.inv_freq.dtype == torch.float32
    torch.testing.assert_close(rotary.inv_freq, expected_frequencies, rtol=0, atol=0)
    assert not any("indexer" in n or ".ple." in n or ".mtp." in n for n, _ in model.named_parameters())
    tokens = torch.randint(2, 64, (2, 12))
    changed = tokens.clone()
    changed[:, 6:] = torch.randint(2, 64, (2, 6))
    with torch.no_grad():
        torch.testing.assert_close(model(tokens).logits[:, :6], model(changed).logits[:, :6], rtol=0, atol=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    losses = []
    for _ in range(6):
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens).logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 64), tokens[:, 1:].reshape(-1))
        losses.append(loss.item())
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        optimizer.step()
    assert losses[-1] < losses[0], losses


def test_checkpointer_materializes_meta_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The framework's parameter-only materializer also restores valid RoPE."""
    # The model's default buffer device follows CUDA availability; this is a CPU test.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with torch.device("meta"):
        model = Qwen3_8_FlashNextMiniForCausalLM(_config())
    Checkpointer.initialize_model_weights(model, torch.device("cpu"))
    expected = _model().model.language_model.rotary_emb.inv_freq
    torch.testing.assert_close(model.model.language_model.rotary_emb.inv_freq, expected, rtol=0, atol=0)
    tokens = torch.randint(2, 64, (2, 8))
    assert torch.isfinite(model(tokens).logits).all()


@pytest.mark.parametrize(
    "use_gr,ffn_width,expected_parameters",
    [(True, 6144, 1_122_283_392), (False, 6144, 982_620_032), (False, 7552, 1_121_032_064)],
)
def test_default_geometry_and_study_arm_parameter_counts(
    use_gr: bool, ffn_width: int, expected_parameters: int
) -> None:
    """The config defaults are the ~1B study geometry, and each arm keeps its parameter budget."""
    config = Qwen3_8_FlashNextMiniConfig(
        text_config=Qwen3_8_FlashNextMiniTextConfig(use_gr=use_gr, intermediate_size=ffn_width)
    )
    with torch.device("meta"):
        model = Qwen3_8_FlashNextMiniForCausalLM(config)
    assert sum(p.numel() for p in model.parameters()) == expected_parameters
    assert config.text_config.vocab_size == 32768 and config.text_config.hidden_size == 2048
    assert config.text_config.hc_count == 4 and config.text_config.hc_lowrank == 256
    assert config.text_config.use_qsa is False and config.text_config.ple_layer_ids == []
    layers = model.model.language_model.layers.values()
    assert [layer.layer_type for layer in layers] == (["linear_attention"] * 3 + ["full_attention"]) * 4
    assert all(layer.ple is None for layer in layers)
    assert all(not hasattr(layer.self_attn, "indexer") for layer in layers if layer.layer_type == "full_attention")
    assert model.lm_head.weight is model.get_input_embeddings().weight


def test_standard_residual_matches_prenorm_equation_and_gradients() -> None:
    """GR-off is x + sublayer(RMSNorm(x)), with no learned read/write gate."""
    from nemo_automodel.components.models.qwen3_8_flash_next_mini.layers import StandardResidual

    residual = StandardResidual(16, 1e-6)
    x = torch.randn(2, 7, 16, requires_grad=True)
    reference = x.detach().clone().requires_grad_(True)
    normalized, state = residual.mix(x)
    actual = residual.combine(normalized.sin(), state)
    expected = reference + F.rms_norm(reference, (16,), eps=1e-6).sin()
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, reference.grad)
    assert sum(p.numel() for p in residual.parameters()) == 16


@pytest.mark.parametrize("use_gr", [False, True])
@pytest.mark.parametrize("memory", [False, True])
def test_residual_memory_ablation_learning_and_roundtrip(tmp_path: Path, use_gr: bool, memory: bool) -> None:
    """All four ablations learn and preserve geometry and logits when reloaded."""
    config = _config()
    config.text_config.use_gr = use_gr
    config.text_config.ple_layer_ids = [2] if memory else []
    config.text_config.ple_embed_dim = 32
    config.text_config.ngram_table_rows = 4096
    model = Qwen3_8_FlashNextMiniForCausalLM(
        config, backend=BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    tokens = torch.randint(2, 64, (2, 12))
    losses = []
    for _ in range(4):
        optimizer.zero_grad()
        logits = model(tokens).logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 64), tokens[:, 1:].reshape(-1))
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    assert config.text_config.residual_streams == (4 if use_gr else 1)
    config.save_pretrained(tmp_path)
    restored = Qwen3_8_FlashNextMiniForCausalLM.from_config(AutoConfig.from_pretrained(tmp_path))
    save_model(model, tmp_path / "model.safetensors")
    load_model(restored, tmp_path / "model.safetensors", strict=True)
    assert model.config.architectures[0] in MODEL_ARCH_MAPPING
    assert restored.config.text_config.use_qsa is False
    assert restored.config.text_config.use_gr == use_gr
    assert restored.config.text_config.ple_layer_ids == ([2] if memory else [])
    assert restored.lm_head.weight is restored.get_input_embeddings().weight
    assert Qwen3_8_FlashNextMiniTextConfig().use_qsa is False
    torch.testing.assert_close(restored(tokens).logits, model(tokens).logits, rtol=0, atol=0)
