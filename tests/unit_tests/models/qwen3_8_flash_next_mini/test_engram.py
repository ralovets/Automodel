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

"""Student-sized Engram/PLE table: sizing, neutrality, and trainability."""

from __future__ import annotations

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_8_flash_next_mini import (
    Qwen3_8_FlashNextMiniConfig,
    Qwen3_8_FlashNextMiniForCausalLM,
    Qwen3_8_FlashNextMiniTextConfig,
)
from nemo_automodel.components.models.qwen3_8_flash_next_mini.engram_support import (
    build_ngram_head_spec,
    neutralize_ple_layer,
)

_BACKEND = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", enable_hf_state_dict_adapter=False)
_HIDDEN = 64
_VOCAB = 256


def _text_config(ple_layer_ids: list[int], table_rows: int = 4096) -> Qwen3_8_FlashNextMiniTextConfig:
    """Build a tiny student text config.

    Args:
        ple_layer_ids: One-based decoder indices carrying an Engram table.
            ``[]`` builds the Engram-free student.
        table_rows: Approximate global Engram row budget.

    Returns:
        A tiny text config with a 3:1 GatedDeltaNet/full-attention pattern.
    """
    return Qwen3_8_FlashNextMiniTextConfig(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        max_position_embeddings=128,
        hc_count=4,
        hc_lowrank=16,
        indexer_budget=1024,
        indexer_head_dim=16,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        dtype="float32",
        ple_layer_ids=ple_layer_ids,
        ple_embed_dim=32,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_table_rows=table_rows,
        eos_token_id=_VOCAB - 1,
    )


def _build(ple_layer_ids: list[int], table_rows: int = 4096) -> Qwen3_8_FlashNextMiniForCausalLM:
    """Build and initialize a tiny student, optionally carrying Engram.

    Args:
        ple_layer_ids: One-based decoder indices carrying an Engram table.
        table_rows: Approximate global Engram row budget.

    Returns:
        An initialized, evaluated float32 student model.
    """
    config = Qwen3_8_FlashNextMiniConfig(text_config=_text_config(ple_layer_ids, table_rows))
    model = Qwen3_8_FlashNextMiniForCausalLM(config, backend=_BACKEND)
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    return model.eval()


def test_head_spec_matches_serving_layout() -> None:
    """Saved head layout uses successive primes and contiguous, padded row ranges."""
    sizes, offsets, padded = build_ngram_head_spec(4096, ngram_size=3, heads_per_ngram=8)
    expected = [257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349]
    assert list(sizes) == expected
    assert list(offsets) == [sum(expected[:i]) for i in range(len(expected))]
    assert sum(expected) <= padded < sum(expected) + 128
    assert padded % 128 == 0


def test_head_spec_rejects_budget_that_starves_heads() -> None:
    """A row budget too small for two rows per head is an error, not a silent clamp."""
    with pytest.raises(ValueError, match="fewer than two rows"):
        build_ngram_head_spec(8, ngram_size=3, heads_per_ngram=8)


def test_engram_layer_is_constructed_only_where_configured() -> None:
    """Only the configured one-based decoder index carries a PLE module."""
    model = _build([2])
    layers = model.model.language_model.layers
    assert layers["1"].ple is not None, "ple_layer_ids is one-based"
    assert [idx for idx, layer in layers.items() if layer.ple is not None] == ["1"]
    assert layers["1"]._nemo_disable_activation_checkpointing is True
    assert layers["0"]._nemo_disable_activation_checkpointing is False


def test_engram_hashes_stay_inside_the_table() -> None:
    """Every hashed row ID lands within the allocated table height."""
    model = _build([2])
    embedding = model.model.language_model.layers["1"].ple.ple_embedding
    input_ids = torch.randint(0, _VOCAB, (4, 32))
    row_ids = embedding._hash_input_ids(input_ids)
    assert row_ids.shape == (4, 32, 16)
    assert int(row_ids.min()) >= 0
    assert int(row_ids.max()) < embedding.ngram_embedding.weight.shape[0]


def test_neutral_engram_leaves_outputs_unchanged() -> None:
    """A neutrally initialized Engram layer does not shift the model's logits."""
    input_ids = torch.randint(0, _VOCAB, (2, 16))
    torch.manual_seed(0)
    without = _build([])
    torch.manual_seed(0)
    with_engram = _build([2])
    # Copy the shared decoder weights across so only Engram differs.
    shared = {k: v for k, v in without.state_dict().items()}
    missing = with_engram.load_state_dict(shared, strict=False).missing_keys
    assert all(".ple." in key or key == "lm_head.weight" for key in missing)
    with_engram.tie_weights()

    with torch.no_grad():
        base = without(input_ids).logits
        engram = with_engram(input_ids).logits
    torch.testing.assert_close(engram, base)


def test_neutral_engram_still_receives_gradient() -> None:
    """Zero-initialized PLE projections are trainable, not dead."""
    model = _build([2])
    ple = model.model.language_model.layers["1"].ple
    assert torch.count_nonzero(ple.value_proj.weight) == 0
    assert torch.count_nonzero(ple.conv1d.weight) == 0

    input_ids = torch.randint(0, _VOCAB, (2, 16))
    model.train()
    model(input_ids).logits.square().mean().backward()

    assert ple.value_proj.weight.grad is not None
    assert torch.count_nonzero(ple.value_proj.weight.grad) > 0, "neutral Engram must train away from zero"


def test_engram_changes_outputs_once_trained_away_from_zero() -> None:
    """A non-zero value projection makes Engram contribute to the forward pass."""
    input_ids = torch.randint(0, _VOCAB, (2, 16))
    model = _build([2])
    with torch.no_grad():
        neutral = model(input_ids).logits.clone()
        torch.nn.init.normal_(model.model.language_model.layers["1"].ple.value_proj.weight, std=0.05)
        active = model(input_ids).logits
    assert not torch.allclose(neutral, active), "a trained Engram must affect the output"


def test_neutralize_is_idempotent_and_restores_a_trained_layer() -> None:
    """Neutralizing a perturbed PLE layer returns it to an exact no-op."""
    input_ids = torch.randint(0, _VOCAB, (2, 16))
    model = _build([2])
    ple = model.model.language_model.layers["1"].ple
    with torch.no_grad():
        neutral = model(input_ids).logits.clone()
        torch.nn.init.normal_(ple.value_proj.weight, std=0.05)
        torch.nn.init.normal_(ple.conv1d.weight, std=0.05)
        neutralize_ple_layer(ple)
        restored = model(input_ids).logits
    torch.testing.assert_close(restored, neutral)


def test_engram_parameter_count_tracks_the_row_budget() -> None:
    """Table parameters scale with the configured row budget."""
    small = _build([2], table_rows=4096)
    large = _build([2], table_rows=16384)

    def table_size(model: Qwen3_8_FlashNextMiniForCausalLM) -> int:
        return model.model.language_model.layers["1"].ple.ple_embedding.ngram_embedding.weight.numel()

    assert table_size(large) > 3 * table_size(small)
