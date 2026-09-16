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

"""Registration, sharding-strategy, and weight-tying contracts for the student.

Each check here pins something a caller reaches through framework machinery
rather than through the model's own API: which parallelization strategy FSDP2
selects, which config class ``AutoConfig`` resolves from a saved checkpoint, and
whether the tied head survives construction.
"""

from __future__ import annotations

import json
import pathlib

import pytest
import torch

from nemo_automodel.components.distributed.parallelizer import (
    Qwen3_5ParallelizationStrategy,
    get_parallelization_strategy,
)
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_8_flash_next_mini import (
    Qwen3_8_FlashNextMiniConfig,
    Qwen3_8_FlashNextMiniForCausalLM,
    Qwen3_8_FlashNextMiniTextConfig,
)

transformers = pytest.importorskip("transformers")

_TINY = dict(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    linear_num_key_heads=2,
    linear_num_value_heads=2,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    hc_count=4,
    hc_lowrank=16,
    ple_layer_ids=[],
    ngram_table_rows=1024,
)
_BACKEND = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", enable_hf_state_dict_adapter=False)


def _text_config(**overrides) -> Qwen3_8_FlashNextMiniTextConfig:
    return Qwen3_8_FlashNextMiniTextConfig(**{**_TINY, **overrides})


def _causal_lm() -> Qwen3_8_FlashNextMiniForCausalLM:
    return Qwen3_8_FlashNextMiniForCausalLM(Qwen3_8_FlashNextMiniConfig(text_config=_text_config()), backend=_BACKEND)


def test_student_selects_the_mixed_dtype_strategy() -> None:
    """The student needs dtype-split FSDP units, not the default strategy.

    Every decoder layer holds fp32 ``_fp32_params`` beside bf16 weights, and
    plain ``fully_shard`` rejects a unit of mixed parameter dtype.
    """
    assert isinstance(get_parallelization_strategy(_causal_lm()), Qwen3_5ParallelizationStrategy)


@pytest.mark.parametrize(
    "layer_types",
    [None, ["full_attention", "linear_attention", "linear_attention", "full_attention"]],
)
def test_hybrid_layer_types_survive_config_reload(tmp_path: pathlib.Path, layer_types: list[str] | None) -> None:
    """DeltaNet construction can index the mixer pattern before and after saving."""
    config = _text_config(num_hidden_layers=4, layer_types=layer_types)
    expected = layer_types or ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    assert config.layer_types == expected
    config.save_pretrained(tmp_path)
    restored = Qwen3_8_FlashNextMiniTextConfig.from_pretrained(tmp_path)
    assert restored.layer_types == expected


def test_decoder_layers_hold_mixed_parameter_dtypes() -> None:
    """The premise of the test above: a decoder layer really is mixed-dtype."""
    layer = _causal_lm().model.language_model.layers["0"]
    dtypes = {parameter.dtype for parameter in layer.parameters()}
    assert torch.float32 in dtypes and torch.bfloat16 in dtypes, dtypes


def test_autoconfig_resolves_the_student_from_a_saved_checkpoint(tmp_path: pathlib.Path) -> None:
    """A checkpoint-style ``config.json`` must resolve to the local config class.

    ``qwen3_8_flash_next_mini`` is not in the installed Transformers
    ``CONFIG_MAPPING``, so without the ``_CUSTOM_CONFIG_REGISTRATIONS`` entry
    ``AutoConfig`` cannot load a saved student.
    """
    from transformers import AutoConfig

    Qwen3_8_FlashNextMiniConfig(text_config=_text_config()).save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["model_type"] == "qwen3_8_flash_next_mini"

    config = AutoConfig.from_pretrained(tmp_path)
    assert isinstance(config, Qwen3_8_FlashNextMiniConfig)
    assert isinstance(config.text_config, Qwen3_8_FlashNextMiniTextConfig)
    assert config.text_config.intermediate_size == _TINY["intermediate_size"]


def test_tied_head_aliases_the_embedding_table() -> None:
    """``TIED_ONLY`` means one tensor, not two tensors that happen to match."""
    model = _causal_lm()
    assert model.lm_head.weight is model.get_input_embeddings().weight

    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    assert model.lm_head.weight is model.get_input_embeddings().weight, "initialize_weights must re-tie"

    state_dict = model.state_dict()
    head, embedding = state_dict["lm_head.weight"], state_dict["model.language_model.embed_tokens.weight"]
    assert head.data_ptr() == embedding.data_ptr(), "the saved head must share storage with the embedding table"
    assert "lm_head.weight" in Qwen3_8_FlashNextMiniForCausalLM._tied_weights_keys


def test_untied_config_is_rejected() -> None:
    """The student ships no separate head, so untying must fail at construction."""
    config = Qwen3_8_FlashNextMiniConfig(text_config=_text_config(tie_word_embeddings=False))
    config.tie_word_embeddings = False
    with pytest.raises(NotImplementedError, match="tie_word_embeddings"):
        Qwen3_8_FlashNextMiniForCausalLM(config, backend=_BACKEND)


def test_context_parallelism_is_not_advertised() -> None:
    """CP needs a model-owned sharder the student does not implement.

    ``supports_cp`` and the ``_owns_cp_attention`` marker each steer the recipe
    down a Flash-Next CP path that requires ``prepare_model_inputs_for_cp``.
    Until that is ported, none of the three may be present.
    """
    assert Qwen3_8_FlashNextMiniForCausalLM.ModelCapabilities().supports_cp is False
    assert not hasattr(Qwen3_8_FlashNextMiniForCausalLM, "_owns_cp_attention")
    assert not hasattr(Qwen3_8_FlashNextMiniForCausalLM, "prepare_model_inputs_for_cp")


def test_engram_owner_sharding_is_not_exposed() -> None:
    """Owner-sharded Engram needs a DTensor hook the dense strategy never calls."""
    config = Qwen3_8_FlashNextMiniConfig(text_config=_text_config())
    with pytest.raises(TypeError, match="engram_process_group"):
        Qwen3_8_FlashNextMiniForCausalLM(config, backend=_BACKEND, engram_process_group=object())
