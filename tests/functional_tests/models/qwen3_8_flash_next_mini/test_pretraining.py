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

"""Exercise scratch initialization through the public AutoModel GPU entry point."""

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_8_flash_next_mini import (
    Qwen3_8_FlashNextMiniConfig,
    Qwen3_8_FlashNextMiniTextConfig,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="AutoModel construction and BF16 kernels require CUDA")
@pytest.mark.parametrize("use_gr,memory", [(True, False), (False, False), (True, True), (False, True)])
def test_pretraining_from_config_forward_backward(use_gr: bool, memory: bool) -> None:
    """The public meta-materialization path initializes a trainable hybrid."""
    config = Qwen3_8_FlashNextMiniConfig(
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
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            hc_lowrank=4,
            use_gr=use_gr,
            ple_layer_ids=[2] if memory else [],
            ple_embed_dim=32,
            ngram_table_rows=4096,
        ),
    )
    model = NeMoAutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.bfloat16,
        use_liger_kernel=False,
        attn_implementation="sdpa",
        backend=BackendConfig(attn="sdpa", linear="torch", rms_norm="torch"),
    )
    assert model.lm_head.weight is model.get_input_embeddings().weight
    assert 0.015 < model.get_input_embeddings().weight.float().std().item() < 0.025
    assert not any("indexer" in name for name, _ in model.named_parameters())
    tokens = torch.randint(2, 64, (2, 128), device="cuda")
    logits = model(tokens).logits
    loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, 64), tokens[:, 1:].reshape(-1))
    assert 3 < loss.item() < 6
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
