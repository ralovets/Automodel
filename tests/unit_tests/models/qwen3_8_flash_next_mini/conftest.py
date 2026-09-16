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

"""CPU reference kernels shared by the custom-model unit tests."""

import inspect

import pytest

from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import (
    CPAwareGatedDeltaNet,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)
from nemo_automodel.components.models.qwen3_8_flash_next_mini import Qwen3_8_FlashNextMiniTextConfig


@pytest.fixture(autouse=True)
def cpu_delta_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CPU tests on reference kernels even when CUDA extensions are installed."""
    original_init = CPAwareGatedDeltaNet.__init__

    def initialize(self: CPAwareGatedDeltaNet, config: Qwen3_8_FlashNextMiniTextConfig, layer_idx: int) -> None:
        original_init(self, config, layer_idx)
        self.causal_conv1d_fn = None
        # Transformers can decorate even the torch fallbacks with CUDA dispatch.
        self.chunk_gated_delta_rule = inspect.unwrap(torch_chunk_gated_delta_rule)
        self.recurrent_gated_delta_rule = inspect.unwrap(torch_recurrent_gated_delta_rule)

    monkeypatch.setattr(CPAwareGatedDeltaNet, "__init__", initialize)
