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

"""Dense Qwen3.8-Flash-Next research model with optional GR and n-gram memory."""

from nemo_automodel.components.models.qwen3_8_flash_next_mini.config import (
    Qwen3_8_FlashNextMiniConfig,
    Qwen3_8_FlashNextMiniTextConfig,
)
from nemo_automodel.components.models.qwen3_8_flash_next_mini.model import (
    Qwen3_8_FlashNextMiniForCausalLM,
)

__all__ = [
    "Qwen3_8_FlashNextMiniConfig",
    "Qwen3_8_FlashNextMiniForCausalLM",
    "Qwen3_8_FlashNextMiniTextConfig",
]
