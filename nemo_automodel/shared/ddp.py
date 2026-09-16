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

"""Helpers for reading model metadata through the DDP wrapper."""

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


def unwrap_ddp_model(model: nn.Module) -> nn.Module:
    """Return the module that owns model metadata, seeing through DDP.

    ``DistributedDataParallel`` is a real wrapper: ``nn.Module.__getattr__``
    only falls back to parameters, buffers and submodules, so class attributes
    such as ``tie_word_embeddings_support``, ``config`` or ``state_dict_adapter``
    and methods such as ``tie_weights`` never reach ``.module``. FSDP2 is
    unaffected because ``fully_shard`` mutates the module in place.

    Args:
        model: Possibly DDP-wrapped model.

    Returns:
        The underlying module, or ``model`` when it is not DDP-wrapped.
    """
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model
