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

"""Tie metadata must survive a DDP wrapper.

``DistributedDataParallel`` is a real wrapper: ``nn.Module.__getattr__`` falls
back only to parameters, buffers and submodules, so a model's
``tie_word_embeddings_support``, ``config`` and ``tie_weights`` are invisible
through it. When that made the checkpointer read a tied model as untied, the
save deduplicated ``lm_head.weight`` away and the resume failed with
``Missing key in checkpoint state_dict: lm_head.weight``.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from nemo_automodel.shared.tied_weights import (
    TieSupport,
    ensure_tied_lm_head,
    get_lm_head_weight_and_name,
    has_local_tied_lm_head,
    is_tied_word_embeddings,
)

_VOCAB, _HIDDEN = 8, 4


class _TiedModel(nn.Module):
    """Minimal tied-head model with the same contract as a registered one."""

    tie_word_embeddings_support = TieSupport.TIED_ONLY

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(_VOCAB, _HIDDEN)
        self.lm_head = nn.Linear(_HIDDEN, _VOCAB, bias=False)
        self.tie_weights()

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def tie_weights(self) -> None:
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.embed_tokens(input_ids))


@pytest.fixture
def single_rank_group():
    """Initialize a one-rank gloo group without touching the network."""
    if dist.is_initialized():
        pytest.skip("a process group is already initialized")
    dist.init_process_group(backend="gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def test_tie_metadata_survives_the_ddp_wrapper(single_rank_group) -> None:
    """Every tie helper must report the same answer wrapped or unwrapped."""
    model = _TiedModel()
    wrapped = DistributedDataParallel(model)

    # The premise: DDP genuinely hides the attribute these helpers read.
    assert not hasattr(DistributedDataParallel, "tie_word_embeddings_support")

    assert is_tied_word_embeddings(model) is True
    assert is_tied_word_embeddings(wrapped) is True, "a tied model must not read as untied through DDP"
    assert has_local_tied_lm_head(wrapped) is True
    assert ensure_tied_lm_head(wrapped) is True

    # The returned FQN keys into the checkpoint, so it must carry no wrapper prefix.
    _, name = get_lm_head_weight_and_name(wrapped)
    assert name == "lm_head.weight", f"DDP prefix leaked into the checkpoint key: {name}"


def test_untied_model_still_reads_untied_through_ddp(single_rank_group) -> None:
    """Unwrapping must not turn every wrapped model into a tied one."""

    class _Untied(_TiedModel):
        tie_word_embeddings_support = TieSupport.UNTIED_ONLY

    model = _Untied()
    model.lm_head = nn.Linear(_HIDDEN, _VOCAB, bias=False)
    assert is_tied_word_embeddings(DistributedDataParallel(model)) is False
