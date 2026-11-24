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

"""Dense Inkling2 on CUDA: compiled FlexAttention against the eager reference, and BF16 training steps."""

from collections.abc import Iterator

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.inkling2.loss import Inkling2FusedLinearCrossEntropy
from nemo_automodel.components.models.inkling2_dense import Inkling2DenseForCausalLM
from tests.unit_tests.models.inkling2_dense.test_model import LOOP, randomized_model, tiny_config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Compiled FlexAttention requires CUDA")

FLEX = BackendConfig(attn="flex", linear="torch", rms_norm="torch", enable_hf_state_dict_adapter=False)


@pytest.fixture(autouse=True)
def exact_fp32_matmul() -> Iterator[None]:
    # Each case builds its own geometry; keep the guards of one case out of the next.
    torch._dynamo.reset()
    # The framework enables TF32, which rounds the FP32 products both paths are compared on.
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize("rows", ["plain", "documents", "padded"])
def test_flex_attention_matches_eager_logits_and_gradients(rows: str) -> None:
    """Rows of 400 tokens span several kernel blocks, and the 150-token window crosses block edges.

    Covers single-document rows, rows holding two documents, and the same with left and right padding.
    """
    config = tiny_config(LOOP, max_position_embeddings=512, sliding_window_size=150)
    eager = randomized_model(config, seed=23).cuda()
    flex = Inkling2DenseForCausalLM(config, backend=FLEX).cuda()
    flex.load_state_dict(eager.state_dict())
    length = 400
    ids = torch.randint(0, 64, (2, length), device="cuda")
    inputs: dict[str, torch.Tensor] = {}
    keep = torch.ones(2, length, dtype=torch.bool, device="cuda")
    if rows == "documents":
        first = torch.cat((torch.arange(170), torch.arange(230)))
        inputs = {"position_ids": torch.stack((first, torch.arange(length))).long().cuda()}
    elif rows == "padded":
        keep[0, :3] = False
        keep[1, 380:] = False
        first = torch.cat((torch.zeros(3), torch.arange(167), torch.arange(230)))
        inputs = {"attention_mask": keep, "position_ids": torch.stack((first, torch.arange(length))).long().cuda()}
    upstream = torch.randn(2, length, 64, device="cuda") * keep.unsqueeze(-1)
    expected, actual = eager(ids, **inputs), flex(ids, **inputs)
    assert torch.isfinite(actual[keep]).all()
    torch.testing.assert_close(actual[keep], expected[keep], rtol=1e-3, atol=1e-3)
    (expected * upstream).sum().backward()
    (actual * upstream).sum().backward()
    reference = dict(eager.named_parameters())
    for name, parameter in flex.named_parameters():
        gradient = reference[name].grad
        assert (parameter.grad - gradient).abs().max() <= 2e-3 * gradient.abs().max(), name


def test_bf16_training_steps_reduce_the_loss() -> None:
    torch.manual_seed(29)
    model = Inkling2DenseForCausalLM(tiny_config(LOOP, dtype="bfloat16"), backend=FLEX).cuda()
    loss_fn = Inkling2FusedLinearCrossEntropy(model.config.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    ids = torch.randint(0, 64, (4, 48), device="cuda")
    labels = torch.roll(ids, -1, 1)
    losses = []
    for _ in range(30):
        output = model(ids, logits_to_keep=1, output_hidden_states=True)
        loss = loss_fn(output.hidden_states, labels, model.lm_head.weight, num_label_tokens=labels.numel())
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    assert losses[-1] < 0.7 * losses[0]


@pytest.mark.parametrize("loop", [None, LOOP], ids=["no_loop", "loop"])
def test_bf16_flex_path_is_causal_at_full_length(loop) -> None:
    """Later tokens leave earlier hidden states bitwise unchanged at 4,096 tokens.

    Cuts sit at, just before and just after kernel-block and window edges, so the cached block
    masks, the 1,024-token window and the short convolutions' shifts are all exercised.
    """
    config = tiny_config(loop, dtype="bfloat16", max_position_embeddings=4096, sliding_window_size=1024)
    model = Inkling2DenseForCausalLM(config, backend=FLEX).cuda()
    model.load_state_dict(randomized_model(config, seed=31).state_dict())
    length = 4096
    ids = torch.randint(0, 64, (1, length), device="cuda")
    with torch.no_grad():
        base = model(ids, logits_to_keep=1, output_hidden_states=True).hidden_states
        for cut in (1, 127, 128, 129, 1023, 1024, 1025, 2048, 4095):
            changed = ids.clone()
            changed[0, cut:] = (ids[0, cut:] + 1 + torch.randint(0, 63, (length - cut,), device="cuda")) % 64
            other = model(changed, logits_to_keep=1, output_hidden_states=True).hidden_states
            assert torch.equal(base[0, :cut], other[0, :cut]), cut
            assert not torch.equal(base[0, cut:], other[0, cut:]), cut
