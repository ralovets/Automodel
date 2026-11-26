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

"""Bits per byte of dense students built from a released Inkling2 checkpoint, with no training.

    python experiment/transplant.py --model MODEL_DIR --usage usage.npz --heldout heldout.jsonl \\
        --output scores.json --variants teacher random/8 8/zero/untied 8/top/untied 4/top/untied

A student has the teacher's shape outside the feed-forward, and a dense feed-forward as wide as ``k`` experts.
A variant is ``teacher``, ``random/k`` (untrained student) or ``k/ffn/head``:

- ``ffn``: ``random`` keeps the initialization, ``zero`` zeroes the output projection, ``top`` places the layer's
  ``k`` most used experts side by side with fixed weights that sum to the teacher's mean total routing weight, in
  proportion to their use; ``expected`` gives each its mean routing weight over all tokens instead. Use is the
  routing weight in ``--usage`` (from ``expert_usage.py``), summed over domains and loop passes.
- ``head``: ``untied`` copies the teacher's input embedding and output projection separately; ``embed`` and
  ``head`` tie both to the teacher's embedding or to its output projection.

Everything else is copied by name; the fused attention projection changes row order. The first documents of each heldout domain, up to ``--tokens`` teacher tokens,
are tokenized with the teacher's tokenizer and scored as in ``score_bpb.py``.
"""

import argparse
import inspect
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from expert_usage import load
from score_bpb import score

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.inkling2.config import Inkling2TextConfig
from nemo_automodel.components.models.inkling2_dense import Inkling2DenseConfig, Inkling2DenseForCausalLM

_TEACHER_ONLY = (
    "n_routed_experts", "num_experts_per_tok", "n_shared_experts", "route_scale", "use_gate_bias",
    "quantization_config", "tie_word_embeddings", "max_position_embeddings", "model_max_length",
)  # fmt: skip


def expert_weights(
    usage: dict[str, np.ndarray], k: int, expected: bool, selected: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return each layer's ``k`` most used experts [layers, k] and the fixed weight of each [layers, k].

    Args:
        usage: ``weight`` and ``count`` of shape [layers, passes, domains, experts] from ``expert_usage.py``.
        k: Experts kept per layer.
        expected: Weight each expert by its mean routing weight over all tokens; otherwise the kept experts
            share the mean total routing weight of a token in proportion to their use.
        selected: Experts the teacher selects per token.
    """
    weight, count = usage["weight"].sum(axis=(1, 2)), usage["count"].sum(axis=(1, 2))  # [layers, experts]
    chosen = np.argsort(-weight, axis=1)[:, :k]
    kept = np.take_along_axis(weight, chosen, axis=1)
    tokens = count.sum(axis=1, keepdims=True) / selected  # router calls per layer
    if expected:
        return chosen, kept / tokens
    return chosen, kept / kept.sum(axis=1, keepdims=True) * weight.sum(axis=1, keepdims=True) / tokens


def transplant(
    teacher: torch.nn.Module, k: int, ffn: str, head: str, usage: dict[str, np.ndarray] | None, attn: str = "flex"
) -> torch.nn.Module:
    """Return a dense student on the teacher's device, in eval mode, filled as the module docstring describes."""
    config = teacher.text_config
    device = next(teacher.parameters()).device
    # The student takes every teacher setting that is not about experts, quantization or context range.
    names = set(inspect.signature(Inkling2TextConfig.__init__).parameters) - {"self", "kwargs", *_TEACHER_ONLY}
    fields = {name: value for name, value in config.to_dict().items() if name in names}
    fields.update(unpadded_vocab_size=config.vocab_size, intermediate_size=k * config.intermediate_size)
    student_config = Inkling2DenseConfig(**fields)
    backend = BackendConfig(attn=attn, linear="torch", rms_norm="torch", enable_hf_state_dict_adapter=False)
    with torch.device(device):
        student = Inkling2DenseForCausalLM(student_config, backend=backend)
    if ffn == "untrained":
        return student.eval()
    source = teacher.state_dict()
    if ffn in ("top", "expected"):
        assert usage is not None
        chosen, fixed = expert_weights(usage, k, ffn == "expected", config.num_experts_per_tok)
    with torch.no_grad():
        for name, target in student.state_dict().items():
            if name == "lm_head.weight":
                continue
            if name.endswith(".attn.qkvr.weight"):
                # Teacher rows [Q, K, V, R, gate, sink] and zero padding; student rows [Q, R, gate, sink, K, V].
                attention = student.layers[name.split(".")[1]].attn
                q, r, gate, sink = attention.query_sizes
                kv = attention.kv_size
                rows = source[name][: target.shape[0]].split((q, kv, kv, r, gate, sink))
                assert not source[name][target.shape[0] :].any(), name
                target.copy_(torch.cat((rows[0], rows[3], rows[4], rows[5], rows[1], rows[2])))
            elif not name.endswith((".mlp.gate_up.weight", ".mlp.down.weight")):
                assert source[name].shape == target.shape, name
                target.copy_(source[name])
            elif ffn == "zero" and name.endswith(".mlp.down.weight"):
                target.zero_()
            elif ffn in ("top", "expected"):
                layer = int(name.split(".")[1])
                experts = torch.from_numpy(chosen[layer]).to(device)
                if name.endswith(".mlp.gate_up.weight"):
                    # Teacher [experts, hidden, 2 * width], gate and up interleaved; student rows: gates, then ups.
                    fused = source[f"layers.{layer}.mlp.experts.gate_and_up_projs"][experts]
                    rows = torch.cat((fused[..., 0::2], fused[..., 1::2]), dim=0)  # [2k, hidden, width]
                    target.copy_(rows.permute(0, 2, 1).flatten(0, 1))
                else:
                    # Teacher [experts, width, hidden]; the routing weight scales the activation, so it folds in here.
                    down = source[f"layers.{layer}.mlp.experts.down_projs"][experts].float()
                    scale = torch.from_numpy(fixed[layer]).to(device, torch.float32)[:, None, None]
                    target.copy_((down * scale).flatten(0, 1).T)
        tied = {"embed": source["embed_tokens.weight"], "head": source["lm_head.weight"]}.get(head)
        student.embed_tokens.weight.copy_(source["embed_tokens.weight"] if tied is None else tied)
        if tied is None:
            # The dense model ties its head; this comparison needs the teacher's separate output projection.
            student.lm_head.weight = torch.nn.Parameter(source["lm_head.weight"].to(student.lm_head.weight.dtype))
    return student.eval()


class UsedVocabulary(torch.nn.Module):
    """A dense student whose logits for the teacher's unused vocabulary rows are negative infinity."""

    def __init__(self, model: torch.nn.Module, used: int) -> None:
        super().__init__()
        self.model, self.used = model, used

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return FP32 logits [batch, sequence, vocab] for ``input_ids`` [batch, sequence]."""
        logits = self.model(input_ids)
        logits[..., self.used :] = float("-inf")
        return logits


def main() -> None:
    """Score the teacher and the requested students on the heldout subset, one window per forward: the logits
    of one 4,096-token window over this vocabulary take 3 GiB."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="Released Inkling2 checkpoint directory")
    parser.add_argument("--usage", type=Path, required=True, help="usage.npz from expert_usage.py")
    parser.add_argument("--heldout", type=Path, required=True, help="Frozen heldout JSONL from build_heldout.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--tokens", type=float, default=3e5, help="Teacher tokens scored per domain")
    parser.add_argument("--window", type=int, default=4096)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    records = [json.loads(line) for line in args.heldout.open(encoding="utf-8")]
    streams = {}
    for domain in sorted({record["domain"] for record in records}):
        texts, ids, total = [record["text"] for record in records if record["domain"] == domain], [], 0
        for text in texts:
            if total >= args.tokens:
                break
            ids.append([*tokenizer(text, add_special_tokens=False)["input_ids"], tokenizer.eos_token_id])
            total += len(ids[-1])
        nbytes = sum(len(text.encode("utf-8")) for text in texts[: len(ids)])
        streams[domain] = (np.concatenate([np.asarray(x) for x in ids]), np.array([len(x) for x in ids]), nbytes)
    usage = dict(np.load(args.usage))
    teacher = load(args.model, "eager", torch.device("cuda"))
    results = json.loads(args.output.read_text()) if args.output.exists() else {}
    for variant in args.variants:
        began = time.time()
        parts = variant.split("/")
        if variant == "teacher":
            model = teacher
        else:
            k, ffn, head = (
                (int(parts[1]), "untrained", "embed") if parts[0] == "random" else (int(parts[0]), *parts[1:])
            )
            model = UsedVocabulary(transplant(teacher, k, ffn, head, usage), teacher.text_config.unpadded_vocab_size)
        row, nll_all, bytes_all = {}, 0.0, 0
        for domain, (stream, lengths, nbytes) in streams.items():
            owner = np.repeat(np.arange(len(lengths)), lengths)
            total = score(model, stream, owner, len(lengths), args.window, 1).sum()
            row[domain] = total / math.log(2) / nbytes
            nll_all, bytes_all = nll_all + total, bytes_all + nbytes
        row["all"] = nll_all / math.log(2) / bytes_all
        results[variant] = row
        args.output.write_text(json.dumps(results, indent=1) + "\n")
        print(f"{variant:22s} bpb {row['all']:.4f}  " + " ".join(f"{d} {v:.3f}" for d, v in row.items() if d != "all"),
              f" {time.time() - began:.0f}s", flush=True)  # fmt: skip
        if model is not teacher:
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
