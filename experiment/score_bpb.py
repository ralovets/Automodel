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

"""Bits per byte of checkpoints and outside models on the frozen heldout.

    python experiment/score_bpb.py --heldout heldout.jsonl --dataset DATA_DIR --output scores.json \\
        --models inkling:OUTPUT_DIR/RUN/epoch_0_step_13999/model/consolidated Qwen/Qwen2.5-Coder-0.5B

Each domain's documents are concatenated in heldout order, each ending with EOS, and cut
into fixed 4,096-token windows as in the recipe's validation, but every target is scored
exactly once (the recipe drops partial windows). An Inkling checkpoint reads the dataset's
validation token streams, which ``prepare_data.py`` wrote from the same heldout; other
models tokenize the heldout text themselves. Bits per byte = summed NLL / ln 2 / UTF-8
bytes. Each target's NLL is also credited to the document that holds it, in
``<output stem>.documents.jsonl``, for paired comparisons across documents.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load(spec: str) -> tuple[torch.nn.Module, object | None]:
    """Return a CUDA model in eval mode and its tokenizer (None for an Inkling checkpoint)."""
    if spec.startswith("inkling:"):
        from safetensors.torch import load_file

        from nemo_automodel.components.models.common import BackendConfig
        from nemo_automodel.components.models.inkling2_dense import Inkling2DenseConfig, Inkling2DenseForCausalLM

        path = Path(spec.removeprefix("inkling:"))
        backend = BackendConfig(attn="flex", linear="torch", rms_norm="torch", enable_hf_state_dict_adapter=False)
        with torch.device("cuda"):
            model = Inkling2DenseForCausalLM(Inkling2DenseConfig.from_pretrained(path), backend=backend)
        state = {key: value for file in sorted(path.glob("*.safetensors")) for key, value in load_file(file).items()}
        own = model.state_dict()
        assert all(own[key].dtype == value.dtype for key, value in state.items() if key in own)
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected and set(missing) <= {"lm_head.weight"}, (missing, unexpected)
        model.tie_weights()
        return model.eval(), None
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(spec, dtype=torch.bfloat16, attn_implementation="sdpa")
    return model.cuda().eval(), AutoTokenizer.from_pretrained(spec)


@torch.no_grad()
def token_nll(model: torch.nn.Module, windows: torch.Tensor) -> torch.Tensor:
    """Return the FP32 NLL [batch, window] of each next-token target of ``windows`` [batch, window + 1]."""
    out = model(input_ids=windows[:, :-1])
    logits = out if torch.is_tensor(out) else out.logits
    return F.cross_entropy(logits.float().transpose(1, 2), windows[:, 1:], reduction="none")


def score(model, stream: np.ndarray, owner: np.ndarray, documents: int, window: int, batch: int) -> np.ndarray:
    """Return the summed NLL per document; padding and the stream's first token are never targets."""
    tokens = torch.from_numpy(stream.astype(np.int64))
    rows = [(start, min(window, len(tokens) - 1 - start)) for start in range(0, len(tokens) - 1, window)]
    rows += [(0, 0)] * (-len(rows) % batch)  # fixed shapes, so compiled kernels are reused
    nll = np.zeros(documents)
    for i in range(0, len(rows), batch):
        group = rows[i : i + batch]
        windows = torch.stack([F.pad(tokens[s : s + n + 1], (0, window - n)) for s, n in group]).cuda()
        losses = token_nll(model, windows).double().cpu().numpy()
        for (start, targets), row in zip(group, losses):
            nll += np.bincount(owner[start + 1 : start + 1 + targets], row[:targets], minlength=documents)
    return nll


def main() -> None:
    """Score every model on every domain and write the totals and per-document NLL."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--heldout", type=Path, required=True, help="Frozen heldout JSONL from build_heldout.py")
    parser.add_argument("--dataset", type=Path, required=True, help="prepare_data.py output for Inkling checkpoints")
    parser.add_argument("--models", nargs="+", required=True, help="HF model IDs or inkling:<consolidated dir>")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=4096)
    args = parser.parse_args()
    from nemo_automodel.components.datasets.llm.megatron.indexed_dataset import IndexedDataset

    records = [json.loads(line) for line in args.heldout.open(encoding="utf-8")]
    domains = sorted({record["domain"] for record in records})
    stats = json.loads((args.dataset / "stats.json").read_text())["validation"]
    results = json.loads(args.output.read_text()) if args.output.exists() else {}
    per_document = args.output.with_suffix(".documents.jsonl").open("a", encoding="utf-8")
    for spec in args.models:
        began = time.time()
        model, tokenizer = load(spec)
        row = {}
        for domain in domains:
            texts = [record for record in records if record["domain"] == domain]
            nbytes = np.array([len(record["text"].encode("utf-8")) for record in texts])
            assert stats[domain]["documents"] == len(texts) and stats[domain]["bytes"] == nbytes.sum(), domain
            if tokenizer is None:
                data = IndexedDataset(str(args.dataset / "validation" / domain.replace("-", "_")))
                lengths = np.asarray(data.sequence_lengths)
                stream = np.concatenate([data[i] for i in range(len(data))])
                batch = 8
            else:
                ids = tokenizer([record["text"] for record in texts], add_special_tokens=False)["input_ids"]
                lengths = np.array([len(x) + 1 for x in ids])
                stream = np.concatenate([np.asarray([*x, tokenizer.eos_token_id]) for x in ids])
                batch = 4
            assert len(lengths) == len(texts)
            owner = np.repeat(np.arange(len(texts)), lengths)
            nll = score(model, stream, owner, len(texts), args.window, batch)
            total = nll.sum()
            row[domain] = {
                "documents": len(texts), "bytes": int(nbytes.sum()), "tokens": len(stream),
                "nats_per_token": total / (len(stream) - 1), "bpb": total / math.log(2) / nbytes.sum(),
            }  # fmt: skip
            for record, value, size in zip(texts, nll, nbytes):
                per_document.write(
                    json.dumps(
                        {"model": spec, "sha256": record["sha256"], "domain": domain, "bytes": int(size), "nll": value}
                    )
                    + "\n"
                )
            print(f"{spec} {domain:13s} tokens {len(stream):8d}  bpb {row[domain]['bpb']:.4f}", flush=True)
        results[spec] = row
        args.output.write_text(json.dumps(results, indent=1) + "\n")
        print(f"{spec}: {time.time() - began:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()
    per_document.close()


if __name__ == "__main__":
    main()
