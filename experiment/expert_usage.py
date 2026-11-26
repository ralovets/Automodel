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

"""Which experts a released Inkling2 checkpoint uses on our text, per layer, loop pass and domain.

    python experiment/expert_usage.py --model MODEL_DIR --sample train-00000-of-00032.jsonl --output OUT_DIR

Documents are read from the start of ``--sample`` (JSONL with ``text`` and ``domain``) until ``--tokens`` tokens,
tokenized with the checkpoint's tokenizer, joined with EOS between them and cut into fixed windows as in training;
positions restart at each document and each window. A forward hook on every layer's router records, for each token,
the experts it selected and their routing weights. Writes ``usage.npz`` (``weight`` and ``count`` of shape
[layers, passes, domains, experts]; a layer outside the loop fills pass 0 only) and ``summary.json``, and prints the
share of routing weight held by the most used experts. The mean next-token loss is printed as a check on the load.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TOP = (4, 8, 16, 32)


def read_stream(sample: Path, tokenizer, budget: int, eos: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return the token stream, each token's domain index and position in its document, and the domain names."""
    ids, owners, positions, names, total = [], [], [], [], 0
    with sample.open(encoding="utf-8") as lines:
        while total < budget:
            records = [json.loads(line) for _, line in zip(range(64), lines)]
            if not records:
                break
            encoded = tokenizer([r["text"] for r in records], add_special_tokens=False)["input_ids"]
            for record, tokens in zip(records, encoded):
                if record["domain"] not in names:
                    names.append(record["domain"])
                ids.append(np.asarray([*tokens, eos], dtype=np.int64))
                owners.append(np.full(len(tokens) + 1, names.index(record["domain"]), dtype=np.int64))
                positions.append(np.arange(len(tokens) + 1, dtype=np.int64))
                total += len(tokens) + 1
    stream, owners, positions = (np.concatenate(parts)[:budget] for parts in (ids, owners, positions))
    return stream, owners, positions, names


def load(path: Path, attn: str, device: torch.device) -> torch.nn.Module:
    """Return a released looped checkpoint's text decoder in eval mode, experts decoded to its compute dtype."""
    from nemo_automodel.components.checkpoint.config import CheckpointingConfig
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.inkling2.config import Inkling2LoopConfiguration
    from nemo_automodel.components.models.inkling2.model import Inkling2ForCausalLM

    backend = BackendConfig(attn=attn, linear="torch", rms_norm="torch", experts="torch", dispatcher="torch")
    config = Inkling2LoopConfiguration.from_pretrained(str(path))
    config.name_or_path = str(path)  # the constructor reads hf_quant_config.json from here
    with torch.device(device):
        model = Inkling2ForCausalLM(config, backend=backend)
    checkpointer = CheckpointingConfig(
        is_async=False, model_repo_id=str(path), original_model_root_dir=str(path), model_cache_dir=str(path)
    ).build(0, 0, 0)
    checkpointer.load_base_model(model, device, str(path), str(path))
    return model.eval()


@torch.no_grad()
def collect(
    model: torch.nn.Module,
    windows: torch.Tensor,
    positions: torch.Tensor,
    owners: torch.Tensor,
    domains: int,
    batch: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return routing weight and selection count [layers, passes, domains, experts] and the mean next-token loss.

    Args:
        model: Inkling2 text decoder; its device receives each batch.
        windows: Token IDs [rows, window].
        positions: Position of each token in its document, restarting at each window [rows, window].
        owners: Domain index of each token [rows, window].
        domains: Number of domains.
        batch: Rows per forward.
    """
    config = model.text_config
    device = next(model.parameters()).device
    loop = config.loop_config
    experts, passes = config.n_routed_experts, loop["num_passes"] if loop else 1
    weight = torch.zeros(config.num_hidden_layers, passes, domains * experts, dtype=torch.float64, device=device)
    count = torch.zeros_like(weight)
    calls = [0] * config.num_hidden_layers
    owner = torch.empty(0)

    def record(layer: int):
        def hook(_module, _inputs, output) -> None:
            weights, indices = output[0], output[1]  # [tokens, selected]
            keys = (owner[:, None] * experts + indices).flatten()
            weight[layer, calls[layer]].index_add_(0, keys, weights.flatten().double())
            count[layer, calls[layer]].index_add_(0, keys, torch.ones_like(keys, dtype=torch.float64))
            calls[layer] += 1

        return hook

    handles = [model.layers[str(i)].mlp.gate.register_forward_hook(record(i)) for i in range(config.num_hidden_layers)]
    nll = 0.0
    for start in range(0, len(windows), batch):
        rows = windows[start : start + batch].to(device)
        owner = owners[start : start + batch].to(device).flatten()
        calls[:] = [0] * len(calls)
        logits = model(rows, position_ids=positions[start : start + batch].to(device))
        nll += F.cross_entropy(logits[:, :-1].float().flatten(0, 1), rows[:, 1:].flatten(), reduction="sum").item()
    for handle in handles:
        handle.remove()
    shape = (config.num_hidden_layers, passes, domains, experts)
    return weight.view(shape).cpu().numpy(), count.view(shape).cpu().numpy(), nll / (windows[:, 1:].numel())


def coverage(weight: np.ndarray) -> dict[str, float]:
    """Return the share of the routing weight [experts] held by its most used experts."""
    ranked = np.sort(weight)[::-1] / weight.sum()
    return {f"top{k}": float(ranked[:k].sum()) for k in TOP if k < len(ranked)}


def summarize(weight: np.ndarray, names: list[str], groups: dict[str, list[str]]) -> dict:
    """Return, per group of domains, layer and pass: coverage and the 8 most used experts, most used first.

    A looped layer also gets a ``passes`` row with its passes summed, the ranking a student that reuses one
    feed-forward across passes would take.
    """
    summary = {}
    for group, members in groups.items():
        picked = weight[:, :, [names.index(m) for m in members if m in names]].sum(axis=2)  # [layers, passes, experts]
        rows = []
        for layer in range(picked.shape[0]):
            used = [p for p in range(picked.shape[1]) if picked[layer, p].sum() > 0]
            views = {f"pass{p}": picked[layer, p] for p in used}
            if len(used) > 1:
                views["passes"] = picked[layer].sum(axis=0)
            for view, values in views.items():
                top = np.argsort(values)[::-1][:8].tolist()
                rows.append({"layer": layer, "view": view, **coverage(values), "experts": top})
        summary[group] = rows
    return summary


def main() -> None:
    """Collect expert usage over the sample and write the arrays, the summary and a table."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="Released Inkling2 checkpoint directory")
    parser.add_argument("--sample", type=Path, required=True, help="JSONL with text and domain, in random order")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=float, default=4e6, help="Tokens to read from the start of the sample")
    parser.add_argument("--window", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--attn", default="eager", choices=("eager", "flex"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    stream, owners, positions, names = read_stream(args.sample, tokenizer, int(args.tokens), tokenizer.eos_token_id)
    rows = len(stream) // args.window  # the partial last window is dropped
    cut = lambda a: torch.from_numpy(a[: rows * args.window]).view(rows, args.window)  # noqa: E731
    windows, owners, positions = cut(stream), cut(owners), cut(positions)
    # A document that continues from the previous window starts again at position 0.
    first = positions - torch.arange(args.window) == positions[:, :1]
    positions = positions - first * positions[:, :1]
    print(f"{rows} windows of {args.window} tokens; domains {names}", flush=True)
    began = time.time()
    model = load(args.model, args.attn, torch.device(args.device))
    print(f"loaded in {time.time() - began:.0f}s", flush=True)
    began = time.time()
    weight, count, loss = collect(model, windows, positions, owners, len(names), args.batch)
    seconds = time.time() - began
    print(f"loss {loss:.4f} nats per token; {windows.numel() / seconds:.0f} tokens/s", flush=True)

    code = [n for n in names if n in ("python", "code-other", "code_qa")]
    groups = {"all": names, "code": code, "code+math": [*code, "math"], **{n: [n] for n in names}}
    summary = {
        "model": str(args.model), "sample": str(args.sample), "tokens": int(windows.numel()), "window": args.window,
        "loss": loss, "tokens_per_second": windows.numel() / seconds, "domains": names,
        "domain_tokens": {n: int((owners == i).sum()) for i, n in enumerate(names)},
        "usage": summarize(weight, names, groups),
    }  # fmt: skip
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / "usage.npz", weight=weight, count=count, domains=np.array(names))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for group in ("all", "code", "code+math"):
        print(f"\n{group}: share of routing weight\nlayer view    " + "".join(f"{f'top{k}':>7s}" for k in TOP))
        for row in summary["usage"][group]:
            shares = "".join(f"{row[f'top{k}']:7.3f}" for k in TOP if f"top{k}" in row)
            print(f"{row['layer']:5d} {row['view']:7s}{shares}  {row['experts']}")


if __name__ == "__main__":
    main()
