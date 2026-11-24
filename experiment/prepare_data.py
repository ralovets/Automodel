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

"""Tokenize JSONL documents into the Megatron shards the experiment recipes read.

Input lines are JSON objects with ``text`` and ``domain``. Every document gets the
tokenizer's EOS appended. The output directory holds ``train/train-NNNNN.{bin,idx}``,
one heldout per domain under ``validation/`` (every record of the frozen heldout
file from ``build_heldout.py``, in order, so every tokenizer sees the same bytes), a
copy of the tokenizer, and ``stats.json`` with documents, UTF-8 bytes and tokens per
split and domain, which is what turns a per-token loss into bits per byte.

    python experiment/prepare_data.py --tokenizer TOKENIZER_DIR \
        --train train.jsonl --holdout heldout.jsonl --output DATA_DIR
"""

import argparse
import json
import shutil
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_automodel.components.datasets.llm.megatron.indexed_dataset import IndexedDatasetBuilder


def read_batches(path: Path, batch_size: int) -> Iterator[list[dict[str, str]]]:
    """Yield lists of at most ``batch_size`` parsed JSONL records."""
    batch: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            batch.append(json.loads(line))
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def encode(tokenizer: PreTrainedTokenizerBase, texts: list[str]) -> list[np.ndarray]:
    """Return one uint16 array of token IDs per text, each ending with EOS."""
    encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
    return [np.asarray([*ids, tokenizer.eos_token_id], dtype=np.uint16) for ids in encoded]


def main() -> None:
    """Write the training shards, the per-domain heldouts, the tokenizer copy and the statistics."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", type=Path, required=True, help="Transformers tokenizer directory")
    parser.add_argument("--train", type=Path, required=True, help="Training JSONL")
    parser.add_argument("--holdout", type=Path, required=True, help="Frozen heldout JSONL from build_heldout.py")
    parser.add_argument("--output", type=Path, required=True, help="Dataset directory to create")
    parser.add_argument("--train-shards", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096, help="Documents per encode call")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if len(tokenizer) > np.iinfo(np.uint16).max + 1:
        raise ValueError("Token IDs are stored as uint16")
    stats: dict[str, dict[str, dict[str, int]]] = {
        split: defaultdict(lambda: {"documents": 0, "bytes": 0, "tokens": 0}) for split in ("train", "validation")
    }

    def count(split: str, record: dict[str, str], ids: np.ndarray) -> None:
        entry = stats[split][record["domain"]]
        entry["documents"] += 1
        entry["bytes"] += len(record["text"].encode("utf-8"))
        entry["tokens"] += len(ids)

    (args.output / "train").mkdir(parents=True)
    (args.output / "validation").mkdir()
    prefixes = [args.output / "train" / f"train-{index:05d}" for index in range(args.train_shards)]
    builders = [IndexedDatasetBuilder(f"{prefix}.bin", dtype=np.uint16) for prefix in prefixes]
    for index, batch in enumerate(read_batches(args.train, args.batch_size)):
        for record, ids in zip(batch, encode(tokenizer, [record["text"] for record in batch])):
            builders[index % args.train_shards].add_document(ids, [len(ids)])
            count("train", record, ids)
    for prefix, builder in zip(prefixes, builders):
        builder.finalize(f"{prefix}.idx")

    heldouts: dict[str, IndexedDatasetBuilder] = {}
    for batch in read_batches(args.holdout, args.batch_size):
        for record, ids in zip(batch, encode(tokenizer, [record["text"] for record in batch])):
            domain = record["domain"]
            if domain not in heldouts:
                heldouts[domain] = IndexedDatasetBuilder(
                    str(args.output / "validation" / f"{domain.replace('-', '_')}.bin"), dtype=np.uint16
                )
            heldouts[domain].add_document(ids, [len(ids)])
            count("validation", record, ids)
    for domain, builder in heldouts.items():
        builder.finalize(str(args.output / "validation" / f"{domain.replace('-', '_')}.idx"))

    shutil.copytree(args.tokenizer, args.output / "tokenizer")
    (args.output / "stats.json").write_text(json.dumps(stats, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
