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

"""Freeze per-domain heldouts from a corpus sample, without near-copies of its training split.

Usage (word hashes need a fixed seed):
    PYTHONHASHSEED=0 python experiment/build_heldout.py --holdout SAMPLE/holdout.jsonl \\
        --train SAMPLE/train-a.jsonl --output SAMPLE/heldout.jsonl

Documents of the sample's holdout split are taken per domain in a fixed random order
(``random.Random(0)`` shuffle) until ``--domain-bytes`` of text are selected:

- A document longer than ``--max-bytes`` contributes one line-aligned excerpt of at most
  that size, starting at an offset derived from its sha256 so file headers are not favoured.
  A document with no such excerpt of at least half that size (long-line generated files)
  is skipped.
- Every candidate's lowercased word 13-grams are looked up in the whole training split.
  A document is rejected when half or more of its 13-grams, or of its excerpt's, occur
  anywhere in training.

The output has one record per selected document: ``sha256`` (of the whole document),
``source``, ``domain``, ``origin``, the UTF-8 byte range ``start``/``end``, the overlap
``union`` of the document and of the excerpt, and the excerpt ``text``. ``prepare_data.py``
and ``score_bpb.py`` read it as is, so every tokenizer is scored on the same bytes.
A summary is written next to it as ``<output stem>.report.json``.
"""

import argparse
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

N = 13
P = np.uint64(0x9E3779B97F4A7C15)
WORD = re.compile(r"\w+")
SHARED: dict = {}  # set in each scan worker by share()


def shingles(text: str) -> np.ndarray:
    """Return the sorted unique hashes of the text's lowercased word 13-grams."""
    words = WORD.findall(text.lower())
    if len(words) < N:
        return np.empty(0, dtype=np.uint64)
    h = np.fromiter((hash(w) for w in words), dtype=np.int64, count=len(words)).view(np.uint64)
    acc = np.zeros(len(words) - N + 1, dtype=np.uint64)
    for i in range(N):
        acc = acc * P + h[i : i + len(acc)]
    return np.unique(acc)


def excerpt(data: bytes, sha256: str, cap: int) -> tuple[int, int] | None:
    """Return a line-aligned byte range of at most ``cap`` bytes, or None if there is none."""
    if len(data) <= cap:
        return 0, len(data)
    offset = int(sha256[:16], 16) % (len(data) - cap + 1)
    start = 0 if offset == 0 else data.find(b"\n", offset - 1) + 1
    if start == 0 and offset:
        return None
    end = len(data) if start + cap >= len(data) else data.rfind(b"\n", start, start + cap) + 1
    return (start, end) if end - start >= cap // 2 else None


def share(keys: np.ndarray, train: str) -> None:
    """Give a scan worker the sorted heldout 13-gram hashes and the training file path."""
    SHARED.update(keys=keys, train=train)


def scan(span: tuple[int, int]) -> np.ndarray:
    """Return the indices of heldout 13-grams that occur in one byte span of the training file."""
    start, end = span
    keys, hit = SHARED["keys"], []
    with open(SHARED["train"], "rb") as f:
        if start:
            f.seek(start - 1)
            f.readline()
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            found = shingles(json.loads(line)["text"])
            index = np.searchsorted(keys, found).clip(max=len(keys) - 1)
            hit.append(index[keys[index] == found])
    return np.unique(np.concatenate(hit)) if hit else np.empty(0, dtype=np.int64)


def union(found: np.ndarray, keys: np.ndarray, hits: np.ndarray) -> float | None:
    """Share of 13-grams that occur in training; None for texts shorter than 13 words."""
    if not len(found):
        return None
    index = np.searchsorted(keys, found).clip(max=len(keys) - 1)
    return float((hits[index] & (keys[index] == found)).mean())


def main() -> None:
    """Select, scan and write the heldout records and their report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--holdout", type=Path, required=True, help="Sample holdout JSONL")
    parser.add_argument("--train", type=Path, required=True, help="Sample training JSONL the model trains on")
    parser.add_argument("--output", type=Path, required=True, help="Heldout JSONL to write")
    parser.add_argument("--domain-bytes", type=int, default=4 << 20, help="UTF-8 bytes selected per domain")
    parser.add_argument("--max-bytes", type=int, default=64 << 10, help="Excerpt cap per document")
    parser.add_argument("--max-union", type=float, default=0.5, help="Reject at this overlap or above")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") in (None, "random"):
        raise SystemExit("Set PYTHONHASHSEED (for example 0) so the word hashes are reproducible")

    # Candidates: twice the budget per domain, in draw order, so rejections can be replaced.
    with args.holdout.open(encoding="utf-8") as lines:
        records = [json.loads(line) for line in lines]
    random.Random(0).shuffle(records)
    pool_bytes: Counter = Counter()
    skipped: Counter = Counter()
    candidates = []
    for record in records:
        domain = record["domain"]
        if pool_bytes[domain] >= 2 * args.domain_bytes:
            continue
        data = record["text"].encode("utf-8")
        span = excerpt(data, record["sha256"], args.max_bytes)
        if span is None:
            skipped[domain] += 1
            continue
        pool_bytes[domain] += span[1] - span[0]
        candidates.append((record, data, span))
    del records
    whole = [shingles(record["text"]) for record, _, _ in candidates]
    parts = [shingles(data[start:end].decode("utf-8")) for _, data, (start, end) in candidates]
    keys = np.unique(np.concatenate(whole + parts))
    print(f"{len(candidates)} candidates, {len(keys)} distinct 13-grams", flush=True)

    size = args.train.stat().st_size
    spans = [(size * i // (args.workers * 16), size * (i + 1) // (args.workers * 16)) for i in range(args.workers * 16)]
    hits = np.zeros(len(keys), dtype=bool)
    began = time.time()
    with Pool(args.workers, initializer=share, initargs=(keys, str(args.train))) as pool:
        for done, found in enumerate(pool.imap_unordered(scan, spans), 1):
            hits[found] = True
            if done % args.workers == 0:
                print(f"{done}/{len(spans)} training spans, {time.time() - began:.0f}s", flush=True)

    selected, taken = [], Counter()
    rejected: Counter = Counter()
    for (record, data, (start, end)), doc_shingles, part_shingles in zip(candidates, whole, parts):
        domain = record["domain"]
        if taken[domain] >= args.domain_bytes:
            continue
        doc_union, part_union = union(doc_shingles, keys, hits), union(part_shingles, keys, hits)
        if max(doc_union or 0.0, part_union or 0.0) >= args.max_union:
            rejected[domain] += 1
            continue
        taken[domain] += end - start
        selected.append({
            "sha256": record["sha256"], "source": record["source"], "domain": domain, "origin": record.get("origin"),
            "document_bytes": len(data), "start": start, "end": end,
            "union": doc_union, "excerpt_union": part_union, "text": data[start:end].decode("utf-8"),
        })  # fmt: skip
    with args.output.open("w", encoding="utf-8") as out:
        for row in selected:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    domains = defaultdict(Counter)
    for row in selected:
        stats = domains[row["domain"]]
        stats["documents"] += 1
        stats["bytes"] += row["end"] - row["start"]
        stats["excerpts"] += row["end"] - row["start"] < row["document_bytes"]
    for domain, stats in domains.items():
        stats["rejected_overlap"] = rejected[domain]
        stats["skipped_long_lines"] = skipped[domain]
        if stats["bytes"] < args.domain_bytes:
            print(f"warning: {domain} has only {stats['bytes']} bytes", flush=True)
    report = {
        "holdout": str(args.holdout), "train": str(args.train), "domain_bytes": args.domain_bytes,
        "max_bytes": args.max_bytes, "max_union": args.max_union, "ngram": N,
        "domains": {domain: dict(stats) for domain, stats in sorted(domains.items())},
    }  # fmt: skip
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["domains"], indent=1), flush=True)


if __name__ == "__main__":
    main()
