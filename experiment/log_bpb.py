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

"""Log validation bits per byte of every checkpoint of a run to W&B.

    python experiment/log_bpb.py --run OUTPUT_DIR/RUN --dataset DATA_DIR --project corpus-v1-tokenizers

Bits per byte = validation loss x tokens / bytes / ln 2, with the loss from each checkpoint's
``losses.json`` and tokens and bytes from ``DATA_DIR/stats.json``; ``bpb/all`` weights the
heldout sets by bytes. The values go to a companion W&B run named ``<RUN>-bpb`` (a second
process must not write to the live training run), with the training step as the W&B step.
Run it again after new checkpoints: steps already logged are skipped.
"""

import argparse
import json
import math
from pathlib import Path

import wandb


def main() -> None:
    """Log the checkpoints the companion run does not hold yet."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run", type=Path, required=True, help="Run output directory holding epoch_*_step_* checkpoints"
    )
    parser.add_argument("--dataset", type=Path, required=True, help="prepare_data.py output of the run's tokenizer")
    parser.add_argument("--project", required=True)
    args = parser.parse_args()
    stats = {
        k.replace("-", "_"): v for k, v in json.loads((args.dataset / "stats.json").read_text())["validation"].items()
    }
    total = sum(v["bytes"] for v in stats.values())
    name = f"{args.run.name}-bpb"
    run = wandb.init(project=args.project, id=name, name=name, resume="allow", job_type="bpb")
    done = run.summary.get("train_step", -1)
    checkpoints = sorted((int(p.parent.name.rsplit("_", 1)[1]), p) for p in args.run.glob("epoch_*_step_*/losses.json"))
    for step, path in checkpoints:
        if step <= done:
            continue
        losses = json.loads(path.read_text())
        bpb = {d: losses[d] * s["tokens"] / s["bytes"] / math.log(2) for d, s in stats.items()}
        row = {f"bpb/{d}": v for d, v in bpb.items()}
        row["bpb/all"] = sum(bpb[d] * s["bytes"] for d, s in stats.items()) / total
        run.log({**row, "train_step": step}, step=step)
        print(step, round(row["bpb/all"], 4), flush=True)
    run.finish()


if __name__ == "__main__":
    main()
