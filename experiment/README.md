# Dense Inkling2 coder: micro and nano

Scratch pretraining of a dense, text-only model on the Inkling-2-Mini decoder
design. Two recipes, one launcher:

| Recipe | Parameters | Width / layers / feed-forward | Attention heads |
|---|---:|---|---|
| [micro.yaml](micro.yaml) | 1,642,194,944 | 2048 / 24 / 8192 | 16 query, 4 key-value |
| [nano.yaml](nano.yaml) | 427,815,936 | 1024 / 24 / 4096 | 8 query, 2 key-value |

Both keep the Inkling-2-Mini skeleton: five local-attention layers (window
1024) then one full-attention layer, four times; learned relative positions
instead of RoPE; short convolutions; attention residuals over blocks of six
layers; softcapped SwiGLU and logits. They differ from it in a dense
feed-forward instead of routed experts, tied cl32k embeddings (32,768), a
4,096-token context and no vision tower. The loop over layers 6 to 17 is off
by default; `loop_config` in the recipe turns it on.

Model code is `nemo_automodel/components/models/inkling2_dense`. It reuses the
upstream `inkling2` attention, convolution, residual and loop modules, and the
micro geometry is the config class default.

## Milestones

| # | Milestone | Status |
|---|---|---|
| 1 | Branch from current main, DDP tied-weight fix carried over | done |
| 2 | Dense model package, unit tests | done |
| 3 | Recipes, launcher, this runbook | done |
| 4 | Training image on PyTorch 26.11 | pending |
| 5 | Two-GPU DDP smoke, nano and micro | pending |
| 6 | Throughput tuning, both recipes | pending |
| 7 | Low-precision training (MXFP8, NVFP4), throughput measured | pending |

## Image

From the repository root, with the upstream Dockerfile unchanged:

```bash
docker build -f docker/Dockerfile -t automodel:mini2 \
  --build-arg BASE_IMAGE=pytorch \
  --build-arg INSTALL_FA3=false --build-arg INSTALL_FA4=false .
```

`BASE_IMAGE=pytorch` selects `nvcr.io/nvidia/pytorch:26.11-py3`.

## Data

[`ralovets/cl32k-python-english-50b-v2`](https://huggingface.co/datasets/ralovets/cl32k-python-english-50b-v2)
is 102.5 GB of tokenized Megatron binaries: 32 training shards (49.7B tokens)
and per-source heldouts `l2`, `l3` and `english`. The recipes read every shard
present under `train/`, so a few shards are enough for smoke and throughput
runs:

```bash
export DATA_DIR="$HOME/data/datasets/cl32k-python-english-50b-v2"
uvx --from huggingface_hub hf download ralovets/cl32k-python-english-50b-v2 \
  --repo-type dataset --revision c0601cec4e506527515c47a3c07268afb14186a2 \
  --local-dir "$DATA_DIR" \
  --include "tokenizer/*" --include "validation-small/*" --include "SHA256SUMS" \
  --include "train/train-0000[0-3]-of-00032*"
```

Drop `--include` for the whole dataset, then `(cd "$DATA_DIR" && sha256sum -c SHA256SUMS)`.

## Launch

`run.sh` takes a run name and optional recipe overrides. Machine settings are
environment variables with defaults at the top of the script: `IMAGE`,
`NPROC`, `RECIPE` (default `experiment/micro.yaml`), `DATA_DIR`, `OUTPUT_DIR`,
`OMP_NUM_THREADS`, and the W&B variables `WANDB_API_KEY`, `WANDB_BASE_URL`,
`WANDB_ENTITY`, `WANDB_MODE`.

```bash
RECIPE=experiment/nano.yaml bash experiment/run.sh nano-smoke
bash experiment/run.sh micro-smoke
```

Inside the container the corpus is `/data` (read-only), outputs `/outputs` and
the checkout `/opt/Automodel`. Metrics go to W&B and to JSONL files under
`OUTPUT_DIR/RUN_NAME/`; the console log is `OUTPUT_DIR/RUN_NAME.log`. The run
name is also the W&B run ID, so use a new name for a new run. Global batch
must divide by local batch times rank count.

Both recipes are 200-step smoke schedules: LR 6e-4 with 32 warmup steps,
global batch 16 (65,536 tokens per update), validation every 100 updates, no
checkpoints. Pick a token budget and LR schedule before a long run.

## Tests

```bash
uv run pytest -q tests/unit_tests/models/inkling2_dense
```
