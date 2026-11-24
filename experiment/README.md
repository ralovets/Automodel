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

Model code is `nemo_automodel/components/models/inkling2_dense`; the micro
geometry is the config class default. The package has its own layers, written
for training throughput, and its unit tests check them against the upstream
`inkling2` forward for logits and gradients.

## Milestones

| # | Milestone | Status |
|---|---|---|
| 1 | Branch from current main, DDP tied-weight fix carried over | done |
| 2 | Dense model package, unit tests | done |
| 3 | Recipes, launcher, this runbook | done |
| 4 | Training image on PyTorch 26.11 (`automodel:mini2`, TE 2.21.0) | done |
| 5 | Two-GPU DDP smoke, nano and micro | done |
| 6 | Throughput tuning, both recipes | done: 65K (nano) and 24K (micro) tokens/s |
| 7 | Low-precision training, throughput measured | done for the feed-forward: NVFP4 and FP8; MXFP8 is not available on these cards |

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

Inside the container the corpus keeps its host path (read-only, so Megatron's
index caches differ per corpus), outputs are `/outputs` and the checkout
`/opt/Automodel`. Metrics go to W&B and to JSONL files under
`OUTPUT_DIR/RUN_NAME/`; the console log is `OUTPUT_DIR/RUN_NAME.log`. The run
name is also the W&B run ID, so use a new name for a new run. Global batch
must divide by local batch times rank count.

Both recipes are 200-step smoke schedules: LR 6e-4 with 32 warmup steps,
global batch 16 (65,536 tokens per update), validation every 100 updates, no
checkpoints. Pick a token budget and LR schedule before a long run.

## Tokenizer trial on corpus-v1

[nano_tok16k.yaml](nano_tok16k.yaml) is nano with a 16,384-entry vocabulary
(411,038,720 parameters). It reads a dataset written by
[prepare_data.py](prepare_data.py), so one recipe serves the Claude-style
tokenizer and the BPE baseline; `DATA_DIR` selects which.

The Claude-style tokenizer needs its Rust core, which the training image does
not have. Build the wheel from the tokenizer project and add it to the image:

```bash
uvx maturin build --release -m native/Cargo.toml -i python3.12 --out wheels
docker build -f experiment/tokenizer.Dockerfile -t automodel:mini2-tok wheels
```

Trial data is built in three steps, inside that image:

1. Sample corpus-v1 in its own proportions with `sample_corpus` from the
   datasets repository: a 120M-token holdout split and a 2.4B-token training
   split (o200k tokens), excluding the tokenizers' training sample.
2. [build_heldout.py](build_heldout.py) freezes 4 MiB per domain of the
   holdout split, in a fixed random order. A document over 64 KiB contributes
   one line-aligned excerpt at a hash-derived offset. A document, or its
   excerpt, is rejected when half or more of its word 13-grams occur in the
   training split. Every tokenizer is scored on this one file.
3. [prepare_data.py](prepare_data.py) tokenizes the training split and the
   frozen heldout.

```bash
S=/home/roman/ext/corpus-v1-sample-20261124
python -m tokenizer_lab.experiments.sample_corpus --corpus /home/roman/storage/corpus-v1-20261014 \
  --output $S --train-splits 1 --exclude TOKENIZER_SAMPLE/train-a.jsonl
PYTHONHASHSEED=0 python experiment/build_heldout.py \
  --holdout $S/holdout.jsonl --train $S/train-a.jsonl --output $S/heldout.jsonl
python experiment/prepare_data.py --tokenizer TOKENIZER_DIR \
  --train $S/train-a.jsonl --holdout $S/heldout.jsonl --output DATA_DIR
```

On roman-ws, trial C (2026-11-24, corpus-v1 with 2% multilingual) is under
`~/data/datasets/corpus-v1-trial-c/`, tokenized with `claude-1124-16k` and
`bpe-lines-1124-16k`. These two were retrained from scratch on a separate
2.4B-token sample in the final source proportions, sharing no document with
trial C (datasets repository, `tokenizer/configs/run_corpus_v1_ws.sh`). Against
the 2026-11-23 `a-16k` pair they use about 9% fewer multilingual tokens and up
to 0.4% more elsewhere; BPE still needs 6.3% fewer tokens than Claude-style
(7.5% on python). `claude-a-16k` and `bpe-lines-a-16k` copies are there for
reference. By bytes
it is 24.8% python, 19.8% other code, 21.3% code Q&A, 17.3% tech, 10.3% math,
4.2% general and 2.3% multilingual. Its heldout dropped 54 near-copies and
holds 6,662 documents. Trial B (`corpus-v1-trial-b`, 0.5% multilingual, an
unfiltered heldout) is the data of the runs below.

```bash
D=$HOME/data/datasets/corpus-v1-trial-c
IMAGE=automodel:mini2-tok RECIPE=experiment/nano_tok16k.yaml DATA_DIR=$D/claude-1124-16k \
  bash experiment/run.sh tok-claude-1124
IMAGE=automodel:mini2-tok RECIPE=experiment/nano_tok16k.yaml DATA_DIR=$D/bpe-lines-1124-16k \
  bash experiment/run.sh tok-bpe-1124
```

Loss per token is not comparable across tokenizers. In-training validation
gives curves: bits per byte = loss × tokens / bytes / ln 2, with tokens and
bytes from `DATA_DIR/stats.json`; it drops each heldout's partial last window.
For final numbers, [score_bpb.py](score_bpb.py) scores every target once,
takes Inkling checkpoints or Hugging Face models, and writes per-document NLL
for paired comparisons:

```bash
python experiment/score_bpb.py --heldout $S/heldout.jsonl --dataset DATA_DIR --output scores.json \
  --models inkling:OUTPUT_DIR/RUN/epoch_0_step_13999/model/consolidated Qwen/Qwen2.5-Coder-0.5B
```

The 200-step smoke of each ran at 66K tokens/s. Bits per byte at step 199, one
seed, 13M training tokens, so a check that the path works and not a verdict:

| Heldout | claude-a-16k | bpe-lines-a-16k |
|---|---:|---:|
| python | 1.694 | 1.748 |
| code_other | 1.882 | 1.946 |
| code_qa | 1.564 | 1.621 |
| tech | 1.949 | 1.990 |
| math | 1.831 | 1.914 |
| general | 2.137 | 2.195 |
| multilingual | 3.291 | 3.257 |

Baseline `nano-claude-a-baseline-1123` (trial B): 14,000 updates, 917M
tokens, cosine LR 6e-4 to 6e-5 after 500 warmup steps, 4 h 47 min at 65K
tokens/s ([W&B](https://roman-ws/ralovets/corpus-v1-tokenizers/runs/quhhxdat)).
Bits per byte from `score_bpb.py`, with two outside models for scale:

| Model | python | code_other | code_qa | tech | math | general | multilingual |
|---|---:|---:|---:|---:|---:|---:|---:|
| nano, claude-a-16k, 0.92B tokens | 0.415 | 0.494 | 0.472 | 0.949 | 0.653 | 1.102 | 1.414 |
| Qwen2.5-Coder-0.5B | 0.320 | 0.330 | 0.426 | 0.761 | 0.526 | 0.834 | 0.959 |
| Qwen3.5-0.8B | 0.378 | 0.440 | 0.469 | 0.780 | 0.522 | 0.868 | 0.891 |

## Results

200-step smoke runs, DDP on two RTX PRO 6000 Blackwell cards, local batch 1,
65,536 tokens per update. Losses are at step 199; `l2`, `l3` and `english` are
the heldouts.

| Run | Tokens/s | Memory per GPU | Train loss | l2 / l3 / english |
|---|---:|---:|---:|---|
| nano, BF16 | 65K | 16.6 GiB | 4.89 | 4.58 / 3.65 / 5.76 |
| micro, BF16 | 24K | 46.3 GiB | 4.77 | 4.41 / 3.60 / 5.68 |
| micro, NVFP4 feed-forward | 30K | 46.3 GiB | 4.84 | 4.49 / 3.64 / 5.72 |

Micro starts at a loss of 16.3, above the uniform 10.4: with tied embeddings
the untrained model predicts the input token. It is below 8.4 after 12 updates.

## Low precision

The feed-forward projections are built by the linear backend, so a Transformer
Engine recipe applies to them (they are about three quarters of the matrix
multiplies in micro). Attention projections, norms and the output head stay in
BF16.

```bash
bash experiment/run.sh micro-nvfp4 \
  --model.backend.linear te --model.backend.te_nvfp4.random_hadamard_transform false
bash experiment/run.sh micro-fp8 --model.backend.linear te --model.backend.te_fp8.recipe current
```

Tokens/s on two GPUs with the recipes; the FP8 run and the nano NVFP4 run are
50 to 60 updates long:

| Feed-forward precision | nano | micro |
|---|---:|---:|
| BF16 | 65K | 24K |
| FP8, current scaling | not run | 28K |
| NVFP4, no Hadamard transform | 73K | 30K |

On these cards (SM120) TE 2.21 refuses MXFP8, and NVFP4 with its default
random Hadamard transform fails an internal assertion, so the transform is
off. That transform is there to protect the quantized gradient GEMM. After 200
updates the NVFP4 micro run is 0.03 to 0.08 behind BF16 on the heldouts (table
above); compare on a longer run before relying on it.

## Throughput notes

Upstream's `inkling2` layers are an FP32 reference: with them nano ran at 24.6K
tokens/s on two GPUs and needed 30 GiB per sequence. What the dense package
does instead, with nano single-GPU forward and backward at each step:

| Change | Tokens/s |
|---|---:|
| Upstream layers | 11.7K |
| Model-dtype activations, FP32 only in reductions; shift-and-add short convolution; BF16 FlexAttention inputs; block masks cached per batch shape; K and V project only their rows | 25.2K |
| Activation, norms, clips and convolution taps as fused compiled functions; depth blocks stacked so a mix is two matmuls | 35.3K |

Tried and left out:

- Whole-model `compile.enabled`: 9% faster on one GPU, no gain under DDP.
- FlexAttention `ROWS_GUARANTEED_SAFE`: 12% faster, but NaN for sliding-window layers.
- Local batch above 1: slightly slower (33.8K at 4), and memory scales with it.

What is left: the attention kernel is about a fifth of nano's step, and most of
that is the per-pair relative-position lookup in its backward.

With the loop at two passes (36 layer applications instead of 24), single-GPU
forward and backward is 19.6K tokens/s for nano and 7.6K for micro, against
35.3K and 13.4K without it. The message gates between passes still use the
upstream FP32 modules and have not been tuned.

## Tests

```bash
uv run pytest -q tests/unit_tests/models/inkling2_dense
```

`tests/functional_tests/models/inkling2_dense` needs CUDA: it compares the
FlexAttention path with the eager one and runs BF16 training steps.
