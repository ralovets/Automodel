# Context: dense Inkling2 coder (mini2), as of 2026-11-24

Handover from the Claude session `c6256e05` that ran on the Mac. That session's
transcript and memory are copied to
`~/.claude/projects/-home-roman-workspace-Automodel/` (`claude --resume`).
The runbook with commands and results is [README.md](README.md); this file adds
the state of the work, the decisions behind it and what is open.

## Where things are

- Branch `ralovets/exp/mini2`, 21 commits on main, head `d0bd1975a`. Not pushed
  to GitHub. The Mac checkout `/Users/roman/Projects/Automodel` is at the same
  commit. This checkout was on `dev/flash`; its uncommitted
  `experiment/tokenizer/README.md` edit is `stash@{0}`
  (`git switch dev/flash && git stash pop` to go back).
- The trial C runs used the code copy at
  `~/workspace/checkpoints/mini2-20261123/source` (same code as this commit,
  not a git checkout).
- The earlier 1B prototype on the Qwen3.8-Flash-Next design (Gated DeltaNet, GR
  residual streams, Engram table; branches `ralovets/exp/micro`, `dev/flash`,
  `experiment/baseline1b/`) is dropped. The model is now the Inkling-2-Mini
  design, kept at its skeleton.

## Architecture

Model code: `nemo_automodel/components/models/inkling2_dense/` (config, layers,
attention, model). The micro geometry is the config class default.

| | Inkling 2 Mini | micro (`micro.yaml`) | nano (`nano.yaml`) |
|---|---|---|---|
| Stored parameters | 26.39B | 1.642B | 0.428B (0.411B with 16k vocab) |
| Active per token | 3.46B | all | all |
| Hidden size | 2304 | 2048 | 1024 |
| Layers | 24 | 24 | 24 |
| Heads (query / key-value, dim 128) | 24 / 8 | 16 / 4 | 8 / 2 |
| Attention pattern | 5 local (window 1024) + 1 full, ×4 | same | same |
| Positions | relative bias (16 dims, extent 1024), no RoPE | same | same |
| Short convolutions | kernel 4 | same | same |
| Residual path | attention residuals over blocks of 6 | same | same |
| Feed-forward | MoE, 128 experts, top 8, width 1152 | dense SwiGLU 8192 | dense SwiGLU 4096 |
| Softcaps | gate 4, up 25, logits 30 | same | same |
| Loop | layers 6–17 run 3 times (48 applications) | off by default | off by default |
| Vocabulary / embedding | 201,024, untied 463M + 463M | cl32k 32,768, tied | cl32k, tied; 16,384 in `nano_tok16k.yaml` |
| Context | 1,048,576 | 4,096 | 4,096 |
| Vision tower, drafter | 318M each | none | none |
| Weights | NVFP4 / MXFP8 experts | BF16, FP32 master | BF16, FP32 master |

With the loop at two passes micro is 1.667B and nano 0.440B.

Mini's distinctive parts, kept here:
- Loop: every pass restarts from the prelude output (state entering layer 6).
  From pass 2 on, attention q, k, v and the MLP input are blended with a message
  from the previous pass by rank-64 gates,
  `(1 − tanh(forget)) · input + tanh(gain) · message`.
- Block attention residuals: each sublayer input is a softmax mix over completed
  block outputs plus the running state, one learned query per sublayer.
- Relative-bias attention: each token projects a 16-dim vector per head that
  selects a learned bias-vs-distance profile; L2-clipped q/k, RMS-normed values,
  sigmoid output gate, attention sink.
- Short convolutions on keys, values, attention output and MLP output.

Loop facts for training (no published recipe from the authors):
- Backprop goes through every pass; memory and time grow with layer
  applications. Two passes measured at 56% of no-loop speed, about twice the
  activation memory. The message gates still use upstream FP32 modules, untuned.
- Core weights get summed gradients from every pass; LR and clipping may need
  retuning (untested).
- Gates start near zero, so pass 2 starts as nearly a repeat of pass 1. Open:
  train with the loop from the start, or attach it to a trained no-loop model.
- Pass count is fixed; no pipeline parallelism with the loop.

## Training stack facts

- Image `automodel:mini2` (PyTorch 26.11, TE 2.21; upstream Dockerfile,
  `BASE_IMAGE=pytorch`, FA3/FA4 off). `automodel:mini2-tok` adds the
  Claude-style tokenizer's Rust core (`experiment/tokenizer.Dockerfile`).
- DDP on 2 GPUs, local batch 1, global 16, 65,536 tokens per update. FusedAdam
  with FP32 master weights and moments.
- Throughput (2 GPUs, BF16): nano 65K tokens/s, micro 24K. Upstream `inkling2`
  layers are an FP32 reference (nano 24.6K, 30 GiB per sequence); the dense
  package's own layers are checked against upstream for logits and gradients.
- Low precision: only the feed-forward goes through the TE linear backend. On
  SM120, TE refuses MXFP8 and NVFP4 needs the random Hadamard transform off.
  NVFP4 micro 30K, FP8 micro 28K, NVFP4 nano 73K. NVFP4 was 0.03–0.08 behind
  BF16 after 200 steps; not validated on a long run.
- Left out: whole-model compile (no gain under DDP); FlexAttention
  `ROWS_GUARANTEED_SAFE` (12% faster but NaN on sliding-window layers); local
  batch above 1. Remaining cost: attention kernel, about a fifth of nano's step,
  mostly the relative-position lookup in its backward.
- Shared framework fixes on the branch: DDP wrapper metadata (tied weights,
  `logits_to_keep`, forward kwargs, TE quantization config), compiled-model
  forward signature, and the loss check under `torch.compile(DDP(model))`.
- Tests: `uv run pytest -q tests/unit_tests/models/inkling2_dense`;
  `tests/functional_tests/models/inkling2_dense` needs CUDA (FlexAttention vs
  eager parity, BF16 training steps, full-length causality at 4,096 tokens).
- Micro's first-step loss is 16.3 (above uniform 10.4): with tied embeddings the
  untrained model predicts the input token. Below 8.4 after 12 updates.

## Current experiment: tokenizer trial C (paused)

Question: does the Claude-style 16k tokenizer beat a BPE 16k for this model?
Nano with `nano_tok16k.yaml`, one run per GPU, 14,000 updates (917M tokens),
cosine LR 6e-4 → 6e-5 after 500 warmup, validation and checkpoint every 1,000,
3 kept, consolidated export at the end.

| Run | GPU | W&B | Stopped at |
|---|---|---|---|
| `nano-claude-1124-trialc` (`claude-1124-16k`) | 0 | https://roman-ws/ralovets/corpus-v1-tokenizers/runs/pyhzkq7i | step 6,013 |
| `nano-bpe-1124-trialc` (`bpe-lines-1124-16k`) | 1 | https://roman-ws/ralovets/corpus-v1-tokenizers/runs/xean0gt4 | step 6,000 |

Paused 2026-11-23 23:54 PST for the corpus drive failure (below). Outputs:
`~/workspace/checkpoints/mini2-trialc/<run>/`, checkpoints `epoch_0_step_4999`,
`epoch_0_step_5999` and the stop checkpoint (`LATEST`). The trainer saves a
checkpoint on SIGTERM, so `docker stop` loses no steps.

Resume (only when asked). `run.sh` truncates `<run>.log`, so copy the logs aside
first; the W&B flags continue the same runs:

```bash
cd ~/workspace/checkpoints/mini2-trialc && for r in claude bpe; do cp nano-$r-1124-trialc.log nano-$r-1124-trialc.part1.log; done
cd ~/workspace/checkpoints/mini2-20261123/source
D=$HOME/data/datasets/corpus-v1-trial-c O=$HOME/workspace/checkpoints/mini2-trialc
ARGS="--step_scheduler.max_steps 14000 --step_scheduler.val_every_steps 1000 --step_scheduler.ckpt_every_steps 1000 --lr_scheduler.lr_decay_style cosine --lr_scheduler.lr_warmup_steps 500 --lr_scheduler.init_lr 0.0 --lr_scheduler.min_lr 6e-05 --checkpoint.enabled true --checkpoint.max_recent_checkpoints 3 --checkpoint.save_consolidated final"
GPUS=device=0 NPROC=1 IMAGE=automodel:mini2-tok RECIPE=experiment/nano_tok16k.yaml DATA_DIR=$D/claude-1124-16k OUTPUT_DIR=$O \
  setsid nohup bash experiment/run.sh nano-claude-1124-trialc $ARGS --wandb.id pyhzkq7i --wandb.resume allow >/dev/null 2>&1 </dev/null &
GPUS=device=1 NPROC=1 IMAGE=automodel:mini2-tok RECIPE=experiment/nano_tok16k.yaml DATA_DIR=$D/bpe-lines-1124-16k OUTPUT_DIR=$O \
  setsid nohup bash experiment/run.sh nano-bpe-1124-trialc $ARGS --wandb.id xean0gt4 --wandb.resume allow >/dev/null 2>&1 </dev/null &
```

Run it from an interactive shell (`bash -ic`): only that shell has the W&B key.
About 2.0–2.1 s per update at full GPU power, so about 4.5 h to finish.

Bits per byte at step 5,999 (in-training validation; lower is better):

| Domain | Claude-style | BPE | Claude vs BPE |
|---|---:|---:|---:|
| python | 0.539 | 0.542 | −0.5% |
| code_other | 0.581 | 0.582 | −0.2% |
| code_qa | 0.547 | 0.551 | −0.8% |
| tech | 1.063 | 1.069 | −0.5% |
| math | 0.719 | 0.726 | −1.0% |
| general | 1.234 | 1.244 | −0.8% |
| multilingual | 1.512 | 1.533 | −1.4% |

Claude-style led on every domain at every eval, though it has read about 6%
fewer bytes at equal steps. The python gap shrank from −2.0% (step 1,999) to
−0.5%. One seed each.

After the runs finish:
- Score both final checkpoints with `score_bpb.py` on the trial C heldout
  (`/home/roman/ext/corpus-v1-sample-20261124/heldout.jsonl`); use the
  per-document NLL file for paired uncertainty.
- Primary comparison at equal steps. Secondary at equal text: BPE near step
  13,108 (14,000 × 3.207 / 3.425 bytes per token), from its 12,999 / 13,999
  checkpoints.
- If the gap is within noise, a second pair of runs (another seed).

bpb = loss × tokens / bytes / ln 2, tokens and bytes from `DATA_DIR/stats.json`.
Per-token loss is not comparable across tokenizers.

Earlier reference (trial B, `nano-claude-a-baseline-1123`, 14,000 updates,
https://roman-ws/ralovets/corpus-v1-tokenizers/runs/quhhxdat), scored with
`score_bpb.py`: python 0.415, code_other 0.494, code_qa 0.472, tech 0.949,
math 0.653, general 1.102, multilingual 1.414. Qwen2.5-Coder-0.5B is better on
every domain (python 0.320), Qwen3.5-0.8B too (python 0.378). A future-token
leak was investigated and ruled out (causality test now in the suite).

## Data and tokenizers

- Full corpus: `/home/roman/storage/corpus-v1-20261014` (2.47 TB parquet, on the
  Crucial T710 `nvme0`).
- Model sample (2026-11-24, final proportions with 2% multilingual):
  `/home/roman/ext/corpus-v1-sample-20261124/` with `holdout.jsonl` (101,075
  docs), `train-a.jsonl` (2.02M docs, 2.4B o200k tokens), `heldout.jsonl` (frozen
  v2: 6,662 docs, 4 MiB per domain, near-copies of training removed).
- Tokenizer sample, disjoint from the model sample:
  `/home/roman/ext/corpus-v1-tokenizer-sample-20261124/train-a.jsonl`.
- Tokenized trial C: `~/data/datasets/corpus-v1-trial-c/{claude-1124-16k,bpe-lines-1124-16k}`
  (3.06B and 2.87B training tokens), plus `claude-a-16k` and `bpe-lines-a-16k`
  references. Trial B: `~/data/datasets/corpus-v1-trial-b/`.
- Tokenizers were trained here in `~/workspace/tokenizer-20261124/` (runner
  `tokenizer/configs/run_corpus_v1_ws.sh` in the datasets repo). Bytes per
  token: 3.207 Claude-style, 3.425 BPE (BPE uses 6.3% fewer tokens, 7.5% fewer
  on python; equal on multilingual).
- Rules. Claude-style: word-start / word-end markers, implied spaces, a shift
  mark for capitals, camelCase and acronym splits, `_` always its own token,
  fewest-pieces segmentation. BPE lines: o200k-style regex, leading space on
  words, `_` kept with identifier parts, digits in right-aligned groups of 3,
  line breaks isolated.
- The tokenizer decision lives in the datasets repo; cl32k (used by `micro.yaml`
  and `nano.yaml`) is the older choice.

## Decontamination (paused at 42%)

Scripts: datasets repo `scripts/decontam-20261124/` (uncommitted on the Mac),
mirrored with outputs in `/home/roman/data/analysis/decontam-20261124/`.
- Benchmarks (18 repos, revisions in `fetch.json`):
  `/home/roman/data/datasets/benchmarks-20261124/`. HumanEval(+), MBPP(+),
  LiveCodeBench, BigCodeBench, APPS, CodeContests, CRUXEval, DS-1000, MultiPL-E,
  MATH/MATH-500, GSM8K, MMLU-Redux, MMLU-Pro, ARC, HellaSwag: 92,182 texts,
  8.74M word 13-grams.
- Rule: ignore 13-grams shared by 3 or more benchmark texts; flag a document
  holding at least half of a benchmark text's remaining 13-grams and at least 8
  of them. Output is a removal list by document ID; the corpus is not modified.
- State: 838 of 2,284 shards scanned (1,028 of 2,474 GB) when the corpus drive
  hung. Resumable:
  `cd /home/roman/data/analysis/decontam-20261124 && STEP=scan WORKERS=12 bash run.sh`,
  then `STEP=report bash run.sh` (writes `remove.parquet`, `report.json`).
  12 workers instead of 24 to go easier on the drive and leave CPU for training
  (24 workers slowed training by about 25%).
- Early finding: `dolmino-code-meta-reasoning` contains APPS problems and
  solutions (about 14% of its first shard).
- Proposed, not approved: HumanEval / MBPP bits per byte on reference solutions
  and pass@k sampling, after decontamination.

## Machine (roman-ws)

- 2× RTX PRO 6000 Blackwell (SM120), Threadripper 9960X (24 cores; use at most 24
  threads for CPU jobs), Ubuntu 24.04.5. The monitor is on GPU 1.
- GPU power limit: the user set 360 W on 2026-11-23; after the reboot it read
  420 W (default 600). Throughput depends on it, and the cards drift about 2.5%
  under a cap, so compare throughput with interleaved runs.
- Corpus drive `nvme0` (Crucial T710, firmware PBCR5103) hung under the scan on
  2026-11-23 23:30 and came back after a cold power-off. SMART clean: no media
  errors, no thermal events. A warm reboot did not recover it. Its fstab line
  had no `nofail`, which stalled boot without ssh.
- W&B is the local server behind Caddy: links are `https://roman-ws/...` (never
  `:8080`). Only an interactive shell has the API key (`bash -ic "..."`).
- Docker containers: `wandb-local`, `caddy`; training containers are named
  `mini2-<run>`.

## Open items

- Finish trial C and score it (above). Decide the tokenizer.
- Finish decontamination, review the removal list, decide how to apply it.
- Choose the token budget and LR schedule for a real micro (1.64B) run; full
  cl32k data is not on this machine (4 of 32 shards in
  `~/data/datasets/cl32k-python-english-50b-v2`), and a new tokenizer means
  re-tokenizing the corpus anyway.
- Loop: tune the message gates, decide when to enable it.
- Deferred review items: full-attention layers bypass LoRA adapters (matters
  only for adapter tuning); no BF16-vs-FP32 check on a trained checkpoint.
- For RL later: no vLLM/SGLang model class for `inkling2_dense`; context must
  grow past 4,096; keep RL in BF16.
- Datasets repo work (sampler, tokenizer runners, pickling fix, demo,
  decontamination) is uncommitted on the Mac; do not commit there without asking.

## Working conventions

- Plain prose, no bold-everywhere formatting, in chat and docs.
- `uv`, never pip. Commits signed off (`git commit -s`), Conventional Commits,
  no attribution lines. Do not push without being asked.
- Always log training to W&B and share the run link on every start, restart or
  status report.
- No auto-restart watchdogs. Detach long jobs (`setsid nohup`, `docker -d`).
- Keep experiment scaffolding minimal: env-var `run.sh`, one YAML, short README.
- Show the commands being run in chat.
