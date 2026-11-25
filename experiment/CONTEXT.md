# Context: dense Inkling2 coder (mini2), as of 2026-11-24

Handover from the Claude session `c6256e05` that ran on the Mac. That session's
transcript and memory are copied to
`~/.claude/projects/-home-roman-workspace-Automodel/` (`claude --resume`).
The runbook with commands and results is [README.md](README.md); this file adds
the state of the work, the decisions behind it and what is open.

## Where things are

- Branch `ralovets/exp/mini2`, 21 commits on main, head `d0bd1975a`. Not pushed
  to GitHub. The Mac is no longer used (2026-11-24): all work is on roman-ws.
  This checkout was on `dev/flash`; its uncommitted
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
- Decided 2026-11-24: use 3 passes, as Mini does; no sweep over the pass count.
  Not measured yet: speed and memory at 3 passes (estimate 40% of no-loop speed,
  about three times the activations; micro at 4,096 may not fit one 98 GB GPU).
  Still open: loop from scratch against attaching it later.
- Lead candidate chosen 2026-11-24 ("Mini attention"): hidden 2304, 24 query /
  8 key-value heads of dim 128 (local and full layers), dense feed-forward
  5632, 3 passes over layers 6-17, 16k vocabulary: 1.648B parameters (feed-
  forward 0.941B), counted with the model code. Everything else as micro.
  Matrix work per token is about 2.0 times today's no-loop micro. Not trained
  or timed yet; Recipe: `miniattn_tok16k.yaml` (1,647,834,624 parameters).
  Write-up: `ARCHITECTURE.md`. Comparison arm: micro shape with feed-forward
  6144 and 8 key-value heads, 3 passes (1.383B).

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

## Current experiment: micro on the decontaminated sample (running)

Started 2026-11-24 11:52 PST (container logs are in UTC: 19:52). Question as before (Claude-style 16k against BPE
16k), now with micro and decontaminated data. The user dropped the earlier
trials (2026-11-24): do not resume or compare against them unless asked.

[micro_tok16k.yaml](micro_tok16k.yaml): micro with the 16,384 vocabulary,
1,608,640,512 parameters. One run per GPU (`NPROC=1`), 14,000 updates of 65,536
tokens (917M tokens, 30% of one pass), cosine LR 6e-4 -> 6e-5 after 500 warmup,
validation and checkpoint every 1,000, 3 kept, consolidated export at the end.

| Run | GPU | Data | W&B |
|---|---|---|---|
| `micro-claude-1124-decontam` | 0 | `claude-1124-16k` | https://roman-ws/ralovets/corpus-v1-tokenizers/runs/srdnjwlx |
| `micro-bpe-1124-decontam` | 1 | `bpe-lines-1124-16k` | https://roman-ws/ralovets/corpus-v1-tokenizers/runs/21t31wuv |

- Data: `~/data/datasets/corpus-v1-decontam/<tokenizer>` (3.06B Claude-style and
  2.87B BPE training tokens), tokenized from
  `/home/roman/ext/corpus-v1-sample-20261124-decontam/` with the frozen heldout.
- Outputs and logs: `~/workspace/checkpoints/mini2-decontam/`; code snapshot of
  commit `e9e81b468` in its `source/`; `launch.sh` there holds the exact
  commands (run it as `bash -ic "bash launch.sh"` for the W&B key).
- Speed at the 420 W limit: about 13.3K tokens/s per GPU, 5.1 s per update,
  42.6 GiB; about 20.5 hours in total, so it should finish near 08:30 PST on
  2026-11-25.
- To resume after a stop, add `--wandb.id <id> --wandb.resume allow` to each
  command and copy the `.log` files aside first (`run.sh` truncates them).

## Earlier: tokenizer trial C (stopped, dropped)

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
- Mixed 32B sample for sweeps (2026-11-24, use this one for new runs):
  `/home/roman/ext/corpus-v1-mix-32b-20261124/`, text only, not tokenized. 21.2M
  documents, 95.7 GiB, 25.1B o200k tokens (about 32B with the 16k tokenizers),
  same mix and exclusions as the decontaminated sample, which it contains. One
  fixed random order in 32 JSONL shards (`train/`), so the first k shards are a
  sample with the mix of the whole; one shard is about 1B 16k-tokens. Same
  holdout and frozen heldout. Details and checks in its `README.md`. Shuffled
  with `tokenizer_lab.experiments.shuffle_sample` (datasets repo, uncommitted).
  `prepare_data.py` takes one `--train` JSONL, so it needs a small change (or a
  concatenated file) to tokenize shards.
- Decontaminated model sample (the two micro runs use it):
  `/home/roman/ext/corpus-v1-sample-20261124-decontam/`, see Decontamination.
- Trial C model sample (2026-11-24, final proportions with 2% multilingual):
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

## Decontamination (removal list applied to the model sample 2026-11-24)

Scripts and outputs: `/home/roman/data/analysis/decontam-20261124/` (`run.sh`,
`source/`). The datasets repo `~/workspace/training-datasets` has the scripts
under `scripts/decontam-20261124/` (commit `51abe77` has the current rule and the
spot-check scripts; `export.py` is in its working tree, uncommitted).
- Benchmarks (18 repos, revisions in `fetch.json`):
  `/home/roman/data/datasets/benchmarks-20261124/`. HumanEval(+), MBPP(+),
  LiveCodeBench, BigCodeBench, APPS, CodeContests, CRUXEval, DS-1000, MultiPL-E,
  MATH/MATH-500, GSM8K, MMLU-Redux, MMLU-Pro, ARC, HellaSwag: 92,182 texts,
  8.74M word 13-grams. APPS is the test split only.
- Scan: all 2,284 shards (2,474 GB) finished 2026-11-24 05:30. The resume after
  the drive hang ran with 12 workers in about 3 h 45 min, CPU-bound (about 5 MB/s
  per worker); the drive stayed clean. `STEP=report bash run.sh` rebuilds the
  list from `hits/` in about 90 seconds without rescanning.
- Rule (`source/report.py`): ignore 13-grams shared by 3 or more benchmark
  texts; flag a document holding at least half of a text's remaining 13-grams
  and at least min(8, remaining) of them. On top of that:
  - APPS/CodeContests solutions and MultiPL-E prompts need at least 80% and at
    least 13, with no short-text exception.
  - Problem statements and questions (texts without reference code) are also
    flagged at 30% with at least 20 matches. This catches a problem copied
    without its answer and statements with the LaTeX stripped.
  - A HumanEval or MBPP problem repeats across HumanEval+, MBPP sanitized, MBPP+
    and the MultiPL-E translations, so its prompt was ignored as shared text
    (HumanEval kept 47% of its 13-grams, mostly the reference solution). These
    texts are scored a second time with the copies of one problem counted once;
    a document is flagged when either scoring flags it.
  - Two texts are exempt from the 30% rule: BigCodeBench/922 (it contains the
    NLTK stopword list) and MMLU-Pro 4919 (the opening of A Tale of Two Cities).
- Result (`remove.parquet`, `report.json`): 535,349 of 1.53B documents, 2.17B
  of 1.81T o200k tokens (0.12%).
- APPS is most of it: 417K documents and 1.81B tokens, 17,088 of 22,110 APPS
  texts found. `opencodereasoning` loses 25% of its documents (1.41B tokens)
  and `dolmino-code-meta-reasoning` 15% (229M); no other source loses more than
  0.1%. Every HumanEval problem (20,442 documents) and every GSM8K problem
  (7,916) is in the corpus; also MBPP 611 of 974 texts (34,744 documents),
  LiveCodeBench 781 of 1,055 (21,292), CRUXEval 646 of 800, MATH 982 of 5,000,
  MMLU-Pro 1,166, MMLU-Redux 922, HellaSwag 184.
- How the rule got here, with the samples behind each step:
  - First list: 512,991 documents, 2.04B tokens (`remove-rule1.parquet`). A
    36-document sample (`spotcheck-20261124.json`) had 29 real copies and 7 weak
    code matches (C++ templates, `a, b, c = map(int, input().split())`), which
    led to the 80% rule for solutions.
  - Second list: 461,089 documents, 1.97B tokens (`remove-rule2.parquet`). 10 of
    10 sampled dropped documents were false matches
    (`spotcheck-dropped-20261124.json`). A review then found it missed HumanEval
    and MBPP prompts without the reference solution, and problem-only copies
    under 50%.
  - Current list: adds 74,260 documents (198M tokens) and drops none. In a
    sample of 51 added documents, 3 per benchmark
    (`spotcheck-added-20261124.json`), 44 are copies of the benchmark problem, 6
    hold the public text an item quotes without the question (wikiHow articles
    for HellaSwag, speeches for MMLU) and 1 is false (a bubble sort against
    MBPP 101).
  - The 30% rule on texts with reference code was tried and dropped: it flagged
    about 38K more documents for generic code (bubble sort, Miller-Rabin, a
    URL regex, an all-zeros matrix) and a Wikipedia sentence on the Collatz
    conjecture.
- Limits: only near-verbatim copies are found, not paraphrases or translations
  into prose. 910 benchmark texts have no usable 13-grams (mostly CRUXEval, ARC,
  MMLU-Redux). An MBPP prompt is one sentence, so an MBPP prompt without the
  reference solution is caught only in its docstring form (MultiPL-E). A partial
  copy of a DS-1000 or BigCodeBench prompt under 50% is not flagged.
- Applied: `source/export.py` writes `remove.jsonl` (source and text sha256 per
  flagged document), which `sample_corpus --exclude` reads. The decontaminated
  model sample is `/home/roman/ext/corpus-v1-sample-20261124-decontam/`, drawn
  like the trial C sample but also excluding `remove.jsonl` and the 2026-11-24
  tokenizer sample (the command is in its `report.json` `exclude` list; the
  first exclude is `tokenizer-corpus-v1-20261123/sample/train-a.jsonl`).
  - `train-a.jsonl`: 2,023,369 documents, 2.401B tokens, same domain mix. It
    shares 2,022,638 documents with the trial C split; 739 left (699 flagged,
    40 from budget shifts) and 731 came in. `holdout.jsonl`: 101,082 documents,
    33 flagged ones replaced. Neither holds a flagged document or a document of
    the tokenizer sample.
  - `heldout.jsonl` is the frozen trial C file, copied unchanged so scores stay
    comparable: it has no flagged document and no near-copy of the 731 new
    training documents. Rebuilding it reshuffles the draw (only 1,327 of 6,667
    documents in common), so do not rerun `build_heldout.py` there.
  - Tokenized as `~/data/datasets/corpus-v1-decontam/`. The corpus itself is not rewritten.
    The tokenizer sample was not decontaminated.
- Trial C and the baseline trained on the sample taken before this list, with
  699 flagged documents among 2.02M (0.03%), so the effect on their HumanEval /
  MBPP numbers is small; the tokenizer comparison is not affected.
- `opencodereasoning` and `dolmino-code-meta-reasoning` are kept, filtered; drop
  them whole only if APPS or CodeContests become headline evaluations.
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
- Score the two micro runs when they finish (`score_bpb.py`, bits per byte).
- Choose the token budget and LR schedule for a real micro (1.64B) run; full
  cl32k data is not on this machine (4 of 32 shards in
  `~/data/datasets/cl32k-python-english-50b-v2`), and a new tokenizer means
  re-tokenizing the corpus anyway.
- Loop: tune the message gates, decide when to enable it.
- Deferred review items: full-attention layers bypass LoRA adapters (matters
  only for adapter tuning); no BF16-vs-FP32 check on a trained checkpoint.
- For RL later: no vLLM/SGLang model class for `inkling2_dense`; context must
  grow past 4,096; keep RL in BF16.
- Datasets repo: `~/workspace/training-datasets` on roman-ws. The sampler,
  tokenizer runners, pickling fix, demo and decontamination scripts are
  committed there. Do not commit there without asking.

## Working conventions

- Plain prose, no bold-everywhere formatting, in chat and docs.
- `uv`, never pip. Commits signed off (`git commit -s`), Conventional Commits,
  no attribution lines. Do not push without being asked.
- Always log training to W&B and share the run link on every start, restart or
  status report.
- No auto-restart watchdogs. Detach long jobs (`setsid nohup`, `docker -d`).
- Keep experiment scaffolding minimal: env-var `run.sh`, one YAML, short README.
- Show the commands being run in chat.
