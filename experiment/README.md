# Experimental ~1B coding model

Scratch pretraining of a dense hybrid model to study four-branch Gated Residuals
(GR) and n-gram memory (Engram). This directory holds one recipe,
[base.yaml](base.yaml), and a Docker launcher, [run.sh](run.sh). Model code lives
in `nemo_automodel/components/models/qwen3_8_flash_next_mini`, and its config
class defaults are the geometry below.

| Setting | Value |
|---|---|
| Layers | 16; repeated 3 Gated DeltaNet then 1 full attention |
| Hidden / FFN | 2048 / 6144; dense SwiGLU |
| Full attention | 8 query / 2 KV heads, head dimension 256 |
| DeltaNet | 16 QK / 16 V heads, dimension 128, convolution 4, SiLU output gate |
| Residuals | Four-branch GR, rank 256; `use_gr: false` gives ordinary pre-norm residuals |
| Position / context | Partial RoPE (64 of 256 dimensions), theta 10M; 4096 tokens |
| Vocabulary / precision | cl32k, tied 32,768-entry embeddings; BF16 compute, FP32 Adam state |
| QSA / MTP / Engram | Off |

The base is a 200-step smoke schedule: LR 6e-4 with 32 warmup steps, global
batch 16 (65,536 tokens per update), validation every 100 updates, no
checkpoints. Pick a token budget and LR schedule before a long run.

## Setup and launch

Build the training image from the repository root:

```bash
docker build -f docker/Dockerfile -t automodel:flash \
  --build-arg BASE_IMAGE=pytorch \
  --build-arg INSTALL_FA3=false --build-arg INSTALL_FA4=false .
```

Download the whole dataset outside the checkout and verify it:

```bash
export DATA_DIR="$HOME/storage/datasets/cl32k-python-english-50b-split-v2-hf-uncompressed"
uvx --from huggingface_hub hf download ralovets/cl32k-python-english-50b-v2 \
  --repo-type dataset --revision c0601cec4e506527515c47a3c07268afb14186a2 \
  --local-dir "$DATA_DIR"
(cd "$DATA_DIR" && sha256sum -c SHA256SUMS)
```

`run.sh` takes a run name and optional recipe overrides. Machine settings are
environment variables with defaults at the top of the script: `IMAGE`, `NPROC`,
`DATA_DIR`, `OUTPUT_DIR`, `OMP_NUM_THREADS`. W&B credentials come from
`WANDB_API_KEY`, `WANDB_BASE_URL`, `WANDB_ENTITY`; set `WANDB_MODE=offline` when
no server is reachable.

```bash
bash experiment/run.sh smoke
NPROC=8 bash experiment/run.sh p5-smoke
bash experiment/run.sh gr-off --model.config.text_config.use_gr false
```

Inside the container the corpus is `/data` (read-only), outputs `/outputs`, and
the checkout `/opt/Automodel`. Metrics go to W&B and to JSONL files under
`OUTPUT_DIR/RUN_NAME/`; the console log is `OUTPUT_DIR/RUN_NAME.log`. The first
launch builds a dataset index cache under `OUTPUT_DIR/dataset-cache` (a few
hundred MB, about a minute). Global batch must divide by microbatch times rank
count. W&B records the YAML as given; model geometry comes from the config class
defaults, so read the parameter count from the log (1,122,283,392 for the base).

A short first check on two GPUs:

```bash
bash experiment/run.sh smoke --step_scheduler.max_steps 24 \
  --lr_scheduler.lr_warmup_steps 4 --step_scheduler.val_every_steps 12
```

A healthy run reaches step 0 within a minute or two, drops the training loss
from about 10.5 to about 8.1 over 24 updates, and reports finite l2, l3 and
english losses at steps 12 and 24.

## Dataset and tokenizer

[`ralovets/cl32k-python-english-50b-v2`](https://huggingface.co/datasets/ralovets/cl32k-python-english-50b-v2)
is about 102.5 GB of already-tokenized Megatron mmap binaries.

| Package path | Contents |
|---|---|
| `train/` | 32 shards; 49.7B training tokens |
| `validation/{l2,l3,english}` | 200M tokens total |
| `test/{l2,l3,english}` | 50M tokens total |
| `validation-small/{l2,l3,english}` | Fixed 5M-token subset of validation |
| `tokenizer/`, `provenance/`, `SHA256SUMS` | Tokenizer, split records, checksums |

The recipe blends all 32 training shards by size and packs documents into
4096-token sequences. Splits keep related source families together; use
validation for tuning and test only for final comparisons. The cl32k tokenizer
is byte-level BPE with cl100k-style splitting; IDs 0 to 4 are EOS/PAD/FIM
markers and the vocabulary is 32,768. Details and licensing are in the dataset
card.

## Validation and checkpoints

The three named heldouts each run one pass over `validation-small`. Checkpoint
selection uses the L2 loss. Changing the context length means updating
`seq_length` in all four dataset blocks and `max_position_embeddings`.

```bash
bash experiment/run.sh checkpoint-smoke \
  --checkpoint.enabled true --step_scheduler.ckpt_every_steps 100
bash experiment/run.sh resumed \
  --checkpoint.enabled true \
  --checkpoint.restore_from /outputs/checkpoint-smoke/epoch_0_step_99
```

Checkpoints include model, optimizer, scheduler, RNG and dataloader state; keep
the whole directory to resume. `max_steps` is the total target, not additional
steps. Keep dataset, seeds, batch size and rank count fixed for exact
continuation.

## Study arms

| Arm | GR | FFN width | Parameters | Overrides |
|---|---|---:|---:|---|
| A | Off | 6144 | 982,620,032 | `--model.config.text_config.use_gr false` |
| B | On, rank 256 | 6144 | 1,122,283,392 | none (base) |
| C | Off | 7552 | 1,121,032,064 | `--model.config.text_config.use_gr false --model.config.text_config.intermediate_size 7552` |

Screen the arms at LR 4e-4, 6e-4 and 1.2e-3 with identical schedules and data,
select on heldout loss, then confirm at 1B tokens with 2 to 3 seeds. Judge on
per-source heldout loss plus HumanEval+/MBPP+ pass@1; better training loss
alone does not establish better code.

The Engram arm adds token bigram/trigram memory at one-based layer 2:

```bash
--model.config.text_config.ple_layer_ids '[2]' \
--model.config.text_config.ngram_size 3 \
--model.config.text_config.heads_per_ngram 8 \
--model.config.text_config.ple_embed_dim 2048 \
--model.config.text_config.ngram_table_rows 7812500
```

Total parameters are 1,991,167,872 with GR off or 2,143,457,152 with GR on.

## Tests

```bash
uv run pytest -q tests/unit_tests/experiment tests/unit_tests/models/qwen3_8_flash_next_mini
```

`tests/functional_tests/models/qwen3_8_flash_next_mini/test_pretraining.py`
needs CUDA and covers all four GR/Engram combinations. Keep datasets, outputs
and scratch scripts outside the checkout.
