# The 1.65B architecture

Written 2026-11-24. This records the model shape we chose for the next round of
training and why. The shape is decided; it has not been trained or timed yet.

## The choice in one line

Inkling 2 Mini's attention, residual and loop design at Mini's own width, with
the mixture-of-experts feed-forward replaced by a narrow dense one and a small
tied vocabulary. 1.648B parameters, counted by building the model.

## Parameters

| | Inkling 2 Mini | Chosen | Micro (running now) |
|---|---|---|---|
| Stored parameters | 26.39B | 1.648B | 1.609B |
| Hidden size | 2304 | 2304 | 2048 |
| Layers | 24 | 24 | 24 |
| Query heads x dim | 24 x 128 | 24 x 128 | 16 x 128 |
| Key-value heads x dim | 8 x 128 | 8 x 128 | 4 x 128 |
| Attention pattern | 5 local (window 1024) + 1 full, x4 | same | same |
| Positions | relative bias, 16 dims, extent 1024 | same | same |
| Short convolutions | kernel 4 | same | same |
| Residual path | attention residuals over blocks of 6 | same | same |
| Loop | layers 6-17, 3 passes | same | off |
| Feed-forward | 128 experts, top 8, width 1152 | dense SwiGLU 5632 | dense SwiGLU 8192 |
| Vocabulary | 201,024, untied | 16,384, tied | 16,384, tied |

Recipe: [miniattn_tok16k.yaml](miniattn_tok16k.yaml). Settings that differ from `micro_tok16k.yaml`:

```yaml
hidden_size: 2304
intermediate_size: 5632
num_attention_heads: 24
num_key_value_heads: 8
swa_num_attention_heads: 24
swa_num_key_value_heads: 8
loop_config: {core_start: 6, core_end: 18, num_passes: 3, gate_rank: 64, is_varloop: false}
```

Where the parameters go: feed-forward 0.94B (57%), attention about 0.64B (39%),
embedding 0.04B, loop gates 0.03B. In the running micro the feed-forward is 75%
and attention 23%.

## Why these values

Hidden size 2304 and heads 24 / 8. These are Mini's values. With them, every
part of the model the authors tuned (softcaps, clipping thresholds, gate rank,
relative-bias size) runs at the shape it was tuned for; the only substitution
is the feed-forward. It also makes query width (24 x 128 = 3072) larger than
the residual stream, which is a deliberate feature of Mini that micro dropped.
Eight key-value heads give each attention layer more distinct things to
retrieve, which is what a long-context model depends on. A side benefit:
hidden states and attention weights line up one to one with Mini, should we
want to distill from it or initialise from its weights.

Three loop passes. Mini uses three, and we decided to follow it without a
sweep over the pass count. The loop adds reasoning depth (48 layer applications
from 24 stored layers) for about 25M parameters.

Dense feed-forward. Mini stores 26B parameters to activate 3.5B per token. We
want a model that is 1.6B in total, so the experts are replaced by one dense
layer.

Feed-forward width 5632 (2.44 x hidden). Under a loop, feed-forward width in
the 12 core layers is paid three times in compute for every stored parameter,
so width is the expensive thing to keep. The feed-forward mostly holds
memorised knowledge; for code and math we judge reasoning steps and retrieval
to matter more, so parameters moved from the feed-forward into attention. 5632
is the width that lands the model on 1.65B. Mini's 4x active width comes from
routing 8 experts and is not a constraint a dense model has.

16k tied vocabulary. From the tokenizer work: both 16k tokenizers are in
training comparisons now, and a small tied embedding (38M) leaves the budget
for the layers.

## Alternatives considered

All with three passes; parameters counted with the model code.

| Variant | Parameters | Matrix work per token vs micro today |
|---|---:|---:|
| Micro shape, feed-forward 8192 | 1.634B | 2.0x |
| Micro shape, feed-forward 6144, 8 key-value heads | 1.383B | 1.65x |
| Mini attention, feed-forward 4608 | 1.478B | 1.8x |
| Mini attention, feed-forward 5632 (chosen) | 1.648B | 2.0x |
| Mini attention, feed-forward 6144 | 1.733B | 2.1x |

6144 is the usual SwiGLU ratio (8/3) for a 2304 hidden size but goes past 1.7B.
The 1.383B variant is the cheaper arm we will compare against.

## Costs we accept

- About twice the matrix work per token of the running micro. The real slowdown
  will be larger: the growth is in attention, our slowest component, and the
  relative-position lookup scales with the number of heads (16 to 24).
- Memory: three passes roughly triple the activations and the wider hidden size
  adds about 12%. We expect it not to fit a 98 GB GPU at a 4096 context without
  gradient checkpointing; a 141 GB H200 should hold it. Not measured.
- The attention cache at inference doubles with 8 key-value heads.

## What is not known yet

- Quality. No run of this shape exists. The reasons above are expectations.
- Speed and memory at three passes. Measured so far: two passes on the micro
  shape ran at 56% of no-loop speed. The loop's message gates still use
  untuned FP32 modules.
- Whether to train with the loop from the start or attach it to a trained
  no-loop model.
- Learning rate and clipping with the loop: core weights receive summed
  gradients from all three passes.

## Next steps

1. Short probe of speed and memory for this shape, once a GPU is free.
2. Compare it with the 1.383B arm on the 4B prefix of the mixed 32B sample
   (`/home/roman/ext/corpus-v1-mix-32b-20261124`), read at equal tokens and at
   equal training time.
