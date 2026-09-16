# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Student-sized Engram/PLE table construction.

The released Qwen3.8-Flash-Next table is 320,001,536 rows of width 160 -- around
40% of that model's weights and far larger than the ~0.9B student. This module
sizes an equivalent table for an arbitrary row budget: it picks one prime
modulus per hash head, packs the heads into a contiguous global row space, and
builds the PLE layer around them.

The PLE delta is ``gated_value + causal_conv(norm(gated_value))`` with
``gated_value = gate * value_proj(embeddings)``. Zeroing ``value_proj`` (and the
already zero-initialized convolution) therefore makes the whole block an exact
no-op while leaving its gradient non-zero, so Engram can be retrofitted onto a
trained checkpoint without perturbing it. See :func:`neutralize_ple_layer`.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_8_flash_next.engram import (
    QWEN3_8_FLASH_NEXT_LAYER_MULTIPLIERS,
    Qwen3_8_FlashNextEngramTableConfig,
    Qwen3_8_FlashNextNGramEmbedding,
    Qwen3_8_FlashNextPLELayer,
)

from .config import Qwen3_8_FlashNextMiniTextConfig

logger = logging.getLogger(__name__)


def _is_prime(value: int) -> bool:
    """Return whether ``value`` is prime by trial division.

    Args:
        value: Candidate integer.

    Returns:
        ``True`` when ``value`` is prime.
    """
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def _next_prime_after(value: int) -> int:
    """Return the smallest prime strictly greater than ``value``.

    Args:
        value: Exclusive lower bound.

    Returns:
        The smallest prime ``> value``.
    """
    candidate = max(int(value), 1) + 1
    if candidate <= 2:
        return 2
    if candidate % 2 == 0:
        candidate += 1
    while not _is_prime(candidate):
        candidate += 2
    return candidate


def build_ngram_head_spec(
    total_rows: int,
    ngram_size: int,
    heads_per_ngram: int,
    divisible_by: int = 128,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Size one hash head per n-gram order and pack them into a row space.

    Args:
        total_rows: Approximate global row budget across every head.
        ngram_size: Largest n-gram order, including the current token.
        heads_per_ngram: Hash heads per order from two through ``ngram_size``.
        divisible_by: Padding granularity for the returned global row count, so
            the table can be sharded evenly across an owner process group.

    Returns:
        A triple ``(head_vocab_sizes, head_offsets, padded_rows)``. Each head
        gets a distinct prime modulus and a contiguous, non-overlapping slice of
        the global row space; ``padded_rows`` is the allocated table height.

    Raises:
        ValueError: If the budget cannot give every head at least two rows.
    """
    if ngram_size < 2:
        raise ValueError(f"ngram_size must be at least 2, got {ngram_size}")
    if heads_per_ngram <= 0:
        raise ValueError(f"heads_per_ngram must be positive, got {heads_per_ngram}")
    if divisible_by <= 0:
        raise ValueError(f"divisible_by must be positive, got {divisible_by}")

    num_heads = (ngram_size - 1) * heads_per_ngram
    per_head = total_rows // num_heads
    if per_head < 2:
        raise ValueError(f"total_rows={total_rows} leaves fewer than two rows for each of {num_heads} heads")

    # Distinct primes keep two heads from colliding on identical hash residues.
    # The heads walk *upward* from the budget: head i takes the (i+1)-th prime
    # above per_head - 1. That is the rule the serving runtime applies, deriving
    # every size from a single ngram_vocab_size_base, so a checkpoint written
    # here reconstructs bit-identically there. Choosing primes downward instead
    # would give an equally valid table that no serving stack could rebuild.
    sizes: list[int] = []
    offsets: list[int] = []
    cursor = 0
    prime = per_head - 1
    for _ in range(num_heads):
        prime = _next_prime_after(prime)
        sizes.append(prime)
        offsets.append(cursor)
        cursor += prime
    padded_rows = ((cursor + divisible_by - 1) // divisible_by) * divisible_by
    return tuple(sizes), tuple(offsets), padded_rows


def build_ple_layer(
    config: Qwen3_8_FlashNextMiniTextConfig,
    backend: BackendConfig,
    dtype: torch.dtype,
) -> Qwen3_8_FlashNextPLELayer:
    """Build a student-sized Engram/PLE layer.

    Args:
        config: Student text config supplying ``ngram_table_rows``,
            ``ngram_size``, ``heads_per_ngram``, ``ple_embed_dim``,
            ``hidden_size``, ``hc_count``, ``ple_conv_kernel_size``,
            ``rms_norm_eps``, ``eos_token_id``, and ``initializer_range``.
        backend: Linear backend configuration for the PLE projections.
        dtype: Parameter dtype for the table and projections.

    Returns:
        A PLE layer whose forward maps HC state ``[batch, sequence, hc_count *
        hidden]`` and IDs ``[batch, sequence]`` to a delta of the same HC shape.
    """
    ngram_size = config.ngram_size
    heads_per_ngram = config.heads_per_ngram
    ngram_heads = (ngram_size - 1) * heads_per_ngram
    ple_embed_dim = config.ple_embed_dim
    if ple_embed_dim % ngram_heads != 0:
        raise ValueError(f"ple_embed_dim must divide evenly over {ngram_heads} n-gram heads, got {ple_embed_dim}")

    total_rows = config.ngram_table_rows
    sizes, offsets, padded_rows = build_ngram_head_spec(
        total_rows,
        ngram_size,
        heads_per_ngram,
        divisible_by=config.make_ngram_vocab_size_divisible_by,
    )
    head_dim = ple_embed_dim // ngram_heads
    table = Qwen3_8_FlashNextEngramTableConfig(
        num_embeddings=padded_rows,
        embedding_dim=head_dim,
        initializer_range=config.initializer_range,
    ).build(process_group=None, dtype=dtype)

    embedding = Qwen3_8_FlashNextNGramEmbedding(
        table,
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        eos_token_id=config.eos_token_id,
        layer_multipliers=QWEN3_8_FLASH_NEXT_LAYER_MULTIPLIERS[:ngram_size],
        ngram_heads_vocab_sizes=sizes,
        ngram_heads_offsets=offsets,
    )
    logger.info(
        "Built student Engram table: %d rows x %d dim (%.1f M params) across %d heads",
        padded_rows,
        head_dim,
        padded_rows * head_dim / 1e6,
        ngram_heads,
    )
    return Qwen3_8_FlashNextPLELayer(
        embedding,
        hidden_size=config.hidden_size,
        hc_count=config.residual_streams,
        ple_embed_dim=ple_embed_dim,
        conv_kernel_size=config.ple_conv_kernel_size,
        rms_norm_eps=config.rms_norm_eps,
        backend=backend,
        dtype=dtype,
    )


@torch.no_grad()
def neutralize_ple_layer(ple: Qwen3_8_FlashNextPLELayer) -> None:
    """Make a PLE layer an exact no-op without freezing it.

    Zeroes ``value_proj`` and the causal convolution, which drives both terms of
    the PLE delta to zero. The gradient with respect to ``value_proj`` stays
    non-zero because it is multiplied by the (non-zero) gate and table values,
    so the layer trains away from zero as soon as optimization starts. This is
    what lets Engram be retrofitted onto a trained checkpoint without changing
    its outputs.

    Args:
        ple: The PLE layer to neutralize in place.
    """
    nn.init.zeros_(ple.value_proj.weight)
    nn.init.zeros_(ple.conv1d.weight)
