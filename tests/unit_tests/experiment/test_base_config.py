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

"""The experiment recipe resolves to the intended model, data and validation setup."""

from pathlib import Path

import pytest
import torch

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.datasets.llm.megatron_dataset import MegatronPretrainingConfig
from nemo_automodel.components.models.qwen3_8_flash_next_mini import Qwen3_8_FlashNextMiniForCausalLM
from nemo_automodel.recipes._typed_config import RecipeConfig

BASE_YAML = Path(__file__).resolve().parents[3] / "experiment" / "base.yaml"
TE_OPTIMIZER = "transformer_engine.pytorch.optimizers.FusedAdam"


@pytest.fixture
def recipe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> RecipeConfig:
    """Load base.yaml as the launcher would, with the CUDA-only optimizer swapped out."""
    monkeypatch.setenv("DATA_DIR", "/data")
    monkeypatch.setenv("OUTPUT_DIR", "/outputs")
    monkeypatch.setenv("RUN_NAME", "smoke")
    # Config loading resolves every _target_; TransformerEngine is not importable on CPU hosts.
    config = tmp_path / "base.yaml"
    config.write_text(BASE_YAML.read_text().replace(TE_OPTIMIZER, "torch.optim.AdamW"))
    return RecipeConfig(parse_args_and_load_config(str(config), argv=[]))


def test_model_block_builds_arm_b(recipe: RecipeConfig) -> None:
    """The base recipe is study arm B: GR on, FFN 6144, no QSA, no Engram."""
    config = recipe._raw.model.config.instantiate()
    with torch.device("meta"):
        model = Qwen3_8_FlashNextMiniForCausalLM(config)
    assert sum(p.numel() for p in model.parameters()) == 1_122_283_392
    text = config.text_config
    assert text.use_gr is True and text.use_qsa is False and text.ple_layer_ids == []
    assert text.vocab_size == 32768 and text.max_position_embeddings == 4096
    layers = model.model.language_model.layers.values()
    assert [layer.layer_type for layer in layers] == (["linear_attention"] * 3 + ["full_attention"]) * 4


def test_runtime_paths_and_heldouts_resolve(recipe: RecipeConfig) -> None:
    """Data, cache and output paths interpolate, and the three heldouts are named for checkpoint selection."""
    train = recipe.dataloader.dataset_config
    assert isinstance(train, MegatronPretrainingConfig)
    assert train.paths == "/data/train/*.bin" and train.splits_to_build == "train"
    assert train.index_mapping_dir == "/outputs/dataset-cache"
    assert recipe.dataloader.dataset_build_schedule.max_steps == 200
    assert recipe._raw.dataset.tokenizer.pretrained_model_name_or_path == "/data/tokenizer"

    loaders = recipe.validation_dataloaders
    assert set(loaders) == {"l2", "l3", "english"}
    assert recipe.checkpoint.best_metric_key in loaders
    for source, loader in loaders.items():
        dataset = loader.dataset_config
        assert dataset.paths == {"validation": [f"/data/validation-small/{source}"]}
        assert dataset.splits_to_build == "validation"
        assert dataset.trainer_limit_val_batches == 1.0 and isinstance(dataset.trainer_limit_val_batches, float)

    assert recipe.checkpoint.checkpoint_dir == "/outputs/smoke"
    assert recipe.wandb.name == "smoke" and recipe.wandb.extra["dir"] == "/outputs/wandb"
