# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig

pytestmark = pytest.mark.skip_global_cleanup


def _make_lilicorr_config() -> dict:
    return {
        "speculators_model_type": "lilicorr",
        "architectures": ["LiLiCorrSpeculator"],
        "draft_vocab_size": 64,
        "mask_token_id": 63,
        "block_size": 16,
        "aux_hidden_state_layer_ids": [2, 10, 18, 26, 34],
        "sample_from_anchor": False,
        "sliding_window_non_causal": False,
        "lilicorr_candidate_topk": 8,
        "lilicorr_hidden_size": 1024,
        "lilicorr_factor_dim": 1024,
        "lilicorr_num_layers": 2,
        "lilicorr_num_heads": 8,
        "lilicorr_mlp_ratio": 2.0,
        "lilicorr_logit_scale": 8.0,
        "lilicorr_vector_eps": 1e-4,
        "transformer_layer_config": {
            "model_type": "qwen3",
            "hidden_size": 16,
        },
        "speculators_config": {
            "algorithm": "lilicorr",
            "proposal_methods": [{"proposal_type": "greedy", "speculative_tokens": 15}],
            "default_proposal_method": "greedy",
            "verifier": {"name_or_path": "Qwen/Qwen3-8B", "architectures": []},
        },
    }


def test_lilicorr_updater_targets_lilicorr_draft_model() -> None:
    outer_config = _make_lilicorr_config()
    config = SpeculatorsConfig.extract_transformers_pre_trained_config(outer_config)

    assert config["architectures"] == ["LiLiCorrDraftModel"]
    assert config["eagle_aux_hidden_state_layer_ids"] == [2, 10, 18, 26, 34]
    dflash_config = config["dflash_config"]
    assert dflash_config["target_layer_ids"] == [1, 9, 17, 25, 33]
    assert dflash_config["block_size"] == 16
    assert dflash_config["lilicorr_enabled"] is True
    for key, value in outer_config.items():
        if key.startswith("lilicorr_"):
            assert dflash_config[key] == value


def test_lilicorr_runs_on_the_dflash_runtime() -> None:
    spec = SpeculatorsConfig.build_vllm_speculative_config(_make_lilicorr_config())

    assert spec["method"] == "dflash"
    assert spec["num_speculative_tokens"] == 15
