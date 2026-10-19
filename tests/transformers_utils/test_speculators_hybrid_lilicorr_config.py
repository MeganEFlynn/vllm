# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig

pytestmark = pytest.mark.skip_global_cleanup


def _make_combined_config() -> dict:
    return {
        "speculators_model_type": "mamba_attn_hybrid",
        "architectures": ["MambaAttnHybridSpeculator"],
        "draft_vocab_size": 64,
        "mask_token_id": 63,
        "block_size": 8,
        "block_pattern": [
            "mamba",
            "mlp",
            "mamba",
            "attention",
            "mlp",
            "mamba",
            "attention",
            "mlp",
            "mamba",
            "attention",
            "mlp",
        ],
        "attn_kv_layer_ids": [17, 25, 34],
        "latent_fusion_layer_ids": [1, 9, 17, 25, 34],
        "markov_rank": 0,
        "enable_lilicorr": True,
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
            "algorithm": "mamba_attn_hybrid",
            "proposal_methods": [{"proposal_type": "greedy", "speculative_tokens": 7}],
            "default_proposal_method": "greedy",
            "verifier": {"name_or_path": "Qwen/Qwen3-4B", "architectures": []},
        },
    }


def test_combined_updater_keeps_hybrid_architecture() -> None:
    config = SpeculatorsConfig.extract_transformers_pre_trained_config(
        _make_combined_config()
    )

    # The combined checkpoint stays on the HSpec backbone: the architecture
    # must NOT be translated to LiLiCorrDraftModel (which would select the
    # DFlash backbone).
    assert config["architectures"] == ["MambaAttnHybridDraftModel"]
    assert config["latent_fusion_layer_ids"] == [1, 9, 17, 25, 34]
    assert config["eagle_aux_hidden_state_layer_ids"] == [1, 9, 17, 25, 34]

    dflash_config = config["dflash_config"]
    assert dflash_config["lilicorr_enabled"] is True
    assert dflash_config["block_size"] == 8
    assert dflash_config["mask_token_id"] == 63
    assert dflash_config["target_layer_ids"] == [0, 8, 16, 24, 33]
    for key, value in _make_combined_config().items():
        if key.startswith("lilicorr_"):
            assert dflash_config[key] == value


def test_combined_runs_on_the_dspark_runtime() -> None:
    spec = SpeculatorsConfig.build_vllm_speculative_config(_make_combined_config())

    assert spec["method"] == "mamba_attn_hybrid"
    assert spec["num_speculative_tokens"] == 7


def test_lilicorr_disabled_leaves_no_head_fields() -> None:
    outer_config = _make_combined_config()
    outer_config["enable_lilicorr"] = False
    outer_config.pop("lilicorr_candidate_topk")
    config = SpeculatorsConfig.extract_transformers_pre_trained_config(outer_config)

    dflash_config = config["dflash_config"]
    assert "lilicorr_enabled" not in dflash_config
    assert "block_size" not in dflash_config
    assert not [k for k in dflash_config if k.startswith("lilicorr_")]
