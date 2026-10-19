# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba-Attn-Hybrid (H-Spec) speculator.

Extends the DSpark speculator (which reuses the DFlash fill-in machinery)
with:

  * a latent seed: the fused verifier hidden states at the anchor position
    (``latent_fusion_layer_ids``) are passed to the draft model as the
    ``latent_seed`` forward argument, where they seed the mamba SSM states;
  * zero-copy KV sharing: the draft's attention sub-layers read their mapped
    verifier layer's KV cache (``attn_kv_layer_ids``) instead of allocating
    their own KV storage, so the draft never stores context KV;
  * no context-KV precompute: with KV sharing active,
    ``precompute_and_store_context_kv`` is a no-op on the draft model.

One block = 1 anchor + N proposals: the anchor is the bonus token (DFlash
``1 + N`` fill-in layout), it is not an 8th proposal. The latent seed is
gathered from the last committed (non-rejected) target token's hidden states
each step, so rejection feeds back into the next draft.
"""

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.mamba_attn_hybrid.utils import (
    load_mamba_attn_hybrid_model,
    wire_zero_copy_kv,
)
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


class MambaAttnHybridSpeculator(DSparkSpeculator):
    _speculator_name = "MambaAttnHybrid"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        hf_config = self.draft_model_config.hf_config

        # H-Spec uses the DFlash fill-in layout: 1 anchor (bonus) + N mask
        # tokens, with every position a prediction after the anchor.
        self.sample_from_anchor = False
        self.num_query_per_req = 1 + self.num_speculative_steps
        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )

        fusion_layer_ids = getattr(
            hf_config, "latent_fusion_layer_ids", None
        ) or getattr(hf_config, "eagle_aux_hidden_state_layer_ids", None)
        if not fusion_layer_ids:
            raise ValueError(
                "mamba_attn_hybrid needs latent_fusion_layer_ids (or "
                "eagle_aux_hidden_state_layer_ids) to size the latent seed."
            )
        self.num_fusion_layers = len(fusion_layer_ids)
        hidden = self.draft_model_config.get_hidden_size()

        # The latent seed replaces DSpark's pooled hidden-state buffer: the
        # draft consumes the *fused* aux hidden states directly.
        self.hidden_states = torch.zeros(
            self.max_num_tokens,
            self.num_fusion_layers * hidden,
            dtype=self.dtype,
            device=device,
        )
        self.latent_seed = torch.zeros(
            self.max_num_reqs,
            self.num_fusion_layers * hidden,
            dtype=self.dtype,
            device=device,
        )

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        if get_pp_group().world_size != 1:
            raise NotImplementedError(
                "mamba_attn_hybrid does not support pipeline parallelism."
            )
        model = load_mamba_attn_hybrid_model(target_model, self.vllm_config)
        # Wire draft attention sub-layers to their verifier KV layers. This
        # runs before get_kv_cache_spec(), so the shared layers contribute no
        # KV cache group and allocate no KV storage.
        kv_ids = wire_zero_copy_kv(model, target_model)

        if self.draft_logits is not None and model.draft_id_to_target_id is not None:
            d2t = model.draft_id_to_target_id
            self._d2t_scatter_index = (
                torch.arange(d2t.shape[0], device=d2t.device) + d2t
            )
            self._draft_scatter_buf = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits.dtype,
                device=self.device,
            )
            logger.info(
                "MambaAttnHybrid: reduced draft vocab (%d -> %d); probabilistic "
                "scatter map built",
                d2t.shape[0],
                self.vocab_size,
            )

        self.use_confidence_head = False

        logger.info(
            "MambaAttnHybrid: %d attention sub-layers <- verifier KV %s, F=%d "
            "fusion layers, zero-copy KV sharing enabled, draft sampling=%s",
            len(kv_ids),
            kv_ids,
            self.num_fusion_layers,
            "sequential Markov"
            if getattr(model, "has_markov_head", False)
            else "parallel",
        )
        return model

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
    ) -> None:
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        # V2 builds one attention metadata for the whole draft, so a recipe
        # mixing sliding and full attention is not supported yet.
        sliding = getattr(self.model, "sliding_attention_layer_names", set())
        if sliding and len(sliding) != len(self.model.model.attn_sublayers):
            raise NotImplementedError(
                "V2 builds one attention metadata for the whole draft, so a "
                "recipe mixing sliding and full attention is not supported yet "
                f"({len(sliding)} of {len(self.model.model.attn_sublayers)} "
                "sub-layers are sliding). Train with --sliding-window-indices "
                "covering all attention blocks, or none."
            )
        self._hybrid_all_sliding = bool(sliding)

    @property
    def attn_vllm_config(self) -> VllmConfig:
        # The draft's full-attention sub-layers read the verifier KV with a
        # non-causal (bidirectional over the block) query pattern.
        from copy import copy

        from vllm.config import replace

        config = copy(super().attn_vllm_config)
        config.attention_config = replace(config.attention_config, use_non_causal=True)
        return config

    def capture(self) -> None:
        self.latent_seed.zero_()
        super().capture()

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> torch.Tensor:
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
            is_padding=self.input_buffers.is_padding[:num_tokens],
        ):
            return self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                inputs_embeds=None,
                latent_seed=self.latent_seed,
            )

    def _prepare_graph_forward(
        self, num_reqs: int, attn_metadata: dict[str, Any] | None
    ) -> None:
        """The FULL graph's context step for the hybrid drafter.

        The draft stores no context KV (it reads the verifier's), so this only
        refreshes the captured decode attention metadata.
        """
        if attn_metadata is not None and self.supports_decode_metadata_update:
            self._update_draft_decode_metadata(attn_metadata, num_reqs)

    def _prepare_graph_replay(
        self,
        input_batch: InputBatch,
        num_graph_reqs: int,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_rejected: torch.Tensor,
        dummy_run: bool,
    ) -> None:
        """Eager work before a FULL replay: stage the latent seed.

        The hybrid drafter reads no context KV of its own, so nothing else
        needs storing; only the fused aux hidden states at the anchor row are
        gathered into the latent-seed buffer the graph reads.
        """
        num_reqs = input_batch.num_reqs
        if aux_hidden_states:
            fused = torch.cat(
                [h[: input_batch.num_tokens] for h in aux_hidden_states], dim=-1
            )
            anchor = (
                input_batch.query_start_loc[1 : num_reqs + 1]
                - 1
                - num_rejected[:num_reqs]
            )
            self.latent_seed[:num_reqs].copy_(fused[anchor])
        else:
            self.latent_seed.zero_()

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        # The Markov head path (when the checkpoint has one) matches DSpark;
        # otherwise fall back to DFlash's parallel sampling.
        if getattr(self.model, "has_markov_head", False):
            super()._generate_draft(
                num_reqs,
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
        else:
            # Without a Markov head, sample in parallel from the model output
            # hidden states (the DFlash path).
            from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
                DFlashSpeculator,
            )

            DFlashSpeculator._generate_draft(
                self,
                num_reqs,
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync_state=None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        # Gather the latent seed from the fused verifier hidden states at the
        # anchor (last committed token, incl. after rejection) position.
        if aux_hidden_states:
            num_reqs = input_batch.num_reqs
            fused = torch.cat(aux_hidden_states, dim=-1)
            anchor = (
                input_batch.query_start_loc[1 : num_reqs + 1]
                - 1
                - num_rejected[:num_reqs]
            )
            self.latent_seed[:num_reqs].copy_(fused[anchor])
        else:
            # Memory profiling / dummy path: no aux states available. Zero the
            # seed so shapes stay right; the outputs are discarded.
            self.latent_seed.zero_()
            last_hidden_states = last_hidden_states.new_zeros(
                last_hidden_states.shape[0], self.hidden_states.shape[1]
            )

        return super().propose(
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            dp_sync_state=dp_sync_state,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            mm_inputs=mm_inputs,
            is_profile=is_profile,
            num_speculative_tokens=num_speculative_tokens,
        )
