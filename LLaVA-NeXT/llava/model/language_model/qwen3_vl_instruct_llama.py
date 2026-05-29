"""
Custom model wrapper for Qwen3-VL-Instruct to handle faithful evaluation parameters.
"""

from typing import Dict, Optional, Union

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

# Support multiple naming conventions across transformers versions.
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    try:
        from transformers import Qwen3_VLForConditionalGeneration as Qwen3VLForConditionalGeneration
    except ImportError:
        # Fallback keeps this module importable on older environments.
        from transformers import Qwen2_5_VLForConditionalGeneration as Qwen3VLForConditionalGeneration

try:
    from transformers import Qwen3VLConfig
except ImportError:
    try:
        from transformers import Qwen3_VLConfig as Qwen3VLConfig
    except ImportError:
        try:
            from transformers import Qwen2_5_VLConfig as Qwen3VLConfig
        except ImportError:
            from transformers import PretrainedConfig as Qwen3VLConfig


class Qwen3VLInstructCustomConfig(Qwen3VLConfig):
    model_type = "qwen3_vl_instruct_llama"


class Qwen3VLInstructCustomForCausalLM(Qwen3VLForConditionalGeneration):
    config_class = Qwen3VLInstructCustomConfig

    def __init__(self, config):
        super().__init__(config)
        config.model_type = "qwen3_vl_instruct_llama"

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        mask_info: Optional[dict] = None,
        gate_truthful_head: Optional[bool] = None,
        hyperparams: Optional[Dict] = None,
        truthful_head: Optional[torch.Tensor] = None,
        num_visual_tokens: Optional[int] = None,
        vis_start_idx: Optional[int] = None,
        **kwargs,
    ) -> Union[tuple, CausalLMOutputWithPast]:
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
            gate_truthful_head=gate_truthful_head,
            truthful_head=truthful_head,
            hyperparams=hyperparams,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        mask_info: Optional[dict] = None,
        gate_truthful_head: Optional[bool] = None,
        hyperparams: Optional[Dict] = None,
        truthful_head: Optional[torch.Tensor] = None,
        num_visual_tokens: Optional[int] = None,
        vis_start_idx: Optional[int] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        return super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
            gate_truthful_head=gate_truthful_head,
            truthful_head=truthful_head,
            hyperparams=hyperparams,
            **kwargs,
        )


AutoConfig.register("qwen3_vl_instruct_llama", Qwen3VLInstructCustomConfig)
AutoModelForCausalLM.register(Qwen3VLInstructCustomConfig, Qwen3VLInstructCustomForCausalLM)
