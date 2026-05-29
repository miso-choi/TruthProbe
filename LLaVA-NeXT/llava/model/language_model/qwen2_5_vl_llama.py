#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM
from transformers import Qwen2_5_VLForConditionalGeneration

# Try to import Qwen2_5_VLConfig, fallback to base config if not available
try:
    from transformers import Qwen2_5_VLConfig
except ImportError:
    from transformers import PretrainedConfig as Qwen2_5_VLConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput


class Qwen2_5_VLCustomConfig(Qwen2_5_VLConfig):
    model_type = "qwen2_5_vl_llama"


class Qwen2_5_VLCustomForCausalLM(Qwen2_5_VLForConditionalGeneration):
    config_class = Qwen2_5_VLCustomConfig

    def __init__(self, config):
        super().__init__(config)
        
        # configure default generation settings
        config.model_type = "qwen2_5_vl_llama"

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        # inputs_embeds: Optional[torch.FloatTensor] = None,
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
        # Custom parameters for faithful evaluation
        mask_info: Optional[dict] = None,
        gate_truthful_head: Optional[bool] = None,
        hyperparams: Optional[dict] = None,
        truthful_head: Optional[torch.Tensor] = None,
        num_visual_tokens: Optional[int] = None,
        vis_start_idx: Optional[int] = None,
        **kwargs,
    ) -> Union[tuple, CausalLMOutputWithPast]:

        # Store custom parameters as instance variables for potential use
        if gate_truthful_head is not None:
            self._current_gate_truthful_head = gate_truthful_head
        if truthful_head is not None:
            self._current_truthful_head = truthful_head
        if hyperparams is not None:
            self._current_hyperparams = hyperparams
        if mask_info is not None:
            self._current_mask_info = mask_info

        # Call the parent forward method with standard parameters only
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            # inputs_embeds=inputs_embeds,
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
        # inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        # Custom parameters for faithful evaluation
        mask_info: Optional[dict] = None,
        gate_truthful_head: Optional[bool] = None,
        hyperparams: Optional[dict] = None,
        truthful_head: Optional[torch.Tensor] = None,
        num_visual_tokens: Optional[int] = None,
        vis_start_idx: Optional[int] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        # Store custom parameters as instance variables
        if gate_truthful_head is not None:
            self._current_gate_truthful_head = gate_truthful_head
        if truthful_head is not None:
            self._current_truthful_head = truthful_head
        if hyperparams is not None:
            self._current_hyperparams = hyperparams
        if mask_info is not None:
            self._current_mask_info = mask_info

        # Call the parent generate method with standard parameters only
        return super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            # inputs_embeds=inputs_embeds,
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


AutoConfig.register("qwen2_5_vl_llama", Qwen2_5_VLCustomConfig)
AutoModelForCausalLM.register(Qwen2_5_VLCustomConfig, Qwen2_5_VLCustomForCausalLM)
