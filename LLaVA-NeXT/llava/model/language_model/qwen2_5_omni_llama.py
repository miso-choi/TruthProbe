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
from transformers import Qwen2_5OmniForConditionalGeneration

# Import Qwen2_5OmniConfig - this should be available with the model
from transformers import Qwen2_5OmniConfig

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput


class Qwen2_5OmniCustomConfig(Qwen2_5OmniConfig):
    model_type = "qwen2_5_omni_llama"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "qwen2_5_omni_llama"


class Qwen2_5OmniCustomForCausalLM(Qwen2_5OmniForConditionalGeneration):
    config_class = Qwen2_5OmniCustomConfig

    def __init__(self, config):
        super().__init__(config)
        
        # configure default generation settings
        config.model_type = "qwen2_5_omni_llama"
        print(f"[DEBUG] Qwen2_5OmniCustomForCausalLM initialized with model_type: {config.model_type}")

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
        # Qwen2.5-Omni specific parameters
        return_audio: Optional[bool] = None,
        use_audio_in_video: Optional[bool] = None,
        thinker_do_sample: Optional[bool] = None,
        # Custom parameters for faithful evaluation
        mask_info: Optional[dict] = None,
        gate_truthful_head: Optional[bool] = None,
        hyperparams: Optional[dict] = None,
        truthful_head: Optional[torch.Tensor] = None,
        # num_visual_tokens: Optional[int] = None,
        # vis_start_idx: Optional[int] = None,
        **kwargs,
    ) -> Union[tuple, CausalLMOutputWithPast]:


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
             # Qwen2.5-Omni specific parameters
            return_audio=return_audio,
            use_audio_in_video=use_audio_in_video,
            thinker_do_sample=thinker_do_sample,
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
        # Qwen2.5-Omni specific parameters
        return_audio: Optional[bool] = None,
        use_audio_in_video: Optional[bool] = None,
        thinker_do_sample: Optional[bool] = None,
        # Custom parameters for faithful evaluation
        mask_info: Optional[dict] = None,
        gate_truthful_head: Optional[bool] = None,
        hyperparams: Optional[dict] = None,
        truthful_head: Optional[torch.Tensor] = None,
        # num_visual_tokens: Optional[int] = None,
        # vis_start_idx: Optional[int] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:


        # Call the thinker directly instead of the parent generate method
        result = self.thinker.generate(
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
            gate_truthful_head=gate_truthful_head,
            truthful_head=truthful_head,
            hyperparams=hyperparams,
            **kwargs,
        )
        return result


AutoConfig.register("qwen2_5_omni_llama", Qwen2_5OmniCustomConfig)
AutoModelForCausalLM.register(Qwen2_5OmniCustomConfig, Qwen2_5OmniCustomForCausalLM)
