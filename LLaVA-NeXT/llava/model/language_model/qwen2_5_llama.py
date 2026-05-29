"""
Custom model wrapper for Qwen2_5 to handle faithful evaluation parameters.
"""

import sys
import os
from typing import Optional, List, Union, Dict, Any

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GenerationMixin,
    PretrainedConfig,
)

# Add the LLaVA-NeXT directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

try:
    from transformers import Qwen2Config, Qwen2ForCausalLM
except ImportError:
    from transformers import PretrainedConfig as Qwen2Config
    from transformers import AutoModelForCausalLM as Qwen2ForCausalLM

class Qwen2_5CustomConfig(Qwen2Config):
    model_type = "qwen2_5_llama"

class Qwen2_5CustomForCausalLM(Qwen2ForCausalLM):
    config_class = Qwen2_5CustomConfig

    def __init__(self, config):
        super().__init__(config)
        config.model_type = "qwen2_5_llama"


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        gate_truthful_head: Optional[bool] = None,
        truthful_head: Optional[torch.Tensor] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):


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
			gate_truthful_head=gate_truthful_head,
			truthful_head=truthful_head,
			hyperparams=hyperparams,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        gate_truthful_head: Optional[bool] = None,
        truthful_head: Optional[torch.Tensor] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):


        # Call the parent generate method with standard parameters only
        return super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            # inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
			gate_truthful_head=gate_truthful_head,
			truthful_head=truthful_head,
			hyperparams=hyperparams,
            **kwargs,
        )

# Register the custom model with transformers
AutoConfig.register("qwen2_5_llama", Qwen2_5CustomConfig)
AutoModelForCausalLM.register(Qwen2_5CustomConfig, Qwen2_5CustomForCausalLM)
