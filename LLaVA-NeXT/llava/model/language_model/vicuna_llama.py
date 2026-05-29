"""
Custom model wrapper for Vicuna to handle faithful evaluation parameters.
This file extends the base AutoModelForCausalLM to accept custom parameters
for faithful evaluation without modifying the transformers library.
"""

import torch
from typing import Optional, Dict, Any, List, Union
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig

# Try to import specific config classes, fallback to generic ones if not available
try:
    from transformers import LlamaConfig, LlamaForCausalLM
except ImportError:
    from transformers import PretrainedConfig as LlamaConfig
    from transformers import AutoModelForCausalLM as LlamaForCausalLM

class VicunaCustomConfig(LlamaConfig):
    """Custom config class for Vicuna model wrapper."""
    model_type = "vicuna_llama"

class VicunaCustomForCausalLM(LlamaForCausalLM):
    """Custom model wrapper for Vicuna that handles faithful evaluation parameters."""
    config_class = VicunaCustomConfig

    def __init__(self, config):
        super().__init__(config)
        config.model_type = "vicuna_llama"
    
    def get_model(self):
        """Return the underlying model for embedding access."""
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
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
            inputs_embeds=inputs_embeds,
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
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        gate_truthful_head: Optional[bool] = None,
        truthful_head: Optional[torch.Tensor] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):

        # Get inputs_embeds if input_ids is provided
        if input_ids is not None:
            inputs_embeds = self.get_model().embed_tokens(input_ids)
        else:
            inputs_embeds = None

        # Call the parent generate method with standard parameters only
        return super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            gate_truthful_head=gate_truthful_head,
            truthful_head=truthful_head,
            hyperparams=hyperparams,
            **kwargs,
        )

# Register the custom model with transformers
AutoConfig.register("vicuna_llama", VicunaCustomConfig)
AutoModelForCausalLM.register(VicunaCustomConfig, VicunaCustomForCausalLM)
