"""
Custom model wrapper for Qwen3 to handle faithful evaluation parameters.
"""

from typing import Any, Dict, List, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM

try:
    from transformers import Qwen3Config, Qwen3ForCausalLM
except ImportError:
    # Fallback for environments where Qwen3 classes are unavailable.
    from transformers import Qwen2Config as Qwen3Config
    from transformers import Qwen2ForCausalLM as Qwen3ForCausalLM


class Qwen3CustomConfig(Qwen3Config):
    model_type = "qwen3_llama"


class Qwen3CustomForCausalLM(Qwen3ForCausalLM):
    config_class = Qwen3CustomConfig

    def __init__(self, config):
        super().__init__(config)
        config.model_type = "qwen3_llama"

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        gate_truthful_head: Optional[bool] = None,
        truthful_head: Optional[torch.Tensor] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
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
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        gate_truthful_head: Optional[bool] = None,
        truthful_head: Optional[torch.Tensor] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        return super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            gate_truthful_head=gate_truthful_head,
            truthful_head=truthful_head,
            hyperparams=hyperparams,
            **kwargs,
        )


AutoConfig.register("qwen3_llama", Qwen3CustomConfig)
AutoModelForCausalLM.register(Qwen3CustomConfig, Qwen3CustomForCausalLM)
