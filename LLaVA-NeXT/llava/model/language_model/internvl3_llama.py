"""
Custom compatibility wrapper for InternVL3-9B.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, PretrainedConfig


class InternVL3CustomConfig(PretrainedConfig):
    model_type = "internvl3_llama"


class InternVL3CustomForCausalLM(nn.Module):
    config_class = InternVL3CustomConfig

    def __init__(self, config: InternVL3CustomConfig, model: Optional[nn.Module] = None):
        super().__init__()
        self.config = config
        self.model = model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, config: Optional[PretrainedConfig] = None, **kwargs):
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        config.model_type = "internvl3_llama"
        model = AutoModel.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True, **kwargs)
        return cls(config=config, model=model)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        num_patches_list: Optional[list[int]] = None,
        gate_truthful_head: Optional[bool] = None,
        truthful_head: Optional[torch.Tensor] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if self.model is None:
            raise RuntimeError("Underlying model is not initialized.")
        try:
            return self.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,
                gate_truthful_head=gate_truthful_head,
                truthful_head=truthful_head,
                hyperparams=hyperparams,
                **kwargs,
            )
        except TypeError:
            return self.model.forward(**kwargs)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        num_patches_list: Optional[list[int]] = None,
        gate_truthful_head: Optional[bool] = None,
        truthful_head: Optional[torch.Tensor] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if self.model is None:
            raise RuntimeError("Underlying model is not initialized.")
        try:
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,
                gate_truthful_head=gate_truthful_head,
                truthful_head=truthful_head,
                hyperparams=hyperparams,
                **kwargs,
            )
        except TypeError:
            return self.model.generate(**kwargs)

    @torch.no_grad()
    def chat(
        self,
        tokenizer=None,
        pixel_values: Optional[torch.Tensor] = None,
        question: Optional[str] = None,
        generation_config: Optional[dict] = None,
        num_patches_list: Optional[list[int]] = None,
        history=None,
        return_history: Optional[bool] = None,
        gate_truthful_head: Optional[bool] = None,
        truthful_head: Optional[torch.Tensor] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if self.model is None:
            raise RuntimeError("Underlying model is not initialized.")
        try:
            return self.model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=generation_config,
                num_patches_list=num_patches_list,
                history=history,
                return_history=return_history,
                gate_truthful_head=gate_truthful_head,
                truthful_head=truthful_head,
                hyperparams=hyperparams,
                **kwargs,
            )
        except TypeError:
            return self.model.chat(**kwargs)


AutoConfig.register("internvl3_llama", InternVL3CustomConfig)
AutoModelForCausalLM.register(InternVL3CustomConfig, InternVL3CustomForCausalLM)
