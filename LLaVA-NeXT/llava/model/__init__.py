import os

AVAILABLE_MODELS = {
    "llava_llama": "LlavaLlamaForCausalLM, LlavaConfig",
    "qwen3_llama": "Qwen3CustomForCausalLM, Qwen3CustomConfig",
    "qwen3_vl_instruct_llama": "Qwen3VLInstructCustomForCausalLM, Qwen3VLInstructCustomConfig",
    "qwen3_vl_thinking_llama": "Qwen3VLThinkingCustomForCausalLM, Qwen3VLThinkingCustomConfig",
    "internvl3_llama": "InternVL3CustomForCausalLM, InternVL3CustomConfig",
    "internlm3_llama": "InternLM3CustomForCausalLM, InternLM3CustomConfig",
    "llava_qwen": "LlavaQwenForCausalLM, LlavaQwenConfig",
    "llava_mistral": "LlavaMistralForCausalLM, LlavaMistralConfig",
    "llava_mixtral": "LlavaMixtralForCausalLM, LlavaMixtralConfig",
    # LLaVA-Med (Mistral) uses llava_mistral weights; do not add "llava_med" here — there is no language_model/llava_med.py.
    
    # "llava_qwen_moe": "LlavaQwenMoeForCausalLM, LlavaQwenMoeConfig",    
    # Add other models as needed
}

for model_name, model_classes in AVAILABLE_MODELS.items():
    try:
        exec(f"from .language_model.{model_name} import {model_classes}")
    except Exception as e:
        print(f"Failed to import {model_name} from llava.language_model.{model_name}. Error: {e}")
